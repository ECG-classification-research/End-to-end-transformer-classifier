"""Legacy ONNX export, INT8 constant storage and structural coverage checks."""
import copy
import json
import warnings
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from torch import nn
from .linear import QuantLinear, quantizers

METADATA_KEY = 'ecg_metadata'


def cpu_session(path, threads=1, profile=False, optimization='all'):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimization == 'all'
                                        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    options.enable_profiling = profile
    if profile:
        options.profile_file_prefix = str(Path(path).parent / 'ort_profile')
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def _fold_constants(graph):
    """Fold only constants/aliases and weight Q; never evaluate activation graphs."""
    values = {w.name: numpy_helper.to_array(w) for w in graph.initializer}
    retained = []
    for node in graph.node:
        result = None
        if node.op_type == 'Constant':
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if 'value' in attrs:
                result = numpy_helper.to_array(attrs['value'])
        elif node.op_type == 'Identity' and node.input[0] in values:
            result = values[node.input[0]]
        elif node.op_type == 'Cast' and node.input[0] in values:
            dtype = next(a.i for a in node.attribute if a.name == 'to')
            result = values[node.input[0]].astype(onnx.helper.tensor_dtype_to_np_dtype(dtype))
        elif node.op_type == 'QuantizeLinear' and all(name in values for name in node.input):
            weight, scale, zero = (values[name] for name in node.input)
            if zero.dtype != np.int8 or weight.ndim != 2:
                raise ValueError('Only constant S8 per-channel Linear weight Q folding is supported.')
            # Restricted symmetric weight range matches WeightFakeQuant.forward.
            result = np.clip(np.rint(weight / scale.reshape(-1, 1)) + zero.reshape(-1, 1), -127, 127).astype(np.int8)
        if result is not None:
            values[node.output[0]] = result
        else:
            retained.append(node)
    # Backward liveness removes unused FP32 weights/observer state and constant nodes.
    needed = {o.name for o in graph.output}
    live = []
    for node in reversed(retained):
        if any(o in needed for o in node.output):
            live.append(node)
            needed.update(node.input)
    del graph.node[:]
    graph.node.extend(reversed(live))
    del graph.initializer[:]
    graph.initializer.extend(numpy_helper.from_array(np.array(value), name)
                             for name, value in values.items() if name in needed)


def coverage_report(graph, expected, quantized):
    producers = {o: n for n in graph.node for o in n.output}
    consumers = {}
    constants = {w.name: numpy_helper.to_array(w) for w in graph.initializer}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    records, covered = [], set()
    def module_scope(node):
        # Legacy exporter preserves ModuleList/Sequential names as blocks.0
        # and may repeat a Sequential scope (ffn/ffn.0).
        scopes = []
        for part in node.name.strip('/').split('/')[:-1]:
            if scopes and part.startswith(scopes[-1]+'.'):
                scopes[-1] = part
            else:
                scopes.append(part)
        return '.'.join(scopes)
    for name in expected:
        candidates = [n for n in graph.node if module_scope(n) == name and n.op_type in ('Gemm', 'MatMul')]
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one exported Linear operation for {name}; got {len(candidates)}.')
        node = candidates[0]
        record = dict(module=name, node=node.name, op_type=node.op_type, quantized=False)
        if quantized:
            input_dq = producers.get(node.input[0])
            weight_dq = producers.get(node.input[1])
            if weight_dq is not None and weight_dq.op_type == 'Transpose':
                weight_dq = producers.get(weight_dq.input[0])
            output_nodes = consumers.get(node.output[0], [])
            # Three-dimensional Linear exports MatMul followed by bias Add.
            if len(output_nodes) == 1 and output_nodes[0].op_type == 'Add':
                output_nodes = consumers.get(output_nodes[0].output[0], [])
            output_q = next((n for n in output_nodes if n.op_type == 'QuantizeLinear'), None)
            input_q = producers.get(input_dq.input[0]) if input_dq is not None else None
            weight = constants.get(weight_dq.input[0]) if weight_dq is not None else None
            if (input_dq is None or input_dq.op_type != 'DequantizeLinear'
                    or input_q is None or input_q.op_type != 'QuantizeLinear'
                    or weight_dq is None or weight_dq.op_type != 'DequantizeLinear'
                    or weight is None or weight.dtype != np.int8 or output_q is None):
                raise ValueError(f'Incomplete activation/weight/output INT8 boundaries for {name}.')
            record.update(quantized=True, weight_shape=list(weight.shape), weight_dtype='int8')
            covered.add(node.name)
        records.append(record)
    unexpected = []
    if quantized:
        for node in graph.node:
            if node.op_type in ('Conv', 'MatMul', 'Gemm') and node.name not in covered:
                # Non-target arithmetic may consume a dequantized Linear output;
                # it must not itself receive a weight-DQ and output-Q boundary.
                if any(n.op_type == 'QuantizeLinear' for n in consumers.get(node.output[0], [])):
                    unexpected.append(node.name)
    if unexpected:
        raise ValueError(f'Unexpected quantized non-Linear operations: {unexpected}')
    return dict(linear_count=len(records), quantized_linear_count=len(covered),
                linear_layers=records, unexpected_quantized_ops=unexpected,
                runtime_kernel_verification='QDQ coverage only; profile on target Pi to verify fused integer kernels')


def export_onnx(model, path, metadata, examples):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    model = copy.deepcopy(model).cpu().eval()
    for q in quantizers(model):
        if q.observe or not q.fake or not bool(q.initialized):
            raise ValueError('Export requires calibrated frozen observers with fake quant enabled.')
    examples = examples.detach().cpu().to(torch.float32)
    expected = [name for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]
    quantized = metadata['method'] != 'fp32'
    if quantized and any(not isinstance(m, QuantLinear) for m in model.modules() if isinstance(m, nn.Linear)):
        raise ValueError('Every Linear must use static quantization for INT8 export.')
    try:
        torch.onnx.export(model, examples[:1], str(path), opset_version=17, dynamo=False,
                          input_names=['signals'], output_names=['logits'],
                          dynamic_axes={'signals': {0: 'batch'}, 'logits': {0: 'batch'}},
                          do_constant_folding=True, external_data=False)
        proto = onnx.load(path)
        _fold_constants(proto.graph)
        report = coverage_report(proto.graph, expected, quantized)
        embedded = copy.deepcopy(metadata)
        embedded['coverage'] = report
        onnx.helper.set_model_props(proto, {METADATA_KEY: json.dumps(embedded, allow_nan=False)})
        onnx.checker.check_model(proto)
        onnx.save(proto, path)
        # Verify mathematical export independently from target-specific fused kernels.
        session = cpu_session(path, optimization='disabled')
        optimized_session = cpu_session(path)
        checks = []
        for batch in sorted(set((1, len(examples)))):
            sample = examples[:batch]
            with torch.inference_mode():
                reference = model(sample).numpy()
            actual = session.run(['logits'], {'signals': sample.numpy()})[0]
            np.testing.assert_allclose(actual, reference, atol=2e-4 if quantized else 2e-5,
                                       rtol=2e-3 if quantized else 2e-4)
            checks.append(dict(batch=batch, max_abs_error=float(np.max(np.abs(actual-reference)))))
            optimized = optimized_session.run(['logits'], {'signals': sample.numpy()})[0]
            if not np.isfinite(optimized).all():
                raise ValueError('Optimized runtime produced non-finite logits.')
            checks[-1]['optimized_max_abs_error'] = float(np.max(np.abs(optimized-reference)))
            checks[-1]['optimized_within_export_tolerance'] = bool(np.allclose(optimized, reference,
                                                                              atol=2e-4, rtol=2e-3))
        report['parity_session_optimization'] = 'disabled'
        report['parity_checks'] = checks
        if not all(c['optimized_within_export_tolerance'] for c in checks):
            warnings.warn('ONNX export parity passed, but optimized host kernels differ from QDQ reference. '
                          'Inspect quantization_report.json and measure quality on the target Pi. '
                          'U8S8 saturation can occur on x86 AVX2/AVX512; do not use host accuracy as Pi accuracy.',
                          RuntimeWarning)
        metadata['coverage'] = report
        # Embed the final report, including optimized-vs-reference diagnostics.
        embedded['coverage'] = report
        onnx.helper.set_model_props(proto, {METADATA_KEY: json.dumps(embedded, allow_nan=False)})
        onnx.save(proto, path)
        return report
    except Exception:
        if path.exists():
            path.unlink()  # Only the incomplete artifact exclusively created above.
        raise
