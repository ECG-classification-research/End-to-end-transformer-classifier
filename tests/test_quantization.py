"""CPU regression tests for Linear-only static PTQ/QAT."""
import copy
import tempfile
import unittest
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from model import ECGClassifier
from quantization.attention import explicit_model
from quantization.linear import prepare_qat, calibrate, freeze_observers, QuantLinear
from quantization.export import export_onnx
from evaluate_pi.split_io import CLASS_NAMES, calibration_indices, load_split, sha256, validate_metadata
from utils import save_checkpoint

torch.set_num_threads(1)


def fixture(folder, length=300):
    rng = np.random.default_rng(42)
    labels = np.tile(np.arange(5), 6)
    original = np.arange(20)
    split = dict(segments=rng.normal(0, .2, (30, length)).astype('float32'), labels=labels,
                 raw_labels=np.array(list('NAVFQ') * 6), original_train_indices=original,
                 train_indices=np.tile(original, 2), test_indices=np.arange(20, 30),
                 kept_indices=np.arange(30), records=np.array(['100']), class_names=np.array(CLASS_NAMES))
    path = folder / 'beat_split.npz'
    np.savez_compressed(path, **split)
    model_args = dict(more_cnn=True, transformer_blocks=1, dropout=.2, max_length=100)
    model = ECGClassifier(**model_args)
    config = dict(model_args=model_args, mode='fixed' if length == 300 else 'variable',
                  class_names=list(CLASS_NAMES), split_sha256=sha256(path), split_file=path.name,
                  split='stratified beat-wise 80/20')
    checkpoint = folder / 'model.pt'
    save_checkpoint(checkpoint, model, torch.optim.Adam(model.parameters()), 1, config, {})
    return model, split, path, checkpoint


class QuantizationTests(unittest.TestCase):
    def test_qat_train_mode_keeps_dropout_and_freezes_bn(self):
        from QAT.finetune import train_mode
        model = prepare_qat(explicit_model(ECGClassifier(transformer_blocks=1)))
        train_mode(model)
        self.assertTrue(all(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)))
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)))
        self.assertTrue(all(not p.requires_grad for p in model.cnn_encoder.parameters()))

    def test_attention_equivalence(self):
        for length in (300, 400):
            original = ECGClassifier(transformer_blocks=2).eval()
            converted = explicit_model(original).eval()
            for batch in (1, 3):
                x = torch.randn(batch, 1, length)
                torch.testing.assert_close(original(x), converted(x), atol=2e-6, rtol=2e-5)
            converted(x).sum().backward()
            self.assertIsNotNone(converted.transformer_encoder.blocks[0].attention.q_proj.weight.grad)
            self.assertIsInstance(original.transformer_encoder.blocks[0].attention, torch.nn.MultiheadAttention)

    def test_train_only_calibration_and_hash(self):
        with tempfile.TemporaryDirectory() as t:
            _, split, path, _ = fixture(Path(t))
            loaded = load_split(path)
            idx = calibration_indices(loaded, 10, 42)
            self.assertEqual(len(np.unique(idx)), 10)
            self.assertTrue(np.isin(idx, split['original_train_indices']).all())
            self.assertEqual(set(loaded['labels'][idx]), set(range(5)))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                validate_metadata(loaded, path, dict(split_sha256='wrong'))
            split['train_indices'][0] = 25
            np.savez_compressed(path, **split)
            with self.assertRaises(ValueError):
                load_split(path)

    def test_qat_freeze_and_export(self):
        from quantization.checkpoint import load_source, build_metadata
        import onnxruntime as ort
        import onnx
        for length in (300, 400):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as t:
                folder = Path(t)
                _, split, path, cp = fixture(folder, length)
                model, config, loaded = load_source(cp, path)
                model = prepare_qat(model)
                self.assertEqual(sum(isinstance(m, QuantLinear) for m in model.modules()), 11)
                attention = model.transformer_encoder.blocks[0].attention
                self.assertIs(attention.q_proj.input_quant, attention.k_proj.input_quant)
                self.assertIs(attention.q_proj.input_quant, attention.v_proj.input_quant)
                indices = calibration_indices(loaded, 10, 42)
                x = torch.from_numpy(split['segments'][indices, None])
                calibrate(model, [x])
                before = copy.deepcopy(model.cnn_encoder.state_dict())
                from QAT.finetune import train_mode
                train_mode(model)
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
                optimizer.zero_grad()
                torch.nn.functional.cross_entropy(model(x), torch.from_numpy(split['labels'][indices])).backward()
                optimizer.step()
                for name, tensor in before.items():
                    torch.testing.assert_close(tensor, model.cnn_encoder.state_dict()[name], rtol=0, atol=0)
                freeze_observers(model)
                model.eval()
                meta = build_metadata(model, config, cp, path, 'qat')
                target = folder / 'qat.onnx'
                report = export_onnx(model, target, meta, x[:3])
                self.assertEqual(report['quantized_linear_count'], 11)
                self.assertEqual(report['unexpected_quantized_ops'], [])
                graph = onnx.load(target)
                self.assertTrue(any(w.data_type == onnx.TensorProto.INT8 for w in graph.graph.initializer))
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                options.intra_op_num_threads = 1
                session = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
                for batch in (1, 3):
                    actual = session.run(None, {'signals': x[:batch].numpy()})[0]
                    np.testing.assert_allclose(actual, model(x[:batch]).detach().numpy(), atol=2e-4, rtol=2e-3)

    def test_full_cli_and_torch_free_onnx(self):
        from PTQ.quantize import main as ptq_main
        from QAT.finetune import main as qat_main, load_qat_checkpoint
        from evaluate_pi.evaluate import main as evaluate_main
        from evaluate_pi.adapters import create_adapter
        for length in (300, 400):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as t:
                folder = Path(t)
                _, split, path, cp = fixture(folder, length)
                common = ['--checkpoint_path', str(cp), '--split_path', str(path), '--calibration_samples', '10', '--batch_size', '7']
                ptq_main([*common, '--output_path', str(folder/'ptq')])
                qat_main([*common, '--output_path', str(folder/'qat'), '--epochs', '1', '--device', 'cpu'])
                restored, payload = load_qat_checkpoint(folder/'qat'/'qat_checkpoint.pt')
                from quantization.export import cpu_session
                x = split['segments'][:3, None]
                session = cpu_session(folder/'qat'/'qat_int8.onnx', optimization='disabled')
                np.testing.assert_allclose(session.run(None, {'signals': x})[0], restored(torch.from_numpy(x)).detach().numpy(), atol=2e-4, rtol=2e-3)
                self.assertTrue(set(payload['config']['calibration_indices']).issubset(set(split['original_train_indices'])))
                # Rename suffix: auto detection must examine container, not filename.
                renamed = cp.with_suffix('.zip')
                cp.rename(renamed)
                self.assertEqual(create_adapter(renamed).predict(x).shape, (3, 5))
                with self.assertRaises(ValueError):
                    create_adapter(path)
                for name, model_path in [('torch', renamed), ('fp32', folder/'ptq'/'fp32.onnx'),
                                         ('ptq', folder/'ptq'/'ptq_int8.onnx'), ('qat', folder/'qat'/'qat_int8.onnx')]:
                    output = folder / ('eval_'+name)
                    evaluate_main(['--model_path', str(model_path), '--split_path', str(path),
                                   '--output_path', str(output), '--batch_size', '3', '--warmup', '1', '--benchmark_runs', '3'])
                    metrics = json.loads((output/'metrics.json').read_text())
                    self.assertEqual(metrics['samples'], 10)
                    self.assertEqual(np.load(output/'probabilities.npy').shape, (10, 5))
                    self.assertTrue((output/'confusion_matrix.png').is_file())
                    with self.assertRaises(FileExistsError):
                        evaluate_main(['--model_path', str(model_path), '--split_path', str(path), '--output_path', str(output)])
                # New process with torch/WFDB imports actively rejected.
                code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','wfdb'):
   raise RuntimeError('Blocked deployment dependency: '+fullname)
sys.meta_path.insert(0,Block())
from evaluate_pi.evaluate import main
main(sys.argv[1:])
assert 'torch' not in sys.modules and 'wfdb' not in sys.modules
'''
                result = subprocess.run([sys.executable, '-c', code, '--model_path', str(folder/'ptq'/'ptq_int8.onnx'),
                                         '--split_path', str(path), '--output_path', str(folder/'no_torch'),
                                         '--warmup', '0', '--benchmark_runs', '1'],
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_cli_and_empty_calibration(self):
        from QAT.finetune import parse_args
        from quantization.linear import freeze_observers
        with self.assertRaises(SystemExit):
            parse_args(['--checkpoint_path', 'x.pt', '--output_path', 'x'])
        model = prepare_qat(explicit_model(ECGClassifier(transformer_blocks=1)))
        with self.assertRaises(ValueError):
            calibrate(model, [])
        with self.assertRaises(ValueError):
            freeze_observers(model)


if __name__ == '__main__':
    unittest.main()
