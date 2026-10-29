# Linear INT8: FP32, static PTQ and static QAT

Target: Raspberry Pi 4, Linux **64-bit/aarch64**, CPUExecutionProvider.
Train/export on a host; evaluate on the Pi. Host tests do not establish Pi latency.
Run commands below from the repository root. Every output directory must be new.

## Host setup

```powershell
python -m pip install -r requirements-quantization.txt
```

CPU-tested host versions: Python 3.13, torch 2.11.0+cpu, ONNX 1.23.2,
ONNX Runtime 1.31.0. Export uses the legacy PyTorch ONNX exporter (`dynamo=False`)
with opset 17 because it supports the explicit fake-quant QDQ boundaries. Batch is
dynamic; signal length is fixed by the checkpoint (300 or 400). Export tracing
warnings about fixed dimensions are expected; tests check multiple batch sizes.

## PTQ: no weight updates

```powershell
python -m PTQ.quantize --checkpoint_path outputs/run/model.pt --split_path outputs/run/beat_split.npz --output_path outputs/ptq
```

Produces `fp32.onnx`, `ptq_int8.onnx`, `metadata.json` (per-artifact manifest),
and `quantization_report.json`. Calibration uses only unique
`original_train_indices`: class-balanced selection up to 1024 beats, seed 42,
without replacement. Short classes contribute all available samples and the
remaining quota comes from other train classes. Change with
`--calibration_samples`, `--seed`, `--batch_size` and `--threads`.

PTQ and QAT deliberately share a MinMax observer implementation and exporter.
PTQ runs a calibration pass with fake quant disabled, freezes scales, then exports;
it does not use an optimizer or update model weights. This is static PTQ, not dynamic
quantization. No ONNX Runtime quantize_dynamic/second re-quantization step is needed.

## QAT: fine-tune from the original FP32 checkpoint

```powershell
python -m QAT.finetune --checkpoint_path outputs/run/model.pt --split_path outputs/run/beat_split.npz --output_path outputs/qat --epochs 10 --lr 1e-5 --batch_size 32 --device cuda
```

Use `--device cpu` or `auto` as appropriate. Defaults: seed 42, one CPU thread,
augmentation off. All Linear weights/biases may update; other parameters and
BatchNorm running statistics are frozen. Dropout remains active throughout the
model during training. Bootstrap observers from train calibration, then update
running MinMax in early epochs with fake quant active. The final ceil(20%) epochs
(at least the final epoch) freeze observers. Select the **last epoch**, without test
evaluation, validation-based selection or early stopping.

Outputs: `qat_int8.onnx`, `qat_checkpoint.pt`, `history.csv`, `config.json`,
`metadata.json`, `quantization_report.json`. The `.pt` is a fake-quant training state
with its own `ecg_linear_qat_v1` schema, not an integer inference artifact. Load it
for further work via `QAT.finetune.load_qat_checkpoint`; optimizer state is preserved.
The CLI currently starts from FP32 only; there is no CLI resume option.

Both commands **require** `--split_path`; its SHA-256 must match the source FP32
checkpoint. Train/test partition and sampled indices are validated. Observer updates,
gradients and export examples use train only. Test data are only used by evaluation.
This preserves the original beat-wise split; it does not remove patient overlap or
whole-record preprocessing already present in the original FP32 experiment.

## Quantization scope

- Linear projections Q, K, V and attention output; FFN; head Dense/residual/output.
- Activation U8 [0,255], per-tensor asymmetric; weight S8 [-127,127], per-output-channel
  symmetric. MinMax observers include zero in activation ranges.
- Every target Linear has input/weight/output QDQ boundaries. Q/K/V share the input
  observer/scale. Bias is float32. CNN, normalization, residual arithmetic, softmax
  and attention matrix multiplications stay float operators in the exported graph.
- Weights are stored as INT8 initializers with DequantizeLinear, not as FP32 weights
  with fake-quant nodes. Graph checks require full Linear coverage and no explicit
  non-target quantization boundaries.
- ONNX Runtime may fuse QDQ patterns, including operations whose two inputs already
  have quantized values (such as QK). Complexity metadata describes the exported
  logical graph; it is not a claim about exact CPU instructions. Inspect runtime
  profiling to see which integer kernels actually ran on the target.

The exporter checks ONNX validity, all target weights/boundaries and PyTorch parity
with **graph optimization disabled**, for batch 1 and a larger train batch. It also
checks optimized inference for finite logits and reports optimized error separately.
Quantized floating reference and integer fused kernels need not match bit-for-bit.

U8S8 kernels can exhibit saturation on some x86 AVX2/AVX512 CPUs. Host optimized
accuracy can therefore differ noticeably from the QDQ reference. ARM has a different
kernel path. Export warns if optimized host parity exceeds tolerance; this is recorded
in `quantization_report.json`, not silently treated as a mathematical export pass.
Evaluate accuracy on the Pi before concluding how quantization affects ECG performance.
See [ONNX Runtime datatype guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection).

## Pi setup and evaluate

Copy the repository code, three ONNX models and the **same** `beat_split.npz` to the Pi.
ONNX metadata is embedded, so the JSON sidecar is not required for inference.

```bash
python -m pip install -r evaluate_pi/requirements.txt
python -m evaluate_pi.evaluate --model_path outputs/ptq/fp32.onnx --split_path outputs/run/beat_split.npz --output_path outputs/eval_fp32
python -m evaluate_pi.evaluate --model_path outputs/ptq/ptq_int8.onnx --split_path outputs/run/beat_split.npz --output_path outputs/eval_ptq
python -m evaluate_pi.evaluate --model_path outputs/qat/qat_int8.onnx --split_path outputs/run/beat_split.npz --output_path outputs/eval_qat
```

Torch is optional and only needed for the source FP32 checkpoint adapter:

```bash
python -m evaluate_pi.evaluate --model_path outputs/run/model.pt --backend torch --split_path outputs/run/beat_split.npz --output_path outputs/eval_torch
```

`--backend auto` recognizes PyTorch ZIP contents even if the suffix is `.zip`;
an NPZ split passed as model is rejected. `numpy.load`/`torch.load` also read the
repo's `outputs/beat_split.zip` and `outputs/model.zip` directly by format, so renaming
or extracting those files is unnecessary. Arbitrary third-party ONNX without this
pipeline's metadata/contract is rejected; QAT training checkpoints are not source FP32.

Defaults: metric batch 64; timing batch **1**, one thread, 50 warm-up and 1000 measured
calls. Options: `--batch_size`, `--threads`, `--warmup`, `--benchmark_runs`.
Use the same settings and thermal/power conditions for all three ONNX runs.
Timing includes `adapter.predict` (NumPy/tensor wrapping and synchronous runtime call),
with inputs prepared in RAM. It excludes model/NPZ loading, preprocessing, metric
calculation, file writing and profiling. Test rows are traversed in saved order;
benchmark uses the first min(test size, runs) rows and cycles if necessary.

`--optimization all` is the default; `--optimization disabled` is a diagnostic QDQ
reference path, **not evidence of integer-kernel speedup**. Runtime settings are saved
alongside every result. For integer-kernel evidence, add `--profile`: one extra session
runs after timing and writes `ort_profile*.json`, so profile instrumentation does not
inflate reported latency. Look for QGemm/QLinearMatMul/MatMulIntegerToFloat or equivalent
target kernels; QDQ graph presence alone does not prove hardware acceleration.

## Reports

- Existing classification outputs: `metrics.json`, `summary_metrics.csv`,
  `per_label_metrics.csv`, `predictions.csv`, `probabilities.npy`, `predictions.npy`,
  `confusion_matrix.csv`, `confusion_matrix.png`.
- `benchmark.json`: mean/median/p95/std ms/sample, derived samples/sec, host/runtime info.
- `latencies.csv`: each timed call and its source segment index.
- `model_info.json`: model provenance, class order, quantization coverage, runtime config.

Classification definitions match test.py, including weighted aggregate P/R/F1,
macro F1, per-class one-vs-rest accuracy, ROC-AUC and undefined AUC/kappa as null.
The common runner computes stable cross-entropy from logits, never from clipped probabilities.

`parameters_total` counts original trainable architecture elements (including frozen
ones), excluding buffers/optimizer/observers. `parameters_quantized` counts INT8 Linear
weight elements, not float bias. Quantization does not reduce parameter count.
`model_size_bytes` is the actual file size: compare inference-only ONNX to ONNX;
source `.pt` may contain optimizer/config and is not an equivalent size baseline.

MACs cover Conv, Linear and QK/AV attention matrix products at batch 1. Two operations
per MAC yield `flops_equivalent_per_sample`; this is a **floating arithmetic equivalent**,
not actual floating-point execution for INT8. FP32/INT8 equivalent operation counts
refer to logical exported operator scope. Bias, normalization, activations, softmax,
pooling, residual and Q/DQ overhead are excluded and listed in metadata.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover fixed/variable lengths, dynamic batch, partial metric batches, QAT gradient
and frozen CNN/BN, train-only calibration, checkpoint round-trip, export coverage,
FP32 parity, CLI artifacts, and ONNX evaluation with torch/WFDB imports blocked.
