"""Calibrate static Linear-only INT8 without updating FP32 weights."""
import argparse
from pathlib import Path
import numpy as np
import torch
from evaluate_pi.split_io import calibration_indices, save_json
from quantization.checkpoint import load_source, build_metadata
from quantization.linear import prepare_qat, calibrate
from quantization.export import export_onnx
from QAT.finetune import calibration_batches
from utils import set_seed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint_path', 'split_path', 'output_path'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--calibration_samples', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    if min(args.calibration_samples, args.batch_size, args.threads) <= 0 or args.seed < 0:
        parser.error('Counts/threads must be positive; seed must be non-negative.')
    if args.output_path.exists():
        raise FileExistsError(args.output_path)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    model, config, split = load_source(args.checkpoint_path, args.split_path)
    indices = calibration_indices(split, args.calibration_samples, args.seed)
    examples = next(calibration_batches(split, indices, min(3, len(indices))))
    args.output_path.mkdir(parents=True, exist_ok=False)
    fp32_metadata = build_metadata(model, config, args.checkpoint_path, args.split_path, 'fp32')
    fp32_report = export_onnx(model, args.output_path / 'fp32.onnx', fp32_metadata, examples)
    quantized = prepare_qat(model)
    calibrate(quantized, calibration_batches(split, indices, args.batch_size))
    metadata = build_metadata(quantized, config, args.checkpoint_path, args.split_path, 'ptq')
    metadata['calibration'] = dict(seed=args.seed, indices=indices.tolist(), samples=len(indices),
                                  class_counts=np.bincount(split['labels'][indices], minlength=5).tolist())
    report = export_onnx(quantized, args.output_path / 'ptq_int8.onnx', metadata, examples)
    save_json(args.output_path / 'metadata.json', {'fp32.onnx': fp32_metadata, 'ptq_int8.onnx': metadata})
    save_json(args.output_path / 'quantization_report.json', {'fp32': fp32_report, 'ptq': report})
    print(f'FP32 and static PTQ INT8 exported to {args.output_path}')


if __name__ == '__main__':
    main()
