"""Evaluate FP32 PyTorch or FP32/PTQ/QAT ONNX on the exact saved test split."""
import argparse
import csv
from pathlib import Path
import numpy as np
from tqdm import tqdm
from .adapters import create_adapter
from .benchmark import benchmark
from .metrics import compute_metrics, save_results, logits_to_probabilities, cross_entropy
from .split_io import load_split, validate_metadata, save_json, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model_path', 'split_path', 'output_path'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--backend', choices=['auto', 'torch', 'onnx'], default='auto')
    parser.add_argument('--batch_size', type=int, default=64, help='Metric batch size; latency always uses batch 1.')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--benchmark_runs', type=int, default=1000)
    parser.add_argument('--optimization', choices=['all', 'disabled'], default='all')
    parser.add_argument('--profile', action='store_true', help='Write a separate ONNX runtime profile after timing.')
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads, args.benchmark_runs) <= 0 or args.warmup < 0:
        parser.error('Batch/threads/runs must be positive and warmup non-negative.')
    if args.output_path.exists():
        raise FileExistsError(args.output_path)
    split = load_split(args.split_path)
    adapter = create_adapter(args.model_path, args.backend, args.threads, args.optimization)
    validate_metadata(split, args.split_path, adapter.metadata)
    idx = split['test_indices']
    probabilities = []
    loss_sum = 0.0
    for start in tqdm(range(0, len(idx), args.batch_size), desc='Test', unit='batch'):
        batch_idx = idx[start:start+args.batch_size]
        signals = np.ascontiguousarray(split['segments'][batch_idx, None], dtype=np.float32)
        logits = adapter.predict(signals)
        if logits.shape != (len(batch_idx), 5):
            raise ValueError('Adapter output must be [B,5].')
        probabilities.append(logits_to_probabilities(logits))
        loss_sum += cross_entropy(logits, split['labels'][batch_idx]) * len(batch_idx)
    probabilities = np.concatenate(probabilities)
    metrics = compute_metrics(split['labels'][idx], probabilities, split['class_names'].tolist())
    # Separate batch-one pass, with no profiling overhead during timing.
    bench_idx = idx[:min(len(idx), args.benchmark_runs)]
    summary, raw = benchmark(adapter, split['segments'][bench_idx, None], args.warmup, args.benchmark_runs)
    summary['runtime'] = adapter.runtime
    complexity = adapter.metadata['complexity']
    metrics.update(test_loss=loss_sum/len(idx), model=str(args.model_path.resolve()),
                   model_sha256=sha256(args.model_path), split_file=str(args.split_path.resolve()),
                   split_sha256=adapter.metadata['split_sha256'], method=adapter.metadata['method'],
                   mode=adapter.metadata['mode'], benchmark=summary, **complexity)
    metrics['model_size_bytes'] = args.model_path.stat().st_size
    args.output_path.mkdir(parents=True, exist_ok=False)
    save_results(args.output_path, metrics, split['labels'][idx], probabilities, idx, split['raw_labels'][idx])
    save_json(args.output_path / 'benchmark.json', summary)
    save_json(args.output_path / 'model_info.json', dict(metadata=adapter.metadata, runtime=adapter.runtime,
                                                       model_size_bytes=metrics['model_size_bytes']))
    with (args.output_path / 'latencies.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run', 'segment_index', 'latency_ms'])
        writer.writerows((i, int(bench_idx[i % len(bench_idx)]), latency) for i, latency in enumerate(raw))
    fields = ['method', 'test_loss', 'accuracy', 'precision', 'recall', 'f1', 'f1_macro',
              'parameters_total', 'parameters_quantized', 'model_size_bytes', 'macs_per_sample',
              'flops_equivalent_per_sample', 'fp32_ops_equivalent', 'int8_ops_equivalent',
              'mean_ms_per_sample', 'p95_ms_per_sample']
    with (args.output_path / 'summary_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({name: ({**metrics, **summary})[name] for name in fields})
    if args.profile:
        profile_adapter = create_adapter(args.model_path, args.backend, args.threads, args.optimization,
                                         profile_dir=args.output_path)
        profile_adapter.predict(np.ascontiguousarray(split['segments'][idx[:1], None], dtype=np.float32))
        profile_path = profile_adapter.end_profile()
        print(f'Runtime profile (separate from latency): {profile_path}')
    print(f"{metrics['method']}: accuracy={metrics['accuracy']:.4%}, F1 macro={metrics['f1_macro']:.4%}, "
          f"mean={summary['mean_ms_per_sample']:.4f} ms/sample; output={args.output_path}")


if __name__ == '__main__':
    main()
