"""Fine-tune Linear-only static QAT from an FP32 checkpoint and its saved split."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_pi.split_io import calibration_indices, save_json, sha256
from quantization.checkpoint import load_source, build_metadata
from quantization.linear import prepare_qat, calibrate, freeze_observers, quantizers, QUANTIZATION_CONFIG
from quantization.export import export_onnx
from utils import set_seed, resolve_device


def train_mode(model):
    model.train()
    # Freeze statistics only: Dropout remains active throughout the model.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def calibration_batches(split, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield torch.from_numpy(np.ascontiguousarray(split['segments'][indices[start:start+batch_size], None], dtype=np.float32))


def save_qat_checkpoint(path, model, optimizer, epoch, config, metadata):
    torch.save(dict(format_version='ecg_linear_qat_v1', model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(), epoch=epoch, config=config, metadata=metadata), path)


def load_qat_checkpoint(path):
    from model import ECGClassifier
    from quantization.attention import explicit_model
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 'ecg_linear_qat_v1':
        raise ValueError('Expected ecg_linear_qat_v1 checkpoint.')
    model = prepare_qat(explicit_model(ECGClassifier(**payload['metadata']['model_args'])))
    # Per-channel observer min/max buffers acquire their size during bootstrap.
    state = payload['model_state']
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if name in state and buffer.shape != state[name].shape:
            buffer.resize_(state[name].shape)
    model.load_state_dict(state, strict=True)
    freeze_observers(model)
    return model.eval(), payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_path', type=Path, required=True)
    parser.add_argument('--split_path', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--calibration_samples', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.calibration_samples, args.threads) <= 0 or args.seed < 0:
        parser.error('Counts/threads must be positive; seed must be non-negative.')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('lr must be finite and positive.')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output_path.exists():
        raise FileExistsError(args.output_path)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, source_config, split = load_source(args.checkpoint_path, args.split_path)
    model = prepare_qat(model).to(device)
    indices = calibration_indices(split, args.calibration_samples, args.seed)
    calibrate(model, calibration_batches(split, indices, args.batch_size))
    for q in quantizers(model):
        q.observe, q.fake = True, True
    train = split['train_indices']
    dataset = TensorDataset(torch.from_numpy(split['segments'][train, None]),
                            torch.from_numpy(split['labels'][train]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    # Epochs are 1-indexed: final ceil(20%) epochs use frozen observers.
    freeze_epoch = max(1, args.epochs - math.ceil(args.epochs * .2) + 1)
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_config.update(observer_freeze_epoch=freeze_epoch, checkpoint_selection='final_epoch',
                      train_samples=len(train), calibration_indices=indices.tolist(),
                      calibration_counts=np.bincount(split['labels'][indices], minlength=5).tolist(),
                      split_sha256=sha256(args.split_path), quantization=QUANTIZATION_CONFIG,
                      frozen_modules='All non-Linear parameters and all BatchNorm statistics')
    args.output_path.mkdir(parents=True, exist_ok=False)
    save_json(args.output_path / 'config.json', run_config)
    with (args.output_path / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'train_accuracy', 'observer_frozen'])
        writer.writeheader()
        for epoch in range(1, args.epochs+1):
            if epoch >= freeze_epoch:
                freeze_observers(model)
            train_mode(model)
            total_loss = correct = samples = 0
            for signals, labels in loader:
                signals, labels = signals.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(signals)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite QAT loss.')
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                correct += int((logits.argmax(1) == labels).sum())
                samples += len(labels)
            row = dict(epoch=epoch, train_loss=total_loss/samples, train_accuracy=correct/samples,
                       observer_frozen=epoch >= freeze_epoch)
            writer.writerow(row)
            stream.flush()
            print(row, flush=True)
    model.eval()
    freeze_observers(model)
    metadata = build_metadata(model, source_config, args.checkpoint_path, args.split_path, 'qat')
    metadata['training'] = run_config
    save_qat_checkpoint(args.output_path / 'qat_checkpoint.pt', model, optimizer, args.epochs, run_config, metadata)
    examples = next(calibration_batches(split, indices, min(3, len(indices))))
    report = export_onnx(model, args.output_path / 'qat_int8.onnx', metadata, examples)
    save_json(args.output_path / 'metadata.json', {'qat_int8.onnx': metadata})
    save_json(args.output_path / 'quantization_report.json', report)
    print(f'QAT INT8 exported: {args.output_path / "qat_int8.onnx"}')


if __name__ == '__main__':
    main()
