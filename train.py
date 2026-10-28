"""Train CNN Encoder -> Transformer Encoder jointly using CrossEntropyLoss."""

import argparse
import csv
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CLASS_NAMES, BeatDataset, ECGAugmentation, prepare_data, save_split
from data.pipeline import sampling_statistics
from model import ECGClassifier
from utils import file_sha256, resolve_device, save_checkpoint, save_json, set_seed


def train_epoch(model, loader, optimizer, device, description="Train"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, samples = 0.0, 0, 0
    progress = tqdm(loader, desc=description, unit="batch")
    for signals, labels in progress:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite classification loss; inspect input data and learning rate.")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        samples += len(labels)
        progress.set_postfix(loss=f"{total_loss / samples:.6f}", accuracy=f"{correct / samples:.2%}")
    if samples == 0:
        raise ValueError("Cannot train on an empty dataset.")
    return total_loss / samples, correct / samples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=Path, required=True, help="WFDB folder containing .hea/.dat/.atr triplets.")
    parser.add_argument("--output_path", type=Path, required=True, help="New run directory; must not exist.")
    parser.add_argument("--records", nargs="+", help="Record names without extensions; default: all .hea records.")
    parser.add_argument("--mode", choices=("fixed", "variable"), default="fixed", help="Segmentation mode.")
    parser.add_argument("--sampling", choices=("none", "over", "under", "hybrid"), default="none", help="Train-only sampling.")
    parser.add_argument("--augment", action="store_true", help="Enable noise, scaling, and time shift for train.")
    parser.add_argument("--more_cnn", action="store_true", help="Add two kernel-3, 64-channel CNN layers.")
    parser.add_argument("--transformer_blocks", type=int, default=2, help="Number of Transformer encoder blocks.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Train batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Compute device.")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.transformer_blocks <= 0:
        parser.error("epochs, batch_size, and transformer_blocks must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or args.seed < 0:
        parser.error("lr must be finite and positive; seed must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    output = args.output_path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Run directory already exists; choose another --output_path: {output}")
    data_path = args.data_path.expanduser().resolve()
    split = prepare_data(data_path, records=args.records, mode=args.mode,
                         sampling=args.sampling, seed=args.seed)
    output.mkdir(parents=True, exist_ok=False)
    split_path = output / "beat_split.npz"
    save_split(split_path, split)
    statistics = sampling_statistics(split)
    model_args = {"more_cnn": args.more_cnn, "transformer_blocks": args.transformer_blocks,
                  "dropout": 0.2, "max_length": 100}
    config = {
        **vars(args), "data_path": str(data_path), "output_path": str(output),
        "records": split["records"].tolist(), "device": str(device),
        "class_names": list(CLASS_NAMES), "model_args": model_args,
        "split": "stratified beat-wise 80/20", "split_file": "beat_split.npz",
        "split_sha256": file_sha256(split_path), "sampling_statistics": statistics,
        "preprocessing": "per-record Butterworth order=4, 0.5-40 Hz; min-max then [-1,1]",
        "loss": "CrossEntropyLoss", "optimizer": "Adam", "checkpoint_selection": "final_epoch",
    }
    save_json(output / "config.json", config)
    print(f"WFDB folder: {data_path}; records: {len(split['records'])}; device: {device}")
    for name, counts in statistics.items():
        print(f"{name}: {sum(counts)} beats; {dict(zip(CLASS_NAMES, counts))}")
    augmentation = ECGAugmentation(seed=args.seed) if args.augment else None
    train_set = BeatDataset(split["segments"], split["labels"], split["train_indices"], augmentation)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    model = ECGClassifier(**model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "train_loss", "train_accuracy", "lr"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            loss, accuracy = train_epoch(model, loader, optimizer, device, f"Epoch {epoch}/{args.epochs}")
            writer.writerow({"epoch": epoch, "train_loss": loss, "train_accuracy": accuracy, "lr": args.lr})
            stream.flush()
    metrics = {"final_train_loss": loss, "final_train_accuracy": accuracy}
    save_json(output / "train_metrics.json", metrics)
    save_checkpoint(output / "model.pt", model, optimizer, args.epochs, config, metrics)
    print(f"Final checkpoint saved to {output / 'model.pt'}")
    print("Run test.py with --checkpoint_path to evaluate the saved test split.")


if __name__ == "__main__":
    main()
