"""Evaluate an end-to-end checkpoint on its saved, unchanged ECG test segments."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CLASS_NAMES, BeatDataset, load_split
from model import ECGClassifier
from utils import compute_metrics, file_sha256, load_checkpoint, resolve_device, save_results


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    probabilities, total_loss, samples = [], 0.0, 0
    for signals, labels in tqdm(loader, desc="Test", unit="batch"):
        logits = model(signals.to(device))
        loss = torch.nn.functional.cross_entropy(logits, labels.to(device), reduction="sum")
        if not torch.isfinite(loss):
            raise ValueError("Non-finite test loss.")
        total_loss += loss.item()
        samples += len(labels)
        probabilities.append(logits.softmax(dim=-1).cpu().numpy())
    if samples == 0:
        raise ValueError("Cannot evaluate an empty test dataset.")
    return np.concatenate(probabilities), total_loss / samples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to an end-to-end model.pt.")
    parser.add_argument("--split_path", type=Path, help="Default: beat_split.npz beside the checkpoint.")
    parser.add_argument("--output_path", type=Path, help="Default: <checkpoint folder>/test_results; must not exist.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Compute device.")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("batch_size must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    split_path = args.split_path if args.split_path is not None else checkpoint_path.parent / config["split_file"]
    split_path = split_path.expanduser().resolve()
    if file_sha256(split_path) != config["split_sha256"]:
        raise ValueError("Split SHA-256 differs from the data used to train this checkpoint.")
    split = load_split(split_path)
    if config["class_names"] != list(CLASS_NAMES):
        raise ValueError("Checkpoint AAMI class order differs from the dataset.")
    expected_length = 300 if config["mode"] == "fixed" else 400
    if split["segments"].shape[1] != expected_length:
        raise ValueError("Segment length differs from the checkpoint segmentation mode.")
    indices = split["test_indices"]
    test_set = BeatDataset(split["segments"], split["labels"], indices)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    model = ECGClassifier(**config["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval().requires_grad_(False)
    output = args.output_path if args.output_path is not None else checkpoint_path.parent / "test_results"
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    probabilities, test_loss = evaluate(model, loader, device)
    labels = split["labels"][indices]
    metrics = compute_metrics(labels, probabilities, CLASS_NAMES)
    metrics.update({
        "test_loss": test_loss, "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint": str(checkpoint_path), "split_file": str(split_path),
        "split_sha256": config["split_sha256"], "split": config["split"], "mode": config["mode"],
    })
    save_results(output, metrics, labels, probabilities, indices, split["raw_labels"][indices])
    print(f"Test loss={test_loss:.6f}; accuracy={metrics['accuracy']:.2%}; "
          f"F1 weighted={metrics['f1']:.2%}; F1 macro={metrics['f1_macro']:.2%}")
    print(f"{'Label':<12} {'Precision':>10} {'Recall':>10} {'Accuracy':>10} {'F1':>10} {'Support':>10}")
    for name, scores in metrics["per_class"].items():
        print(f"{name:<12} {scores['precision']:>10.2%} {scores['recall']:>10.2%} "
              f"{scores['accuracy']:>10.2%} {scores['f1']:>10.2%} {scores['support']:>10d}")
    print(f"Test results saved to {output}")


if __name__ == "__main__":
    main()
