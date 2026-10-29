"""Shared random seeds, checkpoints, and classification result exports."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                        else "cpu" if name == "auto" else name)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path, model, optimizer, epoch, config, train_metrics):
    torch.save({
        "format_version": 1, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "epoch": epoch,
        "config": config, "train_metrics": train_metrics,
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected an ecg_end_to_end checkpoint with format_version=1.")
    required = {"model_state", "optimizer_state", "epoch", "config", "train_metrics"}
    if not required.issubset(checkpoint):
        raise ValueError("Incomplete end-to-end checkpoint.")
    return checkpoint


# Preserve the public API used by train.py/test.py.
from evaluate_pi.metrics import compute_metrics, save_results
