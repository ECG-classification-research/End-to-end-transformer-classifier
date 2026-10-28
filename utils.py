"""Shared random seeds, checkpoints, and classification result exports."""

import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)


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


def compute_metrics(labels, probabilities, class_names):
    """Overall and per-class metrics, with undefined AUC/kappa represented by None."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    classes = np.arange(len(class_names))
    if labels.ndim != 1 or labels.size == 0 or probabilities.shape != (len(labels), len(classes)):
        raise ValueError("Expected non-empty labels [N] and probabilities [N, C].")
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= len(classes))):
        raise ValueError("Labels must be integer class IDs in range.")
    if (not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Probabilities must be finite, non-negative, and sum to one.")
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=classes, zero_division=0,
    )
    metrics = {"samples": len(labels), "class_names": list(class_names),
               "accuracy": float(accuracy_score(labels, predictions)), "per_class": {}}
    for average in ("macro", "weighted"):
        p, r, score, _ = precision_recall_fscore_support(
            labels, predictions, labels=classes, average=average, zero_division=0,
        )
        metrics[average] = {"precision": float(p), "recall": float(r), "f1": float(score)}
    metrics.update({
        "precision": metrics["weighted"]["precision"],
        "recall": metrics["weighted"]["recall"], "f1": metrics["weighted"]["f1"],
        "f1_macro": metrics["macro"]["f1"], "precision_recall_f1_average": "weighted",
        "per_class_accuracy_definition": "one-vs-rest: (TP + TN) / N",
    })
    aucs = []
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        fn = int(matrix[index].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = len(labels) - tp - fn - fp
        binary = labels == index
        auc = float(roc_auc_score(binary, probabilities[:, index])) if binary.any() and (~binary).any() else None
        aucs.append(auc)
        metrics["per_class"][name] = {
            "precision": float(precision[index]), "recall": float(recall[index]),
            "accuracy": float((tp + tn) / len(labels)), "f1": float(f1[index]),
            "support": int(support[index]), "roc_auc_ovr": auc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }
    complete = all(auc is not None for auc in aucs)
    metrics["roc_auc_ovr_macro"] = float(np.mean(aucs)) if complete else None
    metrics["roc_auc_ovr_weighted"] = float(np.average(aucs, weights=support)) if complete else None
    metrics["roc_auc_note"] = None if complete else "Aggregate AUC is undefined: one or more test classes are absent."
    kappa = (float(cohen_kappa_score(labels, predictions, labels=classes))
             if len(np.unique(np.concatenate((labels, predictions)))) > 1 else None)
    metrics["cohen_kappa"] = kappa if kappa is not None and np.isfinite(kappa) else None
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics


def save_results(output, metrics, labels, probabilities, indices, raw_labels):
    output = Path(output)
    names = metrics["class_names"]
    predictions = probabilities.argmax(axis=1)
    save_json(output / "metrics.json", metrics)
    with (output / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["test_loss", "precision", "recall", "accuracy", "f1", "f1_macro", "precision_recall_f1_average"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: metrics[field] for field in fields})
    with (output / "per_label_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["label", "precision", "recall", "accuracy", "f1", "support", "tp", "tn", "fp", "fn", "roc_auc_ovr"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in names:
            writer.writerow({"label": name, **metrics["per_class"][name]})
    np.save(output / "probabilities.npy", probabilities)
    np.save(output / "predictions.npy", predictions)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["test_row", "segment_index", "raw_label", "true_label_id", "true_class",
                  "predicted_label_id", "predicted_class"] + [f"prob_{name}" for name in names]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, (label, probability, index, raw) in enumerate(zip(labels, probabilities, indices, raw_labels)):
            prediction = int(probability.argmax())
            entry = {
                "test_row": row, "segment_index": int(index), "raw_label": str(raw),
                "true_label_id": int(label), "true_class": names[int(label)],
                "predicted_label_id": prediction, "predicted_class": names[prediction],
            }
            entry.update({f"prob_{name}": float(probability[i]) for i, name in enumerate(names)})
            writer.writerow(entry)
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *names])
        writer.writerows([name, *row] for name, row in zip(names, metrics["confusion_matrix"]))
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=names).plot(
        ax=ax, cmap="Blues", values_format="d",
    )
    ax.set_title("CNN-Transformer - MIT-BIH AAMI beat-wise test")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)
