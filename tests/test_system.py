"""Behavior checks and full CLI smoke tests using synthetic local WFDB records."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
import wfdb

from data import CLASS_NAMES, BeatDataset, ECGAugmentation, load_split
from data.pipeline import map_aami_labels, prepare_data, sample_train_indices, save_split
from data.preprocessing import preprocess_ecg, segment_ecg
from model import ECGClassifier
from utils import compute_metrics, load_checkpoint, save_checkpoint

torch.set_num_threads(1)


def write_synthetic_record(folder):
    """Two leads, all five classes, unsupported annotation, and boundary beats."""
    time = np.arange(16000) / 360.0
    signal = np.sin(2 * np.pi * 1.2 * time) + 0.05 * np.cos(2 * np.pi * 20 * time)
    wfdb.wrsamp("100", fs=360, units=["mV", "mV"], sig_name=["MLII", "V5"],
                p_signal=np.column_stack((signal, np.full_like(signal, 0.2))),
                fmt=["16", "16"], adc_gain=[1000, 1000], baseline=[0, 0], write_dir=str(folder))
    labels = ["N"] * 20 + ["A"] * 10 + ["V"] * 8 + ["F"] * 6 + ["Q"] * 6
    peaks = [50] + (500 + 250 * np.arange(50)).tolist() + [13750, 15950]
    wfdb.wrann("100", "atr", sample=np.asarray(peaks), symbol=["N", *labels, "+", "N"],
               write_dir=str(folder))


class SystemTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".verification"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def run_script(self, script, *args, success=True):
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                cwd=self.folder, capture_output=True, text=True, timeout=120)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_segmentation_and_constant_record(self):
        signal = np.arange(1000, dtype=np.float64)
        np.testing.assert_array_equal(segment_ecg(signal, 500), signal[300:600])
        self.assertIsNone(segment_ecg(signal, 50))
        self.assertIsNone(segment_ecg(signal, 950))
        expected_rng, actual_rng = np.random.default_rng(42), np.random.default_rng(42)
        for _ in range(10):
            delta = int(expected_rng.integers(-50, 51))
            result = segment_ecg(signal, 500, "variable", actual_rng)
            length = 350 + delta
            self.assertEqual(result.shape, (400,))
            np.testing.assert_array_equal(result[:length], signal[325:675 + delta])
            np.testing.assert_array_equal(result[length:], 0)
        np.testing.assert_array_equal(preprocess_ecg(np.ones(1000)), -np.ones(1000))

    def test_sampling_and_augmentation(self):
        labels = np.array([0] * 10 + [1] * 4 + [2] * 2)
        indices = np.arange(len(labels))
        for mode, count in (("over", 10), ("under", 2), ("hybrid", 4)):
            sampled = sample_train_indices(labels, indices, mode=mode, seed=42)
            np.testing.assert_array_equal(np.bincount(labels[sampled]), [count] * 3)
            self.assertTrue(np.isin(sampled, indices).all())
        np.testing.assert_array_equal(map_aami_labels(["N", "a", "E", "F", "/", "+"]), [0, 1, 2, 3, 4, -1])
        source = np.linspace(-1, 1, 300, dtype=np.float32)[None]
        original = source.copy()
        dataset = BeatDataset(source, np.array([0]), np.array([0, 0]), ECGAugmentation(42))
        first, label = dataset[0]
        second, _ = dataset[1]
        self.assertEqual(first.shape, (1, 300))
        self.assertEqual(label, 0)
        self.assertFalse(torch.equal(first, second))
        np.testing.assert_array_equal(source, original)

    def test_forward_backward_and_checkpoint(self):
        for length in (300, 400):
            for more_cnn in (False, True):
                with self.subTest(length=length, more_cnn=more_cnn):
                    torch.manual_seed(42)
                    args = {"more_cnn": more_cnn, "transformer_blocks": 3, "dropout": 0.2, "max_length": 100}
                    model = ECGClassifier(**args)
                    signals, labels = torch.randn(5, 1, length), torch.arange(5)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                    weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                    logits = model(signals)
                    self.assertEqual(logits.shape, (5, 5))
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    optimizer.step()
                    for prefix in ("cnn_encoder.", "transformer_encoder.", "head."):
                        self.assertTrue(any(not torch.equal(weights[name], p) for name, p in
                                            model.named_parameters() if name.startswith(prefix)), prefix)
                    model.eval()
                    with torch.inference_mode():
                        before = model(signals)
                    path = self.folder / "model.pt"
                    save_checkpoint(path, model, optimizer, 1, {"model_args": args}, {"loss": loss.item()})
                    checkpoint = load_checkpoint(path)
                    restored = ECGClassifier(**checkpoint["config"]["model_args"]).eval()
                    restored.load_state_dict(checkpoint["model_state"])
                    with torch.inference_mode():
                        torch.testing.assert_close(before, restored(signals), rtol=0, atol=0)

    def test_split_integrity(self):
        raw = self.folder / "raw"
        raw.mkdir()
        write_synthetic_record(raw)
        split = prepare_data(raw, mode="variable", sampling="over", seed=42)
        self.assertEqual(split["segments"].shape, (50, 400))
        self.assertEqual(len(split["original_train_indices"]), 40)
        self.assertEqual(len(split["test_indices"]), 10)
        self.assertEqual(np.intersect1d(split["train_indices"], split["test_indices"]).size, 0)
        self.assertGreater(len(split["train_indices"]), len(split["original_train_indices"]))
        path = self.folder / "split.npz"
        save_split(path, split)
        restored = load_split(path)
        for key in split:
            np.testing.assert_array_equal(restored[key], split[key])
        with self.assertRaises(FileExistsError):
            save_split(path, split)
        corrupted = dict(split, test_indices=split["train_indices"][:10])
        with self.assertRaises(ValueError):
            save_split(self.folder / "invalid.npz", corrupted)

    def test_metrics_with_missing_class(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.eye(5)[[0, 1, 1, 1]]
        metrics = compute_metrics(labels, probabilities, CLASS_NAMES)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["per_class"]["Normal"]["recall"], 0.5)
        self.assertIsNone(metrics["per_class"]["Fusion"]["roc_auc_ovr"])
        self.assertIsNone(metrics["roc_auc_ovr_macro"])
        json.dumps(metrics, allow_nan=False)

    def test_cli_train_test(self):
        raw = self.folder / "raw"
        raw.mkdir()
        write_synthetic_record(raw)
        for script in ("train.py", "test.py"):
            self.assertIn("--device", self.run_script(script, "--help").stdout)
        self.run_script("train.py", "--data_path", raw, "--output_path", self.folder / "bad",
                        "--lr", "nan", success=False)
        for mode in ("fixed", "variable"):
            with self.subTest(mode=mode):
                run = self.folder / mode
                extra = ["--more_cnn", "--augment", "--sampling", "hybrid"] if mode == "variable" else []
                self.run_script("train.py", "--data_path", raw, "--output_path", run,
                                "--mode", mode, "--epochs", 1, "--batch_size", 10,
                                "--transformer_blocks", 2, "--device", "cpu", *extra)
                for name in ("config.json", "beat_split.npz", "history.csv", "train_metrics.json", "model.pt"):
                    self.assertTrue((run / name).is_file(), name)
                self.assertFalse((run / "test_results").exists())
                self.run_script("train.py", "--data_path", raw, "--output_path", run, success=False)
                # Relocation must work without using the absolute paths in config.
                moved = self.folder / f"moved_{mode}"
                run.rename(moved)
                self.run_script("test.py", "--checkpoint_path", moved / "model.pt",
                                "--batch_size", 4, "--device", "cpu")
                result = moved / "test_results"
                names = ("metrics.json", "summary_metrics.csv", "per_label_metrics.csv", "predictions.csv",
                         "probabilities.npy", "predictions.npy", "confusion_matrix.csv", "confusion_matrix.png")
                for name in names:
                    self.assertTrue((result / name).is_file(), name)
                metrics = json.loads((result / "metrics.json").read_text())
                self.assertEqual(metrics["samples"], 10)
                self.assertTrue(np.isfinite(metrics["test_loss"]))
                self.assertEqual(np.asarray(metrics["confusion_matrix"]).sum(), 10)
                with (result / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                split = load_split(moved / "beat_split.npz")
                self.assertEqual([int(row["segment_index"]) for row in rows], split["test_indices"].tolist())
                np.testing.assert_allclose(np.load(result / "probabilities.npy").sum(axis=1), 1, atol=1e-5)
                # Repeated test output and unrelated split files must be rejected.
                self.run_script("test.py", "--checkpoint_path", moved / "model.pt", success=False)
                fake_split = self.folder / "unrelated.npz"
                fake_split.write_bytes(b"unrelated split")
                rejection = self.run_script("test.py", "--checkpoint_path", moved / "model.pt",
                                            "--split_path", fake_split, success=False)
                self.assertIn("SHA-256", rejection.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
