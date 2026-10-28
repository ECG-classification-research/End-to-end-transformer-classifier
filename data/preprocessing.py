"""Preserve the reference WFDB loading, filtering, scaling, and segmentation."""

from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def preprocess_ecg(signal: np.ndarray, fs: float = 360.0) -> np.ndarray:
    """Filter a whole record at 0.5-40 Hz, then scale that record to [-1, 1]."""
    signal = np.asarray(signal, dtype=np.float64)
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, signal)
    if np.ptp(signal) == 0:
        return -np.ones_like(filtered)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(filtered.reshape(-1, 1)).ravel()
    return 2 * normalized - 1


def segment_ecg(signal: np.ndarray, r_peak: int, mode: str = "fixed",
                rng: np.random.Generator | None = None) -> np.ndarray | None:
    """Cut 300 samples, or cut a random 300-400 samples and right-pad to 400."""
    if mode == "fixed":
        start, end = r_peak - 200, r_peak + 100
    elif mode == "variable":
        if rng is None:
            rng = np.random.default_rng()
        delta = int(rng.integers(-50, 51))
        start, end = r_peak - 175, r_peak + 175 + delta
    else:
        raise ValueError("mode must be 'fixed' or 'variable'")
    if start < 0 or end > len(signal):
        return None
    segment = signal[start:end].copy()
    if mode == "variable":
        padded = np.zeros(400, dtype=signal.dtype)
        padded[:len(segment)] = segment
        return padded
    return segment


def discover_wfdb_records(data_path, records=None):
    """Discover .hea files directly in data_path and require complete triplets."""
    root = Path(data_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MIT-BIH folder does not exist: {root}")
    names = ([str(name) for name in records] if records is not None
             else sorted(path.stem for path in root.glob("*.hea")))
    if not names:
        raise FileNotFoundError(f"No WFDB .hea records found directly in {root}")
    missing = [str(root / f"{name}.{ext}") for name in names
               for ext in ("hea", "dat", "atr") if not (root / f"{name}.{ext}").is_file()]
    if missing:
        raise FileNotFoundError("Incomplete MIT-BIH records; missing files:\n" + "\n".join(missing))
    return names


def load_segments(records, data_path, mode="fixed", seed=None):
    """Read lead zero and segment every annotation before filtering AAMI labels."""
    if mode not in ("fixed", "variable"):
        raise ValueError("mode must be 'fixed' or 'variable'")
    root = Path(data_path).expanduser().resolve()
    records = discover_wfdb_records(root, records)
    segments, labels = [], []
    rng = np.random.default_rng(seed)
    for name in tqdm(records, desc="Processing Records"):
        record_path = str(root / name)
        record = wfdb.rdrecord(record_path, channels=[0], physical=True)
        annotation = wfdb.rdann(record_path, extension="atr")
        signal = preprocess_ecg(record.p_signal[:, 0], fs=record.fs)
        for r_peak, label in zip(annotation.sample, annotation.symbol):
            segment = segment_ecg(signal, int(r_peak), mode=mode, rng=rng)
            if segment is not None:
                segments.append(segment)
                labels.append(label)
    length = 300 if mode == "fixed" else 400
    return (np.asarray(segments, dtype=np.float32).reshape(-1, length),
            np.asarray(labels, dtype=str))
