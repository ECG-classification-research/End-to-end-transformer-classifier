"""Read splits without importing torch or WFDB."""
import hashlib
import json
from pathlib import Path
import numpy as np

CLASS_NAMES = ("Normal", "SVEB", "VEB", "Fusion", "Unknown")
AAMI_GROUPS = ("NLRej", "AaJS", "VE", "F", "/fQ")
LABEL_TO_CLASS = {symbol: index for index, symbols in enumerate(AAMI_GROUPS) for symbol in symbols}
SPLIT_ARRAYS = (
    "segments", "labels", "raw_labels", "original_train_indices", "train_indices",
    "test_indices", "kept_indices", "records", "class_names",
)


def map_aami_labels(raw_labels):
    """Map unsupported annotations to -1, to be removed after segmentation."""
    return np.asarray([LABEL_TO_CLASS.get(label, -1) for label in raw_labels], dtype=np.int64)


def validate_split(split):
    """Check alignment and train/test separation without changing any arrays."""
    if split["class_names"].tolist() != list(CLASS_NAMES):
        raise ValueError("Expected the five AAMI classes in their original order.")
    segments, labels = split["segments"], split["labels"]
    count = len(labels)
    if segments.ndim != 2 or segments.shape[1] not in (300, 400) or len(segments) != count or count == 0:
        raise ValueError("Expected non-empty segments [N, 300 or 400] aligned with labels.")
    if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected integer labels [N].")
    if np.any((labels < 0) | (labels >= 5)) or not np.isfinite(segments).all():
        raise ValueError("Segments must be finite and labels must be AAMI IDs 0 through 4.")
    if split["raw_labels"].shape != (count,) or split["kept_indices"].shape != (count,):
        raise ValueError("Original annotation labels and kept indices must align with segments.")
    if not np.array_equal(map_aami_labels(split["raw_labels"]), labels):
        raise ValueError("Original annotation labels do not match AAMI labels.")
    for key in ("original_train_indices", "train_indices", "test_indices"):
        indices = split[key]
        if (indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer)
                or np.any((indices < 0) | (indices >= count))):
            raise ValueError(f"Invalid split indices: {key}.")
    original, train, test = (split[key] for key in
                             ("original_train_indices", "train_indices", "test_indices"))
    if len(np.unique(original)) != len(original) or len(np.unique(test)) != len(test):
        raise ValueError("Original train and test indices must be unique within each split.")
    if np.intersect1d(original, test).size or np.union1d(original, test).size != count:
        raise ValueError("Original train/test must form a disjoint partition of all beats.")
    if not np.isin(train, original).all():
        raise ValueError("Sampled train indices must belong to the original train split.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_split(path):
    with np.load(path, allow_pickle=False) as archive:
        missing = set(SPLIT_ARRAYS) - set(archive.files)
        if missing:
            raise ValueError(f'Split missing arrays: {sorted(missing)}')
        split = {key: archive[key] for key in SPLIT_ARRAYS}
    validate_split(split)
    return split


def validate_metadata(split, path, metadata):
    if metadata.get('split_sha256') != sha256(path):
        raise ValueError('Split SHA-256 differs from the source checkpoint/model.')
    if metadata.get('class_names') != list(CLASS_NAMES):
        raise ValueError('Model AAMI class order differs from split.')
    mode = metadata.get('mode')
    if mode not in ('fixed', 'variable'):
        raise ValueError('Model mode must be fixed or variable.')
    if split['segments'].shape[1] != (300 if mode == 'fixed' else 400):
        raise ValueError('Segment length differs from model mode.')


def calibration_indices(split, count=1024, seed=42):
    """Balanced unique train-only selection, with shortages filled from train."""
    if count <= 0 or seed < 0:
        raise ValueError('Calibration count must be positive and seed non-negative.')
    original = split['original_train_indices']
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(original[split['labels'][original] == c])
              for c in np.unique(split['labels'][original])]
    count = min(count, len(original))
    if count < len(groups):
        raise ValueError('Calibration count must cover every observed train class.')
    selected, position = [], 0
    while len(selected) < count:
        for group in groups:
            if position < len(group) and len(selected) < count:
                selected.append(group[position])
        position += 1
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected

