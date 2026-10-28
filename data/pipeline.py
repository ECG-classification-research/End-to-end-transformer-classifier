"""AAMI mapping and the reference 80/20 stratified beat-wise data pipeline."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .preprocessing import discover_wfdb_records, load_segments


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


def sample_train_indices(labels, train_indices, mode="none", seed=42):
    """Select/repeat train indices with exactly the reference sampling rules."""
    rng = np.random.default_rng(seed)
    if mode == "none":
        return train_indices.copy()
    classes, counts = np.unique(labels[train_indices], return_counts=True)
    if mode == "over":
        target = int(counts.max())
    elif mode == "under":
        target = int(counts.min())
    elif mode == "hybrid":
        minority_counts = counts[classes != 0]
        target = int(minority_counts.max() if minority_counts.size else counts.max())
    else:
        raise ValueError("mode must be 'none', 'over', 'under', or 'hybrid'")
    sampled = []
    for label, count in zip(classes, counts):
        indices = train_indices[labels[train_indices] == label]
        if count > target:
            indices = rng.choice(indices, size=target, replace=False)
        elif count < target:
            extra = rng.choice(indices, size=target - count, replace=True)
            indices = np.concatenate([indices, extra])
        sampled.append(indices)
    sampled = np.concatenate(sampled)
    rng.shuffle(sampled)
    return sampled


def sampling_statistics(split):
    return {name: np.bincount(split["labels"][split[key]], minlength=5).tolist()
            for name, key in (("train_before", "original_train_indices"),
                              ("train_after", "train_indices"), ("test", "test_indices"))}


def prepare_data(data_path, records=None, mode="fixed", sampling="none", seed=42):
    records = discover_wfdb_records(data_path, records)
    segments, raw_labels = load_segments(records, data_path, mode=mode, seed=seed)
    labels = map_aami_labels(raw_labels)
    kept_indices = np.flatnonzero(labels >= 0)
    segments, raw_labels, labels = segments[kept_indices], raw_labels[kept_indices], labels[kept_indices]
    if not len(labels):
        raise ValueError("No supported AAMI beats remain after preprocessing and segmentation.")
    try:
        train_indices, test_indices = train_test_split(
            np.arange(len(labels)), test_size=0.2, random_state=seed, stratify=labels,
        )
    except ValueError as error:
        raise ValueError(f"Cannot stratify the selected records: {error}. Select more records.") from error
    original_train_indices = train_indices.copy()
    train_indices = sample_train_indices(labels, train_indices, mode=sampling, seed=seed)
    return {
        "segments": segments, "labels": labels, "raw_labels": raw_labels,
        "original_train_indices": original_train_indices, "train_indices": train_indices,
        "test_indices": test_indices, "kept_indices": kept_indices,
        "records": np.asarray(records, dtype=str), "class_names": np.asarray(CLASS_NAMES),
    }


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


def save_split(path, split):
    validate_split(split)
    # Exclusive creation preserves an existing split, including when called directly.
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **{key: split[key] for key in SPLIT_ARRAYS})


def load_split(path):
    with np.load(path, allow_pickle=False) as archive:
        missing = set(SPLIT_ARRAYS) - set(archive.files)
        if missing:
            raise ValueError(f"Split file is missing arrays: {sorted(missing)}")
        split = {key: archive[key] for key in SPLIT_ARRAYS}
    validate_split(split)
    return split
