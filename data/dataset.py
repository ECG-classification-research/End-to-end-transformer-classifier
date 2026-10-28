"""Access beats through split indices without copying oversampled signals."""

import numpy as np
import torch
from torch.utils.data import Dataset


class BeatDataset(Dataset):
    def __init__(self, segments, labels, indices=None, augmentation=None):
        self.segments = segments
        self.labels = labels
        self.indices = np.arange(len(labels)) if indices is None else np.asarray(indices)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        segment_index = int(self.indices[index])
        # A private copy protects the saved beat from in-place transformations.
        signal = np.array(self.segments[segment_index], dtype=np.float32, copy=True)
        if self.augmentation is not None:
            signal = self.augmentation.forward(signal)
        signal = torch.from_numpy(np.asarray(signal, dtype=np.float32)).unsqueeze(0)
        return signal, int(self.labels[segment_index])
