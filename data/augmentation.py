"""The reference online augmentations, applied in their original order."""

import numpy as np


class ECGAugmentation:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def gaussian_noise(self, x):
        return x + self.rng.normal(0, 0.005, size=x.shape)

    def random_scale(self, x):
        return x * self.rng.uniform(0.95, 1.05)

    def time_shift(self, x):
        shift = int(self.rng.integers(-5, 6))
        x = np.roll(x, shift)
        if shift > 0:
            x[:shift] = 0
        elif shift < 0:
            x[shift:] = 0
        return x

    def forward(self, x):
        return self.time_shift(self.random_scale(self.gaussian_noise(x)))
