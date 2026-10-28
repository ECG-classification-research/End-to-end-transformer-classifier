"""The reference pre-Dense CNN encoder, with optional additional CNN layers."""

import torch
from torch import nn
from torch.nn import functional as F


class CNNEncoder(nn.Module):
    out_channels = 64

    def __init__(self, in_channels=1, dropout=0.2, more_cnn=False):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv1d(in_channels, 128, 5, stride=2)
        self.bn1 = nn.BatchNorm1d(128, eps=1e-3, momentum=0.01)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(128, 64, 5, stride=2)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.extra_convs = nn.Sequential(*[
            layer for _ in range(2 if more_cnn else 0) for layer in (
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(), nn.BatchNorm1d(64, eps=1e-3, momentum=0.01), nn.Dropout(dropout),
            )
        ])

    @staticmethod
    def _same_pad(x):
        total = 3 if x.shape[-1] % 2 == 0 else 4
        return F.pad(x, (total // 2, total - total // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1, L] -> [B, 64, ceil(L / 4)]."""
        if x.ndim != 3 or x.shape[1] != self.in_channels or x.shape[-1] == 0:
            raise ValueError(f"Expected signals [B, {self.in_channels}, L] with L > 0.")
        x = self.dropout1(self.bn1(self.activation(self.conv1(self._same_pad(x)))))
        x = self.dropout2(self.bn2(self.activation(self.conv2(self._same_pad(x)))))
        return self.extra_convs(x)
