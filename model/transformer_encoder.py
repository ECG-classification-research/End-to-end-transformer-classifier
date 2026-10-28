"""Sinusoidal positions and the reference Transformer blocks with outer residuals."""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_length=100):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        rates = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(1, max_length, d_model)
        encoding[0, :, 0::2] = torch.sin(position * rates)
        encoding[0, :, 1::2] = torch.cos(position * rates[:d_model // 2])
        self.register_buffer("encoding", encoding)

    def forward(self, x):
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError("Sequence exceeds positional encoding capacity.")
        return x + self.encoding[:, :x.shape[1]].to(dtype=x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, ff_dim=256, dropout=0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.outer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        skip = x
        attention, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout1(attention))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return self.outer_norm(x + skip)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, num_heads=4, ff_dim=256, num_layers=2,
                 dropout=0.2, max_length=100):
        super().__init__()
        if num_layers <= 0 or d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("Use positive layers/heads and a d_model divisible by num_heads.")
        if ff_dim <= 0 or max_length <= 0:
            raise ValueError("ff_dim and max_length must be positive.")
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, tokens):
        """[B, T, 64] -> [B, T, 64]; all tokens participate, including padding."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.d_model or tokens.shape[1] == 0:
            raise ValueError(f"Expected non-empty tokens [B, T, {self.d_model}].")
        x = self.positional_encoding(tokens)
        for block in self.blocks:
            x = block(x)
        return x
