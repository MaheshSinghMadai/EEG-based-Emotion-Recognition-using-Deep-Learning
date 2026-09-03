"""
CNN-Transformer (Reduced-Electrode)
==================================
Same CNN + Transformer family as CIT-EmotionNet (#4), but operating on the RAW
windowed signal restricted to a 5-electrode consumer-grade subset:

    AF3, F3, F4, AF4, O1
    -> frontal-asymmetry quartet (valence/arousal correlate) + one occipital reference.

Pipeline: temporal CNN stem -> time-token sequence -> Transformer encoder ->
attention-pooled embedding -> linear classifier.

Input : raw windowed signal [B, 5, T]
Output: class logits [B, n_classes]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Exposed so the notebook / data_loader can request exactly these channels.
REDUCED_5_CHANNELS = ["AF3", "F3", "F4", "AF4", "O1"]


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CNNTransformerReduced(nn.Module):
    def __init__(self, n_channels: int = 5, n_samples: int = 512, n_classes: int = 2,
                 dim: int = 64, depth: int = 3, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        # temporal CNN stem: 3 conv/pool blocks -> downsample time by 8
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, dim, kernel_size=7, padding=3), nn.BatchNorm1d(dim), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2), nn.BatchNorm1d(dim), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1), nn.BatchNorm1d(dim), nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.pos = PositionalEncoding(dim, max_len=n_samples)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        # temporal attention pooling
        self.attn_pool = nn.Linear(dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 5, T]
        h = self.stem(x)                    # [B, dim, T/8]
        h = h.transpose(1, 2)              # [B, T', dim]
        h = self.pos(h)
        h = self.encoder(h)               # [B, T', dim]
        w = torch.softmax(self.attn_pool(h).squeeze(-1), dim=1)  # [B, T']
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)                # [B, dim]
        return self.classifier(pooled)


def build(cfg: dict) -> CNNTransformerReduced:
    meta = cfg.get("meta", {})
    shape = meta.get("input_shape", (5, 512))
    return CNNTransformerReduced(
        n_channels=shape[0], n_samples=shape[1],
        n_classes=cfg.get("n_classes", 2),
        dim=cfg.get("dim", 64), depth=cfg.get("depth", 3),
        heads=cfg.get("heads", 4), dropout=cfg.get("dropout", 0.2),
    )
