"""
DAH-Net — Dual-Attention Hybrid Network (arXiv:2602.06411, 2026 preprint).

Reproduced from the paper (Section III, Table I, Table II):
    * Input: an engineered EEG feature vector treated as a 1-D sequence.
    * 1D-CNN encoder: 4 convolutional layers + 1 residual block, ReLU (Sec. III-C).
    * BiLSTM: 2 bidirectional layers, 256 units per direction.
    * Dual multi-head self-attention: 16 heads, then 8 heads (residual + LayerNorm).
    * Global max + average pooling, dropout 0.28, softmax classifier.
    * Training recipe (see `RECIPE` in benchmark_proposal): AdamW (lr 8e-4,
      weight decay 5e-5), cosine annealing warm restarts (T0 = 50), label smoothing
      0.01, gradient clipping 1.0, batch size 32, Gaussian-noise + scaling augmentation.

Documented deviations:
    * The paper uses a 988-d Kaggle feature set (3 classes, unknown subject structure).
      Here the four feature categories it names (statistical, frequency-domain,
      covariance/connectivity, eigenvalue) are recomputed per 4 s window from the 14
      shared channels -> 273-d vector (see `preprocessing.handcrafted_features`).
    * Conv channel widths, kernel sizes and pooling are not reported; the values in
      `build` are ours.
    * Max 400 epochs / patience 75 (best validation accuracy) are replaced by the benchmark-wide epoch cap and
      patience so all models get the same stopping rule.
    * The paper's evaluation is a random 80/20 split; its 99.19% is therefore a
      window-level (leaky) estimate, not a subject-independent one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .._common import make_builder


class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2), nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2), nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + out)


class DAHNet(nn.Module):
    def __init__(self, n_features: int = 273, n_classes: int = 2, cnn_channels=(64, 128, 128, 128),
                 lstm_hidden: int = 256, lstm_layers: int = 2, heads=(16, 8), dropout: float = 0.28):
        super().__init__()
        c1, c2, c3, c4 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(1, c1, 5, padding=2), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, 3, padding=1), nn.BatchNorm1d(c2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, 3, padding=1), nn.BatchNorm1d(c3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c3, c4, 3, padding=1), nn.BatchNorm1d(c4), nn.ReLU(), nn.MaxPool1d(2),
            ResidualBlock1d(c4),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(c4, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                              bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        dim = 2 * lstm_hidden
        self.attention = nn.Sequential(*(AttentionBlock(dim, h, dropout) for h in heads))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * dim, n_classes))
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"DAH-Net expects [batch, {self.n_features}], got {tuple(x.shape)}")
        h = self.cnn(x.unsqueeze(1)).transpose(1, 2)              # [B, L, C]
        h, _ = self.bilstm(h)
        h = self.attention(h)
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=1)
        return self.classifier(pooled)


build = make_builder(
    DAHNet,
    default_shape=(273,),
    shape_kwargs=lambda s: {"n_features": s[0]},
    params={"cnn_channels": (64, 128, 128, 128), "lstm_hidden": 256, "lstm_layers": 2,
            "heads": (16, 8), "dropout": 0.28},
)
