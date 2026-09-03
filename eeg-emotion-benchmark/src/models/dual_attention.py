"""
Dual-Attention Transformer-CNN-BiLSTM
===================================
A hybrid sequence model with two explicit attention mechanisms:

    raw [B, 14, T]
      -> (1) CHANNEL attention   : squeeze-and-excitation weights over the 14 electrodes
      -> CNN feature extractor    : temporal conv stack -> [B, F, T']
      -> BiLSTM                    : bidirectional temporal sequence modelling
      -> Transformer encoder block : global temporal self-attention
      -> (2) TEMPORAL attention   : additive attention pooling over time steps
      -> linear classifier -> logits [B, n_classes]

Input : raw windowed signal [B, 14, T]
Output: class logits [B, n_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """SE-style gating over EEG channels (attention mechanism #1)."""

    def __init__(self, n_channels: int, reduction: int = 2):
        super().__init__()
        hidden = max(1, n_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, n_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        s = x.mean(dim=-1)                        # [B, C] channel descriptor
        w = torch.sigmoid(self.mlp(s)).unsqueeze(-1)   # [B, C, 1]
        return x * w, w.squeeze(-1)


class TemporalAttentionPool(nn.Module):
    """Additive (Bahdanau-style) attention pooling over time (attention mechanism #2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, dim]
        e = self.v(torch.tanh(self.proj(h))).squeeze(-1)   # [B, T]
        a = torch.softmax(e, dim=1)
        return (h * a.unsqueeze(-1)).sum(dim=1)            # [B, dim]


class DualAttentionNet(nn.Module):
    def __init__(self, n_channels: int = 14, n_samples: int = 512, n_classes: int = 2,
                 cnn_width: int = 64, lstm_hidden: int = 64, tr_heads: int = 4,
                 tr_depth: int = 1, dropout: float = 0.3):
        super().__init__()
        self.chan_attn = ChannelAttention(n_channels)

        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, cnn_width, 7, padding=3), nn.BatchNorm1d(cnn_width), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_width, cnn_width, 5, padding=2), nn.BatchNorm1d(cnn_width), nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.bilstm = nn.LSTM(cnn_width, lstm_hidden, num_layers=1,
                              batch_first=True, bidirectional=True)
        d = lstm_hidden * 2

        layer = nn.TransformerEncoderLayer(d, tr_heads, d * 4, dropout,
                                           batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, tr_depth)

        self.temporal_pool = TemporalAttentionPool(d)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, n_classes)
        )
        self._last_channel_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x, cw = self.chan_attn(x)
        self._last_channel_weights = cw.detach()
        h = self.cnn(x)                 # [B, cnn_width, T']
        h = h.transpose(1, 2)          # [B, T', cnn_width]
        h, _ = self.bilstm(h)         # [B, T', 2*lstm_hidden]
        h = self.transformer(h)       # [B, T', d]
        pooled = self.temporal_pool(h)
        return self.classifier(pooled)


def build(cfg: dict) -> DualAttentionNet:
    meta = cfg.get("meta", {})
    shape = meta.get("input_shape", (14, 512))
    return DualAttentionNet(
        n_channels=shape[0], n_samples=shape[1],
        n_classes=cfg.get("n_classes", 2),
        cnn_width=cfg.get("cnn_width", 64),
        lstm_hidden=cfg.get("lstm_hidden", 64),
        tr_heads=cfg.get("tr_heads", 4),
        tr_depth=cfg.get("tr_depth", 1),
        dropout=cfg.get("dropout", 0.3),
    )
