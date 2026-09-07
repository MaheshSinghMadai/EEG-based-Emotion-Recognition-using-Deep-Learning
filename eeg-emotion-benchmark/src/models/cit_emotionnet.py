"""
CIT-EmotionNet — CNN-Transformer Interactive network
===================================================
Two parallel encoders over a spatially-arranged DE feature map:
    * CNN branch          : local spatial-spectral patterns (conv stack over the topomap)
    * Transformer branch  : global dependencies (self-attention over electrode tokens)
    * Interaction module  : bidirectional cross-attention that lets each branch attend
                            to the other's representation before fusion + classification.

Input : DE features [B, 14, 5]
        -> scattered onto a [B, 5, H, W] topographic grid (bands = channels).
Output: class logits [B, n_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._common import make_builder

# ASSUMPTION: hand-built Emotiv EPOC topographic grid (rows: frontal -> occipital,
# cols: left -> right). Empty cells are zero-filled. An 8x9 grid keeps the 10-20
# geometry roughly intact while staying small enough to conv over.
EPOC_GRID_HW = (8, 9)
EPOC_GRID_POS = {
    "AF3": (0, 3), "AF4": (0, 5),
    "F7": (1, 1), "F3": (1, 3), "F4": (1, 5), "F8": (1, 7),
    "FC5": (2, 2), "FC6": (2, 6),
    "T7": (3, 0), "T8": (3, 8),
    "P7": (5, 1), "P8": (5, 7),
    "O1": (7, 3), "O2": (7, 5),
}
# Channel order must match preprocessing.CHANNEL_NAMES.
_CHANNEL_ORDER = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                  "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def de_to_grid(x: torch.Tensor, channel_order=None) -> torch.Tensor:
    """[B, C, Bands] -> [B, Bands, H, W] topographic tensor."""
    channel_order = channel_order or _CHANNEL_ORDER
    b, c, n_bands = x.shape
    h, w = EPOC_GRID_HW
    grid = x.new_zeros(b, n_bands, h, w)
    for ci, name in enumerate(channel_order[:c]):
        r, col = EPOC_GRID_POS[name]
        grid[:, :, r, col] = x[:, ci, :]
    return grid


class CNNBranch(nn.Module):
    def __init__(self, in_ch: int = 5, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ELU(),
        )
        self.width = width

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        f = self.net(grid)                      # [B, width, H, W]
        return f.flatten(2).transpose(1, 2)     # [B, H*W, width] token sequence


class TransformerBranch(nn.Module):
    def __init__(self, in_dim: int = 5, dim: int = 64, depth: int = 2, heads: int = 4,
                 n_tokens: int = 14, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        # x_tokens: [B, C, in_dim]  (one token per electrode)
        h = self.embed(x_tokens) + self.pos
        return self.encoder(h)                  # [B, C, dim]


class InteractionModule(nn.Module):
    """Bidirectional cross-attention fusion of the CNN and Transformer token streams."""

    def __init__(self, dim: int = 64, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.c2t = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.t2c = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_c = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)

    def forward(self, cnn_tok: torch.Tensor, tr_tok: torch.Tensor):
        c_att, _ = self.t2c(cnn_tok, tr_tok, tr_tok)   # CNN queries Transformer
        t_att, _ = self.c2t(tr_tok, cnn_tok, cnn_tok)  # Transformer queries CNN
        cnn_tok = self.norm_c(cnn_tok + c_att)
        tr_tok = self.norm_t(tr_tok + t_att)
        return cnn_tok, tr_tok


class CITEmotionNet(nn.Module):
    def __init__(self, n_channels: int = 14, n_bands: int = 5, dim: int = 64,
                 heads: int = 4, depth: int = 2, n_classes: int = 2, dropout: float = 0.2,
                 channel_order=None):
        super().__init__()
        self.channel_order = channel_order or _CHANNEL_ORDER
        self.cnn = CNNBranch(in_ch=n_bands, width=dim)
        self.trans = TransformerBranch(n_bands, dim, depth, heads, n_tokens=n_channels, dropout=dropout)
        self.interact = InteractionModule(dim, heads, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, Bands]
        grid = de_to_grid(x, self.channel_order)
        cnn_tok = self.cnn(grid)               # [B, H*W, dim]
        tr_tok = self.trans(x)                 # [B, C, dim]
        cnn_tok, tr_tok = self.interact(cnn_tok, tr_tok)
        fused = torch.cat([cnn_tok.mean(1), tr_tok.mean(1)], dim=-1)  # [B, 2*dim]
        return self.classifier(fused)


build = make_builder(
    CITEmotionNet,
    default_shape=(14, 5),
    shape_kwargs=lambda s: {"n_channels": s[0], "n_bands": s[1]},
    params={"dim": 64, "heads": 4, "depth": 2, "dropout": 0.2},
)
