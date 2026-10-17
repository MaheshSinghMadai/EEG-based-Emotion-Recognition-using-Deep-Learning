"""
STGATE — Spatial-Temporal Graph Attention network with a Transformer Encoder
(Li et al., Front. Hum. Neurosci. 2023, doi:10.3389/fnhum.2023.1169949).

Structure (from the published description):
    1. Transformer Learning Block (TLB): for every electrode, the sequence of
       frequency-band DE vectors over time (1 s steps inside the 4 s window) is encoded
       by a Transformer encoder -> temporal-frequency node features.
    2. Spatial-Temporal Graph Attention (STGAT): multi-head graph attention over the
       electrodes at every time step. Attention logits are biased by a LEARNABLE
       (dynamic) adjacency matrix, initialised from electrode distances and sparsified
       to the top-k neighbours per node; temporal attention then pools over time.
    3. Readout + fully connected classifier.

Input : de_sequence [B, C, S, Bands]   (C=14 electrodes, S=4 one-second steps)

Documented deviations / to verify: the full paper PDF is not in Papers/. Layer
counts, head counts, the exact sparsification rule and the DE time resolution must be
checked against the paper's methods section and corrected here if they differ.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data_loader import distance_adjacency
from ...preprocessing import COMMON_CHANNELS
from .._common import make_builder


class TransformerLearningBlock(nn.Module):
    def __init__(self, n_bands: int, n_steps: int, dim: int, heads: int, depth: int, dropout: float):
        super().__init__()
        self.embed = nn.Linear(n_bands, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_steps, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # [B*C, S, Bands]
        return self.encoder(self.embed(x) + self.pos)             # [B*C, S, dim]


class DynamicGraphAttention(nn.Module):
    """Multi-head GAT whose logits are biased by a learnable, top-k sparsified adjacency."""

    def __init__(self, dim: int, heads: int, n_nodes: int, top_k: int, dropout: float,
                 init_adjacency: torch.Tensor):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.head_dim, self.top_k = heads, dim // heads, min(top_k, n_nodes)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.att_src = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)
        self.att_dst = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)
        self.adj_logits = nn.Parameter(torch.log(init_adjacency.clamp_min(1e-4)))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def adjacency(self) -> torch.Tensor:
        a = torch.sigmoid(self.adj_logits)
        a = 0.5 * (a + a.T)
        keep = torch.zeros_like(a).scatter_(1, a.topk(self.top_k, dim=1).indices, 1.0)
        return a * keep                                           # sparse, learnable

    def forward(self, h: torch.Tensor) -> torch.Tensor:          # [N, C, dim]
        n, c, _ = h.shape
        wh = self.proj(h).view(n, c, self.heads, self.head_dim)
        e_src = (wh * self.att_src).sum(-1)                       # [N, C, H]
        e_dst = (wh * self.att_dst).sum(-1)
        logits = F.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1), 0.2)   # [N, C, C, H]
        adj = self.adjacency()
        logits = logits + torch.log(adj.clamp_min(1e-9)).unsqueeze(0).unsqueeze(-1)
        alpha = self.drop(torch.softmax(logits, dim=2))
        out = torch.einsum("nijh,njhd->nihd", alpha, wh).reshape(n, c, -1)
        return self.norm(h + F.elu(out))


class STGATE(nn.Module):
    def __init__(self, n_channels: int = 14, n_steps: int = 4, n_bands: int = 4, n_classes: int = 2,
                 dim: int = 64, heads: int = 4, tlb_depth: int = 2, gat_layers: int = 2,
                 top_k: int = 6, dropout: float = 0.2, channel_names=None):
        super().__init__()
        channel_names = list(channel_names or COMMON_CHANNELS)[:n_channels]
        init_adj = torch.from_numpy(distance_adjacency(channel_names))
        self.n_channels, self.n_steps, self.n_bands = n_channels, n_steps, n_bands
        self.tlb = TransformerLearningBlock(n_bands, n_steps, dim, heads, tlb_depth, dropout)
        self.gat = nn.ModuleList(
            DynamicGraphAttention(dim, heads, n_channels, top_k, dropout, init_adj)
            for _ in range(gat_layers)
        )
        self.temporal_score = nn.Linear(dim, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(n_channels * dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, s, f = x.shape
        if (c, s, f) != (self.n_channels, self.n_steps, self.n_bands):
            raise ValueError(f"STGATE expects [B, {self.n_channels}, {self.n_steps}, "
                             f"{self.n_bands}], got {tuple(x.shape)}")
        h = self.tlb(x.reshape(b * c, s, f)).view(b, c, s, -1)    # temporal-frequency nodes
        h = h.permute(0, 2, 1, 3).reshape(b * s, c, -1)           # graph per time step
        for layer in self.gat:
            h = layer(h)
        h = h.view(b, s, c, -1)
        w = torch.softmax(self.temporal_score(h), dim=1)          # temporal attention
        return self.classifier((w * h).sum(1).flatten(1))


build = make_builder(
    STGATE,
    default_shape=(14, 4, 4),
    shape_kwargs=lambda s: {"n_channels": s[0], "n_steps": s[1], "n_bands": s[2]},
    params={"dim": 64, "heads": 4, "tlb_depth": 2, "gat_layers": 2, "top_k": 6, "dropout": 0.2},
)
