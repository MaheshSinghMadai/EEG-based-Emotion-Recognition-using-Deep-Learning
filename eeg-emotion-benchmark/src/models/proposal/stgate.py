"""
STGATE — Spatial-Temporal Graph Attention network with a Transformer Encoder
(Li et al., Front. Hum. Neurosci. 2023, 17:1169949, doi:10.3389/fnhum.2023.1169949).

Reproduced from the paper (Sec. 3, Fig. 1, Sec. 4.2):
    1. Transformer Learning Block (TLB, Sec. 3.1), per electrode:
       multi-kernel 2D convolutions over the (band x time) DE map — kernel sizes 1, 3, 3,
       the 3x3 convolutions followed by a 1x1 convolution — that downsample the frequency
       bands, then a Transformer encoder with a learnable positional embedding over time.
       Output X_h in R^{B x N x C x Tr} (N electrodes, C conv channels, Tr time steps).
    2. Spatial-Temporal Graph Attention (STGAT, Sec. 3.3):
       * spatial attention  S = V . tanh((X_h W1) W2 + b_s)          (Eq. 9)
         A = (S - E[S]) / sqrt(Var[S]) with mini-batch statistics     (Eq. 10)
         keep the top-k = 10 edges of every row of A                  (Eq. 11)
       * temporal attention T = V_t . tanh(...) + b_t, normalised the same way and applied
         as X^_h = T^ X_h                                             (Eqs. 12-14)
       * multi-head graph attention (Eqs. 1-4, head outputs averaged, ELU) on X^_h with
         A as the adjacency matrix.
    3. Fully connected classifier, dropout 0.3.
    Recipe (Sec. 4.2): Adam, lr 1e-5, batch 16, dropout 0.3 (set in benchmark_proposal).

Input : de_sequence [B, N, S, Bands]   (N=14 electrodes, S=4 one-second DE steps)

Documented deviations / interpretations:
    * Eq. 12 as printed gives T the same N x N shape as S (W3 in R^Tr, W4 in R^{C x N}),
      which cannot weight time steps in Eq. 14. The temporal attention therefore uses the
      ASTGCN form the equations are taken from (Guo et al., AAAI 2019):
      T = V_t . tanh(((X_h^T U1) U2)(U3 X_h) + b_t) in R^{Tr x Tr}.
    * Top-k only selects neighbours and is not differentiable, so with a pure mask the
      spatial-attention parameters would receive no gradient. The kept entries of A are
      therefore added to the GAT logits (masked, additive bias), which is how A acts as a
      weighted adjacency matrix.
    * "The initial set of edges ... is determined by the kNN algorithm" (Sec. 3.2):
      b_s is initialised to the kNN graph of the electrode positions, so at initialisation
      the dynamic adjacency favours spatial neighbours.
    * The paper uses 5 bands (SEED); the harmonized pipeline has 4 (theta-gamma, <=45 Hz).
      Conv channel count, Transformer depth/heads, GAT heads and hidden sizes are not
      reported; the values in `build` are ours.
    * Training stops by the benchmark's shared validation rule, not "training loss < 0.15".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data_loader import distance_adjacency
from ...preprocessing import COMMON_CHANNELS
from .._common import make_builder


def knn_adjacency(channel_names, k: int) -> torch.Tensor:
    """Symmetric 0/1 graph linking each electrode to its k nearest scalp neighbours."""
    sim = distance_adjacency(channel_names)                  # larger = closer
    np.fill_diagonal(sim, -np.inf)
    nearest = np.argsort(-sim, axis=1)[:, :k]
    adj = np.zeros_like(sim)
    np.put_along_axis(adj, nearest, 1.0, axis=1)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 1.0)
    return torch.from_numpy(adj.astype(np.float32))


class TransformerLearningBlock(nn.Module):
    """Multi-kernel 2D convolutions over (band x time), then a Transformer over time."""

    def __init__(self, n_steps: int, channels: int, heads: int, depth: int, dropout: float):
        super().__init__()

        def branch(*kernels):
            layers = []
            for i, k in enumerate(kernels):
                layers += [nn.Conv2d(1 if i == 0 else channels, channels, k, padding=k // 2),
                           nn.BatchNorm2d(channels), nn.GELU()]
            return nn.Sequential(*layers)

        self.branches = nn.ModuleList([branch(1), branch(3), branch(3, 3)])
        self.fuse = nn.Sequential(nn.Conv2d(3 * channels, channels, 1), nn.BatchNorm2d(channels), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, n_steps, channels))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(channels, heads, 2 * channels, dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # [B*N, Bands, S]
        h = x.unsqueeze(1)
        h = self.fuse(torch.cat([b(h) for b in self.branches], dim=1))
        h = h.mean(dim=2).transpose(1, 2)                         # pool bands -> [B*N, S, C]
        return self.encoder(h + self.pos)


class SpatialAttention(nn.Module):
    """Eqs. 9-11: input-dependent adjacency, batch-normalised, top-k per row."""

    def __init__(self, n_nodes: int, channels: int, n_steps: int, top_k: int, init_edges: torch.Tensor):
        super().__init__()
        self.top_k = min(top_k, n_nodes)
        self.w1 = nn.Parameter(torch.randn(n_steps) / n_steps ** 0.5)
        self.w2 = nn.Parameter(torch.randn(channels, n_nodes) / channels ** 0.5)
        self.bs = nn.Parameter(init_edges.clone())
        self.v = nn.Parameter(torch.eye(n_nodes) + 0.01 * torch.randn(n_nodes, n_nodes))
        self.norm = nn.BatchNorm1d(n_nodes * n_nodes, affine=False)

    def forward(self, xh: torch.Tensor):                         # [B, N, C, Tr]
        b, n = xh.shape[:2]
        s = self.v @ torch.tanh(torch.einsum("bnct,t->bnc", xh, self.w1) @ self.w2 + self.bs)
        a = self.norm(s.reshape(b, n * n)).view(b, n, n)
        mask = torch.zeros_like(a, dtype=torch.bool).scatter_(2, a.topk(self.top_k, dim=2).indices, True)
        return a, mask


class TemporalAttention(nn.Module):
    """Eqs. 12-14 in the ASTGCN form: a [Tr x Tr] weighting applied to X_h over time."""

    def __init__(self, n_nodes: int, channels: int, n_steps: int):
        super().__init__()
        self.u1 = nn.Parameter(torch.randn(n_nodes) / n_nodes ** 0.5)
        self.u2 = nn.Parameter(torch.randn(channels, n_nodes) / channels ** 0.5)
        self.u3 = nn.Parameter(torch.randn(channels) / channels ** 0.5)
        self.bt = nn.Parameter(torch.zeros(n_steps, n_steps))
        self.v = nn.Parameter(torch.eye(n_steps) + 0.01 * torch.randn(n_steps, n_steps))
        self.norm = nn.BatchNorm1d(n_steps * n_steps, affine=False)

    def forward(self, xh: torch.Tensor) -> torch.Tensor:         # [B, N, C, Tr]
        b, _, _, t = xh.shape
        lhs = torch.einsum("bnct,n->btc", xh, self.u1) @ self.u2              # [B, Tr, N]
        rhs = torch.einsum("bnct,c->bnt", xh, self.u3)                        # [B, N, Tr]
        e = self.v @ torch.tanh(lhs @ rhs + self.bt)
        e = self.norm(e.reshape(b, t * t)).view(b, t, t)
        return torch.einsum("bnct,bts->bncs", xh, e)


class GraphAttention(nn.Module):
    """Multi-head GAT (Eqs. 1-4) restricted to the top-k edges, biased by their weights."""

    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.proj = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, a: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = h.shape
        wh = self.proj(h).view(b, n, self.heads, self.out_dim)
        e = (wh * self.att_dst).sum(-1).unsqueeze(2) + (wh * self.att_src).sum(-1).unsqueeze(1)
        e = F.leaky_relu(e, 0.2) + a.unsqueeze(-1)                            # [B, N, N, H]
        e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = self.drop(torch.softmax(e, dim=2))
        return F.elu(torch.einsum("bijh,bjhd->bihd", alpha, wh).mean(dim=2))  # Eq. 4: head mean


class STGATE(nn.Module):
    def __init__(self, n_channels: int = 14, n_steps: int = 4, n_bands: int = 4, n_classes: int = 2,
                 conv_channels: int = 32, tlb_heads: int = 4, tlb_depth: int = 2,
                 gat_dim: int = 64, gat_heads: int = 4, top_k: int = 10, knn: int = 4,
                 dropout: float = 0.3, channel_names=None):
        super().__init__()
        channel_names = list(channel_names or COMMON_CHANNELS)[:n_channels]
        self.n_channels, self.n_steps, self.n_bands = n_channels, n_steps, n_bands
        self.tlb = TransformerLearningBlock(n_steps, conv_channels, tlb_heads, tlb_depth, dropout)
        self.spatial = SpatialAttention(n_channels, conv_channels, n_steps, top_k,
                                        knn_adjacency(channel_names, knn))
        self.temporal = TemporalAttention(n_channels, conv_channels, n_steps)
        self.gat = GraphAttention(conv_channels * n_steps, gat_dim, gat_heads, dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_channels * gat_dim, n_classes))

    def adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse dynamic adjacency per sample (for visualisation, cf. paper Fig. 3)."""
        a, mask = self.spatial(self._tlb(x))
        return a * mask

    def _tlb(self, x: torch.Tensor) -> torch.Tensor:
        b, c, s, f = x.shape
        if (c, s, f) != (self.n_channels, self.n_steps, self.n_bands):
            raise ValueError(f"STGATE expects [B, {self.n_channels}, {self.n_steps}, "
                             f"{self.n_bands}], got {tuple(x.shape)}")
        h = self.tlb(x.reshape(b * c, s, f).transpose(1, 2))     # [B*N, S, C]
        return h.view(b, c, s, -1).permute(0, 1, 3, 2)            # X_h: [B, N, C, Tr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xh = self._tlb(x)
        a, mask = self.spatial(xh)
        xh = self.temporal(xh)
        h = self.gat(xh.flatten(2), a, mask)                      # [B, N, gat_dim]
        return self.classifier(h.flatten(1))


build = make_builder(
    STGATE,
    default_shape=(14, 4, 4),
    shape_kwargs=lambda s: {"n_channels": s[0], "n_steps": s[1], "n_bands": s[2]},
    params={"conv_channels": 32, "tlb_heads": 4, "tlb_depth": 2, "gat_dim": 64, "gat_heads": 4,
            "top_k": 10, "knn": 4, "dropout": 0.3},
)
