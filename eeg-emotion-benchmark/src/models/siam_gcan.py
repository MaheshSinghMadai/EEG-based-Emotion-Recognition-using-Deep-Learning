"""
Siam-GCAN — Siamese Graph Convolutional Attention Network
=======================================================
Twin-branch, weight-shared GCN encoder with multi-head self-attention over the
electrode graph. Trained on trial pairs with a combined objective:

    L = CE(logits_a, y_a) + CE(logits_b, y_b) + lambda * contrastive(emb_a, emb_b, pair_label)

where `pair_label = 1` if the two windows share the same emotion class.

Input (single branch) : DE features [B, 14, 5]
`forward(x)`           : returns class logits  (so the shared evaluation harness works)
`forward_pair(xa, xb)` : returns (emb_a, emb_b, logits_a, logits_b) for training
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import make_builder


class GraphAttnLayer(nn.Module):
    """GCN propagation on a learnable adjacency + multi-head attention over nodes."""

    def __init__(self, in_dim: int, out_dim: int, n_nodes: int = 14, heads: int = 4):
        super().__init__()
        self.n_nodes = n_nodes
        self.lin = nn.Linear(in_dim, out_dim)
        self.adj = nn.Parameter(torch.full((n_nodes, n_nodes), 0.1) + 0.9 * torch.eye(n_nodes))
        self.attn = nn.MultiheadAttention(out_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(out_dim)

    def _norm_adj(self) -> torch.Tensor:
        A = F.relu(self.adj)
        A = 0.5 * (A + A.t()) + torch.eye(self.n_nodes, device=A.device)
        d = A.sum(-1).clamp(min=1e-6).pow(-0.5)
        return (d.unsqueeze(1) * A * d.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, in_dim]
        A = self._norm_adj()
        h = torch.einsum("nm,bmi->bni", A, x)      # graph propagation
        h = F.elu(self.lin(h))
        a, _ = self.attn(h, h, h)                   # attention over electrodes
        return self.norm(h + a)


class GCANEncoder(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 64, n_nodes: int = 14,
                 heads: int = 4, emb_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.l1 = GraphAttnLayer(in_dim, hidden, n_nodes, heads)
        self.l2 = GraphAttnLayer(hidden, hidden, n_nodes, heads)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.l1(x)
        h = self.drop(h)
        h = self.l2(h)
        h = h.mean(dim=1)                # global mean pool over electrodes
        return self.proj(h)             # [B, emb_dim]


class SiamGCAN(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 64, n_nodes: int = 14,
                 heads: int = 4, emb_dim: int = 64, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = GCANEncoder(in_dim, hidden, n_nodes, heads, emb_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(emb_dim, n_classes),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))

    def forward_pair(self, xa: torch.Tensor, xb: torch.Tensor):
        ea, eb = self.embed(xa), self.embed(xb)
        return ea, eb, self.classifier(ea), self.classifier(eb)


def contrastive_loss(emb_a: torch.Tensor, emb_b: torch.Tensor,
                     pair_label: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Contrastive loss (Hadsell et al.). pair_label = 1 -> similar, 0 -> dissimilar."""
    d = F.pairwise_distance(emb_a, emb_b)
    pos = pair_label * d.pow(2)
    neg = (1 - pair_label) * F.relu(margin - d).pow(2)
    return 0.5 * (pos + neg).mean()


build = make_builder(
    SiamGCAN,
    default_shape=(14, 5),
    shape_kwargs=lambda s: {"n_nodes": s[0], "in_dim": s[1]},
    params={"hidden": 64, "heads": 4, "emb_dim": 64, "dropout": 0.3},
)
