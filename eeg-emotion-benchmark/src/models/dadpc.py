"""
DADPc — Domain Adaptive Deep Possibilistic Clustering
====================================================
Targeted at the subject-independent (LOSO) protocol. A shared feature extractor
maps DE features to an embedding; a linear head classifies the labelled source
(22 training subjects). Cross-subject domain shift is reduced by a clustering
objective in embedding space rather than a plain MMD term:

    L = CE(source)                                   # supervised
      + w_mmd  * MMD(source_emb, target_emb)         # keep a light global alignment
      + w_pc   * possibilistic_clustering(all_emb)   # compact, well-separated clusters
      + w_fe   * fuzzy_entropy(memberships)          # regularizes membership sharpness

`target_emb` is the UNLABELLED left-out subject (transductive DA — the test
subject's windows are visible during training but its labels are never used).

Input : DE features [B, 14, 5]
`forward(x)` -> class logits   (evaluation harness compatible)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import make_builder


class FeatureExtractor(nn.Module):
    def __init__(self, in_dim: int = 70, hidden: int = 128, emb_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.flatten(x, 1))


class DADPc(nn.Module):
    # ASSUMPTION: n_clusters = n_classes (one possibilistic cluster per emotion class).
    # `m` is the possibilistic fuzzifier; `eta` the per-cluster bandwidth (learnable).
    def __init__(self, in_shape=(14, 5), n_classes: int = 2, emb_dim: int = 64,
                 hidden: int = 128, n_clusters: int | None = None, m: float = 2.0,
                 dropout: float = 0.3):
        super().__init__()
        in_dim = in_shape[0] * in_shape[1]
        self.emb_dim = emb_dim
        self.m = m
        self.extractor = FeatureExtractor(in_dim, hidden, emb_dim, dropout)
        self.classifier = nn.Linear(emb_dim, n_classes)
        k = n_clusters or n_classes
        self.centers = nn.Parameter(torch.randn(k, emb_dim) * 0.1)
        self.log_eta = nn.Parameter(torch.zeros(k))    # eta = softplus(log_eta) + eps

    # -- forward / embed -------------------------------------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))

    # -- possibilistic memberships -------------------------------------------------
    def memberships(self, z: torch.Tensor) -> torch.Tensor:
        """u_ik = 1 / (1 + (d_ik^2 / eta_k)^(1/(m-1)))   -> [B, K], NOT summing to 1."""
        d2 = torch.cdist(z, self.centers).pow(2)                 # [B, K]
        eta = F.softplus(self.log_eta) + 1e-6
        ratio = (d2 / eta).clamp(min=1e-8).pow(1.0 / (self.m - 1.0))
        return 1.0 / (1.0 + ratio)

    # -- loss terms --------------------------------------------------------------
    def possibilistic_loss(self, z: torch.Tensor) -> torch.Tensor:
        u = self.memberships(z)
        d2 = torch.cdist(z, self.centers).pow(2)
        eta = F.softplus(self.log_eta) + 1e-6
        # classic PCM objective: sum u^m d^2 + sum eta * sum (u log u - u + 1)
        attract = (u.pow(self.m) * d2).sum(dim=1).mean()
        # per-cluster entropy term [B, K], weighted by that cluster's eta_k (broadcast
        # over dim=1) before summing over clusters and averaging over the batch.
        ent = u * torch.log(u.clamp(min=1e-8)) - u + 1.0
        penalty = (eta * ent).sum(dim=1).mean()
        return attract + penalty

    @staticmethod
    def fuzzy_entropy(u: torch.Tensor) -> torch.Tensor:
        """Mean Bernoulli-style fuzzy entropy of memberships (encourages decisive clusters)."""
        u = u.clamp(1e-8, 1 - 1e-8)
        return -(u * torch.log(u) + (1 - u) * torch.log(1 - u)).mean()

    @staticmethod
    def mmd_loss(zs: torch.Tensor, zt: torch.Tensor, sigmas=(1, 2, 4, 8, 16)) -> torch.Tensor:
        """Multi-kernel (RBF) Maximum Mean Discrepancy between source and target embeddings."""
        def rbf(a, b):
            d2 = torch.cdist(a, b).pow(2)
            return sum(torch.exp(-d2 / (2 * s ** 2)) for s in sigmas) / len(sigmas)
        return rbf(zs, zs).mean() + rbf(zt, zt).mean() - 2 * rbf(zs, zt).mean()

    def dadpc_loss(self, x_src, y_src, x_tgt,
                   w_mmd: float = 1.0, w_pc: float = 0.1, w_fe: float = 0.1):
        """Full training objective. Returns (total, parts_dict)."""
        z_src = self.embed(x_src)
        z_tgt = self.embed(x_tgt)
        ce = F.cross_entropy(self.classifier(z_src), y_src)
        mmd = self.mmd_loss(z_src, z_tgt)
        z_all = torch.cat([z_src, z_tgt], dim=0)
        pc = self.possibilistic_loss(z_all)
        fe = self.fuzzy_entropy(self.memberships(z_all))
        total = ce + w_mmd * mmd + w_pc * pc + w_fe * fe
        return total, {"ce": ce.item(), "mmd": mmd.item(), "pc": pc.item(), "fe": fe.item()}


build = make_builder(
    DADPc,
    default_shape=(14, 5),
    shape_kwargs=lambda s: {"in_shape": tuple(s)},
    params={"emb_dim": 64, "hidden": 128, "n_clusters": None, "m": 2.0, "dropout": 0.3},
)
