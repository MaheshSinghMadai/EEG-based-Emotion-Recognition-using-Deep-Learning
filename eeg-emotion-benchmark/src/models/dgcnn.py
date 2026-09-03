"""
DGCNN — Dynamical Graph Convolutional Neural Network
===================================================
Song, Zheng, Song, Cui (2020), "EEG Emotion Recognition Using Dynamical Graph
Convolutional Neural Networks", IEEE Transactions on Affective Computing.

Key idea: the adjacency matrix over the EEG electrodes is a *trainable parameter*,
learned jointly with the classifier, instead of being fixed from electrode geometry.
Graph filtering uses a Chebyshev polynomial approximation of the graph Laplacian.

Input : DE features [B, 14, 5]   (14 electrodes = graph nodes, 5 bands = node features)
Output: class logits [B, n_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebyGraphConv(nn.Module):
    """K-order Chebyshev graph convolution on a *learnable* adjacency.

    Follows the DGCNN formulation:
      * A is a learnable [N, N] matrix, constrained non-negative (ReLU) and symmetric.
      * L~ = 2L/lambda_max - I  with the normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
      * out = sum_{k=0}^{K-1} T_k(L~) X Theta_k
    """

    # ASSUMPTION: Chebyshev order K=2 (as used in the DGCNN paper's SEED experiments).
    def __init__(self, in_dim: int, out_dim: int, n_nodes: int = 14, K: int = 2):
        super().__init__()
        self.K = K
        self.n_nodes = n_nodes
        self.weight = nn.Parameter(torch.empty(K, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)
        # learnable adjacency, initialised near a weak fully-connected graph
        self.adj = nn.Parameter(torch.full((n_nodes, n_nodes), 0.1) + 0.9 * torch.eye(n_nodes))

    def normalized_laplacian(self) -> torch.Tensor:
        A = F.relu(self.adj)
        A = 0.5 * (A + A.t())                       # symmetrize
        A = A + torch.eye(self.n_nodes, device=A.device)  # self-loops
        deg = A.sum(-1)
        dinv = torch.diag(deg.clamp(min=1e-6).pow(-0.5))
        A_norm = dinv @ A @ dinv
        L = torch.eye(self.n_nodes, device=A.device) - A_norm
        # lambda_max ~ 2 for a normalized Laplacian -> L~ = L - I
        return L - torch.eye(self.n_nodes, device=A.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, in_dim]
        L = self.normalized_laplacian()
        Tx_0 = x
        out = torch.einsum("bni,io->bno", Tx_0, self.weight[0])
        if self.K > 1:
            Tx_1 = torch.einsum("nm,bmi->bni", L, x)
            out = out + torch.einsum("bni,io->bno", Tx_1, self.weight[1])
            Tx_prev, Tx_cur = Tx_0, Tx_1
            for k in range(2, self.K):
                Tx_next = 2 * torch.einsum("nm,bmi->bni", L, Tx_cur) - Tx_prev
                out = out + torch.einsum("bni,io->bno", Tx_next, self.weight[k])
                Tx_prev, Tx_cur = Tx_cur, Tx_next
        return out + self.bias


class DGCNN(nn.Module):
    def __init__(
        self,
        n_nodes: int = 14,
        in_dim: int = 5,
        hidden: int = 64,
        K: int = 2,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(n_nodes * in_dim)
        self.gc1 = ChebyGraphConv(in_dim, hidden, n_nodes, K)
        self.gc2 = ChebyGraphConv(hidden, hidden, n_nodes, K)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(n_nodes * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F]
        b, n, f = x.shape
        x = self.bn_in(x.reshape(b, n * f)).reshape(b, n, f)
        x = self.act(self.gc1(x))
        x = self.drop(x)
        x = self.act(self.gc2(x))
        x = torch.flatten(x, 1)
        return self.head(x)

    @torch.no_grad()
    def learned_adjacency(self) -> torch.Tensor:
        """Symmetrized non-negative adjacency from the first graph-conv layer (for plots)."""
        A = F.relu(self.gc1.adj)
        return (0.5 * (A + A.t())).cpu()


def build(cfg: dict) -> DGCNN:
    meta = cfg.get("meta", {})
    shape = meta.get("input_shape", (14, 5))
    n_nodes, in_dim = shape[0], shape[1]
    return DGCNN(
        n_nodes=n_nodes,
        in_dim=in_dim,
        hidden=cfg.get("hidden", 64),
        K=cfg.get("K", 2),
        n_classes=cfg.get("n_classes", 2),
        dropout=cfg.get("dropout", 0.3),
    )
