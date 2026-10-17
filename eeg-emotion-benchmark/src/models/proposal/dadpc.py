"""
DADPc — Domain Adaptive Deep Possibilistic Clustering (Dan et al., Front. Neurosci. 2025,
doi:10.3389/fnins.2025.1592070).

Implemented from the paper (Sections 3-5, Eq. 1, 8-20, 39):
    * Symmetric MLP auto-encoder on DE features; the bottleneck H^(M/2) is the deep
      feature space (Eq. 8-10).
    * J1  adaptive-loss reconstruction ||H^(M) - X||_sigma on source AND target, sigma=1e-3
      (Eq. 1, 11).
    * J2  cross-entropy on temperature-sharpened source predictions (Eq. 12-13); the
      L2 penalties on W and b (lambda1, lambda2) are realised as optimizer weight decay.
    * J3  possibilistic clustering with fuzzy-entropy regularisation (Eq. 17):
          lambda3 * sum_ik v_ik^2 d(h_i, c_k) + v_ik^2 ln v_ik^2 - v_ik^2
      For fixed features its minimiser is closed-form, v_ik^2 = exp(-d_ik); d is the
      cosine distance between L2-normalised features and centroids (Section 4.4).
    * Centroids initialised from source class means after AE pre-training, then
      recomputed every epoch from a memory bank of all source+target features using
      source labels and target pseudo-labels (Eq. 18-19, 39).

Documented deviations:
    * The paper alternates closed-form/SGD updates (Algorithm 2) and refreshes memory
      bank rows per mini-batch; here the bank is refreshed once per epoch with a full
      no-grad pass, and W/b are updated by Adam with V and C held fixed in-between.
    * Classification uses a linear head on the bottleneck (the paper's notation applies
      the softmax to h^(M), which has input dimensionality).
    * lambda3 is fixed (default 0.1) instead of being cross-validated over [1e-4, 1e5].
    * The paper's own DEAP protocol uses 14 subjects x 3 trials, 3 classes and 1 s DE
      windows; here every subject/trial, binary labels and 4 s windows are used.

Uses UNLABELED target-domain inputs (transductive domain adaptation). The benchmark
passes the test partition's inputs through `UnlabeledDataset`, which carries no labels.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...leakage import assert_unlabeled
from ...training import EarlyStopper, TrainRecipe, make_optimizer, validation_loss
from .._common import make_builder


def adaptive_loss(residual: torch.Tensor, sigma: float = 1e-3) -> torch.Tensor:
    """||Q||_sigma = sum_i (1 + sigma) ||q_i||^2 / (||q_i|| + sigma), averaged over rows."""
    norm = residual.flatten(1).norm(dim=1)
    return ((1.0 + sigma) * norm.pow(2) / (norm + sigma)).mean()


def _mlp(dims: Sequence[int], dropout: float, last_activation: bool) -> nn.Sequential:
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2 or last_activation:
            layers += [nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class DADPc(nn.Module):
    def __init__(self, in_shape=(14, 4), n_classes: int = 2, hidden: Sequence[int] = (256, 128),
                 emb_dim: int = 64, dropout: float = 0.2, tau: float = 0.5,
                 sigma: float = 1e-3, lambda3: float = 0.1):
        super().__init__()
        self.in_shape = tuple(in_shape)
        in_dim = 1
        for d in self.in_shape:
            in_dim *= d
        dims = [in_dim, *hidden, emb_dim]
        self.encoder = _mlp(dims, dropout, last_activation=False)
        self.decoder = _mlp(dims[::-1], dropout, last_activation=False)
        self.classifier = nn.Linear(emb_dim, n_classes)
        self.tau, self.sigma, self.lambda3 = tau, sigma, lambda3
        self.register_buffer("centroids", torch.zeros(n_classes, emb_dim))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))

    # --- loss terms ----------------------------------------------------------------
    def reconstruction(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return adaptive_loss(self.decoder(z) - x.flatten(1), self.sigma)

    def sharpened_ce(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.classifier(z) / self.tau, y)

    def cosine_distance(self, z: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.normalize(z, dim=1) @ F.normalize(self.centroids, dim=1).T   # [B, K]

    def clustering(self, z: torch.Tensor) -> torch.Tensor:
        d = self.cosine_distance(z)
        w = torch.exp(-d).detach()                        # v^2, closed form for fixed features
        return self.lambda3 * (w * d + w * torch.log(w.clamp_min(1e-8)) - w).sum(1).mean()

    @torch.no_grad()
    def update_centroids(self, z_src: torch.Tensor, y_src: torch.Tensor,
                         z_tgt: Optional[torch.Tensor]) -> None:
        """Eq. 39: class means over source labels and target pseudo-labels."""
        z_src = F.normalize(z_src, dim=1)
        feats, labels = [z_src], [y_src]
        if z_tgt is not None and z_tgt.numel():
            feats.append(F.normalize(z_tgt, dim=1))
            labels.append(self.cosine_distance(z_tgt).argmin(dim=1))
        z, y = torch.cat(feats), torch.cat(labels)
        for k in range(self.centroids.shape[0]):
            if (y == k).any():
                self.centroids[k] = z[y == k].mean(0)


build = make_builder(
    DADPc,
    default_shape=(14, 4),
    shape_kwargs=lambda s: {"in_shape": tuple(s)},
    params={"hidden": (256, 128), "emb_dim": 64, "dropout": 0.2, "tau": 0.5,
            "sigma": 1e-3, "lambda3": 0.1},
)


@torch.no_grad()
def _embed_all(model: DADPc, loader: DataLoader, device: torch.device, labeled: bool):
    model.eval()
    zs, ys = [], []
    for batch in loader:
        x = (batch[0] if labeled else batch).to(device)
        zs.append(model.embed(x))
        if labeled:
            ys.append(batch[1].to(device))
    return torch.cat(zs), (torch.cat(ys) if labeled else None)


def fit_dadpc(
    model: DADPc,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    recipe: TrainRecipe,
    unlabeled_loader: Optional[DataLoader] = None,
    pretrain_epochs: int = 10,
    **_: object,
) -> Dict:
    """AE pre-training on source+target, then joint J1 + J2 + J3 optimisation."""
    model.to(device)
    opt = make_optimizer(model, recipe)
    start = time.time()
    pretrain_epochs = min(pretrain_epochs, recipe.max_epochs)

    def target_batches():
        while True:
            if unlabeled_loader is None:
                yield None
                continue
            for xt in unlabeled_loader:
                assert_unlabeled(xt)
                yield xt.to(device)

    tgt = target_batches()

    # 1) auto-encoder pre-training (J1 only)
    for _ in range(pretrain_epochs):
        model.train()
        for xs, _y in train_loader:
            xs = xs.to(device)
            xt = next(tgt)
            x = xs if xt is None else torch.cat([xs, xt])
            loss = model.reconstruction(x, model.embed(x))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # 2) centroids from source class means
    z_src, y_src = _embed_all(model, train_loader, device, labeled=True)
    model.update_centroids(z_src, y_src, None)

    # 3) joint optimisation with memory-bank centroid refresh each epoch
    stopper = EarlyStopper(model, recipe)
    epoch = 0
    for epoch in range(recipe.max_epochs):
        model.train()
        for xs, ys in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            xt = next(tgt)
            zs = model.embed(xs)
            loss = model.reconstruction(xs, zs) + model.sharpened_ce(zs, ys)
            z_all = zs
            if xt is not None:
                zt = model.embed(xt)
                loss = loss + model.reconstruction(xt, zt)
                z_all = torch.cat([zs, zt])
            loss = loss + model.clustering(z_all)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if recipe.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
            opt.step()
        z_src, y_src = _embed_all(model, train_loader, device, labeled=True)
        z_tgt = _embed_all(model, unlabeled_loader, device, labeled=False)[0] if unlabeled_loader else None
        model.update_centroids(z_src, y_src, z_tgt)
        monitor = validation_loss(model, val_loader, device) if val_loader is not None else 0.0
        if stopper.step(epoch, monitor):
            break
    stopper.restore()
    return {"train_time_sec": time.time() - start, "epochs_run": epoch + 1,
            "best_epoch": stopper.best_epoch + 1, "best_val_loss": stopper.best,
            "used_unlabeled_target": unlabeled_loader is not None}
