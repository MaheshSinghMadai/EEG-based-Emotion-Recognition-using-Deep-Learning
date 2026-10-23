"""
CLISA — Contrastive Learning for Inter-Subject Alignment (Shen et al., IEEE TAFFC 2022,
doi:10.1109/TAFFC.2022.3164516).

Reproduced from the paper (Sec. 3, Sec. 4.4, Table 1, Algorithm 1):
    Contrastive stage
      * Data sampler (Sec. 3.1-A): a minibatch is one pair of subjects (A, B); from every
        trial one sample is drawn at the same time position for both subjects, so sample
        i of A and sample i of B saw the same stimulus segment (positive pair) and all
        2(N-1) other samples are negatives. Every subject pair of the training set is
        enumerated once per epoch.
      * Base encoder (Sec. 3.1-B): K1 = 16 spatial filters spanning all M electrodes,
        then K2 = 16 temporal filters of length P1 shared across the latent signals
        (padded to keep length T) -> K2 x K1 x T.
      * Projector (Sec. 3.1-C): ELU + average pooling 1 x S, depthwise spatial conv
        (C = 2 filters of size K1 per map -> 32 maps), ELU, depthwise temporal conv
        (C = 2 filters of length P2 per map -> 64 maps), ELU, flatten.
      * Stratified normalisation (Sec. 4.4) per subject and channel at the base-encoder
        input, after average pooling and after the projector's temporal conv.
      * NT-Xent on cosine similarities (Eqs. 6-8).
      * Adam, lr 7e-4, weight decay 0.015, cosine annealing with three warm restarts,
        100 epochs, early stopping after 30 epochs without validation-accuracy gain.
    Prediction stage
      * Frozen base encoder; "trained DE" features 0.5 log(2 pi e var) of every one of the
        K2 x K1 = 256 encoder maps over 1 s samples (Eqs. 9-10).
      * Three-layer MLP (two hidden layers of 30 ReLU units), Adam lr 5e-4, batch 256,
        cross-entropy (Sec. 4.4).

Both DEAP (preprocessed release, trials sorted by video id) and DREAMER (fixed clip
order) present the same stimuli to every subject, so (trial_index, window_index) is a
valid stimulus-time key. Only TRAINING subjects are ever paired.

Documented deviations:
    * Filter lengths scaled to 128 Hz as the paper does between datasets: P1 = 31
      (60 at 250 Hz, 48 at 200 Hz), S = 15 (30 / 24), P2 = 3 (6 / 4).
    * Contrastive samples are the benchmark's 4 s windows (paper: 5 s on THU-EP, 30 s on
      SEED); the temperature is not reported (0.07 used).
    * The paper LDS-smooths 1 s DE features across consecutive samples of a trial. Here the
      prediction unit is one 4 s window, so the four 1 s DE vectors inside the window are
      averaged; no information crosses window (or trial) boundaries.
    * The paper normalises DE features adaptively with incoming test-subject data
      (decay 0.99). That is unlabeled test-time adaptation; to keep CLISA inductive like the
      other models, features are z-scored with training-set statistics only.
    * The MLP weight decay was chosen by cross-validation from {0.005, ..., 0.125};
      the middle value 0.025 is fixed here. Its early stopping uses the benchmark's shared
      validation rule.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ...training import EarlyStopper, TrainRecipe, make_optimizer
from .._common import make_builder


def stratified_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """z-score every channel/map over (samples, time) of a single-subject batch [B, ..., T]."""
    dims = (0, x.ndim - 1)
    return (x - x.mean(dim=dims, keepdim=True)) / (x.std(dim=dims, keepdim=True) + eps)


class CLISA(nn.Module):
    def __init__(self, n_channels: int = 14, n_samples: int = 512, n_classes: int = 2,
                 n_spatial: int = 16, n_temporal: int = 16, temporal_kernel: int = 31,
                 pool: int = 15, proj_kernel: int = 3, proj_multiplier: int = 2,
                 segment_samples: int = 128, mlp_hidden: int = 30, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.segment_samples = segment_samples
        c, k1, k2 = proj_multiplier, n_spatial, n_temporal
        self.spatial = nn.Conv2d(1, k1, (n_channels, 1), bias=False)                    # Eq. 1
        self.temporal = nn.Conv2d(1, k2, (1, temporal_kernel),
                                  padding=(0, temporal_kernel // 2), bias=False)        # Eq. 2
        self.base = nn.ModuleList([self.spatial, self.temporal])
        self.pool = nn.AvgPool2d((1, pool), stride=(1, pool))                           # Eq. 3
        self.proj_spatial = nn.Conv2d(k2, c * k2, (k1, 1), groups=k2)                   # Eq. 4
        self.proj_temporal = nn.Conv2d(c * k2, c * c * k2, (1, proj_kernel), groups=c * k2)  # Eq. 5
        self.projector = nn.ModuleList([self.proj_spatial, self.proj_temporal])
        self.feature_dim = k1 * k2
        self.register_buffer("feat_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feat_std", torch.ones(self.feature_dim))
        self.classifier = nn.Sequential(                                                # Eq. 11
            nn.Linear(self.feature_dim, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, n_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, M, T] -> aligned representation [B, K2, K1, T] (Eq. 9)."""
        s = self.spatial(x.unsqueeze(1))                       # [B, K1, 1, T]
        return self.temporal(s.transpose(1, 2))                # [B, K2, K1, T]

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Single-subject batch -> L2-normalised projection used by the contrastive loss."""
        h = self.pool(F.elu(self.encode(stratified_norm(x))))
        h = F.elu(self.proj_spatial(stratified_norm(h)))
        h = stratified_norm(F.elu(self.proj_temporal(h)))
        return F.normalize(h.flatten(1), dim=1)

    def contrastive_logits(self, xa: torch.Tensor, xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Row i of xa (subject A) and row i of xb (subject B) share a stimulus segment."""
        z = torch.cat([self.project(xa), self.project(xb)])
        logits = z @ z.T / self.temperature
        logits = logits.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), float("-inf"))
        n = xa.shape[0]
        targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
        return logits, targets

    def contrastive_loss(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        logits, targets = self.contrastive_logits(xa, xb)
        return F.cross_entropy(logits, targets, reduction="sum")                       # Eq. 8

    def de_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trained DE (Eq. 10) of every encoder map over 1 s samples, averaged over the window."""
        h = self.encode(x).flatten(1, 2)                       # [B, K2*K1, T]
        n_seg = max(1, h.shape[-1] // self.segment_samples)
        h = h[..., : n_seg * self.segment_samples].reshape(*h.shape[:2], n_seg, -1)
        de = 0.5 * torch.log(2 * math.pi * math.e * h.var(dim=-1).clamp_min(1e-8))
        return de.mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier((self.de_features(x) - self.feat_mean) / self.feat_std)


build = make_builder(
    CLISA,
    default_shape=(14, 512),
    shape_kwargs=lambda s: {"n_channels": s[0], "n_samples": s[1]},
    params={"n_spatial": 16, "n_temporal": 16, "temporal_kernel": 31, "pool": 15,
            "proj_kernel": 3, "proj_multiplier": 2, "segment_samples": 128,
            "mlp_hidden": 30, "temperature": 0.07},
)


class StimulusPairSampler:
    """Algorithm 1 data sampler: one stimulus-aligned sample per trial for a subject pair."""

    def __init__(self, subject_ids, trial_ids, window_ids, seed: int = 42):
        self.rows: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(dict)
        for row, (s, t, w) in enumerate(zip(subject_ids, trial_ids, window_ids)):
            self.rows[s][(int(t), int(w))] = row
        self.subjects = sorted(self.rows)
        self.rng = np.random.default_rng(seed)
        if len(self.subjects) < 2:
            raise ValueError("CLISA needs at least two subjects for pairing.")

    def sample_pair(self, a: str, b: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        by_trial: Dict[int, List[int]] = defaultdict(list)
        for t, w in set(self.rows[a]) & set(self.rows[b]):
            by_trial[t].append(w)
        if len(by_trial) < 2:
            return None
        keys = [(t, int(self.rng.choice(sorted(ws)))) for t, ws in sorted(by_trial.items())]
        return (np.array([self.rows[a][k] for k in keys]),
                np.array([self.rows[b][k] for k in keys]))

    def epoch(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Every subject pair once, in random order."""
        pairs = list(combinations(self.subjects, 2))
        for i in self.rng.permutation(len(pairs)):
            pair = self.sample_pair(*pairs[i])
            if pair is not None:
                yield pair


def _contrastive_epoch(model: CLISA, x: torch.Tensor, sampler: StimulusPairSampler, device,
                       opt=None, sched=None, epoch: int = 0) -> Tuple[float, float]:
    """One pass over all subject pairs -> (mean loss per sample, top-1 pair accuracy)."""
    batches = list(sampler.epoch())
    loss_sum, correct, count = 0.0, 0, 0
    for i, (ia, ib) in enumerate(batches):
        xa, xb = x[ia].to(device), x[ib].to(device)
        with torch.set_grad_enabled(opt is not None):
            logits, targets = model.contrastive_logits(xa, xb)
            loss = F.cross_entropy(logits, targets, reduction="sum")
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step(epoch + (i + 1) / len(batches))
        loss_sum += loss.item()
        correct += (logits.argmax(1) == targets).sum().item()
        count += len(targets)
    return (loss_sum / count, correct / count) if count else (float("inf"), 0.0)


@torch.no_grad()
def _features(model: CLISA, x: torch.Tensor, device, batch: int = 256) -> torch.Tensor:
    model.eval()
    return torch.cat([model.de_features(x[i:i + batch].to(device)) for i in range(0, len(x), batch)])


def fit_clisa(
    model: CLISA,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    recipe: TrainRecipe,
    pretrain_epochs: int = 100,
    pretrain_patience: int = 30,
    pretrain_lr: float = 7e-4,
    pretrain_weight_decay: float = 0.015,
    warm_restarts: int = 3,
    seed: int = 42,
    **_: object,
) -> Dict:
    model.to(device)
    start = time.time()
    pretrain_epochs = min(pretrain_epochs, recipe.max_epochs)
    tr = train_loader.dataset
    sampler = StimulusPairSampler(tr.subject_ids, tr.trial_ids, tr.window_ids, seed)
    val_sampler = None
    if val_loader is not None and len(set(val_loader.dataset.subject_ids)) >= 2:
        vd = val_loader.dataset
        val_sampler = StimulusPairSampler(vd.subject_ids, vd.trial_ids, vd.window_ids, seed + 1)
        if next(val_sampler.epoch(), None) is None:
            val_sampler = None                                 # no stimulus shared across val subjects

    # Stage 1: inter-subject contrastive learning of base encoder + projector.
    enc_params = list(model.base.parameters()) + list(model.projector.parameters())
    opt = torch.optim.Adam(enc_params, lr=pretrain_lr, weight_decay=pretrain_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, pretrain_epochs // (warm_restarts + 1)))
    best_acc, best_state, stale, pre_epoch = -1.0, None, 0, 0
    for pre_epoch in range(pretrain_epochs):
        model.train()
        train_loss, train_acc = _contrastive_epoch(model, tr.x, sampler, device, opt, sched, pre_epoch)
        model.eval()
        monitor = _contrastive_epoch(model, val_loader.dataset.x, val_sampler, device)[1] \
            if val_sampler else train_acc
        if monitor > best_acc + 1e-4:
            best_acc, stale = monitor, 0
            best_state = {k: v.detach().clone() for k, v in model.base.state_dict().items()}
        else:
            stale += 1
            if stale >= pretrain_patience:
                break
    if best_state is not None:
        model.base.load_state_dict(best_state)

    # Stage 2: frozen encoder -> trained-DE features -> MLP.
    for p in enc_params:
        p.requires_grad_(False)
    f_train = _features(model, tr.x, device)
    model.feat_mean.copy_(f_train.mean(0))
    model.feat_std.copy_(f_train.std(0).clamp_min(1e-6))
    norm = lambda f: (f - model.feat_mean) / model.feat_std  # noqa: E731
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(norm(f_train).cpu(), tr.y), batch_size=recipe.batch_size,
                        shuffle=True, generator=gen)
    if val_loader is not None:
        f_val, y_val = norm(_features(model, val_loader.dataset.x, device)), val_loader.dataset.y.to(device)
    opt = make_optimizer(model, recipe, model.classifier.parameters())
    stopper = EarlyStopper(model, recipe)
    epoch = 0
    for epoch in range(recipe.max_epochs):
        model.classifier.train()
        running, seen = 0.0, 0
        for f, y in loader:
            f, y = f.to(device), y.to(device)
            loss = F.cross_entropy(model.classifier(f), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running, seen = running + loss.item() * len(y), seen + len(y)
        model.classifier.eval()
        if val_loader is not None:
            with torch.no_grad():
                monitor = F.cross_entropy(model.classifier(f_val), y_val).item()
        else:
            monitor = running / seen
        if stopper.step(epoch, monitor):
            break
    stopper.restore()
    return {"train_time_sec": time.time() - start, "epochs_run": epoch + 1,
            "best_epoch": stopper.best_epoch + 1, "best_val_loss": stopper.best,
            "contrastive_epochs": pre_epoch + 1, "contrastive_best_val_acc": best_acc}
