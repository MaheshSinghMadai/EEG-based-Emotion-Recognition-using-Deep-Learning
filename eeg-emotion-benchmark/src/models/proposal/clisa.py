"""
CLISA — Contrastive Learning of Inter-Subject Alignment (Shen et al., IEEE TAFFC 2022,
doi:10.1109/TAFFC.2022.3164516).

Core idea reproduced:
    Stage 1  A convolutional base encoder (spatial conv across electrodes followed by a
             temporal conv) is pre-trained with an inter-subject contrastive loss. Positive
             pairs are windows of the SAME stimulus at the SAME time position recorded
             from TWO DIFFERENT subjects; the other stimulus/time windows in the batch
             are negatives (NT-Xent). This aligns representations across subjects
             without using emotion labels.
    Stage 2  The encoder is frozen; per-window features are the log-variance (Gaussian
             differential entropy) of the encoder's output maps over 1 s segments; an
             MLP classifier is trained on these features with the emotion labels.

Both DEAP (preprocessed release, trials sorted by video id) and DREAMER (fixed clip
order) present the same stimuli to every subject, so (trial_index, window_index) is a
valid stimulus-time key. Only TRAINING subjects are ever paired.

Documented deviations (verify against the full paper before final reporting):
    * The paper's encoder hyper-parameters (filter counts/kernel lengths) were set for
      250 Hz THU-EP/SEED data; here kernels are scaled to 128 Hz.
    * Linear dynamical system (LDS) smoothing of the extracted features and the paper's
      running normalisation are omitted; features are z-scored by BatchNorm.
    * The input is the 4-45 Hz filtered signal shared by all models.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...training import EarlyStopper, TrainRecipe, make_optimizer, validation_loss
from .._common import make_builder


class CLISA(nn.Module):
    def __init__(self, n_channels: int = 14, n_samples: int = 512, n_classes: int = 2,
                 n_spatial: int = 16, temporal_kernel: int = 32, projection_dim: int = 64,
                 segment_samples: int = 128, mlp_hidden: int = 64, dropout: float = 0.3,
                 temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.segment_samples = segment_samples
        self.base = nn.Sequential(                                   # [B, 1, C, T]
            nn.Conv2d(1, n_spatial, (n_channels, 1)),                # spatial filtering
            nn.Conv2d(n_spatial, n_spatial, (1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), groups=n_spatial),  # temporal
            nn.BatchNorm2d(n_spatial),
        )
        self.projector = nn.Sequential(
            nn.ELU(),
            nn.AvgPool2d((1, 16), stride=(1, 16)),
            nn.Conv2d(n_spatial, 2 * n_spatial, (1, 5), padding=(0, 2)),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 4)),
            nn.Flatten(),
            nn.Linear(2 * n_spatial * 4, projection_dim),
        )
        n_segments = max(1, n_samples // segment_samples)
        self.feature_dim = n_spatial * n_segments
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Linear(self.feature_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] -> base feature maps [B, F, T']."""
        return self.base(x.unsqueeze(1)).squeeze(2)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.base(x.unsqueeze(1))), dim=1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """DE (log-variance) of each base feature map over 1 s segments."""
        h = self.encode(x)
        n_seg = max(1, h.shape[-1] // self.segment_samples)
        h = h[..., : n_seg * self.segment_samples].reshape(*h.shape[:2], n_seg, -1)
        return 0.5 * torch.log(2 * np.pi * np.e * h.var(dim=-1).clamp_min(1e-6)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def contrastive_loss(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """NT-Xent: row i of xa (subject A) and row i of xb (subject B) share a stimulus."""
        z = torch.cat([self.project(xa), self.project(xb)])
        logits = z @ z.T / self.temperature
        logits.fill_diagonal_(float("-inf"))
        n = xa.shape[0]
        targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
        return F.cross_entropy(logits, targets)


build = make_builder(
    CLISA,
    default_shape=(14, 512),
    shape_kwargs=lambda s: {"n_channels": s[0], "n_samples": s[1]},
    params={"n_spatial": 16, "temporal_kernel": 32, "projection_dim": 64,
            "segment_samples": 128, "mlp_hidden": 64, "dropout": 0.3, "temperature": 0.07},
)


class StimulusPairSampler:
    """Draw batches of stimulus-aligned window pairs from two different subjects."""

    def __init__(self, subject_ids, trial_ids, window_ids, seed: int = 42):
        self.rows: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(dict)
        for row, (s, t, w) in enumerate(zip(subject_ids, trial_ids, window_ids)):
            self.rows[s][(int(t), int(w))] = row
        self.subjects = sorted(self.rows)
        self.rng = np.random.default_rng(seed)
        if len(self.subjects) < 2:
            raise ValueError("CLISA needs at least two training subjects for pairing.")

    def sample(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        for _ in range(10):
            a, b = self.rng.choice(self.subjects, size=2, replace=False)
            shared = sorted(set(self.rows[a]) & set(self.rows[b]))
            if len(shared) >= 2:
                pick = self.rng.choice(len(shared), size=min(batch_size, len(shared)), replace=False)
                keys = [shared[i] for i in pick]
                return (np.array([self.rows[a][k] for k in keys]),
                        np.array([self.rows[b][k] for k in keys]))
        return None


def _contrastive_epoch(model: CLISA, x: torch.Tensor, sampler: StimulusPairSampler,
                       steps: int, batch_size: int, device, opt=None) -> float:
    losses: List[float] = []
    for _ in range(steps):
        pair = sampler.sample(batch_size)
        if pair is None:
            break
        xa, xb = x[pair[0]].to(device), x[pair[1]].to(device)
        if opt is None:
            with torch.no_grad():
                losses.append(model.contrastive_loss(xa, xb).item())
            continue
        loss = model.contrastive_loss(xa, xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def fit_clisa(
    model: CLISA,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    recipe: TrainRecipe,
    pretrain_epochs: int = 30,
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
    steps = max(1, len(tr) // recipe.batch_size)

    # Stage 1: inter-subject contrastive pre-training of the encoder.
    enc_params = list(model.base.parameters()) + list(model.projector.parameters())
    opt = make_optimizer(model, recipe, enc_params)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(pretrain_epochs):
        model.train()
        train_loss = _contrastive_epoch(model, tr.x, sampler, steps, recipe.batch_size, device, opt)
        model.eval()
        monitor = (_contrastive_epoch(model, val_loader.dataset.x, val_sampler, 10,
                                      recipe.batch_size, device) if val_sampler else train_loss)
        if monitor < best - 1e-4:
            best, stale = monitor, 0
            best_state = {k: v.detach().clone() for k, v in model.base.state_dict().items()}
        else:
            stale += 1
            if stale >= recipe.patience:
                break
    if best_state is not None:
        model.base.load_state_dict(best_state)

    # Stage 2: frozen encoder, supervised MLP on DE features of the encoder output.
    for p in enc_params:
        p.requires_grad_(False)
    opt = make_optimizer(model, recipe, model.classifier.parameters())
    stopper = EarlyStopper(model, recipe)
    epoch = 0
    for epoch in range(recipe.max_epochs):
        model.train()
        model.base.eval()  # keep frozen BatchNorm statistics fixed
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        monitor = validation_loss(model, val_loader, device) if val_loader is not None else loss.item()
        if stopper.step(epoch, monitor):
            break
    stopper.restore()
    return {"train_time_sec": time.time() - start, "epochs_run": epoch + 1,
            "best_epoch": stopper.best_epoch + 1, "best_val_loss": stopper.best,
            "contrastive_best_loss": best}
