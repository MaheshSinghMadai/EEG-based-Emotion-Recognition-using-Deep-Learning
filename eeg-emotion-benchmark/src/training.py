"""
training.py
===========
Per-model training recipes and the shared supervised training loop.

Fairness rule used throughout the benchmark
-------------------------------------------
* Each model keeps its PAPER's optimisation recipe where the paper reports one
  (optimizer, learning rate, batch size, scheduler, label smoothing, augmentation).
* Every model shares the SAME stopping rule: at most `max_epochs` epochs, keep the
  checkpoint with the lowest validation cross-entropy, stop after `patience` epochs
  without improvement. Validation data always comes from the training side of the
  split (never the test subjects / test trials).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .evaluation import _clone_state


@dataclass(frozen=True)
class TrainRecipe:
    optimizer: str = "adam"            # "adam" | "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 15
    min_epochs: int = 5
    scheduler: Optional[str] = None    # None | "cosine_warm_restarts"
    scheduler_t0: int = 50
    label_smoothing: float = 0.0
    grad_clip: Optional[float] = 5.0
    augment: Optional[str] = None      # None | "noise_scale"
    source: str = ""                   # where the values come from (paper section)

    def with_overrides(self, **kw) -> "TrainRecipe":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})

    def to_dict(self) -> Dict:
        return asdict(self)


def make_optimizer(model: nn.Module, recipe: TrainRecipe, params=None) -> torch.optim.Optimizer:
    params = params if params is not None else model.parameters()
    cls = torch.optim.AdamW if recipe.optimizer == "adamw" else torch.optim.Adam
    return cls(params, lr=recipe.lr, weight_decay=recipe.weight_decay)


def make_scheduler(opt: torch.optim.Optimizer, recipe: TrainRecipe):
    if recipe.scheduler == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=recipe.scheduler_t0)
    return None


def augment_batch(x: torch.Tensor, kind: Optional[str]) -> torch.Tensor:
    """Training-time augmentation. "noise_scale" = DAH-Net's Gaussian noise + scaling."""
    if kind == "noise_scale":
        scale = 1.0 + 0.1 * torch.randn(x.shape[0], *([1] * (x.ndim - 1)), device=x.device)
        return x * scale + 0.05 * torch.randn_like(x)
    return x


@torch.no_grad()
def validation_loss(model: nn.Module, loader: DataLoader, device: torch.device,
                    forward: Optional[Callable] = None) -> float:
    model.eval()
    forward = forward or model
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += F.cross_entropy(forward(x), y, reduction="sum").item()
        count += y.numel()
    return total / max(count, 1)


class EarlyStopper:
    """Tracks the best validation loss and the corresponding weights."""

    def __init__(self, model: nn.Module, recipe: TrainRecipe):
        self.model, self.recipe = model, recipe
        self.best = float("inf")
        self.best_state = _clone_state(model)
        self.best_epoch = 0
        self.stale = 0

    def step(self, epoch: int, monitor: float) -> bool:
        """Record one epoch; return True when training should stop."""
        if monitor < self.best - 1e-4:
            self.best, self.best_epoch, self.stale = monitor, epoch, 0
            self.best_state = _clone_state(self.model)
        else:
            self.stale += 1
        return epoch + 1 >= self.recipe.min_epochs and self.stale >= self.recipe.patience

    def restore(self) -> None:
        self.model.load_state_dict(self.best_state)


def fit_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    recipe: TrainRecipe,
    **_: object,
) -> Dict:
    """Plain supervised training with the recipe's optimizer/scheduler/augmentation."""
    model.to(device)
    opt = make_optimizer(model, recipe)
    sched = make_scheduler(opt, recipe)
    stopper = EarlyStopper(model, recipe)
    start = time.time()
    epoch = 0
    for epoch in range(recipe.max_epochs):
        model.train()
        running, seen = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            x, y = augment_batch(x.to(device), recipe.augment), y.to(device)
            loss = F.cross_entropy(model(x), y, label_smoothing=recipe.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if recipe.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
            opt.step()
            if sched is not None:
                sched.step(epoch + i / max(len(train_loader), 1))
            running += loss.item() * y.numel()
            seen += y.numel()
        monitor = validation_loss(model, val_loader, device) if val_loader is not None else running / seen
        if stopper.step(epoch, monitor):
            break
    stopper.restore()
    return {"train_time_sec": time.time() - start, "epochs_run": epoch + 1,
            "best_epoch": stopper.best_epoch + 1, "best_val_loss": stopper.best}
