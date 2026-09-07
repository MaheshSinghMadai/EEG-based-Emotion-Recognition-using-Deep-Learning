"""
evaluation.py
=============
Shared evaluation harness. EVERY benchmark notebook must go through this module so
the numbers in the thesis results table are produced identically.

The benchmark uses a single validation protocol: leave-one-subject-out
(subject-independent). `benchmark_si.run_one` is the one entry point the notebooks
call; it drives `run_loso` below.

What it provides
----------------
* `set_seed`, `get_device`, `count_parameters`
* `make_default_fit` / `default_predict` : a plain supervised training + inference loop
* `compute_metrics` : Accuracy, Precision, Recall, F1 (macro) + confusion matrix
* `run_loso(...)`    : trains a fresh model per LOSO fold, aggregates
* `summarize_per_fold(...)` : same aggregation for callers that run their own loop
* `save_results(...)` : writes results/{model_name}_results.json in the fixed schema:
    {
      model, family, config,
      loso: {
        protocol, accuracy, precision, recall, f1,
        accuracy_std, f1_std, precision_std, recall_std,
        per_fold_scores, confusion_matrix, params_count, train_time_sec
      }
    }

Models that need a custom training step (Siam-GCAN pairs, DADPc domain loss) pass
their own `fit_fn` / `predict_fn` with the same signatures as the defaults.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

RESULTS_DIR_DEFAULT = "results"

FitFn = Callable[[nn.Module, DataLoader, Optional[DataLoader], torch.device], float]
PredictFn = Callable[[nn.Module, DataLoader, torch.device], Tuple[np.ndarray, np.ndarray]]
BuildModelFn = Callable[[Dict], nn.Module]


# --------------------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic-ish; keep cudnn benchmark off for reproducibility.
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


# --------------------------------------------------------------------------------------
# Default training / inference
# --------------------------------------------------------------------------------------
def make_default_fit(
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    criterion: Optional[nn.Module] = None,
    optimizer_name: str = "adam",
    grad_clip: Optional[float] = 5.0,
    early_stop_patience: Optional[int] = 12,
    verbose: bool = False,
) -> FitFn:
    """Build a `fit_fn`. The model's `forward(x)` must return class logits [B, n_classes]."""

    def fit(model: nn.Module, train_loader: DataLoader,
            val_loader: Optional[DataLoader], device: torch.device) -> float:
        model.to(device)
        crit = criterion or nn.CrossEntropyLoss()
        if optimizer_name == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                running += loss.item() * xb.size(0)
            train_loss = running / max(1, len(train_loader.dataset))

            monitor = train_loss
            if val_loader is not None:
                monitor = _eval_loss(model, val_loader, crit, device)
            if verbose:
                print(f"    epoch {ep+1:03d}  train={train_loss:.4f}  monitor={monitor:.4f}")

            if monitor < best_val - 1e-4:
                best_val = monitor
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if early_stop_patience and patience >= early_stop_patience:
                    if verbose:
                        print(f"    early stop @ epoch {ep+1}")
                    break

        model.load_state_dict(best_state)
        return time.time() - t0

    return fit


def _eval_loss(model: nn.Module, loader: DataLoader, crit: nn.Module, device: torch.device) -> float:
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            tot += crit(model(xb), yb).item() * xb.size(0)
    return tot / max(1, len(loader.dataset))


def default_predict(model: nn.Module, loader: DataLoader, device: torch.device
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) as int arrays. Assumes `forward(x) -> logits`."""
    model.to(device).eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            ps.append(pred)
            ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = 2) -> Dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=list(range(n_classes))
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return {
        "accuracy": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def summarize_per_fold(per_fold_scores: List[Dict], params_count: int,
                       train_time_sec: float, protocol: str) -> Dict:
    """Build a schema-compliant results dict from a list of per-fold metric dicts.

    Use this in callers that run their own custom training loop (e.g. DADPc) instead
    of `run_loso`.
    """
    agg = _aggregate(per_fold_scores)
    return {
        "protocol": protocol,
        "accuracy": agg["accuracy"], "precision": agg["precision"],
        "recall": agg["recall"], "f1": agg["f1"],
        "accuracy_std": agg["accuracy_std"], "f1_std": agg["f1_std"],
        "precision_std": agg["precision_std"], "recall_std": agg["recall_std"],
        "confusion_matrix": agg["confusion_matrix"],
        "per_fold_scores": per_fold_scores,
        "params_count": int(params_count),
        "train_time_sec": float(train_time_sec),
    }


def _aggregate(per_unit: List[Dict]) -> Dict:
    """Mean ± std of the headline metrics across folds/subjects, plus summed confusion."""
    keys = ["accuracy", "precision", "recall", "f1"]
    agg = {k: float(np.mean([u[k] for u in per_unit])) for k in keys}
    agg.update({f"{k}_std": float(np.std([u[k] for u in per_unit])) for k in keys})
    cm = np.sum([np.asarray(u["confusion_matrix"]) for u in per_unit], axis=0)
    agg["confusion_matrix"] = cm.tolist()
    return agg


# --------------------------------------------------------------------------------------
# Protocol runner (leave-one-subject-out / subject-independent)
# --------------------------------------------------------------------------------------
def run_loso(
    build_model: BuildModelFn,
    loso_path: str,
    loader_factory: Callable,          # loader_factory(fold) -> (train, val_or_None, test, meta)
    fit_fn: FitFn,
    predict_fn: PredictFn = default_predict,
    device: Optional[torch.device] = None,
    model_kwargs: Optional[Dict] = None,
    n_classes: int = 2,
    folds: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> Dict:
    """Train a FRESH model per LOSO fold; report mean ± std across all folds."""
    device = device or get_device()
    model_kwargs = model_kwargs or {}
    with open(loso_path) as f:
        loso = json.load(f)
    fold_ids = list(folds) if folds is not None else list(range(loso["n_folds"]))

    per_fold_scores = []
    total_train_time = 0.0
    params_count = None

    for fold in fold_ids:
        out = loader_factory(fold=fold)
        tl, vl, el, meta = out
        model = build_model({**model_kwargs, "meta": meta})
        if params_count is None:
            params_count = count_parameters(model)
        t = fit_fn(model, tl, vl, device)
        total_train_time += t
        yt, yp = predict_fn(model, el, device)
        m = compute_metrics(yt, yp, n_classes)
        m["fold"] = fold
        m["test_subject"] = meta.get("test_subject")
        per_fold_scores.append(m)
        if verbose:
            print(f"  [LOSO] fold{fold:02d} ({m['test_subject']}): "
                  f"acc={m['accuracy']:.3f} f1={m['f1']:.3f} ({t:.1f}s)")

    agg = _aggregate(per_fold_scores)
    return {
        "protocol": "loso",
        "accuracy": agg["accuracy"], "precision": agg["precision"],
        "recall": agg["recall"], "f1": agg["f1"],
        "accuracy_std": agg["accuracy_std"], "f1_std": agg["f1_std"],
        "precision_std": agg["precision_std"], "recall_std": agg["recall_std"],
        "confusion_matrix": agg["confusion_matrix"],
        "per_fold_scores": per_fold_scores,
        "params_count": int(params_count or 0),
        "train_time_sec": float(total_train_time),
    }


# --------------------------------------------------------------------------------------
# Results IO
# --------------------------------------------------------------------------------------
def save_results(
    model_name: str,
    loso: Optional[Dict] = None,
    family: str = "",
    config: Optional[Dict] = None,
    results_dir: str = RESULTS_DIR_DEFAULT,
) -> str:
    """Write the LOSO results into results/{model_name}_results.json.

    An existing file is loaded and only the `loso` / `family` / `config` sections
    are refreshed.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{model_name}_results.json")
    blob: Dict = {}
    if os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)

    blob["model"] = model_name
    if family:
        blob["family"] = family
    if config is not None:
        blob["config"] = config
    if loso is not None:
        blob["loso"] = loso

    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"[results] wrote {path}")
    return path


def load_all_results(results_dir: str = RESULTS_DIR_DEFAULT) -> List[Dict]:
    out = []
    for fn in sorted(os.listdir(results_dir)):
        if fn.endswith("_results.json"):
            with open(os.path.join(results_dir, fn)) as f:
                out.append(json.load(f))
    return out


# --------------------------------------------------------------------------------------
# Small helpers reused by notebooks
# --------------------------------------------------------------------------------------
def quick_report(name: str, res: Dict) -> None:
    print(f"=== {name} :: {res['protocol']} ===")
    print(f"  accuracy = {res['accuracy']:.4f} ± {res.get('accuracy_std', 0):.4f}")
    print(f"  f1       = {res['f1']:.4f} ± {res.get('f1_std', 0):.4f}")
    print(f"  params   = {res['params_count']:,}")
    print(f"  train_t  = {res['train_time_sec']:.1f}s")
