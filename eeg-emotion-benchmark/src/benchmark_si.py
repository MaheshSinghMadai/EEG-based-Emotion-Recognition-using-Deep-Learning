"""
benchmark_si.py
===============
**Subject-INDEPENDENT** (leave-one-subject-out) evaluation protocol for the whole
model zoo, in one place, so every algorithm is scored identically and a
comparative analysis can be produced with a single command.

This is the subject-independent counterpart to running notebooks 01-08 by hand. It
reuses the shared harness in `evaluation.py` (`run_loso`, `compute_metrics`,
`save_results`) and only *centralises* the per-model LOSO recipe (feature view,
batch size, epochs, learning rate, and any custom training loop) that was
previously duplicated across the notebooks.

Protocol
--------
23-fold leave-one-subject-out (`data/splits/loso_splits.json`). Fold *k* trains on
22 subjects and tests on the held-out 23rd. Normalisation statistics are fit on
the training subjects only. We report mean +/- std of the headline metrics across
the 23 folds. A `MajorityBaseline` (predict the training-majority class) is scored
on the same folds as a reference floor.

From a notebook / script
------------------------
    from src import benchmark_si as B
    res = B.run_one("EEGNet", label="valence", quick=True)   # 23-fold LOSO for one model;
                                                             # writes results/EEGNet_results.json
    B.run_all(models=["EEGNet", "DGCNN"], label="valence")   # several models + comparison table

`run_one(name, ...)` is the single entry every model notebook calls — it looks up the
per-model recipe in `REGISTRY`, picks the right training loop (plain / Siamese pairs /
DADPc domain loss), scores 23-fold LOSO through `evaluation.run_loso`, and saves.

CLI
---
    python -m src.benchmark_si --quick                  # 3-fold / few-epoch smoke test
    python -m src.benchmark_si                          # full 23-fold run, every model
    python -m src.benchmark_si --models EEGNet DGCNN    # subset
    python -m src.benchmark_si --label arousal          # switch target (valence/arousal/dominance)
    python -m src.benchmark_si --no-skip-existing       # recompute even if a result file exists
    python -m src.benchmark_si --aggregate-only         # just rebuild the comparison table + plots

Outputs
-------
    results/{Model}_results.json                    (loso section; evaluation.save_results schema)
    results/subject_independent_comparison.csv
    results/subject_independent_comparison.md
    results/figures/si_accuracy_by_model.png
    results/figures/si_per_subject_heatmap.png
    results/figures/si_accuracy_vs_params.png
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data_loader import (
    REDUCED_5_CHANNELS,
    SiamesePairDataset,
    build_loso_loaders,
)
from .evaluation import (
    compute_metrics,
    count_parameters,
    default_predict,
    get_device,
    make_default_fit,
    quick_report,
    save_results,
    set_seed,
    summarize_per_fold,
)
from .models import (
    cit_emotionnet,
    cnn_transformer_reduced,
    dadpc,
    dgcnn,
    dual_attention,
    eegnet,
    siam_gcan,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
LOSO_SPLITS = os.path.join(ROOT, "data", "splits", "loso_splits.json")
RESULTS_DIR = os.path.join(ROOT, "results")

SEED = 42
QUICK_EPOCHS = 5
QUICK_DADPC_EPOCHS = 8
QUICK_FOLDS = 3


# --------------------------------------------------------------------------------------
# Per-model LOSO recipe registry
# --------------------------------------------------------------------------------------
@dataclass
class SISpec:
    """Everything needed to run one model through the subject-independent protocol.

    The values mirror the `### Subject-independent (LOSO)` section of each notebook
    so results are identical whether produced here or from the notebook.
    """

    name: str
    family: str
    build: Callable[[dict], nn.Module]
    feature: str                       # "raw" or "de"
    batch_size: int
    epochs: int
    lr: float
    val_frac: float
    channels: Optional[List[str]] = None
    runner: str = "default"            # "default" | "siam" | "dadpc"
    extra_config: Dict = field(default_factory=dict)


REGISTRY: Dict[str, SISpec] = {
    "EEGNet": SISpec(
        "EEGNet", "CNN", eegnet.build,
        feature="raw", batch_size=128, epochs=50, lr=1e-3, val_frac=0.1,
    ),
    "DGCNN": SISpec(
        "DGCNN", "Graph CNN", dgcnn.build,
        feature="de", batch_size=256, epochs=60, lr=1e-3, val_frac=0.1,
    ),
    "SiamGCAN": SISpec(
        "SiamGCAN", "Graph CNN + Siamese", siam_gcan.build,
        feature="de", batch_size=128, epochs=45, lr=1e-3, val_frac=0.0,
        runner="siam", extra_config={"pairs_per_sample": 2, "contrastive_lambda": 1.0},
    ),
    "CIT_EmotionNet": SISpec(
        "CIT_EmotionNet", "CNN-Transformer", cit_emotionnet.build,
        feature="de", batch_size=256, epochs=60, lr=7e-4, val_frac=0.1,
    ),
    "CNN_Transformer_Reduced": SISpec(
        "CNN_Transformer_Reduced", "CNN-Transformer (5ch)", cnn_transformer_reduced.build,
        feature="raw", batch_size=128, epochs=50, lr=7e-4, val_frac=0.1,
        channels=list(REDUCED_5_CHANNELS),
    ),
    "DADPc": SISpec(
        "DADPc", "Domain adaptation", dadpc.build,
        feature="de", batch_size=256, epochs=60, lr=1e-3, val_frac=0.0,
        runner="dadpc", extra_config={"w_mmd": 1.0, "w_pc": 0.1, "w_fe": 0.1},
    ),
    "DualAttention_TransformerCNNBiLSTM": SISpec(
        "DualAttention_TransformerCNNBiLSTM", "CNN-BiLSTM-Transformer", dual_attention.build,
        feature="raw", batch_size=128, epochs=50, lr=7e-4, val_frac=0.1,
    ),
}

# A model whose subject-independent number is only a diagnostic, not a benchmark
# entry, lives in notebook 05 (SHAP) and is intentionally excluded here.


# --------------------------------------------------------------------------------------
# Custom training loops (kept identical to notebooks 03 / 07)
# --------------------------------------------------------------------------------------
def make_siam_fit(epochs: int, lr: float, lam: float = 1.0,
                  pairs_per_sample: int = 2, wd: float = 1e-4):
    """Siamese contrastive + dual cross-entropy loop (Siam-GCAN, notebook 03)."""

    def fit(model: nn.Module, train_loader: DataLoader,
            val_loader: Optional[DataLoader], device: torch.device) -> float:
        model.to(device)
        base = train_loader.dataset
        bs = train_loader.batch_size or 64
        pair_ds = SiamesePairDataset(base, pairs_per_sample=pairs_per_sample, seed=SEED)
        pl = DataLoader(pair_ds, batch_size=bs, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        ce = nn.CrossEntropyLoss()
        t0 = time.time()
        for _ in range(epochs):
            pair_ds.resample()
            model.train()
            for xa, xb, plab, ya, yb in pl:
                xa, xb = xa.to(device), xb.to(device)
                plab, ya, yb = plab.to(device), ya.to(device), yb.to(device)
                opt.zero_grad()
                ea, eb, la, lb = model.forward_pair(xa, xb)
                loss = ce(la, ya) + ce(lb, yb) + lam * siam_gcan.contrastive_loss(ea, eb, plab)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        return time.time() - t0

    return fit


def run_dadpc_loso(spec: SISpec, label: str, quick: bool,
                   folds: Optional[Sequence[int]], device: torch.device) -> Dict:
    """Bespoke LOSO loop for DADPc (notebook 07): the held-out subject's *unlabelled*
    windows are used as the target domain during training (MMD + possibilistic +
    fuzzy-entropy terms), then the same windows are classified for scoring."""
    with open(LOSO_SPLITS) as f:
        loso = json.load(f)
    fold_ids = list(folds) if folds is not None else list(range(loso["n_folds"]))
    w_mmd = spec.extra_config["w_mmd"]
    w_pc = spec.extra_config["w_pc"]
    w_fe = spec.extra_config["w_fe"]
    epochs = QUICK_DADPC_EPOCHS if quick else spec.epochs

    per_fold: List[Dict] = []
    t_total = 0.0
    params_count = None

    for k in fold_ids:
        _tl, _, el, meta = build_loso_loaders(
            PROCESSED_DIR, LOSO_SPLITS, fold=k, label=label, feature=spec.feature,
            batch_size=spec.batch_size, normalize=True, val_frac=0.0,
        )
        tgt_X = torch.tensor(
            np.stack([el.dataset[i][0].numpy() for i in range(len(el.dataset))]),
            dtype=torch.float32,
        )
        model = spec.build({"n_classes": 2, "meta": meta}).to(device)
        if params_count is None:
            params_count = count_parameters(model)
        opt = torch.optim.Adam(model.parameters(), lr=spec.lr, weight_decay=1e-4)

        t0 = time.time()
        for _ in range(epochs):
            model.train()
            for xb, yb in _tl:
                xb, yb = xb.to(device), yb.to(device)
                idx = torch.randint(0, tgt_X.shape[0], (xb.shape[0],))
                xt = tgt_X[idx].to(device)
                opt.zero_grad()
                loss, _ = model.dadpc_loss(xb, yb, xt, w_mmd, w_pc, w_fe)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        t_total += time.time() - t0

        model.eval()
        with torch.no_grad():
            pred = model(tgt_X.to(device)).argmax(1).cpu().numpy()
        yte = np.array([int(el.dataset[i][1]) for i in range(len(el.dataset))])
        m = compute_metrics(yte, pred)
        m["fold"] = k
        m["test_subject"] = meta["test_subject"]
        per_fold.append(m)
        print(f"  [LOSO] fold{k:02d} ({m['test_subject']}): "
              f"acc={m['accuracy']:.3f} f1={m['f1']:.3f}")

    return summarize_per_fold(per_fold, params_count or 0, t_total, protocol="loso")


def run_majority_baseline(label: str, folds: Optional[Sequence[int]]) -> Dict:
    """ZeroR reference: predict the training-majority class for every test window."""
    with open(LOSO_SPLITS) as f:
        loso = json.load(f)
    fold_ids = list(folds) if folds is not None else list(range(loso["n_folds"]))
    per_fold: List[Dict] = []
    t0 = time.time()
    for k in fold_ids:
        _tl, _, el, meta = build_loso_loaders(
            PROCESSED_DIR, LOSO_SPLITS, fold=k, label=label, feature="de",
            batch_size=512, normalize=False, val_frac=0.0,
        )
        y_tr = _tl.dataset.y.numpy()
        y_te = el.dataset.y.numpy()
        maj = int(np.bincount(y_tr, minlength=2).argmax())
        pred = np.full_like(y_te, maj)
        m = compute_metrics(y_te, pred)
        m["fold"] = k
        m["test_subject"] = meta["test_subject"]
        per_fold.append(m)
    return summarize_per_fold(per_fold, 0, time.time() - t0, protocol="loso")


# --------------------------------------------------------------------------------------
# One model
# --------------------------------------------------------------------------------------
def _loso_loader_factory(spec: SISpec, label: str) -> Callable:
    return partial(
        build_loso_loaders,
        processed_dir=PROCESSED_DIR, loso_path=LOSO_SPLITS,
        label=label, feature=spec.feature, batch_size=spec.batch_size,
        channels=spec.channels, normalize=True, val_frac=spec.val_frac,
    )


def run_one(name: str, label: str = "valence", quick: bool = False,
            folds: Optional[Sequence[int]] = None, device: Optional[torch.device] = None,
            save: bool = True) -> Dict:
    """Run one registered model through the 23-fold LOSO protocol and (optionally)
    write `results/{name}_results.json`. Returns the `loso` results dict."""
    from .evaluation import run_loso  # local import keeps module import cheap

    spec = REGISTRY[name]
    device = device or get_device()
    set_seed(SEED)
    epochs = QUICK_EPOCHS if quick else spec.epochs
    fold_sel = list(range(QUICK_FOLDS)) if (quick and folds is None) else folds

    print(f"\n=== {name}  [{spec.family}]  feature={spec.feature} "
          f"batch={spec.batch_size} epochs={epochs} lr={spec.lr} "
          f"folds={'all' if fold_sel is None else list(fold_sel)} ===")
    t0 = time.time()

    if spec.runner == "dadpc":
        loso_res = run_dadpc_loso(spec, label, quick, fold_sel, device)
    else:
        if spec.runner == "siam":
            fit_fn = make_siam_fit(
                epochs=epochs, lr=spec.lr,
                lam=spec.extra_config.get("contrastive_lambda", 1.0),
                pairs_per_sample=spec.extra_config.get("pairs_per_sample", 2),
            )
        else:
            fit_fn = make_default_fit(epochs=epochs, lr=spec.lr)
        loso_res = run_loso(
            build_model=spec.build, loso_path=LOSO_SPLITS,
            loader_factory=_loso_loader_factory(spec, label),
            fit_fn=fit_fn, predict_fn=default_predict, device=device,
            model_kwargs={"n_classes": 2}, folds=fold_sel,
        )

    quick_report(name, loso_res)
    print(f"    wall time: {time.time() - t0:.1f}s")

    if save:
        config = {
            "label": label, "feature": spec.feature, "channels": spec.channels,
            "batch_loso": spec.batch_size, "epochs_loso": epochs, "lr": spec.lr,
            "val_frac": spec.val_frac, "runner": spec.runner, "quick": quick,
            **spec.extra_config,
        }
        save_results(name, loso=loso_res, family=spec.family,
                     config=config, results_dir=RESULTS_DIR)
    return loso_res


# --------------------------------------------------------------------------------------
# All models
# --------------------------------------------------------------------------------------
def _already_done(name: str, label: str, quick: bool) -> bool:
    path = os.path.join(RESULTS_DIR, f"{name}_results.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            blob = json.load(f)
    except Exception:
        return False
    lo = blob.get("loso")
    cfg = blob.get("config", {})
    return bool(lo) and cfg.get("label") == label and bool(cfg.get("quick")) == quick


def run_all(models: Optional[Sequence[str]] = None, label: str = "valence",
            quick: bool = False, folds: Optional[Sequence[int]] = None,
            skip_existing: bool = True, device: Optional[torch.device] = None,
            with_baseline: bool = True) -> Dict[str, Dict]:
    """Run every requested model through the subject-independent protocol.

    `models` : names from `REGISTRY` (default: all, in registry order).
    `skip_existing` : skip a model whose results file already matches (label, quick).
    Returns {name: loso_results_dict} for the models run this call.
    """
    names = list(models) if models else list(REGISTRY)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown model(s): {unknown}. known: {list(REGISTRY)}")

    device = device or get_device()
    print(f"device: {device} | label: {label} | quick: {quick} | models: {names}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out: Dict[str, Dict] = {}

    if with_baseline and not (skip_existing and _already_done("MajorityBaseline", label, quick)):
        print("\n=== MajorityBaseline [reference] ===")
        fold_sel = list(range(QUICK_FOLDS)) if (quick and folds is None) else folds
        base = run_majority_baseline(label, fold_sel)
        quick_report("MajorityBaseline", base)
        save_results("MajorityBaseline", loso=base, family="baseline",
                     config={"label": label, "feature": "-", "quick": quick,
                             "runner": "zeror"},
                     results_dir=RESULTS_DIR)
        out["MajorityBaseline"] = base

    for name in names:
        if skip_existing and _already_done(name, label, quick):
            print(f"\n[skip] {name} — results/{name}_results.json already matches "
                  f"(label={label}, quick={quick}). Use --no-skip-existing to recompute.")
            continue
        out[name] = run_one(name, label=label, quick=quick, folds=folds, device=device)

    build_comparison(RESULTS_DIR, label=label)
    try:
        make_plots(RESULTS_DIR, label=label)
    except Exception as exc:  # plotting must never sink a long run
        print(f"[warn] plotting failed: {exc!r}")
    return out


# --------------------------------------------------------------------------------------
# Comparative analysis
# --------------------------------------------------------------------------------------
def _collect(results_dir: str) -> List[Dict]:
    rows = []
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith("_results.json"):
            continue
        with open(os.path.join(results_dir, fn)) as f:
            blob = json.load(f)
        lo = blob.get("loso")
        if not lo:
            continue
        per_fold = lo.get("per_fold_scores", [])
        rows.append({
            "model": blob.get("model", fn[:-13]),
            "family": blob.get("family", ""),
            "params": lo.get("params_count", 0),
            "acc": lo.get("accuracy", np.nan),
            "acc_std": lo.get("accuracy_std", np.nan),
            "f1": lo.get("f1", np.nan),
            "f1_std": lo.get("f1_std", np.nan),
            "precision": lo.get("precision", np.nan),
            "recall": lo.get("recall", np.nan),
            "min_fold_acc": float(np.min([s["accuracy"] for s in per_fold])) if per_fold else np.nan,
            "max_fold_acc": float(np.max([s["accuracy"] for s in per_fold])) if per_fold else np.nan,
            "n_folds": len(per_fold),
            "train_time_sec": lo.get("train_time_sec", 0.0),
            "label": blob.get("config", {}).get("label", ""),
            "quick": bool(blob.get("config", {}).get("quick", False)),
            "per_fold": per_fold,
        })
    return rows


def build_comparison(results_dir: str = RESULTS_DIR, label: str = "") -> "pd.DataFrame":
    """Build + persist the subject-independent comparison table (CSV + Markdown)."""
    import pandas as pd

    rows = _collect(results_dir)
    if not rows:
        print("[compare] no LOSO results found yet.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop(columns=["per_fold"])
    # models first (by accuracy), baseline pinned to the bottom
    df["_is_base"] = df["model"].eq("MajorityBaseline")
    df = df.sort_values(["_is_base", "acc"], ascending=[True, False]).drop(columns="_is_base")

    csv_path = os.path.join(results_dir, "subject_independent_comparison.csv")
    df.to_csv(csv_path, index=False)

    def pm(m, s):
        return "-" if np.isnan(m) else (f"{m:.3f} ± {s:.3f}" if not np.isnan(s) else f"{m:.3f}")

    pretty = pd.DataFrame({
        "Model": df["model"],
        "Family": df["family"],
        "Params": df["params"].map(lambda p: f"{int(p):,}" if p else "-"),
        "LOSO Acc": [pm(m, s) for m, s in zip(df["acc"], df["acc_std"])],
        "LOSO F1": [pm(m, s) for m, s in zip(df["f1"], df["f1_std"])],
        "Precision": df["precision"].map(lambda v: f"{v:.3f}" if not np.isnan(v) else "-"),
        "Recall": df["recall"].map(lambda v: f"{v:.3f}" if not np.isnan(v) else "-"),
        "Fold range": [f"{lo:.2f}–{hi:.2f}" if not np.isnan(lo) else "-"
                       for lo, hi in zip(df["min_fold_acc"], df["max_fold_acc"])],
        "Folds": df["n_folds"],
        "Train (s)": df["train_time_sec"].map(lambda v: f"{v:,.0f}"),
    })
    md_path = os.path.join(results_dir, "subject_independent_comparison.md")
    hdr = f"# Subject-independent (LOSO) benchmark — DREAMER {label or df['label'].iloc[0]}\n\n"
    if df["quick"].any():
        hdr += "> ⚠️ some rows are from a **QUICK** smoke run (few folds / few epochs).\n\n"
    with open(md_path, "w") as f:
        f.write(hdr + pretty.to_markdown(index=False) + "\n")
    print(f"[compare] wrote {csv_path}\n[compare] wrote {md_path}\n")
    print(pretty.to_markdown(index=False))
    return df


def make_plots(results_dir: str = RESULTS_DIR, label: str = "") -> None:
    """Three comparison figures into results/figures/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _collect(results_dir)
    if not rows:
        return
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    tag = label or (rows[0]["label"] if rows else "")

    models = [r for r in rows if r["model"] != "MajorityBaseline"]
    models.sort(key=lambda r: r["acc"], reverse=True)
    base = next((r for r in rows if r["model"] == "MajorityBaseline"), None)

    # 1) accuracy per model with cross-fold std
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(models)), 5))
    x = np.arange(len(models))
    ax.bar(x, [r["acc"] for r in models], yerr=[r["acc_std"] for r in models],
           capsize=4, color="#4C72B0")
    ax.axhline(0.5, ls="--", lw=1, color="grey", label="chance (0.50)")
    if base:
        ax.axhline(base["acc"], ls=":", lw=1.5, color="crimson",
                   label=f"majority baseline ({base['acc']:.2f})")
    for xi, r in zip(x, models):
        ax.text(xi, r["acc"] + r["acc_std"] + 0.005, f"{r['acc']:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in models], rotation=30, ha="right")
    ax.set_ylabel("LOSO accuracy (mean ± std over folds)")
    ax.set_ylim(0.4, min(1.0, max([r["acc"] + r["acc_std"] for r in models] + [0.6]) + 0.08))
    ax.set_title(f"Subject-independent accuracy by model — DREAMER {tag}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "si_accuracy_by_model.png"), dpi=140)
    plt.close(fig)

    # 2) per-subject (fold) accuracy heatmap
    all_models = models + ([base] if base else [])
    subj_axis = sorted({s.get("test_subject", str(s.get("fold")))
                        for r in all_models for s in r["per_fold"]})
    M = np.full((len(all_models), len(subj_axis)), np.nan)
    for i, r in enumerate(all_models):
        for s in r["per_fold"]:
            key = s.get("test_subject", str(s.get("fold")))
            if key in subj_axis:
                M[i, subj_axis.index(key)] = s["accuracy"]
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(subj_axis)), 0.6 * len(all_models) + 2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.3, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(subj_axis)))
    ax.set_xticklabels([s.replace("subject_", "S") for s in subj_axis], rotation=90, fontsize=8)
    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels([r["model"] for r in all_models])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="accuracy", fraction=0.025, pad=0.02)
    ax.set_title(f"Per-held-out-subject accuracy — DREAMER {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "si_per_subject_heatmap.png"), dpi=140)
    plt.close(fig)

    # 3) accuracy vs parameter count
    e = [r for r in models if r["params"]]
    if e:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar([r["params"] for r in e], [r["acc"] for r in e],
                    yerr=[r["acc_std"] for r in e], fmt="o", capsize=3)
        for r in e:
            ax.annotate(r["model"], (r["params"], r["acc"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        if base:
            ax.axhline(base["acc"], ls=":", color="crimson", lw=1.2, label="majority baseline")
            ax.legend()
        ax.set_xscale("log")
        ax.set_xlabel("parameters (log scale)")
        ax.set_ylabel("LOSO accuracy")
        ax.set_title(f"Accuracy vs efficiency (subject-independent) — DREAMER {tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "si_accuracy_vs_params.png"), dpi=140)
        plt.close(fig)
    print(f"[plots] wrote 3 figures to {fig_dir}/")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Subject-independent (LOSO) benchmark for the DREAMER model zoo.")
    p.add_argument("--models", nargs="+", default=None,
                   help=f"subset of {list(REGISTRY)} (default: all)")
    p.add_argument("--label", default="valence",
                   choices=["valence", "arousal", "dominance"])
    p.add_argument("--quick", action="store_true",
                   help=f"{QUICK_FOLDS}-fold / {QUICK_EPOCHS}-epoch smoke test")
    p.add_argument("--folds", nargs="+", type=int, default=None,
                   help="explicit fold indices (0-22); overrides --quick fold count")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="recompute even if a matching results file exists")
    p.add_argument("--no-baseline", dest="with_baseline", action="store_false")
    p.add_argument("--aggregate-only", action="store_true",
                   help="skip training; only rebuild the comparison table + plots")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if args.aggregate_only:
        build_comparison(RESULTS_DIR, label=args.label)
        make_plots(RESULTS_DIR, label=args.label)
        return
    device = get_device() if args.device == "auto" else torch.device(args.device)
    run_all(
        models=args.models, label=args.label, quick=args.quick, folds=args.folds,
        skip_existing=args.skip_existing, device=device, with_baseline=args.with_baseline,
    )


if __name__ == "__main__":
    main()
