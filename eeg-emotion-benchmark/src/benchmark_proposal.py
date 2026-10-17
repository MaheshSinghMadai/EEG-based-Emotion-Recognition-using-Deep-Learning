"""
benchmark_proposal.py
=====================
Unified, leakage-checked benchmark for the five thesis-proposal models.

Protocols (frozen manifests from `src.splits`):
    loso            strict headline result (subject- and trial-disjoint)
    subject_kfold   strict 5-fold, same train/test budget as the ablations
    trial_kfold     ablation: trial-disjoint, subjects shared       (subject leakage)
    random_window   ablation: random windows                        (trial + subject leakage)
    leakage_suite   runs subject_kfold + trial_kfold + random_window
    transfer        train on one dataset, test on every subject of the other

Every test window is predicted exactly once per seed. Predictions are pooled per
test subject, and the headline metric is the mean +/- std over subjects (the unit
used by the statistical tests in `src.analysis`). A majority-class baseline (the
training-set majority label) is scored on the same windows.

Examples
--------
    # smoke test: one fold, two epochs, one model
    python -m src.benchmark_proposal --protocol loso --dataset DEAP --models STGATE \
        --max-folds 1 --max-epochs 2

    # full strict run, all models, three seeds
    python -m src.benchmark_proposal --protocol loso --dataset DREAMER --seeds 42 43 44

    # leakage ablation and cross-dataset transfer
    python -m src.benchmark_proposal --protocol leakage_suite --dataset DEAP
    python -m src.benchmark_proposal --protocol transfer --source DEAP --target DREAMER
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_loader import UnlabeledDataset, build_fold_loaders, build_transfer_fold_loaders
from .evaluation import compute_metrics, count_parameters, get_device, set_seed
from .models.proposal import cit_emotionnet, clisa, dadpc, dah_net, stgate
from .preprocessing import COMMON_CHANNELS, PREPROCESSING_VERSION
from .splits import ABLATION_PROTOCOLS, ALL_PROTOCOLS, load_manifest
from .training import TrainRecipe, fit_supervised

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = ("DEAP", "DREAMER")
RESULTS_DIR = os.path.join(ROOT, "results", "proposal")
LABELS = ("valence", "arousal", "dominance")

# Benchmark-wide stopping rule (identical for every model; see src/training.py).
MAX_EPOCHS = 100
PATIENCE = 15


@dataclass(frozen=True)
class ProposalModel:
    name: str
    year: int
    family: str
    build: Callable
    feature: str
    recipe: TrainRecipe
    fit: Callable = fit_supervised
    uses_unlabeled_target: bool = False
    fit_kwargs: Dict = field(default_factory=dict)
    paper_protocol: str = ""


MODELS: Dict[str, ProposalModel] = {
    "DADPc": ProposalModel(
        "DADPc", 2025, "Domain adaptation (possibilistic clustering)", dadpc.build, "de",
        TrainRecipe(lr=1e-3, weight_decay=1e-4, batch_size=128,
                    source="Optimizer not reported; sigma=1e-3 from Sec. 6.3"),
        fit=dadpc.fit_dadpc, uses_unlabeled_target=True,
        paper_protocol="LOSO (transductive DA); DEAP subset: 14 subjects, 3 trials, 3 classes",
    ),
    "CIT-EmotionNet": ProposalModel(
        "CIT-EmotionNet", 2023, "Parallel CNN + Transformer fusion", cit_emotionnet.build, "de",
        TrainRecipe(lr=1e-5, weight_decay=0.0, batch_size=32, source="Sec. 3.1: Adam, lr 1e-5, batch 32"),
        paper_protocol="random 6:4 window split (SEED / SEED-IV)",
    ),
    "STGATE": ProposalModel(
        "STGATE", 2023, "Dynamic graph attention + Transformer", stgate.build, "de_seq",
        TrainRecipe(lr=1e-3, weight_decay=1e-4, batch_size=64,
                    source="TO VERIFY against the STGATE paper (PDF not in Papers/)"),
        paper_protocol="subject-dependent and subject-independent (SEED, SEED-IV, DREAMER) - verify",
    ),
    "DAH-Net": ProposalModel(
        "DAH-Net", 2026, "1D-CNN + BiLSTM + dual attention", dah_net.build, "handcrafted",
        TrainRecipe(optimizer="adamw", lr=8e-4, weight_decay=5e-5, batch_size=32,
                    scheduler="cosine_warm_restarts", scheduler_t0=50, label_smoothing=0.01,
                    grad_clip=1.0, augment="noise_scale", source="Sec. III-D, Tables I-II"),
        paper_protocol="random stratified 80/20 split of a 988-d Kaggle feature set",
    ),
    "CLISA": ProposalModel(
        "CLISA", 2022, "Contrastive self-supervised pretraining", clisa.build, "raw",
        TrainRecipe(lr=1e-3, weight_decay=1e-4, batch_size=64,
                    source="TO VERIFY against the CLISA paper (PDF not in Papers/)"),
        fit=clisa.fit_clisa, fit_kwargs={"pretrain_epochs": 30},
        paper_protocol="cross-subject (subject-grouped CV) on THU-EP and SEED - verify",
    ),
}


# --------------------------------------------------------------------------------------
# Paths and preprocessing validation
# --------------------------------------------------------------------------------------
def processed_dir(dataset: str) -> str:
    return os.path.join(ROOT, "data", "processed", dataset)


def manifest_path(dataset: str, protocol: str) -> str:
    return os.path.join(ROOT, "data", "splits", dataset, f"{protocol}.json")


def validate_preprocessing(dataset: str) -> Dict:
    path = os.path.join(processed_dir(dataset), "preprocess_config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}; run `python -m src.preprocessing --dataset {dataset}`.")
    with open(path) as f:
        cfg = json.load(f)
    expected = {"preprocessing_version": PREPROCESSING_VERSION, "fs": 128, "window_sec": 4.0,
                "band_preset": "harmonized", "channel_names": COMMON_CHANNELS}
    wrong = {k: cfg.get(k) for k, v in expected.items() if cfg.get(k) != v}
    if wrong:
        raise ValueError(f"{dataset} was not preprocessed with the harmonized protocol: {wrong}. "
                         "Re-run `python -m src.preprocessing`.")
    return cfg


# --------------------------------------------------------------------------------------
# One training run
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs = []
    for x, _y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu())
    p = torch.cat(probs).numpy()
    return (p >= 0.5).astype(np.int64), p


def train_and_predict(spec: ProposalModel, train, val, test, meta: Dict, device: torch.device,
                      recipe: TrainRecipe, seed: int, adaptation: str) -> Dict:
    set_seed(seed)
    model = spec.build({"n_classes": 2, "meta": meta}).to(device)
    kwargs = dict(spec.fit_kwargs, seed=seed)
    if spec.uses_unlabeled_target and adaptation == "transductive":
        kwargs["unlabeled_loader"] = DataLoader(UnlabeledDataset(test.dataset),
                                                batch_size=recipe.batch_size, shuffle=True,
                                                generator=torch.Generator().manual_seed(seed))
    info = spec.fit(model, train, val, device, recipe, **kwargs)
    y_pred, prob = predict(model, test, device)
    ds = test.dataset
    return {
        "y_true": ds.y.numpy(), "y_pred": y_pred, "prob": prob,
        "y_majority": np.full(len(ds), meta["train_majority_class"]),
        "subject": np.asarray(ds.subject_ids), "trial": np.asarray(ds.trial_ids),
        "window": np.asarray(ds.window_ids), "fit": info,
        "params_count": count_parameters(model),
    }


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def per_subject_metrics(pooled: Dict[str, np.ndarray], pred_key: str = "y_pred") -> List[Dict]:
    rows = []
    for s in sorted(set(pooled["subject"])):
        mask = pooled["subject"] == s
        m = compute_metrics(pooled["y_true"][mask], pooled[pred_key][mask])
        m["subject"] = str(s)
        rows.append(m)
    return rows


def summarize(per_subject: List[Dict]) -> Dict:
    out = {}
    for k in ("accuracy", "f1", "precision", "recall"):
        vals = np.array([r[k] for r in per_subject])
        out[k], out[f"{k}_std"] = float(vals.mean()), float(vals.std())
    return out


def _pool(runs: List[Dict]) -> Dict[str, np.ndarray]:
    keys = ("y_true", "y_pred", "prob", "y_majority", "subject", "trial", "window")
    return {k: np.concatenate([r[k] for r in runs]) for k in keys}


def _check_each_window_once(pooled: Dict[str, np.ndarray], domain: str) -> None:
    keys = list(zip(pooled["subject"], pooled["trial"], pooled["window"]))
    if len(keys) != len(set(keys)):
        raise AssertionError(f"{domain}: some test windows were predicted more than once.")


def aggregate_seeds(seed_results: Dict[int, Dict]) -> Dict:
    """Average per-subject metrics over seeds; also report the spread of the seed means."""
    subjects = [r["subject"] for r in next(iter(seed_results.values()))["per_subject"]]
    per_subject = []
    for i, s in enumerate(subjects):
        row = {"subject": s}
        for k in ("accuracy", "f1", "precision", "recall"):
            row[k] = float(np.mean([r["per_subject"][i][k] for r in seed_results.values()]))
        per_subject.append(row)
    summary = summarize(per_subject)
    summary["seed_std_accuracy"] = float(np.std([r["summary"]["accuracy"] for r in seed_results.values()]))
    summary["seed_std_f1"] = float(np.std([r["summary"]["f1"] for r in seed_results.values()]))
    return {"summary": summary, "per_subject": per_subject}


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[results] {path}")


def _save_predictions(path: str, pooled: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{k: (v.astype(str) if v.dtype == object else v) for k, v in pooled.items()})


# --------------------------------------------------------------------------------------
# Protocol drivers
# --------------------------------------------------------------------------------------
def _model_label(spec: ProposalModel, adaptation: str) -> str:
    return spec.name if not spec.uses_unlabeled_target or adaptation == "transductive" else f"{spec.name}-inductive"


def run_within(dataset: str, protocol: str, model_name: str, label: str = "valence",
               seeds: Sequence[int] = (42,), max_folds: Optional[int] = None,
               recipe_overrides: Optional[Dict] = None, adaptation: str = "transductive",
               device: Optional[torch.device] = None) -> Dict:
    spec = MODELS[model_name]
    validate_preprocessing(dataset)
    mpath = manifest_path(dataset, protocol)
    manifest = load_manifest(mpath, processed_dir(dataset))
    recipe = spec.recipe.with_overrides(max_epochs=MAX_EPOCHS, patience=PATIENCE).with_overrides(**(recipe_overrides or {}))
    folds = list(range(manifest["n_folds"]))[: max_folds or None]
    device = device or get_device()
    name = _model_label(spec, adaptation)
    seed_results, fold_logs = {}, []
    for seed in seeds:
        runs = []
        for fold in folds:
            train, val, test, meta = build_fold_loaders(
                processed_dir(dataset), mpath, fold, label=label, feature=spec.feature,
                batch_size=recipe.batch_size, channels=COMMON_CHANNELS, seed=seed, domain=dataset)
            run = train_and_predict(spec, train, val, test, meta, device, recipe, seed, adaptation)
            runs.append(run)
            acc = float((run["y_true"] == run["y_pred"]).mean())
            fold_logs.append({"seed": seed, "fold": fold, "test_subjects": meta["test_subjects"],
                              "accuracy": acc, "fit": run["fit"], "leakage": meta["leakage"]})
            print(f"  [{dataset}/{protocol}] {name} seed={seed} fold={fold:02d} acc={acc:.3f} "
                  f"({run['fit']['train_time_sec']:.0f}s, {run['fit']['epochs_run']} ep)")
        pooled = _pool(runs)
        _check_each_window_once(pooled, dataset)
        per_subject = per_subject_metrics(pooled)
        seed_results[seed] = {
            "summary": summarize(per_subject), "per_subject": per_subject,
            "majority_baseline": summarize(per_subject_metrics(pooled, "y_majority")),
        }
        _save_predictions(os.path.join(RESULTS_DIR, dataset, protocol, "predictions",
                                       f"{name}_{label}_seed{seed}.npz"), pooled)

    full = max_folds is None and not recipe_overrides
    result = {
        "model": name, "year": spec.year, "family": spec.family, "dataset": dataset,
        "protocol": protocol, "label": label, "seeds": list(seeds),
        "manifest_sha256": manifest["sha256"], "guarantees": manifest["guarantees"],
        "n_folds_run": len(folds), "n_folds_total": manifest["n_folds"],
        "run_status": "full_protocol" if full else "smoke_test_or_partial",
        "feature": spec.feature, "recipe": recipe.to_dict(),
        "adaptation": adaptation if spec.uses_unlabeled_target else "none",
        "paper_protocol": spec.paper_protocol,
        "params_count": runs[0]["params_count"],
        **aggregate_seeds(seed_results),
        "majority_baseline": seed_results[seeds[0]]["majority_baseline"],
        "per_seed": {str(s): r["summary"] for s, r in seed_results.items()},
        "folds": fold_logs,
    }
    save_json(os.path.join(RESULTS_DIR, dataset, protocol, f"{name}_{label}.json"), result)
    return result


def run_transfer(source: str, target: str, model_name: str, label: str = "valence",
                 seeds: Sequence[int] = (42,), recipe_overrides: Optional[Dict] = None,
                 adaptation: str = "transductive", device: Optional[torch.device] = None) -> Dict:
    if source == target:
        raise ValueError("Cross-dataset transfer needs two different datasets.")
    spec = MODELS[model_name]
    validate_preprocessing(source)
    validate_preprocessing(target)
    recipe = spec.recipe.with_overrides(max_epochs=MAX_EPOCHS, patience=PATIENCE).with_overrides(**(recipe_overrides or {}))
    device = device or get_device()
    name = _model_label(spec, adaptation)
    scenario = f"{source}_to_{target}"
    seed_results, logs = {}, []
    for seed in seeds:
        train, val, test, meta = build_transfer_fold_loaders(
            processed_dir(source), processed_dir(target), manifest_path(source, "transfer_source"),
            source, target, label=label, feature=spec.feature, batch_size=recipe.batch_size,
            channels=COMMON_CHANNELS, seed=seed)
        run = train_and_predict(spec, train, val, test, meta, device, recipe, seed, adaptation)
        pooled = _pool([run])
        _check_each_window_once(pooled, target)
        per_subject = per_subject_metrics(pooled)
        seed_results[seed] = {"summary": summarize(per_subject), "per_subject": per_subject,
                              "majority_baseline": summarize(per_subject_metrics(pooled, "y_majority"))}
        logs.append({"seed": seed, "fit": run["fit"], "leakage": meta["leakage"],
                     "manifest_sha256": meta["manifest_sha256"]})
        _save_predictions(os.path.join(RESULTS_DIR, scenario, "predictions",
                                       f"{name}_{label}_seed{seed}.npz"), pooled)
        print(f"  [{scenario}] {name} seed={seed} acc={seed_results[seed]['summary']['accuracy']:.3f}")
    result = {
        "model": name, "year": spec.year, "family": spec.family, "source": source, "target": target,
        "protocol": "transfer", "label": label, "seeds": list(seeds),
        "run_status": "full_protocol" if not recipe_overrides else "smoke_test_or_partial",
        "feature": spec.feature, "recipe": recipe.to_dict(),
        "adaptation": adaptation if spec.uses_unlabeled_target else "none",
        "normalization": "fit on source training subjects only",
        "checkpoint_selection": "minimum source-validation cross-entropy",
        "params_count": run["params_count"],
        **aggregate_seeds(seed_results),
        "majority_baseline": seed_results[seeds[0]]["majority_baseline"],
        "per_seed": {str(s): r["summary"] for s, r in seed_results.items()},
        "runs": logs,
    }
    save_json(os.path.join(RESULTS_DIR, scenario, f"{name}_{label}.json"), result)
    return result


def write_comparison(out_dir: str, label: str) -> None:
    """Rank every model result stored in `out_dir` for `label` (not just the latest run)."""
    results = {}
    for fn in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if fn.endswith(f"_{label}.json"):
            with open(os.path.join(out_dir, fn)) as f:
                r = json.load(f)
            results[r["model"]] = r
    if not results:
        return
    first = next(iter(results.values()))
    rows = [{
        "rank": rank, "model": name,
        "accuracy": r["summary"]["accuracy"], "accuracy_std": r["summary"]["accuracy_std"],
        "f1": r["summary"]["f1"], "f1_std": r["summary"]["f1_std"],
        "run_status": r["run_status"],
    } for rank, (name, r) in enumerate(
        sorted(results.items(), key=lambda kv: kv[1]["summary"]["accuracy"], reverse=True), start=1)]
    base = first["majority_baseline"]
    rows.append({"rank": "-", "model": "Majority baseline", "accuracy": base["accuracy"],
                 "accuracy_std": base["accuracy_std"], "f1": base["f1"], "f1_std": base["f1_std"],
                 "run_status": "reference"})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"comparison_{label}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, f"comparison_{label}.md"), "w") as f:
        f.write(f"# {os.path.relpath(out_dir, RESULTS_DIR)} ({label})\n\n")
        if any(r["run_status"] == "smoke_test_or_partial" for r in rows):
            f.write("> Smoke/partial run only; do not interpret as thesis results.\n\n")
        f.write("| Rank | Model | Accuracy (mean ± std over subjects) | Macro F1 | Run |\n|---:|---|---:|---:|---|\n")
        for r in rows:
            f.write(f"| {r['rank']} | {r['model']} | {r['accuracy']:.4f} ± {r['accuracy_std']:.4f} | "
                    f"{r['f1']:.4f} ± {r['f1_std']:.4f} | {r['run_status']} |\n")
    print(f"[compare] {out_dir}/comparison_{label}.csv")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", choices=list(ALL_PROTOCOLS) + ["leakage_suite", "transfer"], default="loso")
    p.add_argument("--dataset", choices=DATASETS, default="DEAP")
    p.add_argument("--source", choices=DATASETS, default="DEAP")
    p.add_argument("--target", choices=DATASETS, default="DREAMER")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--label", choices=LABELS, default="valence")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--max-folds", type=int, default=None, help="Run only the first N folds (smoke test).")
    p.add_argument("--max-epochs", type=int, default=None, help="Override the epoch cap (smoke test).")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--dadpc-adaptation", choices=["transductive", "none"], default="transductive")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    overrides = {k: v for k, v in {"max_epochs": args.max_epochs, "patience": args.patience}.items() if v}
    device = get_device()
    print(f"[device] {device}")
    if args.protocol == "transfer":
        for m in args.models:
            run_transfer(args.source, args.target, m, args.label, args.seeds, overrides,
                         args.dadpc_adaptation, device)
        write_comparison(os.path.join(RESULTS_DIR, f"{args.source}_to_{args.target}"), args.label)
        return
    protocols = ["subject_kfold", *ABLATION_PROTOCOLS] if args.protocol == "leakage_suite" else [args.protocol]
    for protocol in protocols:
        start = time.time()
        for m in args.models:
            run_within(args.dataset, protocol, m, args.label, args.seeds, args.max_folds,
                       overrides, args.dadpc_adaptation, device)
        write_comparison(os.path.join(RESULTS_DIR, args.dataset, protocol), args.label)
        print(f"[{args.dataset}/{protocol}] done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
