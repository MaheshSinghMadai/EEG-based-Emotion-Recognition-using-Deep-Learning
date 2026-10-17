"""
splits.py
=========
Frozen, hash-stamped evaluation split manifests.

Every protocol is materialised ONCE (after preprocessing) as a JSON manifest under
`data/splits/<DATASET>/<protocol>.json`. Training runs only read manifests, so all
five models are scored on byte-identical partitions, and every result file records
the manifest's SHA-256 so a reader can verify which split produced it.

Protocols
---------
Strict (headline results; subject- AND trial-disjoint):
    loso            leave-one-subject-out; validation = held-out *training* subjects
    subject_kfold   5-fold subject-grouped CV (same 80/20 budget as the leaky variants)

Leakage ablation (deliberately weaker; used only to quantify inflation):
    trial_kfold     5-fold trial-grouped CV, pooled over subjects. Trial-disjoint, but
                    every test subject also contributes training trials
                    (subject leakage only).
    random_window   5-fold random window CV, pooled. Windows of the same trial appear in
                    train and test (trial + subject leakage). This is the split used by
                    e.g. CIT-EmotionNet (random 6:4) and DAH-Net (random 80/20).

Cross-dataset:
    transfer_source source-dataset train/validation subjects; the whole target dataset
                    is the test set (built at run time from the target fingerprint).

Inflation estimates reported in the thesis:
    trial leakage   = acc(random_window) - acc(trial_kfold)
    subject leakage = acc(trial_kfold)   - acc(subject_kfold)
    total           = acc(random_window) - acc(subject_kfold)

Selection format
----------------
A partition is a list of selections over processed subject files:
    {"subject": "s01"}                         all windows of the subject
    {"subject": "s01", "trials": [0, 3, 7]}    all windows of those trials
    {"subject": "s01", "windows": [4, 9, 12]}  explicit row indices into the npz
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np

STRICT_PROTOCOLS = ("loso", "subject_kfold")
ABLATION_PROTOCOLS = ("trial_kfold", "random_window")
ALL_PROTOCOLS = STRICT_PROTOCOLS + ABLATION_PROTOCOLS

GUARANTEES = {
    "loso": {"subject_disjoint": True, "trial_disjoint": True, "window_disjoint": True},
    "subject_kfold": {"subject_disjoint": True, "trial_disjoint": True, "window_disjoint": True},
    "trial_kfold": {"subject_disjoint": False, "trial_disjoint": True, "window_disjoint": True},
    "random_window": {"subject_disjoint": False, "trial_disjoint": False, "window_disjoint": True},
    "transfer_source": {"subject_disjoint": True, "trial_disjoint": True, "window_disjoint": True},
}

N_KFOLD = 5
VAL_FRAC = 0.1


# --------------------------------------------------------------------------------------
# Fingerprint and hashing
# --------------------------------------------------------------------------------------
def dataset_fingerprint(processed_dir: str) -> Dict[str, Dict]:
    """{subject: {"n_windows": N, "trials": [trial ids in file order]}} from the npz files."""
    paths = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No processed subjects in {processed_dir}; run preprocessing.")
    fp = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            trial_index = z["trial_index"]
        fp[os.path.basename(path)[:-4]] = {
            "n_windows": int(trial_index.size),
            "trials": sorted(int(t) for t in np.unique(trial_index)),
        }
    return fp


def manifest_hash(folds: List[Dict]) -> str:
    return hashlib.sha256(json.dumps(folds, sort_keys=True).encode()).hexdigest()


def _n_val(n: int) -> int:
    return min(n - 1, max(1, int(round(VAL_FRAC * n))))


# --------------------------------------------------------------------------------------
# Protocol builders
# --------------------------------------------------------------------------------------
def build_loso(subjects: List[str], seed: int) -> List[Dict]:
    folds = []
    for k, test_subject in enumerate(subjects):
        rest = [s for s in subjects if s != test_subject]
        rng = np.random.default_rng(seed + k)
        val = sorted(rng.permutation(rest)[:_n_val(len(rest))].tolist())
        folds.append({
            "fold": k,
            "train": [{"subject": s} for s in rest if s not in val],
            "val": [{"subject": s} for s in val],
            "test": [{"subject": test_subject}],
        })
    return folds


def build_subject_kfold(subjects: List[str], seed: int, k: int = N_KFOLD) -> List[Dict]:
    rng = np.random.default_rng(seed)
    groups = [sorted(np.asarray(subjects)[idx].tolist())
              for idx in np.array_split(rng.permutation(len(subjects)), k)]
    folds = []
    for i, test in enumerate(groups):
        rest = [s for s in subjects if s not in test]
        val = sorted(np.random.default_rng(seed + i).permutation(rest)[:_n_val(len(rest))].tolist())
        folds.append({
            "fold": i,
            "train": [{"subject": s} for s in rest if s not in val],
            "val": [{"subject": s} for s in val],
            "test": [{"subject": s} for s in test],
        })
    return folds


def build_trial_kfold(fp: Dict[str, Dict], seed: int, k: int = N_KFOLD) -> List[Dict]:
    """Per subject, trials are split into k groups; fold i tests on group i of every subject."""
    rng = np.random.default_rng(seed)
    per_subject = {
        s: [sorted(int(t) for t in g) for g in np.array_split(rng.permutation(info["trials"]), k)]
        for s, info in fp.items()
    }
    folds = []
    for i in range(k):
        fold_rng = np.random.default_rng(seed + i)
        train, val, test = [], [], []
        for s, groups in per_subject.items():
            rest = [t for j, g in enumerate(groups) if j != i for t in g]
            v = sorted(fold_rng.permutation(rest)[:_n_val(len(rest))].tolist())
            train.append({"subject": s, "trials": [t for t in rest if t not in v]})
            val.append({"subject": s, "trials": v})
            test.append({"subject": s, "trials": groups[i]})
        folds.append({"fold": i, "train": train, "val": val, "test": test})
    return folds


def build_random_window(fp: Dict[str, Dict], seed: int, k: int = N_KFOLD) -> List[Dict]:
    """Windows are assigned to folds uniformly at random, ignoring trial and subject."""
    rng = np.random.default_rng(seed)
    per_subject = {
        s: [sorted(int(w) for w in g) for g in np.array_split(rng.permutation(info["n_windows"]), k)]
        for s, info in fp.items()
    }
    folds = []
    for i in range(k):
        fold_rng = np.random.default_rng(seed + i)
        train, val, test = [], [], []
        for s, groups in per_subject.items():
            rest = np.asarray([w for j, g in enumerate(groups) if j != i for w in g])
            v_mask = fold_rng.random(rest.size) < VAL_FRAC
            train.append({"subject": s, "windows": sorted(rest[~v_mask].tolist())})
            val.append({"subject": s, "windows": sorted(rest[v_mask].tolist())})
            test.append({"subject": s, "windows": groups[i]})
        folds.append({"fold": i, "train": train, "val": val, "test": test})
    return folds


def build_transfer_source(subjects: List[str], seed: int) -> List[Dict]:
    val = sorted(np.random.default_rng(seed).permutation(subjects)[:_n_val(len(subjects))].tolist())
    return [{
        "fold": 0,
        "train": [{"subject": s} for s in subjects if s not in val],
        "val": [{"subject": s} for s in val],
        "test": [],  # filled with every target-dataset subject at run time
    }]


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------
def make_manifest(protocol: str, dataset: str, folds: List[Dict], fp: Dict, seed: int) -> Dict:
    return {
        "protocol": protocol,
        "dataset": dataset,
        "guarantees": GUARANTEES[protocol],
        "seed": seed,
        "n_folds": len(folds),
        "fingerprint": {s: info["n_windows"] for s, info in fp.items()},
        "sha256": manifest_hash(folds),
        "folds": folds,
    }


def write_all_manifests(dataset: str, processed_dir: str, splits_dir: str, seed: int = 42) -> Dict[str, str]:
    os.makedirs(splits_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(splits_dir, "*.json")):
        os.remove(stale)
    fp = dataset_fingerprint(processed_dir)
    subjects = sorted(fp)
    builders = {
        "loso": lambda: build_loso(subjects, seed),
        "subject_kfold": lambda: build_subject_kfold(subjects, seed),
        "trial_kfold": lambda: build_trial_kfold(fp, seed),
        "random_window": lambda: build_random_window(fp, seed),
        "transfer_source": lambda: build_transfer_source(subjects, seed),
    }
    paths = {}
    for protocol, build in builders.items():
        manifest = make_manifest(protocol, dataset, build(), fp, seed)
        path = os.path.join(splits_dir, f"{protocol}.json")
        with open(path, "w") as f:
            json.dump(manifest, f)
        paths[protocol] = path
        print(f"[split] {path}  folds={manifest['n_folds']}  sha256={manifest['sha256'][:12]}")
    return paths


def load_manifest(path: str, processed_dir: Optional[str] = None) -> Dict:
    """Load a manifest, verify its hash, and (optionally) that it matches the data on disk."""
    with open(path) as f:
        manifest = json.load(f)
    if manifest_hash(manifest["folds"]) != manifest["sha256"]:
        raise ValueError(f"{path}: fold contents do not match the stored SHA-256 (edited?).")
    if processed_dir is not None:
        current = {s: i["n_windows"] for s, i in dataset_fingerprint(processed_dir).items()}
        if current != manifest["fingerprint"]:
            raise ValueError(
                f"{path} was built for different processed data; re-run preprocessing "
                "or `python -m src.splits` to regenerate the manifests."
            )
    return manifest


def selection_indices(selection: Dict, trial_index: np.ndarray) -> np.ndarray:
    """Row indices of one subject file selected by a manifest entry."""
    if "windows" in selection:
        return np.asarray(selection["windows"], dtype=np.int64)
    if "trials" in selection:
        return np.flatnonzero(np.isin(trial_index, selection["trials"]))
    return np.arange(trial_index.size)


def main() -> None:
    p = argparse.ArgumentParser(description="(Re)build frozen split manifests.")
    p.add_argument("--dataset", choices=["DEAP", "DREAMER"], required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    write_all_manifests(
        args.dataset,
        os.path.join("data", "processed", args.dataset),
        os.path.join("data", "splits", args.dataset),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
