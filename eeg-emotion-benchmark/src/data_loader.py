"""
data_loader.py
==============
PyTorch `Dataset` / `DataLoader` construction for the benchmark.

Feature views of the same windows (all produced by `src.preprocessing`):
    * "raw"         : filtered signal               [C, T]      DAH-Net raw ablation, CLISA
    * "de"          : differential entropy           [C, B]      DADPc, CIT-EmotionNet
    * "de_seq"      : DE per 1 s sub-window          [C, S, B]   STGATE
    * "handcrafted" : engineered feature vector      [F]         DAH-Net

Protocol entry points:
    * `build_fold_loaders`          one fold of a frozen manifest (`src.splits`) on one dataset
    * `build_transfer_fold_loaders` source train/val manifest + every target subject as test

Both fit normalization on the TRAIN partition only and run the automated leakage
assertions in `src.leakage` before returning. `build_loso_loaders` /
`build_transfer_loaders` are the earlier subject-list based helpers, kept for
backward compatibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .leakage import check_partitions
from .splits import load_manifest, selection_indices

LABEL_KEYS = {
    "valence": "labels_valence",
    "arousal": "labels_arousal",
    "dominance": "labels_dominance",
}
FEATURE_KEYS = {
    "raw": "raw_windows",
    "de": "de_features",
    "de_seq": "de_sequence",
    "handcrafted": "handcrafted",
}

# --------------------------------------------------------------------------------------
# Raw npz access
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _load_key(path: str, key: str) -> np.ndarray:
    """Load ONE array of a subject file (cached), so raw windows are only read if used."""
    with np.load(path, allow_pickle=True) as z:
        return z[key]


@lru_cache(maxsize=64)
def _keys_in(path: str) -> Tuple[str, ...]:
    with np.load(path, allow_pickle=True) as z:
        return tuple(z.files)


def _window_index(trial_index: np.ndarray) -> np.ndarray:
    """Position of each window inside its trial (for files written before v2)."""
    out = np.zeros_like(trial_index)
    for t in np.unique(trial_index):
        idx = np.flatnonzero(trial_index == t)
        out[idx] = np.arange(idx.size)
    return out


@dataclass
class SubjectArrays:
    subject_id: str
    path: str
    trial_index: np.ndarray        # [N]
    window_index: np.ndarray       # [N]
    labels: Dict[str, np.ndarray]  # name -> [N] in {0,1}
    channel_names: List[str]
    band_names: List[str]
    fs: int
    window_samples: int

    def feature(self, view: str) -> np.ndarray:
        key = FEATURE_KEYS.get(view)
        if key is None:
            raise ValueError(f"Unknown feature view {view!r}; expected one of {list(FEATURE_KEYS)}.")
        if key not in _keys_in(self.path):
            raise KeyError(
                f"{self.path} has no {key!r}; re-run `python -m src.preprocessing` (v2)."
            )
        return _load_key(self.path, key).astype(np.float32)

    # Backward-compatible attribute access.
    @property
    def raw_windows(self) -> np.ndarray:
        return self.feature("raw")

    @property
    def de_features(self) -> np.ndarray:
        return self.feature("de")


def load_subject(processed_dir: str, sid: str) -> SubjectArrays:
    path = os.path.join(processed_dir, f"{sid}.npz")
    keys = _keys_in(path)
    trial_index = _load_key(path, "trial_index").astype(np.int64)
    window_index = (
        _load_key(path, "window_index").astype(np.int64)
        if "window_index" in keys else _window_index(trial_index)
    )
    return SubjectArrays(
        subject_id=str(_load_key(path, "subject_id")),
        path=path,
        trial_index=trial_index,
        window_index=window_index,
        labels={k: _load_key(path, v).astype(np.int64) for k, v in LABEL_KEYS.items()},
        channel_names=[str(c) for c in _load_key(path, "channel_names")],
        band_names=[str(b) for b in _load_key(path, "band_names")],
        fs=int(_load_key(path, "fs")),
        window_samples=int(_load_key(path, "window_samples")),
    )


def list_subjects(processed_dir: str) -> List[str]:
    return sorted(f[:-4] for f in os.listdir(processed_dir) if f.endswith(".npz"))


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------
@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    feature: str
    n_fit: int = 0  # number of rows the statistics were fitted on (leakage audit)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def fit_normalizer(x_train: np.ndarray, feature: str) -> Normalizer:
    """
    raw         : [N, C, T]    -> per-channel stats over (N, T)
    de          : [N, C, B]    -> per-(channel, band) stats over N
    de_seq      : [N, C, S, B] -> per-(channel, band) stats over (N, S)
    handcrafted : [N, F]       -> per-feature stats over N
    """
    axes = {"raw": (0, 2), "de": (0,), "de_seq": (0, 2), "handcrafted": (0,)}.get(feature)
    if axes is None:
        raise ValueError(feature)
    mean = x_train.mean(axis=axes, keepdims=True)
    std = np.clip(x_train.std(axis=axes, keepdims=True), 1e-6, None)
    return Normalizer(mean=mean.astype(np.float32), std=std.astype(np.float32),
                      feature=feature, n_fit=int(x_train.shape[0]))


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class EEGWindowDataset(Dataset):
    """Tensor-backed dataset of EEG windows.

    x : float32 tensor (shape depends on the feature view)
    y : long scalar in {0, 1}
    `.subject_ids`, `.trial_ids` and `.window_ids` are aligned with samples and drive
    the leakage checks, per-subject scoring and CLISA's stimulus-aligned pairing.
    `.domain` names the dataset the windows came from.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subject_ids: Sequence[str],
        trial_ids: Sequence[int],
        normalizer: Optional[Normalizer] = None,
        window_ids: Optional[Sequence[int]] = None,
        domain: str = "",
    ):
        if normalizer is not None:
            x = normalizer.apply(x)
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        self.subject_ids = list(subject_ids)
        self.trial_ids = list(trial_ids)
        self.window_ids = list(window_ids) if window_ids is not None else list(range(len(self.y)))
        self.domain = domain

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]

    # convenience
    @property
    def n_classes(self) -> int:
        return int(self.y.max().item()) + 1 if len(self) else 0

    def class_counts(self) -> Dict[int, int]:
        vals, cnts = torch.unique(self.y, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}


# --------------------------------------------------------------------------------------
# Assembling arrays for a protocol
# --------------------------------------------------------------------------------------
def _select_feature(arr: SubjectArrays, feature: str, channels: Optional[List[str]]) -> np.ndarray:
    x = arr.feature(feature)
    if channels is None or list(channels) == arr.channel_names:
        return x
    if feature == "handcrafted":
        raise ValueError("The handcrafted view is computed on all channels; cannot subset it.")
    missing = sorted(set(channels) - set(arr.channel_names))
    if missing:
        raise ValueError(f"Subject {arr.subject_id!r} is missing requested channels: {missing}")
    return x[:, [arr.channel_names.index(c) for c in channels]]


def _gather_subject(
    processed_dir: str, sid: str, feature: str, label: str, channels: Optional[List[str]]
):
    a = load_subject(processed_dir, sid)
    x = _select_feature(a, feature, channels)
    y = a.labels[label]
    sids = [sid] * len(y)
    tids = a.trial_index.tolist()
    return x, y, sids, tids


@dataclass
class Partition:
    x: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    window_ids: np.ndarray

    def dataset(self, normalizer: Optional[Normalizer], domain: str) -> "EEGWindowDataset":
        return EEGWindowDataset(self.x, self.y, self.subject_ids, self.trial_ids,
                                normalizer, window_ids=self.window_ids, domain=domain)


def gather_selections(
    processed_dir: str,
    selections: Sequence[Dict],
    feature: str,
    label: str,
    channels: Optional[List[str]] = None,
) -> Partition:
    """Materialise a manifest partition (list of selections) into arrays."""
    xs, ys, sids, tids, wids = [], [], [], [], []
    for sel in selections:
        a = load_subject(processed_dir, sel["subject"])
        idx = selection_indices(sel, a.trial_index)
        if idx.size == 0:
            continue
        xs.append(_select_feature(a, feature, channels)[idx])
        ys.append(a.labels[label][idx])
        sids.append(np.full(idx.size, sel["subject"], dtype=object))
        tids.append(a.trial_index[idx])
        wids.append(a.window_index[idx])
    if not xs:
        raise ValueError("Partition selects no windows.")
    return Partition(np.concatenate(xs), np.concatenate(ys), np.concatenate(sids),
                     np.concatenate(tids), np.concatenate(wids))


def _loaders(train_ds, val_ds, test_ds, batch_size: int, seed: int, num_workers: int = 0):
    common = {"num_workers": num_workers}
    # A trailing batch of one sample would break BatchNorm layers in training mode.
    drop_last = len(train_ds) % batch_size == 1
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last,
                       generator=torch.Generator().manual_seed(seed), **common)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common) if val_ds is not None else None
    test = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train, val, test


def build_fold_loaders(
    processed_dir: str,
    manifest_path: str,
    fold: int,
    label: str = "valence",
    feature: str = "de",
    batch_size: int = 128,
    channels: Optional[List[str]] = None,
    normalize: bool = True,
    seed: int = 42,
    domain: str = "",
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, Dict]:
    """One fold of a frozen manifest -> (train, val, test, meta). Leakage-checked."""
    manifest = load_manifest(manifest_path, processed_dir)
    spec = manifest["folds"][fold]
    tr = gather_selections(processed_dir, spec["train"], feature, label, channels)
    va = gather_selections(processed_dir, spec["val"], feature, label, channels) if spec["val"] else None
    te = gather_selections(processed_dir, spec["test"], feature, label, channels)
    norm = fit_normalizer(tr.x, feature) if normalize else None
    train_ds = tr.dataset(norm, domain)
    val_ds = va.dataset(norm, domain) if va is not None else None
    test_ds = te.dataset(norm, domain)
    leakage = check_partitions(train_ds, val_ds, test_ds, manifest["guarantees"],
                               normalizer_fit_rows=norm.n_fit if norm else None)
    train, val, test = _loaders(train_ds, val_ds, test_ds, batch_size, seed)
    meta = {
        "protocol": manifest["protocol"],
        "dataset": manifest["dataset"],
        "manifest_sha256": manifest["sha256"],
        "fold": fold,
        "test_subjects": sorted(set(test_ds.subject_ids)),
        "test_subject": test_ds.subject_ids[0] if len(set(test_ds.subject_ids)) == 1 else None,
        "validation_subjects": sorted(set(val_ds.subject_ids)) if val_ds is not None else [],
        "label": label,
        "feature": feature,
        "n_train": len(train_ds),
        "n_val": len(val_ds) if val_ds is not None else 0,
        "n_test": len(test_ds),
        "input_shape": tuple(train_ds.x.shape[1:]),
        "train_majority_class": int(np.bincount(tr.y, minlength=2).argmax()),
        "leakage": leakage,
    }
    return train, val, test, meta


def build_transfer_fold_loaders(
    source_dir: str,
    target_dir: str,
    source_manifest_path: str,
    source: str,
    target: str,
    label: str = "valence",
    feature: str = "de",
    batch_size: int = 128,
    channels: Optional[List[str]] = None,
    normalize: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Source train/val from the frozen manifest; all target subjects are the test set.

    Normalization and checkpoint selection use source data only. Partitions are
    namespaced by dataset in the leakage check.
    """
    manifest = load_manifest(source_manifest_path, source_dir)
    spec = manifest["folds"][0]
    tr = gather_selections(source_dir, spec["train"], feature, label, channels)
    va = gather_selections(source_dir, spec["val"], feature, label, channels)
    targets = [{"subject": s} for s in list_subjects(target_dir)]
    te = gather_selections(target_dir, targets, feature, label, channels)
    norm = fit_normalizer(tr.x, feature) if normalize else None
    train_ds, val_ds, test_ds = tr.dataset(norm, source), va.dataset(norm, source), te.dataset(norm, target)
    leakage = check_partitions(train_ds, val_ds, test_ds, manifest["guarantees"],
                               normalizer_fit_rows=norm.n_fit if norm else None)
    train, val, test = _loaders(train_ds, val_ds, test_ds, batch_size, seed)
    meta = {
        "protocol": "cross_dataset",
        "source": source,
        "target": target,
        "manifest_sha256": manifest["sha256"],
        "source_subjects": sorted(set(train_ds.subject_ids)),
        "validation_subjects": sorted(set(val_ds.subject_ids)),
        "target_subjects": sorted(set(test_ds.subject_ids)),
        "test_subjects": sorted(set(test_ds.subject_ids)),
        "label": label,
        "feature": feature,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "input_shape": tuple(train_ds.x.shape[1:]),
        "train_majority_class": int(np.bincount(tr.y, minlength=2).argmax()),
        "leakage": leakage,
    }
    return train, val, test, meta


class UnlabeledDataset(Dataset):
    """Inputs only — used to hand target-domain windows to domain-adaptation models
    without ever exposing their labels (see `leakage.assert_unlabeled`)."""

    def __init__(self, base: EEGWindowDataset):
        self.x = base.x

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.x[i]


def build_loso_loaders(
    processed_dir: str,
    loso_path: str,
    fold: int,
    label: str = "valence",
    feature: str = "raw",
    batch_size: int = 128,
    channels: Optional[List[str]] = None,
    normalize: bool = True,
    val_frac: float = 0.0,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, Dict]:
    """LOSO fold -> (train_loader, val_loader_or_None, test_loader, meta).

    `val_frac` > 0 carves a stratified validation slice out of the 22 training subjects
    (used by DADPc / domain-adaptation notebooks for early stopping).
    """
    with open(loso_path) as f:
        loso = json.load(f)
    spec = loso["folds"][fold]
    test_sid = spec["test_subject"]
    train_sids = spec["train_subjects"]

    xs, ys, sid_list, tid_list = [], [], [], []
    for s in train_sids:
        x, y, sids, tids = _gather_subject(processed_dir, s, feature, label, channels)
        xs.append(x); ys.append(y); sid_list += sids; tid_list += tids
    x_tr = np.concatenate(xs, 0); y_tr = np.concatenate(ys, 0)
    sid_tr = np.array(sid_list); tid_tr = np.array(tid_list)

    x_te, y_te, sid_te, tid_te = _gather_subject(processed_dir, test_sid, feature, label, channels)

    dl_common = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        dl_common["persistent_workers"] = True

    val_loader = None
    if val_frac and val_frac > 0:
        subjects = np.unique(sid_tr)
        if len(subjects) < 2:
            raise ValueError("At least two training subjects are required for validation.")
        n_val_subjects = min(
            len(subjects) - 1,
            max(1, int(round(val_frac * len(subjects)))),
        )
        rng = np.random.default_rng(seed)
        val_subjects = rng.permutation(subjects)[:n_val_subjects]
        val_mask = np.isin(sid_tr, val_subjects)
        v_idx, t_idx = np.flatnonzero(val_mask), np.flatnonzero(~val_mask)
        norm = fit_normalizer(x_tr[t_idx], feature) if normalize else None
        ds_val = EEGWindowDataset(x_tr[v_idx], y_tr[v_idx], sid_tr[v_idx], tid_tr[v_idx], norm)
        ds_tr = EEGWindowDataset(x_tr[t_idx], y_tr[t_idx], sid_tr[t_idx], tid_tr[t_idx], norm)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, **dl_common)
    else:
        norm = fit_normalizer(x_tr, feature) if normalize else None
        ds_tr = EEGWindowDataset(x_tr, y_tr, sid_tr, tid_tr, norm)

    ds_te = EEGWindowDataset(x_te, y_te, sid_te, tid_te, norm)

    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False,
                    generator=g, **dl_common)
    el = DataLoader(ds_te, batch_size=batch_size, shuffle=False, **dl_common)
    meta = {
        "protocol": "loso", "fold": fold, "test_subject": test_sid,
        "n_train_subjects": len(set(ds_tr.subject_ids)),
        "validation_subjects": sorted(set(val_loader.dataset.subject_ids)) if val_loader else [],
        "label": label, "feature": feature,
        "n_train": len(ds_tr), "n_test": len(ds_te),
        "input_shape": tuple(ds_tr.x.shape[1:]),
        # domain ids: integer per training subject, for MMD / domain-adaptation losses
        "train_domain_ids": {
            s: i for i, s in enumerate(sorted(set(ds_tr.subject_ids)))
        },
    }
    return tl, val_loader, el, meta


def build_transfer_loaders(
    source_dir: str,
    target_dir: str,
    source_subjects: Sequence[str],
    target_subjects: Sequence[str],
    label: str = "valence",
    feature: str = "raw",
    batch_size: int = 128,
    channels: Optional[List[str]] = None,
    normalize: bool = True,
    validation_subjects: int = 1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Build source-train/source-validation and held-out-target transfer loaders.

    Validation subjects are selected only from the source data. Target samples are
    never used to fit normalization or select a checkpoint.
    """
    source_subjects = list(source_subjects)
    target_subjects = list(target_subjects)
    if not source_subjects or not target_subjects:
        raise ValueError("Transfer evaluation needs source and target subjects.")
    if validation_subjects < 1 or validation_subjects >= len(source_subjects):
        raise ValueError(
            "validation_subjects must be at least one and leave a source subject for training."
        )

    rng = np.random.default_rng(seed)
    val_subjects = set(rng.permutation(source_subjects)[:validation_subjects].tolist())
    train_subjects = [sid for sid in source_subjects if sid not in val_subjects]

    def gather(directory: str, subjects: Sequence[str]):
        xs, ys, sid_list, tid_list = [], [], [], []
        for sid in subjects:
            x, y, sids, tids = _gather_subject(directory, sid, feature, label, channels)
            xs.append(x)
            ys.append(y)
            sid_list.extend(sids)
            tid_list.extend(tids)
        return (
            np.concatenate(xs, axis=0),
            np.concatenate(ys, axis=0),
            np.asarray(sid_list),
            np.asarray(tid_list),
        )

    x_train, y_train, sid_train, tid_train = gather(source_dir, train_subjects)
    x_val, y_val, sid_val, tid_val = gather(source_dir, sorted(val_subjects))
    x_test, y_test, sid_test, tid_test = gather(target_dir, target_subjects)
    normalizer = fit_normalizer(x_train, feature) if normalize else None

    common = {"num_workers": num_workers, "pin_memory": pin_memory}
    train_ds = EEGWindowDataset(x_train, y_train, sid_train, tid_train, normalizer)
    val_ds = EEGWindowDataset(x_val, y_val, sid_val, tid_val, normalizer)
    test_ds = EEGWindowDataset(x_test, y_test, sid_test, tid_test, normalizer)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), **common,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    meta = {
        "protocol": "cross_dataset",
        "source_subjects": train_subjects,
        "validation_subjects": sorted(val_subjects),
        "target_subjects": target_subjects,
        "label": label,
        "feature": feature,
        "n_train": len(train_ds),
        "n_validation": len(val_ds),
        "n_test": len(test_ds),
        "input_shape": tuple(train_ds.x.shape[1:]),
    }
    return train_loader, val_loader, test_loader, meta


def loso_domain_id_tensor(dataset: EEGWindowDataset, domain_map: Dict[str, int]) -> torch.Tensor:
    """Map each sample's subject id to its integer domain id (for DADPc)."""
    return torch.tensor([domain_map[s] for s in dataset.subject_ids], dtype=torch.long)


# --------------------------------------------------------------------------------------
# Siamese pair wrapper (Siam-GCAN, notebook 03)
# --------------------------------------------------------------------------------------
def sample_siamese_pairs(
    y: np.ndarray,
    pairs_per_sample: int = 1,
    seed: int = 42,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Siamese pair sampling.

    For every window we draw `pairs_per_sample` partners; each partner is same-class
    with probability ~0.5. Returns ``(idx_a, idx_b, pair_label)`` where
    ``pair_label = 1.0`` iff the two windows share an emotion class.

    Distribution-equivalent to the previous per-sample Python loop (identical for the
    2-class DREAMER targets) but ~100x faster, so resampling every epoch is free.
    """
    rng = rng or np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    classes = np.unique(y)
    by_class = {int(c): np.where(y == c)[0] for c in classes}

    idx_a = np.repeat(np.arange(n, dtype=np.int64), pairs_per_sample)
    m = idx_a.size
    ya = y[idx_a]
    want_same = rng.random(m) < 0.5
    idx_b = np.empty(m, dtype=np.int64)

    for c in classes:
        c = int(c)
        pool_same = by_class[c]
        others = (
            np.concatenate([by_class[o] for o in by_class if o != c])
            if len(by_class) > 1 else pool_same
        )
        m_same = want_same & (ya == c)
        m_diff = (~want_same) & (ya == c)
        if m_same.any():
            idx_b[m_same] = rng.choice(pool_same, size=int(m_same.sum()))
        if m_diff.any():
            idx_b[m_diff] = rng.choice(others, size=int(m_diff.sum()))

    plab = (y[idx_a] == y[idx_b]).astype(np.float32)
    return idx_a, idx_b, plab


class SiamesePairDataset(Dataset):
    """Wrap an `EEGWindowDataset`; each item is a pair (x_a, x_b, pair_label, y_a, y_b).

    pair_label = 1 if the two windows share the same emotion class, else 0.
    Pairs are resampled every epoch via `resample()` for stochastic coverage.
    """

    def __init__(self, base: EEGWindowDataset, pairs_per_sample: int = 1, seed: int = 42):
        self.base = base
        self.pairs_per_sample = pairs_per_sample
        self.rng = np.random.default_rng(seed)
        self._y = base.y.numpy()
        self._build()

    def _build(self):
        self.idx_a, self.idx_b, self.plab = sample_siamese_pairs(
            self._y, pairs_per_sample=self.pairs_per_sample, rng=self.rng
        )

    def resample(self):
        self._build()

    def __len__(self):
        return len(self.idx_a)

    def __getitem__(self, i):
        a, b = int(self.idx_a[i]), int(self.idx_b[i])
        xa, ya = self.base[a]
        xb, yb = self.base[b]
        return xa, xb, torch.tensor(self.plab[i]), ya, yb


def make_pair_loader(base: EEGWindowDataset, batch_size: int = 64, shuffle: bool = True,
                     pairs_per_sample: int = 1, seed: int = 42, num_workers: int = 0):
    ds = SiamesePairDataset(base, pairs_per_sample=pairs_per_sample, seed=seed)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# --------------------------------------------------------------------------------------
# Adjacency helpers for GCN-family models
# --------------------------------------------------------------------------------------
# ASSUMPTION: approximate 2-D sensor coordinates for the Emotiv EPOC 14 channels,
# used only to seed a distance-based adjacency where a model wants an initial graph.
# Values are unit-disc projected 10-20 positions (x=right, y=front); good enough for
# a Gaussian-kernel prior — DGCNN then learns its own adjacency from here.
EPOC_COORDS = {
    "AF3": (-0.25, 0.90), "AF4": (0.25, 0.90),
    "F7": (-0.80, 0.55), "F3": (-0.40, 0.55), "F4": (0.40, 0.55), "F8": (0.80, 0.55),
    "FC5": (-0.65, 0.25), "FC6": (0.65, 0.25),
    "T7": (-0.95, 0.00), "T8": (0.95, 0.00),
    "P7": (-0.75, -0.55), "P8": (0.75, -0.55),
    "O1": (-0.30, -0.90), "O2": (0.30, -0.90),
}


def distance_adjacency(channel_names: Sequence[str], sigma: float = 0.5,
                       self_loops: bool = True) -> np.ndarray:
    coords = np.array([EPOC_COORDS[c] for c in channel_names], dtype=np.float32)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    A = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
    if not self_loops:
        np.fill_diagonal(A, 0.0)
    return A
