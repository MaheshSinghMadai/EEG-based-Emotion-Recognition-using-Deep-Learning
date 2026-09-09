"""
data_loader.py
==============
Shared PyTorch `Dataset` / `DataLoader` construction for every benchmark notebook.

Two feature views of the same windows are supported:
    * "raw" : filtered windowed signal, shape [14, window_samples]  (EEGNet, CNN models)
    * "de"  : Differential-Entropy matrix,  shape [14, 5]            (DGCNN, GCN models)

Validation protocol (subject-independent):
    * LOSO : `build_loso_loaders(fold)` using data/splits/loso_splits.json

Normalization statistics are always computed on the TRAIN partition only and applied
to test (no leakage). A Siamese pair wrapper is provided for Siam-GCAN.
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

LABEL_KEYS = {
    "valence": "labels_valence",
    "arousal": "labels_arousal",
    "dominance": "labels_dominance",
}

# 5-electrode consumer-grade subset used by notebook 06 (indices into CHANNEL_NAMES).
# AF3, F3, F4, AF4, O1  -> frontal asymmetry pair + one posterior reference.
REDUCED_5_CHANNELS = ["AF3", "F3", "F4", "AF4", "O1"]


# --------------------------------------------------------------------------------------
# Raw npz access
# --------------------------------------------------------------------------------------
@dataclass
class SubjectArrays:
    subject_id: str
    raw_windows: np.ndarray      # [N, 14, win]
    de_features: np.ndarray      # [N, 14, 5]
    trial_index: np.ndarray      # [N]
    labels: Dict[str, np.ndarray]  # name -> [N] in {0,1}
    channel_names: List[str]
    band_names: List[str]
    fs: int
    window_samples: int


@lru_cache(maxsize=32)
def _load_npz_cached(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def load_subject(processed_dir: str, sid: str) -> SubjectArrays:
    path = os.path.join(processed_dir, f"{sid}.npz")
    z = _load_npz_cached(path)
    return SubjectArrays(
        subject_id=str(z["subject_id"]),
        raw_windows=z["raw_windows"].astype(np.float32),
        de_features=z["de_features"].astype(np.float32),
        trial_index=z["trial_index"].astype(np.int64),
        labels={k: z[LABEL_KEYS[k]].astype(np.int64) for k in LABEL_KEYS},
        channel_names=[str(c) for c in z["channel_names"]],
        band_names=[str(b) for b in z["band_names"]],
        fs=int(z["fs"]),
        window_samples=int(z["window_samples"]),
    )


def list_subjects(processed_dir: str) -> List[str]:
    files = [f for f in os.listdir(processed_dir) if f.startswith("subject_") and f.endswith(".npz")]
    return sorted(f[:-4] for f in files)


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------
@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    feature: str  # "raw" or "de"

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def fit_normalizer(x_train: np.ndarray, feature: str) -> Normalizer:
    """
    raw: x_train is [N, C, T]  -> per-channel stats over (N, T)
    de : x_train is [N, C, B]  -> per-(channel, band) stats over N
    """
    if feature == "raw":
        mean = x_train.mean(axis=(0, 2), keepdims=True)
        std = x_train.std(axis=(0, 2), keepdims=True)
    elif feature == "de":
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)
    else:
        raise ValueError(feature)
    std = np.clip(std, 1e-6, None)
    return Normalizer(mean=mean.astype(np.float32), std=std.astype(np.float32), feature=feature)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class EEGWindowDataset(Dataset):
    """Tensor-backed dataset of EEG windows.

    x : float32 tensor, [C, T] (raw) or [C, B] (de)
    y : long scalar in {0, 1}
    Also exposes `.subject_ids` and `.trial_ids` aligned with samples (handy for
    per-subject SHAP analysis and leakage checks).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subject_ids: Sequence[str],
        trial_ids: Sequence[int],
        normalizer: Optional[Normalizer] = None,
    ):
        if normalizer is not None:
            x = normalizer.apply(x)
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        self.subject_ids = list(subject_ids)
        self.trial_ids = list(trial_ids)

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
    x = arr.raw_windows if feature == "raw" else arr.de_features
    if channels is not None:
        idx = [arr.channel_names.index(c) for c in channels]
        x = x[:, idx, :]
    return x


def _gather_subject(
    processed_dir: str, sid: str, feature: str, label: str, channels: Optional[List[str]]
):
    a = load_subject(processed_dir, sid)
    x = _select_feature(a, feature, channels)
    y = a.labels[label]
    sids = [sid] * len(y)
    tids = a.trial_index.tolist()
    return x, y, sids, tids


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

    norm = fit_normalizer(x_tr, feature) if normalize else None

    dl_common = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        dl_common["persistent_workers"] = True

    val_loader = None
    if val_frac and val_frac > 0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y_tr))
        n_val = int(round(val_frac * len(y_tr)))
        v_idx, t_idx = idx[:n_val], idx[n_val:]
        ds_val = EEGWindowDataset(x_tr[v_idx], y_tr[v_idx], sid_tr[v_idx], tid_tr[v_idx], norm)
        ds_tr = EEGWindowDataset(x_tr[t_idx], y_tr[t_idx], sid_tr[t_idx], tid_tr[t_idx], norm)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, **dl_common)
    else:
        ds_tr = EEGWindowDataset(x_tr, y_tr, sid_tr, tid_tr, norm)

    ds_te = EEGWindowDataset(x_te, y_te, sid_te, tid_te, norm)

    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False,
                    generator=g, **dl_common)
    el = DataLoader(ds_te, batch_size=batch_size, shuffle=False, **dl_common)
    meta = {
        "protocol": "loso", "fold": fold, "test_subject": test_sid,
        "n_train_subjects": len(train_sids), "label": label, "feature": feature,
        "n_train": len(ds_tr), "n_test": len(ds_te),
        "input_shape": tuple(ds_tr.x.shape[1:]),
        # domain ids: integer per training subject, for MMD / domain-adaptation losses
        "train_domain_ids": {s: i for i, s in enumerate(train_sids)},
    }
    return tl, val_loader, el, meta


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
