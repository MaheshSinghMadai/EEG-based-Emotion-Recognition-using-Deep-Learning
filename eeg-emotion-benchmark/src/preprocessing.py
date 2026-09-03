"""
preprocessing.py
================
End-to-end preprocessing for the DREAMER dataset, producing per-subject
`.npz` files plus the split-index files used by every benchmark notebook.

Pipeline (see the thesis methodology chapter):
    1. Load DREAMER.mat (v7.3/HDF5 via `mat73`, or legacy via `scipy.io.loadmat`).
    2. Baseline correction  : stimulus[t, c] -= mean_t(baseline[:, c])
    3. Band-pass filter      : 0.3-50 Hz, Butterworth 4th order, zero-phase.
    4. Segmentation          : non-overlapping windows (default 4 s = 512 samples).
    5. Feature extraction     : Differential Entropy (DE) per channel per band
                               -> [14 channels x 5 bands] matrix per window.
                               Raw filtered windows [14 x win_samples] are also saved.
    6. Labeling              : binarize valence / arousal / dominance (2-class).
    7. Save                  : one `data/processed/subject_XX.npz` per subject.
    8. Splits               : `data/splits/subject_dependent_splits.json`
                               `data/splits/loso_splits.json`

Run:
    python -m src.preprocessing --raw data/raw/DREAMER.mat --out data/processed \
        --splits data/splits --window-sec 4 --label-threshold 3.0

All field-name assumptions are flagged with `# ASSUMPTION:` — verify against your
actual DREAMER.mat export (MATLAB versions differ slightly) before trusting output.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

# --------------------------------------------------------------------------------------
# Constants describing the DREAMER acquisition setup (from the dataset paper).
# --------------------------------------------------------------------------------------
EEG_FS = 128  # Hz, EEG sampling rate
N_CHANNELS = 14
N_TRIALS = 18
N_SUBJECTS = 23

# Emotiv EPOC channel order as documented in the task brief.
CHANNEL_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# 5 standard EEG bands (Hz). Names kept in a fixed order everywhere downstream.
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 31.0),
    "gamma": (31.0, 50.0),
}
BAND_NAMES = list(BANDS.keys())
N_BANDS = len(BAND_NAMES)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    raw_path: str = "data/raw/DREAMER.mat"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"

    fs: int = EEG_FS
    window_sec: float = 4.0            # segmentation window length (configurable)
    window_overlap: float = 0.0        # 0.0 => non-overlapping (default per brief)

    # Broadband band-pass applied before segmentation.
    bp_low: float = 0.3
    bp_high: float = 50.0
    bp_order: int = 4

    # Labeling.
    # label_strategy: "threshold" -> high if score > label_threshold
    #                 "median"    -> high if score > median(scores in `median_scope`)
    label_strategy: str = "threshold"
    label_threshold: float = 3.0
    median_scope: str = "subject"     # "subject" or "global" (only for median strategy)

    # Subject-dependent split.
    # sd_scheme: "holdout" -> single train/test split of size (1 - sd_test_size)
    #            "kfold"   -> sd_kfolds folds
    sd_scheme: str = "holdout"
    sd_test_size: float = 0.2
    sd_kfolds: int = 5
    # split_by: "trial"  -> whole film clips go to train OR test (no window leakage)
    #           "window" -> windows shuffled individually (leaky; provided for ablation)
    split_by: str = "trial"

    seed: int = 42

    # Which label the *default* comparison uses. All three are always saved; this
    # only annotates the split files so notebooks can pick a consistent target.
    default_label: str = "valence"

    def window_samples(self) -> int:
        return int(round(self.window_sec * self.fs))

    def hop_samples(self) -> int:
        step = self.window_sec * (1.0 - self.window_overlap)
        return max(1, int(round(step * self.fs)))


# --------------------------------------------------------------------------------------
# Loading DREAMER.mat
# --------------------------------------------------------------------------------------
def _load_mat(raw_path: str) -> Dict[str, Any]:
    """
    Load DREAMER.mat and return a *normalized* python structure:

        {
          "subjects": [
             {
               "age": <str/num>, "gender": <str>,
               "eeg_baseline": [np.ndarray(bs_samples, 14), ... x18],
               "eeg_stimuli":  [np.ndarray(st_samples, 14), ... x18],
               "valence": np.ndarray(18,), "arousal": np.ndarray(18,), "dominance": np.ndarray(18,),
             },
             ... x23
          ]
        }

    Tries `mat73` first (DREAMER is commonly distributed as MATLAB v7.3 / HDF5),
    then falls back to `scipy.io.loadmat` for legacy exports.
    """
    try:
        import mat73  # type: ignore

        raw = mat73.loadmat(raw_path)
        return _normalize_mat73(raw)
    except Exception as exc_mat73:  # noqa: BLE001 - we want a broad fallback
        try:
            from scipy.io import loadmat

            raw = loadmat(raw_path, struct_as_record=False, squeeze_me=True)
            return _normalize_scipy(raw)
        except Exception as exc_scipy:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load DREAMER.mat with both mat73 and scipy.\n"
                f"  mat73 error : {exc_mat73!r}\n"
                f"  scipy error : {exc_scipy!r}\n"
                "If the file is v7.3, `pip install mat73`. Otherwise verify the path."
            )


def _normalize_mat73(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a `mat73.loadmat` result.

    mat73 returns nested dicts/lists. Expected shape:
        raw["DREAMER"]["Data"]      -> list of 23 dicts
        subj["EEG"]["baseline"]     -> list of 18 arrays (samples x 14)
        subj["EEG"]["stimuli"]      -> list of 18 arrays (samples x 14)
        subj["ScoreValence"]        -> array-like of 18
    """
    # ASSUMPTION: top-level MATLAB struct variable is named "DREAMER".
    dreamer = raw["DREAMER"]
    data = dreamer["Data"]

    # `Data` may come back as a list, or as a 1-element list wrapping the list.
    if isinstance(data, dict):
        data = [data]
    subjects = []
    for subj in data:
        eeg = subj["EEG"]
        # ASSUMPTION: EEG sub-struct has "baseline" and "stimuli" cell arrays.
        baseline = _as_list_of_2d(eeg["baseline"])
        stimuli = _as_list_of_2d(eeg["stimuli"])
        subjects.append(
            {
                "age": subj.get("Age", None),
                "gender": subj.get("Gender", None),
                "eeg_baseline": baseline,
                "eeg_stimuli": stimuli,
                # ASSUMPTION: score field names are ScoreValence / ScoreArousal / ScoreDominance.
                "valence": np.asarray(subj["ScoreValence"], dtype=float).reshape(-1),
                "arousal": np.asarray(subj["ScoreArousal"], dtype=float).reshape(-1),
                "dominance": np.asarray(subj["ScoreDominance"], dtype=float).reshape(-1),
            }
        )
    return {"subjects": subjects}


def _normalize_scipy(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a `scipy.io.loadmat(struct_as_record=False, squeeze_me=True)` result."""
    # ASSUMPTION: top-level MATLAB struct variable is named "DREAMER".
    dreamer = raw["DREAMER"]
    data = dreamer.Data  # object array of length 23 (mat_struct entries)

    subjects = []
    for subj in np.atleast_1d(data):
        eeg = subj.EEG
        baseline = _as_list_of_2d(eeg.baseline)
        stimuli = _as_list_of_2d(eeg.stimuli)
        subjects.append(
            {
                "age": getattr(subj, "Age", None),
                "gender": getattr(subj, "Gender", None),
                "eeg_baseline": baseline,
                "eeg_stimuli": stimuli,
                "valence": np.asarray(subj.ScoreValence, dtype=float).reshape(-1),
                "arousal": np.asarray(subj.ScoreArousal, dtype=float).reshape(-1),
                "dominance": np.asarray(subj.ScoreDominance, dtype=float).reshape(-1),
            }
        )
    return {"subjects": subjects}


def _as_list_of_2d(cell: Any) -> List[np.ndarray]:
    """Coerce a MATLAB cell array of trial matrices into a list of (samples, 14) arrays."""
    # mat73 -> list; scipy -> object ndarray; single trial -> bare 2D array.
    if isinstance(cell, np.ndarray) and cell.dtype != object and cell.ndim == 2:
        return [np.asarray(cell, dtype=float)]
    seq = list(cell) if not isinstance(cell, list) else cell
    out: List[np.ndarray] = []
    for trial in seq:
        arr = np.asarray(trial, dtype=float)
        # ASSUMPTION: trials are stored as [samples x channels]. If your export is
        # transposed ([channels x samples]) this will be caught by the shape guard below.
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D trial matrix, got shape {arr.shape}")
        if arr.shape[1] != N_CHANNELS and arr.shape[0] == N_CHANNELS:
            arr = arr.T  # auto-correct a transposed trial
        if arr.shape[1] != N_CHANNELS:
            raise ValueError(
                f"Trial has {arr.shape[1]} columns; expected {N_CHANNELS} EEG channels. "
                "Check channel axis / field mapping."
            )
        out.append(arr)
    return out


# --------------------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------------------
def _bandpass_sos(low: float, high: float, fs: int, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    lo = max(low / nyq, 1e-6)
    hi = min(high / nyq, 0.999999)
    return butter(order, [lo, hi], btype="bandpass", output="sos")


def bandpass_filter(x: np.ndarray, low: float, high: float, fs: int, order: int) -> np.ndarray:
    """Zero-phase Butterworth band-pass. `x` is [samples, channels]; filter along time."""
    sos = _bandpass_sos(low, high, fs, order)
    # sosfiltfilt operates along the last axis by default -> transpose so time is last.
    return sosfiltfilt(sos, x, axis=0)


def baseline_correct(stimulus: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Subtract the per-channel mean of the baseline recording from the stimulus.

    Removes subject-specific DC offset / resting-state bias (standard for DREAMER).
    `stimulus` : [st_samples, 14]  `baseline` : [bs_samples, 14]
    """
    # ASSUMPTION: baseline-correction is done in the *time domain* on the raw signal,
    # as described in the task brief. An alternative common in the literature is to
    # subtract baseline DE from stimulus DE (feature domain); if you prefer that,
    # set the label `de_baseline_correct=True` path below and disable this call.
    return stimulus - baseline.mean(axis=0, keepdims=True)


def segment(signal: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Slice [samples, channels] -> [n_windows, channels, win] (non-overlapping by default)."""
    n = signal.shape[0]
    if n < win:
        return np.empty((0, signal.shape[1], win), dtype=signal.dtype)
    starts = list(range(0, n - win + 1, hop))
    out = np.stack([signal[s : s + win, :].T for s in starts], axis=0)
    return out  # [n_windows, channels, win]


def differential_entropy(window: np.ndarray, fs: int, order: int = 4) -> np.ndarray:
    """Differential Entropy per channel per band for one window.

    `window` : [channels, win_samples]
    returns  : [channels, n_bands]

    For a signal that is approximately Gaussian within a narrow band, the DE reduces
    to a closed form of the band-power:  DE = 0.5 * ln(2*pi*e*var(x_band)).
    This is the definition used by Duan et al. 2013 / Zheng & Lu 2015 (SEED) and by
    the DGCNN / CIT-EmotionNet family this benchmark compares against.
    """
    n_ch = window.shape[0]
    de = np.zeros((n_ch, N_BANDS), dtype=np.float32)
    x = window  # [channels, win]
    for b, (lo, hi) in enumerate(BANDS.values()):
        sos = _bandpass_sos(lo, hi, fs, order)
        # filter along time (last axis) for all channels at once
        xb = sosfiltfilt(sos, x, axis=-1)
        var = xb.var(axis=-1)
        var = np.clip(var, 1e-10, None)  # guard log(0)
        de[:, b] = 0.5 * np.log(2.0 * np.pi * np.e * var)
    return de


# --------------------------------------------------------------------------------------
# Labeling
# --------------------------------------------------------------------------------------
def binarize_scores(
    scores: np.ndarray, strategy: str, threshold: float, median_ref: float | None
) -> np.ndarray:
    """Return a 0/1 array: 1 = "high", 0 = "low".

    strategy == "threshold": high if score >  threshold  (DREAMER paper convention, thr=3)
    strategy == "median"   : high if score >  median_ref
    """
    if strategy == "threshold":
        return (scores > threshold).astype(np.int64)
    if strategy == "median":
        assert median_ref is not None
        return (scores > median_ref).astype(np.int64)
    raise ValueError(f"Unknown label strategy: {strategy}")


# --------------------------------------------------------------------------------------
# Per-subject processing
# --------------------------------------------------------------------------------------
def process_subject(subj: Dict[str, Any], cfg: PreprocessConfig, global_medians: Dict[str, float]
                    ) -> Dict[str, np.ndarray]:
    """Run the full pipeline for one subject and return arrays ready for np.savez."""
    win = cfg.window_samples()
    hop = cfg.hop_samples()

    raw_windows: List[np.ndarray] = []   # each [n_win, 14, win]
    de_windows: List[np.ndarray] = []    # each [n_win, 14, 5]
    trial_index: List[np.ndarray] = []   # each [n_win]

    n_trials = len(subj["eeg_stimuli"])
    for t in range(n_trials):
        stim = subj["eeg_stimuli"][t]
        base = subj["eeg_baseline"][t]

        # 2. baseline correction (time domain)
        stim_bc = baseline_correct(stim, base)

        # 3. broadband band-pass 0.3-50 Hz
        stim_f = bandpass_filter(stim_bc, cfg.bp_low, cfg.bp_high, cfg.fs, cfg.bp_order)

        # 4. segmentation
        wins = segment(np.asarray(stim_f), win, hop)  # [n_win, 14, win]
        if wins.shape[0] == 0:
            continue

        # 5. DE features per window
        de = np.stack([differential_entropy(w, cfg.fs) for w in wins], axis=0)  # [n_win,14,5]

        raw_windows.append(wins.astype(np.float32))
        de_windows.append(de.astype(np.float32))
        trial_index.append(np.full(wins.shape[0], t, dtype=np.int64))

    raw_arr = np.concatenate(raw_windows, axis=0)
    de_arr = np.concatenate(de_windows, axis=0)
    tidx = np.concatenate(trial_index, axis=0)

    # 6. labels — expand per-trial rating to per-window
    labels = {}
    for name in ("valence", "arousal", "dominance"):
        scores = subj[name]  # [18]
        if cfg.label_strategy == "median":
            ref = float(np.median(scores)) if cfg.median_scope == "subject" else global_medians[name]
        else:
            ref = None
        trial_labels = binarize_scores(scores, cfg.label_strategy, cfg.label_threshold, ref)
        labels[name] = trial_labels[tidx].astype(np.int64)

    return {
        "raw_windows": raw_arr,               # [N, 14, win_samples]
        "de_features": de_arr,                # [N, 14, 5]
        "labels_valence": labels["valence"],  # [N]
        "labels_arousal": labels["arousal"],  # [N]
        "labels_dominance": labels["dominance"],  # [N]
        "trial_index": tidx,                  # [N]  (0..17)
        "channel_names": np.array(CHANNEL_NAMES),
        "band_names": np.array(BAND_NAMES),
        "fs": np.array(cfg.fs),
        "window_samples": np.array(win),
        "window_sec": np.array(cfg.window_sec),
    }


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
def _kfold_indices(items: np.ndarray, k: int, rng: np.random.Generator) -> List[np.ndarray]:
    perm = rng.permutation(len(items))
    return [items[perm[i::k]] for i in range(k)]  # simple round-robin folds


def make_subject_dependent_splits(
    per_subject: Dict[str, Dict[str, np.ndarray]], cfg: PreprocessConfig
) -> Dict[str, Any]:
    """Build per-subject train/test index lists.

    Indices refer to rows of that subject's processed arrays. When `split_by == "trial"`
    entire film clips are assigned to a side (prevents windows of the same clip leaking
    across the train/test boundary — recommended for an honest subject-dependent number).
    """
    rng = np.random.default_rng(cfg.seed)
    out: Dict[str, Any] = {
        "protocol": "subject_dependent",
        "scheme": cfg.sd_scheme,
        "split_by": cfg.split_by,
        "default_label": cfg.default_label,
        "subjects": {},
    }

    for sid, data in per_subject.items():
        tidx = data["trial_index"]
        n = len(tidx)
        all_rows = np.arange(n)

        if cfg.split_by == "trial":
            units = np.unique(tidx)
        else:
            units = all_rows

        if cfg.sd_scheme == "kfold":
            fold_units = _kfold_indices(units, cfg.sd_kfolds, rng)
            folds = []
            for i in range(cfg.sd_kfolds):
                test_u = set(fold_units[i].tolist())
                if cfg.split_by == "trial":
                    test_rows = all_rows[np.isin(tidx, list(test_u))]
                    train_rows = all_rows[~np.isin(tidx, list(test_u))]
                else:
                    test_rows = np.array(sorted(test_u), dtype=int)
                    train_rows = np.array([r for r in all_rows if r not in test_u], dtype=int)
                folds.append(
                    {"train": train_rows.tolist(), "test": test_rows.tolist()}
                )
            out["subjects"][sid] = {"folds": folds}
        else:  # holdout
            perm = rng.permutation(len(units))
            n_test = max(1, int(round(cfg.sd_test_size * len(units))))
            test_u = set(units[perm[:n_test]].tolist())
            if cfg.split_by == "trial":
                test_rows = all_rows[np.isin(tidx, list(test_u))]
                train_rows = all_rows[~np.isin(tidx, list(test_u))]
            else:
                test_rows = np.array(sorted(test_u), dtype=int)
                train_rows = np.array([r for r in all_rows if r not in test_u], dtype=int)
            out["subjects"][sid] = {
                "folds": [{"train": train_rows.tolist(), "test": test_rows.tolist()}]
            }

    return out


def make_loso_splits(subject_ids: List[str], cfg: PreprocessConfig) -> Dict[str, Any]:
    """23-fold leave-one-subject-out. Fold k tests on subject k, trains on the other 22."""
    folds = []
    for k, sid in enumerate(subject_ids):
        folds.append(
            {
                "fold": k,
                "test_subject": sid,
                "train_subjects": [s for s in subject_ids if s != sid],
            }
        )
    return {
        "protocol": "loso",
        "n_folds": len(folds),
        "default_label": cfg.default_label,
        "folds": folds,
    }


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run(cfg: PreprocessConfig) -> None:
    os.makedirs(cfg.processed_dir, exist_ok=True)
    os.makedirs(cfg.splits_dir, exist_ok=True)

    print(f"[load] {cfg.raw_path}")
    mat = _load_mat(cfg.raw_path)
    subjects = mat["subjects"]
    print(f"[load] {len(subjects)} subjects")
    if len(subjects) != N_SUBJECTS:
        print(f"  WARNING: expected {N_SUBJECTS} subjects, got {len(subjects)}")

    # Global medians (only used when label_strategy='median' and median_scope='global').
    global_medians = {}
    for name in ("valence", "arousal", "dominance"):
        allscores = np.concatenate([np.asarray(s[name]).reshape(-1) for s in subjects])
        global_medians[name] = float(np.median(allscores))
    print(f"[label] global medians: {global_medians}")

    per_subject: Dict[str, Dict[str, np.ndarray]] = {}
    subject_ids: List[str] = []
    for i, subj in enumerate(subjects, start=1):
        sid = f"subject_{i:02d}"
        subject_ids.append(sid)
        print(f"[proc] {sid} ...", end="", flush=True)
        arrays = process_subject(subj, cfg, global_medians)
        arrays["subject_id"] = np.array(sid)
        arrays["age"] = np.array(str(subj.get("age")))
        arrays["gender"] = np.array(str(subj.get("gender")))
        out_path = os.path.join(cfg.processed_dir, f"{sid}.npz")
        np.savez_compressed(out_path, **arrays)
        per_subject[sid] = {
            "trial_index": arrays["trial_index"],
        }
        print(f" {arrays['raw_windows'].shape[0]} windows -> {out_path}")

    # Splits
    sd = make_subject_dependent_splits(per_subject, cfg)
    sd_path = os.path.join(cfg.splits_dir, "subject_dependent_splits.json")
    with open(sd_path, "w") as f:
        json.dump(sd, f, indent=2)
    print(f"[split] {sd_path}")

    loso = make_loso_splits(subject_ids, cfg)
    loso_path = os.path.join(cfg.splits_dir, "loso_splits.json")
    with open(loso_path, "w") as f:
        json.dump(loso, f, indent=2)
    print(f"[split] {loso_path}")

    # Persist the exact config used (reproducibility for the thesis appendix).
    with open(os.path.join(cfg.processed_dir, "preprocess_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print("[done]")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess DREAMER.mat for the benchmark.")
    p.add_argument("--raw", dest="raw_path", default=PreprocessConfig.raw_path)
    p.add_argument("--out", dest="processed_dir", default=PreprocessConfig.processed_dir)
    p.add_argument("--splits", dest="splits_dir", default=PreprocessConfig.splits_dir)
    p.add_argument("--window-sec", dest="window_sec", type=float, default=PreprocessConfig.window_sec)
    p.add_argument("--overlap", dest="window_overlap", type=float, default=PreprocessConfig.window_overlap)
    p.add_argument("--label-strategy", dest="label_strategy",
                   choices=["threshold", "median"], default=PreprocessConfig.label_strategy)
    p.add_argument("--label-threshold", dest="label_threshold", type=float,
                   default=PreprocessConfig.label_threshold)
    p.add_argument("--median-scope", dest="median_scope",
                   choices=["subject", "global"], default=PreprocessConfig.median_scope)
    p.add_argument("--sd-scheme", dest="sd_scheme",
                   choices=["holdout", "kfold"], default=PreprocessConfig.sd_scheme)
    p.add_argument("--sd-test-size", dest="sd_test_size", type=float, default=PreprocessConfig.sd_test_size)
    p.add_argument("--sd-kfolds", dest="sd_kfolds", type=int, default=PreprocessConfig.sd_kfolds)
    p.add_argument("--split-by", dest="split_by",
                   choices=["trial", "window"], default=PreprocessConfig.split_by)
    p.add_argument("--default-label", dest="default_label",
                   choices=["valence", "arousal", "dominance"], default=PreprocessConfig.default_label)
    p.add_argument("--seed", dest="seed", type=int, default=PreprocessConfig.seed)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(PreprocessConfig(**vars(args)))
