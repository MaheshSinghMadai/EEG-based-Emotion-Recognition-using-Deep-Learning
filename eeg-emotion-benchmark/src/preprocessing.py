"""
preprocessing.py
================
Harmonized preprocessing for DEAP and DREAMER. Both datasets go through the SAME
`process_trials` function so that every downstream model sees identically prepared
inputs and cross-dataset transfer is meaningful.

Harmonization protocol (thesis methodology, "channel alignment & label harmonization")
--------------------------------------------------------------------------------------
1. Channels  : the 14 Emotiv EPOC electrodes, which are a subset of DEAP's 32-channel
               BioSemi montage. DEAP channels are selected by NAME from the official
               DEAP (Geneva) channel order; see `DEAP_CHANNEL_NAMES`.
2. Sampling  : 128 Hz (DEAP preprocessed release and DREAMER are both 128 Hz).
3. Baseline  : per-trial mean of the pre-stimulus baseline is subtracted
               (DEAP: first 3 s of each trial; DREAMER: the separate 61 s baseline).
4. Reference : common average reference over the 14 shared channels (DEAP ships
               CAR over 32 channels, DREAMER is mastoid-referenced; re-referencing
               both to the same 14-channel average removes that mismatch).
5. Filter    : 4-45 Hz zero-phase Butterworth. DEAP's preprocessed release is already
               band-limited to 4-45 Hz, so a delta band would be empty for DEAP but
               not for DREAMER. The harmonized preset therefore uses four bands
               (theta, alpha, beta, gamma<=45 Hz). `--band-preset classic` restores
               the original 0.3-50 Hz / five-band setting for within-dataset ablations.
6. Windows   : 4 s non-overlapping windows, never crossing a trial boundary.
7. Labels    : high iff rating is strictly above the instrument midpoint
               (DEAP 1-9 scale -> > 5; DREAMER 1-5 scale -> > 3).

Saved per window (one `.npz` per subject):
    raw_windows        [N, 14, 512]          filtered signal (DAH-Net raw ablation, CLISA)
    de_features        [N, 14, B]            differential entropy per band, 4 s
    de_sequence        [N, 14, 4, B]         DE per 1 s sub-window (STGATE temporal input)
    handcrafted        [N, F]                engineered feature vector (DAH-Net, see below)
    trial_index        [N]                   trial = stimulus id (same clip order for all
                                             subjects in both datasets)
    window_index       [N]                   position of the window inside its trial
    labels_{valence,arousal,dominance} [N]   binary labels
    ratings_{valence,arousal,dominance} [N]  raw ratings (for label-scheme sensitivity)

Run:
    python -m src.preprocessing --dataset DEAP
    python -m src.preprocessing --dataset DREAMER
Each run also writes the frozen evaluation split manifests via `src.splits`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis, skew

PREPROCESSING_VERSION = "harmonized-v2"

# --------------------------------------------------------------------------------------
# Acquisition constants
# --------------------------------------------------------------------------------------
EEG_FS = 128  # Hz, both datasets
N_CHANNELS = 14
DREAMER_N_SUBJECTS = 23

# Emotiv EPOC channel order (DREAMER storage order and the model input order).
CHANNEL_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
COMMON_CHANNELS = CHANNEL_NAMES

# Official channel order of the DEAP "data_preprocessed_python" release (Geneva order,
# DEAP website, "Channel layout" table). NOTE: earlier versions of this file listed a
# wrong order for indices 18-31 (Fz and Cz were missing), which silently replaced F4,
# F8, FC6, T8, P8 and O2 with neighbouring electrodes. The order below was verified
# empirically: each listed electrode correlates most strongly with its scalp neighbours
# (see tests/test_proposal_pipeline.py::test_deap_channel_indices).
DEAP_CHANNEL_NAMES = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]
DEAP_COMMON_CHANNEL_INDICES = [DEAP_CHANNEL_NAMES.index(name) for name in COMMON_CHANNELS]
DEAP_BASELINE_SEC = 3.0

BAND_PRESETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Harmonized (default): DEAP is already 4-45 Hz, so no delta and gamma capped at 45.
    "harmonized": {
        "theta": (4.0, 8.0),
        "alpha": (8.0, 14.0),
        "beta": (14.0, 31.0),
        "gamma": (31.0, 45.0),
    },
    # Original five-band setting (within-dataset ablation only).
    "classic": {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 14.0),
        "beta": (14.0, 31.0),
        "gamma": (31.0, 50.0),
    },
}
BROADBAND = {"harmonized": (4.0, 45.0), "classic": (0.3, 50.0)}

# Rating scale midpoints: label = 1 iff rating > midpoint.
LABEL_MIDPOINTS = {"DEAP": 5.0, "DREAMER": 3.0}
LABEL_NAMES = ("valence", "arousal", "dominance")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    dataset: str = "DREAMER"
    raw_path: str = "data/raw/DREAMER.mat"
    processed_dir: str = "data/processed/DREAMER"
    splits_dir: str = "data/splits/DREAMER"

    fs: int = EEG_FS
    window_sec: float = 4.0
    window_overlap: float = 0.0       # 0.0 => non-overlapping
    sub_window_sec: float = 1.0       # DE sequence resolution inside a window
    band_preset: str = "harmonized"
    bp_order: int = 4
    common_average_reference: bool = True

    seed: int = 42
    channel_names: List[str] = field(default_factory=lambda: list(COMMON_CHANNELS))

    def window_samples(self) -> int:
        return int(round(self.window_sec * self.fs))

    def hop_samples(self) -> int:
        step = self.window_sec * (1.0 - self.window_overlap)
        return max(1, int(round(step * self.fs)))

    @property
    def bands(self) -> Dict[str, Tuple[float, float]]:
        return BAND_PRESETS[self.band_preset]

    @property
    def broadband(self) -> Tuple[float, float]:
        return BROADBAND[self.band_preset]

    @property
    def label_midpoint(self) -> float:
        return LABEL_MIDPOINTS[self.dataset]


# --------------------------------------------------------------------------------------
# Loading DREAMER.mat
# --------------------------------------------------------------------------------------
def _load_mat(raw_path: str) -> Dict[str, Any]:
    """Load DREAMER.mat into {"subjects": [{eeg_baseline, eeg_stimuli, valence, ...}]}.

    Tries `mat73` first (v7.3 / HDF5), then `scipy.io.loadmat` for legacy exports.
    """
    try:
        import mat73  # type: ignore

        return _normalize_mat73(mat73.loadmat(raw_path))
    except Exception as exc_mat73:  # noqa: BLE001 - broad fallback is intended
        try:
            from scipy.io import loadmat

            raw = loadmat(raw_path, struct_as_record=False, squeeze_me=True)
            return _normalize_scipy(raw)
        except Exception as exc_scipy:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load DREAMER.mat with both mat73 and scipy.\n"
                f"  mat73 error : {exc_mat73!r}\n"
                f"  scipy error : {exc_scipy!r}"
            )


def _normalize_mat73(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = raw["DREAMER"]["Data"]
    if isinstance(data, dict):
        data = [data]
    subjects = []
    for subj in data:
        eeg = subj["EEG"]
        subjects.append({
            "age": subj.get("Age", None),
            "gender": subj.get("Gender", None),
            "eeg_baseline": _as_list_of_2d(eeg["baseline"]),
            "eeg_stimuli": _as_list_of_2d(eeg["stimuli"]),
            "valence": np.asarray(subj["ScoreValence"], dtype=float).reshape(-1),
            "arousal": np.asarray(subj["ScoreArousal"], dtype=float).reshape(-1),
            "dominance": np.asarray(subj["ScoreDominance"], dtype=float).reshape(-1),
        })
    return {"subjects": subjects}


def _normalize_scipy(raw: Dict[str, Any]) -> Dict[str, Any]:
    subjects = []
    for subj in np.atleast_1d(raw["DREAMER"].Data):
        eeg = subj.EEG
        subjects.append({
            "age": getattr(subj, "Age", None),
            "gender": getattr(subj, "Gender", None),
            "eeg_baseline": _as_list_of_2d(eeg.baseline),
            "eeg_stimuli": _as_list_of_2d(eeg.stimuli),
            "valence": np.asarray(subj.ScoreValence, dtype=float).reshape(-1),
            "arousal": np.asarray(subj.ScoreArousal, dtype=float).reshape(-1),
            "dominance": np.asarray(subj.ScoreDominance, dtype=float).reshape(-1),
        })
    return {"subjects": subjects}


def _as_list_of_2d(cell: Any) -> List[np.ndarray]:
    """Coerce a MATLAB cell array of trial matrices into a list of (samples, 14) arrays."""
    if isinstance(cell, np.ndarray) and cell.dtype != object and cell.ndim == 2:
        return [np.asarray(cell, dtype=float)]
    seq = list(cell) if not isinstance(cell, list) else cell
    out: List[np.ndarray] = []
    for trial in seq:
        arr = np.asarray(trial, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D trial matrix, got shape {arr.shape}")
        if arr.shape[1] != N_CHANNELS and arr.shape[0] == N_CHANNELS:
            arr = arr.T
        if arr.shape[1] != N_CHANNELS:
            raise ValueError(
                f"Trial has {arr.shape[1]} columns; expected {N_CHANNELS} EEG channels."
            )
        out.append(arr)
    return out


# --------------------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------------------
def _bandpass_sos(low: float, high: float, fs: int, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    return butter(order, [max(low / nyq, 1e-6), min(high / nyq, 0.999999)],
                  btype="bandpass", output="sos")


def bandpass_filter(x: np.ndarray, low: float, high: float, fs: int, order: int) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0 of a [samples, channels] array."""
    return sosfiltfilt(_bandpass_sos(low, high, fs, order), x, axis=0)


def baseline_correct(stimulus: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Subtract the per-channel baseline mean. Both arrays are [samples, channels]."""
    return stimulus - baseline.mean(axis=0, keepdims=True)


def common_average_reference(x: np.ndarray) -> np.ndarray:
    """Re-reference [samples, channels] to the mean over channels."""
    return x - x.mean(axis=1, keepdims=True)


def segment(signal: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Slice [samples, channels] -> [n_windows, channels, win]."""
    n = signal.shape[0]
    if n < win:
        return np.empty((0, signal.shape[1], win), dtype=signal.dtype)
    starts = range(0, n - win + 1, hop)
    return np.stack([signal[s:s + win, :].T for s in starts], axis=0)


def _de_from_variance(var: np.ndarray) -> np.ndarray:
    """Differential entropy of a Gaussian: 0.5 * ln(2*pi*e*var)."""
    return 0.5 * np.log(2.0 * np.pi * np.e * np.clip(var, 1e-10, None))


def differential_entropy(window: np.ndarray, fs: int, bands: Dict[str, Tuple[float, float]],
                         order: int = 4) -> np.ndarray:
    """DE per channel per band for a single [channels, samples] window (reference impl)."""
    de = np.zeros((window.shape[0], len(bands)), dtype=np.float32)
    for b, (lo, hi) in enumerate(bands.values()):
        xb = sosfiltfilt(_bandpass_sos(lo, hi, fs, order), window, axis=-1)
        de[:, b] = _de_from_variance(xb.var(axis=-1))
    return de


def handcrafted_features(windows: np.ndarray, band_windows: np.ndarray) -> np.ndarray:
    """Engineered per-window feature vector for DAH-Net.

    DAH-Net (2026) consumes a 988-d vector of statistical, frequency-domain,
    covariance/connectivity and eigenvalue features from an unspecified 3-class
    corpus. The same four categories are reproduced here for 14 channels:

        statistical    : mean, log-variance, skewness, kurtosis    4 x 14 = 56
        frequency      : log band power, relative band power       2 x B x 14
        connectivity   : upper-triangle Pearson correlations       91
        eigenvalue     : log eigenvalues of the correlation matrix 14

    windows      : [N, C, T] broadband windows
    band_windows : [N, B, C, T] band-filtered windows
    returns      : [N, F] float32, F = 161 + 28 * B  (273 for the 4-band preset)
    """
    n, c, _ = windows.shape
    stats = np.concatenate([
        windows.mean(axis=-1),
        np.log(np.clip(windows.var(axis=-1), 1e-10, None)),
        skew(windows, axis=-1),
        kurtosis(windows, axis=-1),
    ], axis=1)
    band_power = np.clip(band_windows.var(axis=-1), 1e-10, None)          # [N, B, C]
    log_power = np.log(band_power).reshape(n, -1)
    rel_power = (band_power / band_power.sum(axis=1, keepdims=True)).reshape(n, -1)
    iu = np.triu_indices(c, k=1)
    corr_feats, eig_feats = [], []
    for w in windows:
        r = np.nan_to_num(np.corrcoef(w), nan=0.0)
        corr_feats.append(r[iu])
        eig_feats.append(np.log(np.clip(np.linalg.eigvalsh(r), 1e-6, None)))
    return np.concatenate(
        [stats, log_power, rel_power, np.asarray(corr_feats), np.asarray(eig_feats)], axis=1
    ).astype(np.float32)


def process_trials(
    stimuli: Sequence[np.ndarray],
    baselines: Sequence[np.ndarray],
    ratings: Dict[str, np.ndarray],
    cfg: PreprocessConfig,
) -> Dict[str, np.ndarray]:
    """Shared pipeline for one subject. `stimuli`/`baselines` are [samples, 14] arrays.

    Every operation is confined to a single trial, so no information crosses trial
    (or subject) boundaries before the evaluation split is applied.
    """
    win, hop = cfg.window_samples(), cfg.hop_samples()
    sub = int(round(cfg.sub_window_sec * cfg.fs))
    if win % sub:
        raise ValueError("window_sec must be a multiple of sub_window_sec.")
    n_sub = win // sub
    lo, hi = cfg.broadband
    band_sos = [_bandpass_sos(b_lo, b_hi, cfg.fs, cfg.bp_order) for b_lo, b_hi in cfg.bands.values()]

    out: Dict[str, List[np.ndarray]] = {k: [] for k in (
        "raw_windows", "de_features", "de_sequence", "handcrafted", "trial_index", "window_index")}
    for trial_id, (stim, base) in enumerate(zip(stimuli, baselines)):
        x = baseline_correct(np.asarray(stim, float), np.asarray(base, float))
        if cfg.common_average_reference:
            x = common_average_reference(x)
        x = bandpass_filter(x, lo, hi, cfg.fs, cfg.bp_order)
        windows = segment(x, win, hop)                                    # [n, C, win]
        if windows.shape[0] == 0:
            continue
        # Filter each band on the whole trial once (avoids per-window edge effects).
        band_windows = np.stack(
            [segment(sosfiltfilt(sos, x, axis=0), win, hop) for sos in band_sos], axis=1
        )                                                                 # [n, B, C, win]
        de = _de_from_variance(band_windows.var(axis=-1)).transpose(0, 2, 1)   # [n, C, B]
        sub_var = band_windows.reshape(*band_windows.shape[:3], n_sub, sub).var(axis=-1)
        de_seq = _de_from_variance(sub_var).transpose(0, 2, 3, 1)          # [n, C, S, B]

        n = windows.shape[0]
        out["raw_windows"].append(windows.astype(np.float32))
        out["de_features"].append(de.astype(np.float32))
        out["de_sequence"].append(de_seq.astype(np.float32))
        out["handcrafted"].append(handcrafted_features(windows, band_windows))
        out["trial_index"].append(np.full(n, trial_id, dtype=np.int64))
        out["window_index"].append(np.arange(n, dtype=np.int64))

    if not out["raw_windows"]:
        raise ValueError("No complete windows were produced for this subject.")
    arrays = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    tidx = arrays["trial_index"]
    for name in LABEL_NAMES:
        scores = np.asarray(ratings[name], dtype=float).reshape(-1)
        arrays[f"ratings_{name}"] = scores[tidx].astype(np.float32)
        arrays[f"labels_{name}"] = (scores[tidx] > cfg.label_midpoint).astype(np.int64)
    arrays.update(
        channel_names=np.asarray(cfg.channel_names),
        band_names=np.asarray(list(cfg.bands)),
        fs=np.array(cfg.fs),
        window_samples=np.array(win),
        window_sec=np.array(cfg.window_sec),
    )
    return arrays


# --------------------------------------------------------------------------------------
# Dataset drivers
# --------------------------------------------------------------------------------------
def _save_subject(arrays: Dict[str, np.ndarray], subject_id: str, cfg: PreprocessConfig) -> None:
    arrays["subject_id"] = np.array(subject_id)
    path = os.path.join(cfg.processed_dir, f"{subject_id}.npz")
    np.savez_compressed(path, **arrays)
    print(f"[proc] {subject_id}: {arrays['raw_windows'].shape[0]} windows -> {path}")


def run_dreamer(cfg: PreprocessConfig) -> List[str]:
    print(f"[load] {cfg.raw_path}")
    subjects = _load_mat(cfg.raw_path)["subjects"]
    if len(subjects) != DREAMER_N_SUBJECTS:
        print(f"  WARNING: expected {DREAMER_N_SUBJECTS} subjects, got {len(subjects)}")
    ids = []
    for i, subj in enumerate(subjects, start=1):
        sid = f"subject_{i:02d}"
        arrays = process_trials(subj["eeg_stimuli"], subj["eeg_baseline"],
                                {k: subj[k] for k in LABEL_NAMES}, cfg)
        arrays["age"] = np.array(str(subj.get("age")))
        arrays["gender"] = np.array(str(subj.get("gender")))
        _save_subject(arrays, sid, cfg)
        ids.append(sid)
    return ids


def load_deap_subject(path: str) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, np.ndarray]]:
    """Return (stimuli, baselines, ratings) for one official DEAP preprocessed pickle."""
    with open(path, "rb") as f:
        payload = pickle.load(f, encoding="latin1")
    data = np.asarray(payload["data"], dtype=np.float64)
    scores = np.asarray(payload["labels"], dtype=np.float64)
    if data.ndim != 3 or data.shape[1] < len(DEAP_CHANNEL_NAMES):
        raise ValueError(f"{path}: expected [trials, >=32 channels, samples], got {data.shape}")
    if scores.shape != (data.shape[0], 4):
        raise ValueError(f"{path}: expected ratings of shape ({data.shape[0]}, 4), got {scores.shape}")
    n_base = int(round(DEAP_BASELINE_SEC * EEG_FS))
    eeg = data[:, DEAP_COMMON_CHANNEL_INDICES, :]
    stimuli = [trial[:, n_base:].T for trial in eeg]
    baselines = [trial[:, :n_base].T for trial in eeg]
    ratings = {"valence": scores[:, 0], "arousal": scores[:, 1], "dominance": scores[:, 2]}
    return stimuli, baselines, ratings


def run_deap(cfg: PreprocessConfig) -> List[str]:
    paths = sorted(glob.glob(os.path.join(cfg.raw_path, "s*.dat")))
    if not paths:
        raise FileNotFoundError(f"No DEAP subject files (s*.dat) found in {cfg.raw_path!r}.")
    ids = []
    for path in paths:
        sid = os.path.splitext(os.path.basename(path))[0].lower()
        stimuli, baselines, ratings = load_deap_subject(path)
        _save_subject(process_trials(stimuli, baselines, ratings, cfg), sid, cfg)
        ids.append(sid)
    return ids


def run(cfg: PreprocessConfig) -> None:
    os.makedirs(cfg.processed_dir, exist_ok=True)
    # Remove stale subject files so a re-run never mixes preprocessing versions.
    for stale in glob.glob(os.path.join(cfg.processed_dir, "*.npz")):
        os.remove(stale)
    ids = run_deap(cfg) if cfg.dataset == "DEAP" else run_dreamer(cfg)

    meta = asdict(cfg)
    meta.update(
        preprocessing_version=PREPROCESSING_VERSION,
        raw_path=os.path.abspath(cfg.raw_path),
        bands={k: list(v) for k, v in cfg.bands.items()},
        broadband=list(cfg.broadband),
        label_midpoint=cfg.label_midpoint,
        label_rule="label = 1 iff rating > midpoint",
        subjects=ids,
    )
    if cfg.dataset == "DEAP":
        meta["deap_channel_indices"] = DEAP_COMMON_CHANNEL_INDICES
    with open(os.path.join(cfg.processed_dir, "preprocess_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    from .splits import write_all_manifests  # local import avoids a cycle at import time
    write_all_manifests(cfg.dataset, cfg.processed_dir, cfg.splits_dir, seed=cfg.seed)
    print(f"[done] {cfg.dataset}: {len(ids)} subjects")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Harmonized DEAP / DREAMER preprocessing.")
    p.add_argument("--dataset", choices=["DEAP", "DREAMER"], default="DREAMER")
    p.add_argument("--raw", default=None, help="DREAMER .mat file or DEAP subject directory.")
    p.add_argument("--out", dest="processed_dir", default=None)
    p.add_argument("--splits", dest="splits_dir", default=None)
    p.add_argument("--window-sec", type=float, default=PreprocessConfig.window_sec)
    p.add_argument("--overlap", type=float, default=PreprocessConfig.window_overlap)
    p.add_argument("--band-preset", choices=list(BAND_PRESETS), default="harmonized")
    p.add_argument("--no-car", action="store_true", help="Disable common average reference.")
    p.add_argument("--seed", type=int, default=PreprocessConfig.seed)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    default_raw = "data/raw/DEAP" if args.dataset == "DEAP" else "data/raw/DREAMER.mat"
    run(PreprocessConfig(
        dataset=args.dataset,
        raw_path=args.raw or default_raw,
        processed_dir=args.processed_dir or os.path.join("data", "processed", args.dataset),
        splits_dir=args.splits_dir or os.path.join("data", "splits", args.dataset),
        window_sec=args.window_sec,
        window_overlap=args.overlap,
        band_preset=args.band_preset,
        common_average_reference=not args.no_car,
        seed=args.seed,
    ))
