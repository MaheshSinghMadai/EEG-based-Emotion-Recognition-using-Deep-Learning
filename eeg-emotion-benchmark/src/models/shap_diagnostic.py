"""
SHAP Diagnostic Study — multi-model comparison for explaining the SD/LOSO gap
==========================================================================
Not a single architecture: this module bundles the four comparators used in
notebook 05 (KNN, Random Forest, LDA, and a small CNN), all trained on the
flattened DE feature vector [14 x 5] -> 70 dims, plus helpers to pull SHAP values.

The point of the notebook is *diagnosis*: after training under both protocols we
compute SHAP values per model and inspect how much SHAP-value attributions vary
across subjects — large cross-subject variance in the important features is the
mechanism behind the subject-dependent vs subject-independent accuracy drop.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------------------
# Classic ML comparators (sklearn)
# --------------------------------------------------------------------------------------
def make_sklearn_models(seed: int = 42) -> Dict[str, Pipeline]:
    return {
        "KNN": Pipeline([("scale", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "RandomForest": Pipeline([("scale", StandardScaler()),
                                  ("clf", RandomForestClassifier(
                                      n_estimators=300, max_depth=None,
                                      n_jobs=-1, random_state=seed))]),
        "LDA": Pipeline([("scale", StandardScaler()),
                         ("clf", LinearDiscriminantAnalysis())]),
    }


# --------------------------------------------------------------------------------------
# Small CNN comparator (PyTorch)
# --------------------------------------------------------------------------------------
class SimpleDECNN(nn.Module):
    """Tiny 2-D CNN over the [14 x 5] DE matrix (treated as a 1-channel image)."""

    def __init__(self, n_channels: int = 14, n_bands: int = 5, n_classes: int = 2,
                 width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(width * 2 * 4 * 2, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:               # [B, C, Bands] -> [B, 1, C, Bands]
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))


def build(cfg: dict) -> SimpleDECNN:
    meta = cfg.get("meta", {})
    shape = meta.get("input_shape", (14, 5))
    return SimpleDECNN(
        n_channels=shape[0], n_bands=shape[1],
        n_classes=cfg.get("n_classes", 2),
        width=cfg.get("width", 32), dropout=cfg.get("dropout", 0.3),
    )


# --------------------------------------------------------------------------------------
# Helpers: loader <-> flat numpy arrays (for sklearn + SHAP KernelExplainer)
# --------------------------------------------------------------------------------------
DE_FEATURE_NAMES = [f"{ch}-{band}"
                    for ch in ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                               "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
                    for band in ["delta", "theta", "alpha", "beta", "gamma"]]


def loader_to_xy(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate a loader into (X_flat [N,70], y [N], subject_ids [N])."""
    xs, ys = [], []
    ds = loader.dataset
    for xb, yb in loader:
        xs.append(xb.reshape(xb.shape[0], -1).numpy())
        ys.append(yb.numpy())
    sids = np.array(getattr(ds, "subject_ids", ["?"] * sum(len(y) for y in ys)))
    return np.concatenate(xs), np.concatenate(ys), sids


def torch_predict_proba(model: nn.Module, device: torch.device):
    """Return a `f(X_flat_np) -> proba_np` callable for SHAP's KernelExplainer."""
    model.eval()

    def f(x_flat: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(x_flat, dtype=torch.float32, device=device)
            t = t.reshape(t.shape[0], 14, 5)
            logits = model(t)
            return torch.softmax(logits, dim=1).cpu().numpy()

    return f


def subject_shap_variance(shap_values: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """Per-feature variance of the mean |SHAP| across subjects.

    shap_values : [N, n_features]  (use the positive-class SHAP matrix)
    returns     : [n_features]     — higher => that feature's importance is subject-specific,
                  i.e. a driver of the subject-dependent / independent generalization gap.
    """
    subs = np.unique(subject_ids)
    per_subj_mean = np.stack([
        np.abs(shap_values[subject_ids == s]).mean(axis=0) for s in subs
    ], axis=0)                                   # [n_subjects, n_features]
    return per_subj_mean.var(axis=0)
