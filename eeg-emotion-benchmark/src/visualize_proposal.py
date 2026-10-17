"""Create plots from proposal benchmark comparison CSV files.

Run from the repository root:
    python -m src.visualize_proposal --label valence
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COMPARISON_ROOT = ROOT / "results" / "proposal"
OUTPUT_ROOT = COMPARISON_ROOT / "visualizations"
REQUIRED_COLUMNS = {
    "rank",
    "model",
    "accuracy",
    "accuracy_std",
    "f1",
    "f1_std",
    "run_status",
}
SCENARIO_ORDER = tuple(
    f"{d}/{p}" for d in ("DEAP", "DREAMER")
    for p in ("loso", "subject_kfold", "trial_kfold", "random_window")
) + ("DEAP_to_DREAMER", "DREAMER_to_DEAP")
MODEL_COLORS = {
    "DADPc": "#4C78A8",
    "CIT-EmotionNet": "#F58518",
    "STGATE": "#54A24B",
    "DAH-Net": "#E45756",
    "CLISA": "#B279A2",
}


def load_comparisons(label: str) -> Dict[str, List[Dict[str, str]]]:
    """Load and validate every comparison CSV matching the requested label."""
    pattern = f"comparison_{label}.csv"
    paths = sorted(p for p in COMPARISON_ROOT.rglob(pattern)
                   if "visualizations" not in p.parts and "analysis" not in p.parts)
    if not paths:
        raise FileNotFoundError(
            f"No {pattern} files found under {COMPARISON_ROOT}. "
            "Run the benchmark first."
        )
    comparisons: Dict[str, List[Dict[str, str]]] = {}
    for path in paths:
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing comparison columns: {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"Comparison file is empty: {path}")
        comparisons[path.parent.relative_to(COMPARISON_ROOT).as_posix()] = rows
        print(f"[load] {path} ({len(rows)} models)")
    ordered = sorted(
        comparisons,
        key=lambda name: (SCENARIO_ORDER.index(name) if name in SCENARIO_ORDER else 99, name),
    )
    return {name: comparisons[name] for name in ordered}


def _metric(row: Dict[str, str], name: str) -> float:
    return float(row[name])


def _std(row: Dict[str, str], name: str) -> float:
    return max(0.0, float(row[name]))


def _models(comparisons: Dict[str, List[Dict[str, str]]]) -> List[str]:
    return list(dict.fromkeys(
        row["model"]
        for rows in comparisons.values()
        for row in rows
    ))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def plot_scenario_rankings(
    comparisons: Dict[str, List[Dict[str, str]]], output_dir: Path, label: str
) -> None:
    """One accuracy and macro-F1 ranking panel for each available comparison."""
    fig, axes = plt.subplots(
        len(comparisons), 2,
        figsize=(14, max(4.2, 3.7 * len(comparisons))),
        squeeze=False,
    )
    fig.suptitle(
        f"{label.title()} model comparisons by scenario",
        fontsize=17,
        fontweight="bold",
    )
    for row_index, (scenario, rows) in enumerate(comparisons.items()):
        for col_index, (metric, std_key, title) in enumerate((
            ("accuracy", "accuracy_std", "Accuracy"),
            ("f1", "f1_std", "Macro F1"),
        )):
            axis = axes[row_index, col_index]
            ordered = sorted(rows, key=lambda row: _metric(row, metric))
            names = [row["model"] for row in ordered]
            values = [_metric(row, metric) for row in ordered]
            errors = [_std(row, std_key) for row in ordered]
            y_pos = np.arange(len(ordered))
            axis.barh(
                y_pos,
                values,
                xerr=errors,
                color=[MODEL_COLORS.get(name, "#888888") for name in names],
                capsize=3,
                alpha=0.9,
            )
            axis.set_yticks(y_pos, names)
            axis.set_xlim(0, 1)
            axis.set_xlabel(f"{title} (mean; error bars = reported std)")
            scenario_title = (
                scenario.replace("_to_", " → ")
                if "_to_" in scenario
                else f"{scenario} (within)"
            )
            axis.set_title(f"{scenario_title} · {title}")
            axis.grid(axis="x", alpha=0.25)
            for y, value in zip(y_pos, values):
                axis.text(min(value + 0.012, 0.90), y, f"{value:.3f}", va="center", fontsize=8)
    fig.text(
        0.5, 0.006,
        "All inputs currently mark smoke/partial runs. These rankings are diagnostic only.",
        ha="center", color="#9b2c2c", fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    _save(fig, output_dir / f"{label}_scenario_rankings.png")


def plot_metric_heatmaps(
    comparisons: Dict[str, List[Dict[str, str]]], output_dir: Path, label: str
) -> None:
    """Compare accuracy and macro F1 for every model across all scenarios."""
    model_names = _models(comparisons)
    scenario_names = list(comparisons)
    matrices = {}
    for metric in ("accuracy", "f1"):
        matrix = np.full((len(model_names), len(scenario_names)), np.nan)
        for col, scenario in enumerate(scenario_names):
            by_model = {row["model"]: row for row in comparisons[scenario]}
            for row, model_name in enumerate(model_names):
                if model_name in by_model:
                    matrix[row, col] = _metric(by_model[model_name], metric)
        matrices[metric] = matrix

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.2, 0.7 * len(model_names))))
    fig.suptitle(f"{label.title()} performance across scenarios", fontsize=16, fontweight="bold")
    for axis, metric, title in zip(
        axes, ("accuracy", "f1"), ("Accuracy", "Macro F1")
    ):
        image = axis.imshow(matrices[metric], vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        axis.set_xticks(
            np.arange(len(scenario_names)),
            [name.replace("_", "\n→ ") for name in scenario_names],
        )
        axis.set_yticks(np.arange(len(model_names)), model_names)
        axis.set_title(title)
        for row in range(len(model_names)):
            for col in range(len(scenario_names)):
                value = matrices[metric][row, col]
                if np.isfinite(value):
                    axis.text(
                        col, row, f"{value:.3f}",
                        ha="center", va="center",
                        color="white" if value > 0.62 else "black",
                        fontsize=9,
                    )
        fig.colorbar(image, ax=axis, shrink=0.82, label=title)
    fig.text(
        0.5, 0.015,
        "Smoke/partial results only; within-dataset cells currently use one held-out fold.",
        ha="center", color="#9b2c2c", fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    _save(fig, output_dir / f"{label}_scenario_heatmaps.png")


def plot_transfer_directions(
    comparisons: Dict[str, List[Dict[str, str]]], output_dir: Path, label: str
) -> None:
    """Compare forward/reverse results for every available dataset pair."""
    pairs: Dict[tuple[str, str], List[str]] = {}
    for scenario in comparisons:
        if "_to_" not in scenario:
            continue
        source, target = scenario.split("_to_", maxsplit=1)
        pair = tuple(sorted((source, target)))
        pairs.setdefault(pair, []).append(scenario)
    complete_pairs = {
        pair: sorted(
            directions,
            key=lambda name: (SCENARIO_ORDER.index(name) if name in SCENARIO_ORDER else 99, name),
        )
        for pair, directions in pairs.items()
        if len(directions) == 2
    }
    if not complete_pairs:
        print("[skip] no dataset pair has both transfer directions")
        return

    fig, axes = plt.subplots(
        len(complete_pairs), 2,
        figsize=(13, max(5.5, 4.4 * len(complete_pairs))),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(
        f"{label.title()} cross-dataset transfer by direction",
        fontsize=16,
        fontweight="bold",
    )
    for row_index, (pair, directions) in enumerate(complete_pairs.items()):
        names_per_direction = [
            {item["model"] for item in comparisons[direction]}
            for direction in directions
        ]
        model_names = sorted(set.intersection(*names_per_direction))
        direction_labels = [
            f"{direction.split('_to_')[0]} → {direction.split('_to_')[1]}"
            for direction in directions
        ]
        for col_index, (metric, std_key, title) in enumerate((
            ("accuracy", "accuracy_std", "Accuracy"),
            ("f1", "f1_std", "Macro F1"),
        )):
            axis = axes[row_index, col_index]
            for model_name in model_names:
                values = []
                errors = []
                for direction in directions:
                    item = next(
                        row for row in comparisons[direction]
                        if row["model"] == model_name
                    )
                    values.append(_metric(item, metric))
                    errors.append(_std(item, std_key))
                color = MODEL_COLORS.get(model_name, "#888888")
                axis.plot(
                    range(len(directions)),
                    values,
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=model_name,
                )
                axis.errorbar(
                    range(len(directions)), values, yerr=errors, fmt="none",
                    capsize=3, color=color,
                )
            axis.set_xticks(range(len(directions)), direction_labels)
            axis.set_ylim(0, 1)
            axis.set_ylabel(title)
            axis.set_title(f"{pair[0]} ↔ {pair[1]} · {title}")
            axis.grid(axis="y", alpha=0.25)
            if row_index == 0 and col_index == 1:
                axis.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.text(
        0.5, 0.01,
        "Smoke/partial results only. Error bars show reported subject standard deviation.",
        ha="center", color="#9b2c2c", fontsize=9, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 0.86, 0.93))
    _save(fig, output_dir / f"{label}_transfer_direction.png")


def plot_accuracy_f1_scatter(
    comparisons: Dict[str, List[Dict[str, str]]], output_dir: Path, label: str
) -> None:
    """Show accuracy/macro-F1 balance, preserving scenario-level identity."""
    n_cols = min(2, len(comparisons))
    n_rows = (len(comparisons) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(
        f"{label.title()} accuracy vs macro F1 by scenario",
        fontsize=16,
        fontweight="bold",
    )
    label_offsets = ((6, 6), (6, -11), (-42, 7), (-42, -11), (6, 16))
    for axis, (scenario, rows) in zip(axes.flat, comparisons.items()):
        for index, row in enumerate(rows):
            model_name = row["model"]
            accuracy = _metric(row, "accuracy")
            f1 = _metric(row, "f1")
            axis.errorbar(
                accuracy,
                f1,
                xerr=_std(row, "accuracy_std"),
                yerr=_std(row, "f1_std"),
                fmt="o",
                markersize=8,
                capsize=3,
                color=MODEL_COLORS.get(model_name, "#888888"),
            )
            axis.annotate(
                model_name,
                (accuracy, f1),
                xytext=label_offsets[index % len(label_offsets)],
                textcoords="offset points",
                fontsize=8,
            )
        axis.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_title(scenario.replace("_", " → "))
        axis.set_xlabel("Accuracy")
        axis.set_ylabel("Macro F1")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(comparisons):]:
        axis.set_visible(False)
    fig.text(
        0.5, 0.01,
        "Smoke/partial runs; error bars show reported std. The diagonal is accuracy = macro F1.",
        ha="center", color="#9b2c2c", fontsize=9, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    _save(fig, output_dir / f"{label}_accuracy_vs_macro_f1.png")


def visualize(label: str) -> List[Path]:
    comparisons = load_comparisons(label)
    output_dir = OUTPUT_ROOT / label
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_scenario_rankings(comparisons, output_dir, label)
    plot_metric_heatmaps(comparisons, output_dir, label)
    plot_transfer_directions(comparisons, output_dir, label)
    plot_accuracy_f1_scatter(comparisons, output_dir, label)
    return sorted(output_dir.glob("*.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default="valence",
        help="Label whose comparison CSVs should be plotted (default: valence).",
    )
    args = parser.parse_args()
    paths = visualize(args.label)
    print(f"[done] generated {len(paths)} plots in {OUTPUT_ROOT / args.label}")


if __name__ == "__main__":
    main()
