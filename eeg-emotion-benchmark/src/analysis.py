"""
analysis.py
===========
Turns `results/proposal/**.json` into the thesis tables and figures that answer the
research question:

  T1  strict within-dataset results (LOSO), with the majority-class floor
  T2  leakage inflation per model: subject_kfold vs trial_kfold vs random_window,
      with paired Wilcoxon tests over subjects (random_window vs subject_kfold)
  T3  ranking shift: model ranks per protocol, Kendall tau between strict and leaky
      rankings, and between the papers' reported ranking and the strict ranking
  T4  significance on the strict protocol: Friedman test over subjects + pairwise
      Wilcoxon signed-rank with Holm correction
  T5  cross-dataset generalization: within-dataset LOSO on the target vs
      source -> target transfer (generalization gap)

Run:  python -m src.analysis --label valence
Writes results/proposal/analysis/<label>/{tables.md, *.csv, *.png}.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import friedmanchisquare, kendalltau, wilcoxon

from .benchmark_proposal import MODELS, RESULTS_DIR, ROOT

METRIC = "accuracy"
PAPER_REPORTED = os.path.join(ROOT, "docs", "paper_reported.json")
# Reference palette, categorical slots 1-3 (validated all-pairs, light mode).
PROTOCOL_COLORS = {"subject_kfold": "#2a78d6", "trial_kfold": "#eb6834", "random_window": "#1baf7a"}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_results(label: str) -> List[Dict]:
    out = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "**", f"*_{label}.json"), recursive=True):
        if "/analysis/" in path:
            continue
        with open(path) as f:
            r = json.load(f)
        r["_scenario"] = (f"{r['source']}_to_{r['target']}" if r["protocol"] == "transfer"
                          else f"{r['dataset']}/{r['protocol']}")
        out.append(r)
    return out


def index(results: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    by: Dict[str, Dict[str, Dict]] = {}
    for r in results:
        by.setdefault(r["_scenario"], {})[r["model"]] = r
    return by


def per_subject(r: Dict, metric: str = METRIC) -> Dict[str, float]:
    return {row["subject"]: row[metric] for row in r["per_subject"]}


def holm(pvalues: List[float]) -> List[float]:
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def paired(a: Dict[str, float], b: Dict[str, float]):
    keys = sorted(set(a) & set(b))
    return np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 5 or np.allclose(x, y):
        return None
    return float(wilcoxon(x, y).pvalue)


def fmt(r: Optional[Dict], metric: str = METRIC) -> str:
    if r is None:
        return "—"
    s = r["summary"]
    flag = "" if r.get("run_status") == "full_protocol" else "†"
    return f"{s[metric]:.3f} ± {s[metric + '_std']:.3f}{flag}"


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def table_strict(by, datasets) -> List[str]:
    lines = ["## T1 — Strict within-dataset (LOSO, subject- and trial-disjoint)", "",
             "| Model | " + " | ".join(f"{d} acc | {d} macro-F1" for d in datasets) + " |",
             "|---|" + "---:|---:|" * len(datasets)]
    for m in MODELS:
        cells = []
        for d in datasets:
            r = by.get(f"{d}/loso", {}).get(m)
            cells += [fmt(r), fmt(r, "f1")]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    base = []
    for d in datasets:
        r = next(iter(by.get(f"{d}/loso", {}).values()), None)
        base += ([f"{r['majority_baseline']['accuracy']:.3f}", f"{r['majority_baseline']['f1']:.3f}"]
                 if r else ["—", "—"])
    lines.append("| *Majority class* | " + " | ".join(base) + " |")
    return lines + [""]


def table_leakage(by, datasets) -> tuple[List[str], List[Dict]]:
    lines = ["## T2 — Leakage inflation (5-fold, equal train/test budget)", "",
             "Δtrial = random_window − trial_kfold; Δsubject = trial_kfold − subject_kfold; "
             "p = Wilcoxon over subjects, random_window vs subject_kfold.", "",
             "| Dataset | Model | subject_kfold | trial_kfold | random_window | Δtrial | Δsubject | Δtotal | p |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for d in datasets:
        for m in MODELS:
            rs = {p: by.get(f"{d}/{p}", {}).get(m) for p in PROTOCOL_COLORS}
            if not all(rs.values()):
                continue
            acc = {p: r["summary"][METRIC] for p, r in rs.items()}
            p = safe_wilcoxon(*paired(per_subject(rs["random_window"]), per_subject(rs["subject_kfold"])))
            row = {"dataset": d, "model": m, **acc,
                   "d_trial": acc["random_window"] - acc["trial_kfold"],
                   "d_subject": acc["trial_kfold"] - acc["subject_kfold"],
                   "d_total": acc["random_window"] - acc["subject_kfold"], "p": p}
            rows.append(row)
            lines.append(f"| {d} | {m} | {fmt(rs['subject_kfold'])} | {fmt(rs['trial_kfold'])} | "
                         f"{fmt(rs['random_window'])} | {row['d_trial']:+.3f} | {row['d_subject']:+.3f} | "
                         f"{row['d_total']:+.3f} | {'—' if p is None else f'{p:.3g}'} |")
    return lines + [""], rows


def _ranking(models_scores: Dict[str, float]) -> List[str]:
    return [m for m, _ in sorted(models_scores.items(), key=lambda kv: kv[1], reverse=True)]


def table_ranking(by, datasets) -> List[str]:
    lines = ["## T3 — Ranking shift", ""]
    protocols = ["random_window", "trial_kfold", "subject_kfold", "loso"]
    for d in datasets:
        ranks = {}
        for p in protocols:
            scores = {m: r["summary"][METRIC] for m, r in by.get(f"{d}/{p}", {}).items() if m in MODELS}
            if len(scores) >= 2:
                ranks[p] = _ranking(scores)
        if not ranks:
            continue
        lines += [f"**{d}** (1 = best accuracy)", "", "| Model | " + " | ".join(ranks) + " |",
                  "|---|" + "---:|" * len(ranks)]
        for m in MODELS:
            lines.append(f"| {m} | " + " | ".join(str(r.index(m) + 1) if m in r else "—" for r in ranks.values()) + " |")
        if "random_window" in ranks and "loso" in ranks:
            common = [m for m in ranks["loso"] if m in ranks["random_window"]]
        if "random_window" in ranks and "loso" in ranks and len(common) >= 3:
            tau = kendalltau([ranks["random_window"].index(m) for m in common],
                             [ranks["loso"].index(m) for m in common])
            lines.append(f"\nKendall τ (random_window vs LOSO ranking) = {tau.statistic:.2f} (p = {tau.pvalue:.2g})")
        lines.append("")

    if os.path.isfile(PAPER_REPORTED):
        with open(PAPER_REPORTED) as f:
            reported = {k: v for k, v in json.load(f)["models"].items() if v.get("accuracy") is not None}
        lines += ["**Reported in the original papers vs. this benchmark (strict LOSO)**", "",
                  "| Model | Paper accuracy | Paper dataset / protocol | Strict LOSO (" + ", ".join(datasets) + ") |",
                  "|---|---:|---|---|"]
        for m, v in reported.items():
            strict = ", ".join(fmt(by.get(f"{d}/loso", {}).get(m)) for d in datasets)
            lines.append(f"| {m} | {v['accuracy']:.2f}% | {v['dataset']} / {v['protocol']} | {strict} |")
        lines += ["", "Paper numbers come from different datasets and protocols; the comparison is of "
                  "rank and magnitude of claims, not a like-for-like score.", ""]
    return lines


def table_significance(by, datasets) -> List[str]:
    lines = ["## T4 — Significance on the strict protocol (LOSO)", ""]
    for d in datasets:
        rs = {m: r for m, r in by.get(f"{d}/loso", {}).items() if m in MODELS}
        if len(rs) < 3:
            continue
        subjects = sorted(set.intersection(*(set(per_subject(r)) for r in rs.values())))
        mat = {m: [per_subject(r)[s] for s in subjects] for m, r in rs.items()}
        if len(subjects) < 5:
            lines.append(f"{d}: too few subjects ({len(subjects)}) for the tests.\n")
            continue
        stat = friedmanchisquare(*mat.values())
        lines += [f"**{d}** — Friedman χ² = {stat.statistic:.2f}, p = {stat.pvalue:.3g} "
                  f"({len(subjects)} subjects × {len(mat)} models)", "",
                  "| Pair | Wilcoxon p | Holm-adjusted p |", "|---|---:|---:|"]
        pairs = list(combinations(mat, 2))
        raw = [safe_wilcoxon(np.array(mat[a]), np.array(mat[b])) for a, b in pairs]
        valid = [p for p in raw if p is not None]
        adj = iter(holm(valid)) if valid else iter([])
        for (a, b), p in zip(pairs, raw):
            lines.append(f"| {a} vs {b} | {'—' if p is None else f'{p:.3g}'} | "
                         f"{'—' if p is None else f'{next(adj):.3g}'} |")
        lines.append("")
    return lines


def table_transfer(by) -> tuple[List[str], List[Dict]]:
    lines = ["## T5 — Cross-dataset generalization", "",
             "Gap = strict LOSO accuracy on the target dataset − accuracy when trained on the other dataset.", "",
             "| Direction | Model | Transfer acc | Transfer macro-F1 | Target LOSO acc | Gap |",
             "|---|---|---:|---:|---:|---:|"]
    rows = []
    for scenario in sorted(s for s in by if "_to_" in s):
        target = scenario.split("_to_")[1]
        for m, r in by[scenario].items():
            within = by.get(f"{target}/loso", {}).get(m)
            gap = within["summary"][METRIC] - r["summary"][METRIC] if within else None
            rows.append({"direction": scenario, "model": m, "transfer": r["summary"][METRIC], "gap": gap})
            lines.append(f"| {scenario.replace('_to_', ' → ')} | {m} | {fmt(r)} | {fmt(r, 'f1')} | "
                         f"{fmt(within)} | {'—' if gap is None else f'{gap:+.3f}'} |")
    return lines + [""], rows


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
def plot_leakage(rows: List[Dict], out_dir: str, label: str) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 4.2), squeeze=False,
                             facecolor=SURFACE)
    width = 0.26
    for ax, d in zip(axes[0], datasets):
        sub = [r for r in rows if r["dataset"] == d]
        x = np.arange(len(sub))
        for i, p in enumerate(PROTOCOL_COLORS):
            vals = [r[p] for r in sub]
            bars = ax.bar(x + (i - 1) * width, vals, width - 0.02, color=PROTOCOL_COLORS[p],
                          label=p.replace("_", " "), edgecolor=SURFACE, linewidth=2)
            if p == "random_window":
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                            fontsize=7, color=INK_MUTED)
        ax.axhline(0.5, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xticks(x, [r["model"] for r in sub], fontsize=8, color=INK)
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel(f"{label} accuracy (mean over subjects)", color=INK_MUTED, fontsize=9)
        ax.set_title(f"{d}: accuracy rises as the split leaks", color=INK, fontsize=11, loc="left")
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_MUTED)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left", title="split (strict → leaky)",
                      title_fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "leakage_inflation.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] {path}")


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="valence")
    args = p.parse_args()
    by = index(load_results(args.label))
    if not by:
        raise SystemExit("No results found; run src.benchmark_proposal first.")
    datasets = sorted({s.split("/")[0] for s in by if "/" in s})
    out_dir = os.path.join(RESULTS_DIR, "analysis", args.label)
    os.makedirs(out_dir, exist_ok=True)

    leak_lines, leak_rows = table_leakage(by, datasets)
    transfer_lines, transfer_rows = table_transfer(by)
    lines = [f"# Benchmark analysis — {args.label}", "",
             "† = smoke/partial run (fold subset or reduced epochs); not a thesis result.", ""]
    lines += table_strict(by, datasets) + leak_lines + table_ranking(by, datasets)
    lines += table_significance(by, datasets) + transfer_lines
    with open(os.path.join(out_dir, "tables.md"), "w") as f:
        f.write("\n".join(lines))
    write_csv(leak_rows, os.path.join(out_dir, "leakage_inflation.csv"))
    write_csv(transfer_rows, os.path.join(out_dir, "transfer_gap.csv"))
    plot_leakage(leak_rows, out_dir, args.label)
    print(f"[done] {out_dir}/tables.md")


if __name__ == "__main__":
    main()
