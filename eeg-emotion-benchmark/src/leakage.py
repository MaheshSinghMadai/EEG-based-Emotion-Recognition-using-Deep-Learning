"""
leakage.py
==========
Automated leakage assertions, executed on EVERY fold before a model is trained.

Checks
------
window_disjoint   no (subject, trial, window) key appears in two partitions.
trial_disjoint    no (subject, trial) pair contributes windows to test AND train/val.
subject_disjoint  no subject contributes windows to test AND train/val.
normalizer        normalization statistics were fitted on exactly the training rows.
target_labels     domain-adaptation models only ever receive unlabeled target tensors.

A protocol declares which guarantees it promises (see `splits.GUARANTEES`). A
promised guarantee that does not hold raises `LeakageError` and stops the run.
Guarantees a protocol does NOT promise (the deliberately leaky ablations) are still
measured and written into the result file, so the amount of overlap is documented.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Set, Tuple


class LeakageError(AssertionError):
    """Raised when a partition violates a guarantee its protocol promises."""


def _keys(subject_ids: Sequence[str], trial_ids: Sequence[int], window_ids: Sequence[int],
          domain: str = "") -> Tuple[Set, Set, Set]:
    """(subjects, (subject, trial) pairs, (subject, trial, window) triples) for one partition.

    `domain` (e.g. the dataset name) namespaces keys so cross-dataset partitions with
    colliding subject ids are not reported as overlapping.
    """
    subjects = {(domain, s) for s in subject_ids}
    trials = {(domain, s, int(t)) for s, t in zip(subject_ids, trial_ids)}
    windows = {(domain, s, int(t), int(w)) for s, t, w in zip(subject_ids, trial_ids, window_ids)}
    return subjects, trials, windows


def partition_keys(dataset, domain: str = "") -> Tuple[Set, Set, Set]:
    return _keys(dataset.subject_ids, dataset.trial_ids, dataset.window_ids,
                 domain or getattr(dataset, "domain", ""))


def check_partitions(
    train,
    val,
    test,
    guarantees: Dict[str, bool],
    normalizer_fit_rows: Optional[int] = None,
) -> Dict:
    """Validate train/val/test `EEGWindowDataset`s against the declared guarantees.

    Returns a JSON-serialisable report. Raises `LeakageError` on a broken promise.
    """
    parts = {"train": partition_keys(train), "test": partition_keys(test)}
    if val is not None and len(val):
        parts["val"] = partition_keys(val)
    fit_side = ["train"] + (["val"] if "val" in parts else [])

    def overlap(level: int, a: Iterable[str], b: str) -> int:
        seen = set().union(*(parts[p][level] for p in a))
        return len(seen & parts[b][level])

    report = {
        "subject_overlap_test": overlap(0, fit_side, "test"),
        "trial_overlap_test": overlap(1, fit_side, "test"),
        "window_overlap_test": overlap(2, fit_side, "test"),
        "window_overlap_train_val": (
            len(parts["train"][2] & parts["val"][2]) if "val" in parts else 0
        ),
        "subject_overlap_train_val": (
            len(parts["train"][0] & parts["val"][0]) if "val" in parts else 0
        ),
        "n_train": len(train),
        "n_val": len(val) if val is not None else 0,
        "n_test": len(test),
    }
    report["subject_disjoint"] = report["subject_overlap_test"] == 0
    report["trial_disjoint"] = report["trial_overlap_test"] == 0
    report["window_disjoint"] = (
        report["window_overlap_test"] == 0 and report["window_overlap_train_val"] == 0
    )

    violations = [g for g, promised in guarantees.items() if promised and not report.get(g, False)]
    if normalizer_fit_rows is not None:
        report["normalizer_fit_rows"] = int(normalizer_fit_rows)
        report["normalizer_fit_on_train_only"] = int(normalizer_fit_rows) == len(train)
        if not report["normalizer_fit_on_train_only"]:
            violations.append("normalizer_fit_on_train_only")
    report["guarantees_promised"] = dict(guarantees)
    report["passed"] = not violations
    if violations:
        raise LeakageError(f"Leakage check failed: {violations}. Report: {report}")
    return report


def assert_unlabeled(batch) -> None:
    """Domain-adaptation loaders must yield bare input tensors, never (x, y) pairs."""
    if isinstance(batch, (tuple, list)):
        raise LeakageError(
            "Unlabeled target loader yielded a tuple; target labels must never reach training."
        )
