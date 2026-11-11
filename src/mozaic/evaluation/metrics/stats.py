"""Metric helpers shared by every task. Pure numpy, no sklearn."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def _r(x: float | None) -> float | None:
    return None if x is None or math.isnan(x) else round(float(x), 4)


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson interval of k successes out of n."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def accuracy(correct: Sequence[bool]) -> dict[str, Any]:
    n, k = len(correct), int(sum(correct))
    return {"n": n, "accuracy": _r(_div(k, n)) if n else None, "ci95": wilson(k, n)}


# --------------------------------------------------------------------------- #
# Ranking metrics: need a score per item (higher = more likely positive)
# --------------------------------------------------------------------------- #
def auroc(gt: Sequence[int], score: Sequence[float]) -> float | None:
    """Area under the ROC curve (Mann-Whitney, ties count half)."""
    y, s = np.asarray(gt), np.asarray(score, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return _r((greater + 0.5 * ties) / (len(pos) * len(neg)))


def roc_points(gt: Sequence[int], score: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """False and true positive rates at every threshold, from (0, 0) to (1, 1)."""
    y, s = np.asarray(gt), np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    last_of_tie = np.r_[np.diff(s) != 0, True]
    tp = np.cumsum(y)[last_of_tie]
    fp = np.cumsum(1 - y)[last_of_tie]
    tpr = np.r_[0, tp / max(y.sum(), 1)]
    fpr = np.r_[0, fp / max((1 - y).sum(), 1)]
    return fpr, tpr


def eer(gt: Sequence[int], score: Sequence[float]) -> float | None:
    """Equal error rate: where the false positive rate equals the false negative rate."""
    y = np.asarray(gt)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    fpr, tpr = roc_points(gt, score)
    fnr = 1 - tpr
    d = fpr - fnr
    i = int(np.argmax(d >= 0))
    if i == 0 or d[i] == 0:
        return _r(fpr[i])
    t = d[i - 1] / (d[i - 1] - d[i])  # interpolate between the two points around the crossing
    return _r(fpr[i - 1] + t * (fpr[i] - fpr[i - 1]))


def average_precision(gt: Sequence[int], score: Sequence[float]) -> float | None:
    """Area under the precision-recall curve (step-wise)."""
    y, s = np.asarray(gt), np.asarray(score, dtype=float)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    last_of_tie = np.r_[np.diff(s) != 0, True]
    tp = np.cumsum(y)[last_of_tie]
    fp = np.cumsum(1 - y)[last_of_tie]
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    return _r(np.sum(np.diff(np.r_[0, recall]) * precision))


# --------------------------------------------------------------------------- #
# Binary: positive = fake = 1
# --------------------------------------------------------------------------- #
def binary(gt: Sequence[int], pred: Sequence[int | None], score: Sequence[float] | None = None) -> dict[str, Any]:
    """Every binary metric. An unreadable prediction (None) counts as wrong.

    Without a score, the hard prediction is used as the score: AUROC then equals
    balanced accuracy, and score_source says so.
    """
    g = [int(x) for x in gt]
    p = [x if x is not None else 1 - gi for x, gi in zip(pred, g)]  # unreadable = wrong
    tp = sum(gi == 1 and pi == 1 for gi, pi in zip(g, p))
    tn = sum(gi == 0 and pi == 0 for gi, pi in zip(g, p))
    fp = sum(gi == 0 and pi == 1 for gi, pi in zip(g, p))
    fn = sum(gi == 1 and pi == 0 for gi, pi in zip(g, p))

    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    source = "model" if score is not None else "hard_label"
    s = list(score) if score is not None else [float(x) for x in p]
    return {
        **accuracy([gi == pi for gi, pi in zip(g, p)]),
        "positives": tp + fn,
        "negatives": tn + fp,
        "precision": _r(precision),
        "recall": _r(recall),                  # true positive rate: fakes caught
        "specificity": _r(specificity),        # true negative rate: reals kept
        "fpr": _r(1 - specificity),            # reals called fake
        "fnr": _r(1 - recall),                 # fakes missed
        "f1": _r(_div(2 * precision * recall, precision + recall)),
        "balanced_accuracy": _r((recall + specificity) / 2),
        "mcc": _r(_div(tp * tn - fp * fn, mcc_den)),
        "auroc": auroc(g, s),
        "eer": eer(g, s),
        "ap": average_precision(g, s),
        "score_source": source,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


# --------------------------------------------------------------------------- #
# Multi-class
# --------------------------------------------------------------------------- #
def confusion(gt: Sequence[str], pred: Sequence[str | None], classes: Sequence[str]) -> dict[str, dict[str, int]]:
    """gt class -> predicted class -> count. Unreadable replies go to 'none'."""
    table = {g: {c: 0 for c in [*classes, "none"]} for g in classes}
    for g, p in zip(gt, pred):
        table[g][p if p in classes else "none"] += 1
    return table


def multiclass(
    gt: Sequence[str],
    pred: Sequence[str | None],
    classes: Sequence[str],
    scores: Sequence[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Per-class precision, recall, F1 and one-vs-rest AUROC; macro and weighted averages; kappa."""
    n = len(gt)
    per: dict[str, Any] = {}
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gt, pred))
        fp = sum(g != c and p == c for g, p in zip(gt, pred))
        fn = sum(g == c and p != c for g, p in zip(gt, pred))
        precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
        per[c] = {
            "support": tp + fn,
            "precision": _r(precision),
            "recall": _r(recall),
            "f1": _r(_div(2 * precision * recall, precision + recall)),
            "auroc": auroc([g == c for g in gt], [s[c] for s in scores]) if scores else None,
        }

    def avg(key: str, weighted: bool) -> float | None:
        w = [per[c]["support"] if weighted else 1 for c in classes]
        vals = [per[c][key] for c in classes]
        if any(v is None for v in vals) or not sum(w):
            return None
        return _r(sum(v * wi for v, wi in zip(vals, w)) / sum(w))

    observed = _div(sum(g == p for g, p in zip(gt, pred)), n)
    expected = sum(_div(sum(g == c for g in gt), n) * _div(sum(p == c for p in pred), n) for c in classes)
    return {
        "per_class": per,
        "macro": {k: avg(k, False) for k in ("precision", "recall", "f1", "auroc")},
        "weighted": {k: avg(k, True) for k in ("precision", "recall", "f1")},
        "balanced_accuracy": avg("recall", False),
        "cohen_kappa": _r(_div(observed - expected, 1 - expected)),
        "confusion": confusion(gt, pred, classes),
    }


def by_group(items: Sequence[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    """Accuracy per group, e.g. per video method."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for item in items:
        groups[str(key(item))].append(item["correct"])
    return {g: accuracy(v) for g, v in sorted(groups.items())}
