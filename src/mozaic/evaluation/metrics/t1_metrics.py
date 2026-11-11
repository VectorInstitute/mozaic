"""T1 scores: the 4-way answer, real vs fake, and the video and audio parts on their own.

A prediction may carry score: {"RVRA": p, "RVFA": p, "FVRA": p, "FVFA": p} (class
probabilities, e.g. from a classifier). Then AUROC, EER and AP use them. Without it,
they are computed from the hard answer and marked score_source: hard_label.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from mozaic.evaluation.metrics.stats import accuracy, binary, by_group, multiclass

LABELS = ["RVRA", "RVFA", "FVRA", "FVFA"]


def _fake_scores(scores: list[dict[str, float]] | None) -> dict[str, list[float]] | None:
    """Turn 4-class probabilities into fake probabilities per view."""
    if scores is None:
        return None
    return {
        "any": [1 - s["RVRA"] for s in scores],
        "video": [s["FVRA"] + s["FVFA"] for s in scores],
        "audio": [s["RVFA"] + s["FVFA"] for s in scores],
    }


def score(questions: dict[str, dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for p in predictions:
        q = questions[p["id"]]
        items.append({
            "gt": q["gt_label"],
            "pred": p["pred_label"],
            "gt_target": q["gt_target"],
            "pred_target": p["pred_target"] or {},
            "score": p.get("score"),
            "correct": p["pred_label"] == q["gt_label"],
            "gt_position": q["options"].index(q["gt_answer"]) + 1,
            "pred_position": q["options"].index(p["pred_answer"]) + 1 if p["pred_answer"] else None,
            "meta": q["meta"],
            "usage": p.get("usage") or {},
        })

    gt = [i["gt"] for i in items]
    pred = [i["pred"] for i in items]
    scores = [i["score"] for i in items] if items and all(i["score"] for i in items) else None
    fake = _fake_scores(scores)

    def stream(name: str) -> dict[str, Any]:
        key = f"{name}_fake"
        return binary(
            [i["gt_target"][key] for i in items],
            [i["pred_target"].get(key) for i in items],
            fake[name] if fake else None,
        )

    return {
        "questions": len(questions),
        "answered": len(items),
        "missing": len(questions) - len(items),
        "unreadable": sum(p is None for p in pred),
        **accuracy([i["correct"] for i in items]),
        **multiclass(gt, pred, LABELS, scores),
        "real_vs_fake": binary(
            [int(g != "RVRA") for g in gt],
            [None if p is None else int(p != "RVRA") for p in pred],
            fake["any"] if fake else None,
        ),
        "video": stream("video"),
        "audio": stream("audio"),
        "per_method": by_group(items, lambda i: i["meta"]["video_method"]),
        "per_arm": by_group(items, lambda i: i["meta"]["arm"]),
        "per_frames": by_group(items, lambda i: i["meta"]["frames"]),
        "per_label": by_group(items, lambda i: i["gt"]),
        "pred_labels": dict(Counter(str(p) for p in pred)),  # answer bias; "None" = unreadable
        "pred_positions": dict(sorted(Counter(str(i["pred_position"]) for i in items).items())),  # order bias
        "gt_positions": dict(sorted(Counter(str(i["gt_position"]) for i in items).items())),
        "tokens": {k: sum(i["usage"].get(k, 0) for i in items)
                   for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
    }