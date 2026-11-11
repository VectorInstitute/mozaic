"""T2 scores: the 4-way manipulation type, plus real vs fake.

Classes come from the target index (gt_target, pred_target), not the stored names, so
predictions saved under earlier class names score the same.

A prediction may carry score: {"real": p, "face_swap": p, "fully_generated": p,
"full_body_swap": p}. Then AUROC, EER and AP use them. Without it, they are
computed from the hard answer and marked score_source: hard_label.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from mozaic.annotation.t2_manipulation_type.build_questions import (
    TARGET_NAMES as CLASSES,
)
from mozaic.evaluation.metrics.stats import accuracy, binary, by_group, multiclass


def score(questions: dict[str, dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for p in predictions:
        q = questions[p["id"]]
        pred = None if p["pred_target"] is None else CLASSES[p["pred_target"]]
        items.append({
            "gt": CLASSES[q["gt_target"]],
            "pred": pred,
            "score": p.get("score"),
            "correct": p["pred_target"] == q["gt_target"],
            "gt_position": q["options"].index(q["gt_answer"]) + 1,
            "pred_position": q["options"].index(p["pred_answer"]) + 1 if p["pred_answer"] else None,
            "meta": q["meta"],
            "usage": p.get("usage") or {},
        })

    gt = [i["gt"] for i in items]
    pred = [i["pred"] for i in items]
    scores = [i["score"] for i in items] if items and all(i["score"] for i in items) else None

    return {
        "questions": len(questions),
        "answered": len(items),
        "missing": len(questions) - len(items),
        "unreadable": sum(p is None for p in pred),
        **accuracy([i["correct"] for i in items]),
        **multiclass(gt, pred, CLASSES, scores),
        # fake = any manipulation; fpr = real videos called fake, recall = fakes caught
        "real_vs_fake": binary(
            [int(g != "real") for g in gt],
            [None if p is None else int(p != "real") for p in pred],
            [1 - s["real"] for s in scores] if scores else None,
        ),
        "per_method": by_group(items, lambda i: i["meta"]["video_method"]),
        "per_frames": by_group(items, lambda i: i["meta"]["frames"]),
        "per_arm": by_group(items, lambda i: i["meta"]["arm"]),
        "pred_classes": dict(Counter(str(p) for p in pred)),            # answer bias; "None" = unreadable
        "pred_positions": dict(sorted(Counter(str(i["pred_position"]) for i in items).items())),  # order bias
        "gt_positions": dict(sorted(Counter(str(i["gt_position"]) for i in items).items())),
        "tokens": {k: sum(i["usage"].get(k, 0) for i in items)
                   for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
    }