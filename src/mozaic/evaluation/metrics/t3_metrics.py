"""T3 scores: the 4-way answer, and its two halves on their own: real or fake, same
person or not.

The classes are the question file's gt_target_name: real_same_person,
real_different_person, fake_this_person, fake_different_person. A prediction may carry
score: {class name: p} (class probabilities, e.g. from a classifier); then AUROC, EER
and AP use them. Without it, they are computed from the hard answer and marked
score_source: hard_label.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from mozaic.annotation.t3_identity_check.build_questions import TARGET_NAMES as CLASSES
from mozaic.evaluation.metrics.stats import accuracy, binary, by_group, multiclass

VOICE_KIND = {"none": "real", "bigvgan": "vocoded", "seed_vc": "converted"}  # audio_method -> clip 1's sound


def shows(q: dict[str, Any]) -> str:
    """What clip 1 shows, e.g. "face facefusion" or "voice converted"."""
    meta = q["meta"]
    return f"{q['part']} " + (meta["video_method"] if q["part"] == "face" else VOICE_KIND[meta["audio_method"]])


def _half_scores(scores: list[dict[str, float]] | None) -> dict[str, list[float]] | None:
    """Turn 4-class probabilities into a probability per half."""
    if scores is None:
        return None
    return {
        "fake": [s["fake_this_person"] + s["fake_different_person"] for s in scores],
        "same_person": [s["real_same_person"] + s["fake_this_person"] for s in scores],
    }


def score(questions: dict[str, dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for p in predictions:
        q = questions[p["id"]]
        items.append({
            "gt": q["gt_target_name"],
            "pred": p["pred_target_name"],
            "gt_target": q["gt_target"],
            "pred_target": p["pred_target"] or {},
            "pred_answer": p["pred_answer"],
            "score": p.get("score"),
            "correct": p["pred_target_name"] == q["gt_target_name"],
            "gt_position": q["options"].index(q["gt_answer"]) + 1,
            "pred_position": q["options"].index(p["pred_answer"]) + 1 if p["pred_answer"] else None,
            "part": q["part"],
            "shows": shows(q),
            "label": q["gt_label"],
            "right_answer": q["gt_answer"],
            "meta": q["meta"],
            "usage": p.get("usage") or {},
        })

    gt = [i["gt"] for i in items]
    pred = [i["pred"] for i in items]
    scores = [i["score"] for i in items] if items and all(i["score"] for i in items) else None
    half = _half_scores(scores)

    def view(key: str) -> dict[str, Any]:
        return binary(
            [i["gt_target"][key] for i in items],
            [i["pred_target"].get(key) for i in items],
            half[key] if half else None,
        )

    prompt_tokens: dict[str, set[Any]] = defaultdict(set)
    for i in items:
        prompt_tokens[f"{i['part']} {i['meta']['frames']}"].add(i["usage"].get("prompt_tokens"))

    return {
        "questions": len(questions),
        "answered": len(items),
        "missing": len(questions) - len(items),
        "unreadable": sum(p is None for p in pred),
        **accuracy([i["correct"] for i in items]),
        **multiclass(gt, pred, CLASSES, scores),
        "fake": view("fake"),  # real or fake: fake = 1
        "same_person": view("same_person"),  # same person as clip 2 or not: same = 1
        "per_part": by_group(items, lambda i: i["part"]),
        "per_clip1": by_group(items, lambda i: i["shows"]),  # e.g. "voice converted"
        "per_label": by_group(items, lambda i: i["label"]),  # clip 1's label
        "per_clip2_person": by_group(items, lambda i: f"{i['part']} {i['meta']['clip2_person']}"),
        "per_right_answer": by_group(items, lambda i: f"{i['part']} {i['right_answer']}"),
        "per_frames": by_group(items, lambda i: i["meta"]["frames"]),
        "pred_answers": dict(Counter(str(i["pred_answer"]) for i in items)),  # answer bias; "None" = unreadable
        "pred_positions": dict(sorted(Counter(str(i["pred_position"]) for i in items).items())),  # order bias
        "gt_positions": dict(sorted(Counter(str(i["gt_position"]) for i in items).items())),
        "tokens": {k: sum(i["usage"].get(k, 0) for i in items)
                   for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        # Every question of one part and length should cost the same prompt tokens; a
        # voice count that does not grow with length means the sound did not arrive.
        "prompt_tokens": {k: sorted(v, key=str) for k, v in sorted(prompt_tokens.items())},
    }
