"""T4 scores. The answer is an explanation per part (pred_target.sections: video, audio), scored against the
reference (gt_target.sections), per part and for both together:

    text       ROUGE-L, BLEU-4, CIDEr-D (text_metrics.py): word overlap, computed here
    cosine     of the reference's and the prediction's embeddings, from judge.jsonl (judge/run_judge.py)
    judge      gpt-5.4-mini's 0-10 per part (mean), from judge.jsonl

A prediction that could not be read counts as an empty text: 0 on every score.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from mozaic.evaluation.metrics.stats import accuracy
from mozaic.evaluation.metrics.text_metrics import bleu_4, cider_d, rouge_l

PARTS = ("video", "audio")


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def parse_rate(parsed: list[bool]) -> dict[str, Any]:
    """The share of replies that parse: a rate of the format, not an accuracy of the answer."""
    a = accuracy(parsed)
    return {"n": a["n"], "rate": a["accuracy"], "ci95": a["ci95"]}


def text_scores(refs: list[str], preds: list[str]) -> dict[str, Any]:
    if not refs:
        return {"n": 0, "rouge_l": None, "bleu_4": None, "cider_d": None}
    return {"n": len(refs), "rouge_l": _r(mean(rouge_l(p, r) for p, r in zip(preds, refs, strict=True))),
            "bleu_4": _r(bleu_4(preds, refs)), "cider_d": _r(mean(cider_d(preds, refs)))}


def judge_scores(items: list[dict[str, Any]], judged: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not judged:
        return {"judge": "not run yet", "cosine": "not run yet"}
    ok = [i for i in items if i["id"] in judged and judged[i["id"]]["ok"]]
    out: dict[str, Any] = {"judged": len(ok), "not_judged": len(items) - len(ok),
                           "failed": sum(not judged[i["id"]]["ok"] for i in items if i["id"] in judged)}
    if not ok:
        return out
    rows = [judged[i["id"]] for i in ok]
    out["judge"] = {p: {"mean": _r(mean(r["scores"][p]["score"] for r in rows))} for p in PARTS}
    out["judge"]["both"] = {"mean": _r(mean(mean(r["scores"][p]["score"] for p in PARTS) for r in rows))}
    embedded = all(r["cosine"] is not None for r in rows)  # judged without an embedder: no cosine
    out["cosine"] = {p: _r(mean(r["cosine"][p] for r in rows)) for p in PARTS} if embedded else "not run"
    cells: dict[str, list[float]] = defaultdict(list)
    for i, r in zip(ok, rows, strict=True):
        cells[i["cell"]].append(mean(r["scores"][p]["score"] for p in PARTS))
    out["judge_per_cell"] = {c: {"n": len(v), "mean": _r(mean(v))} for c, v in sorted(cells.items())}
    return out


def per_group(items: list[dict[str, Any]], key) -> dict[str, Any]:
    """The parse rate per group, e.g. per label x video method."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for i in items:
        groups[str(key(i))].append(i["parsed"])
    return {g: parse_rate(v) for g, v in sorted(groups.items())}


def score(questions: dict[str, dict[str, Any]], predictions: list[dict[str, Any]],
          judged: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    items = []
    for p in predictions:
        q = questions[p["id"]]
        sections = (p["pred_target"] or {}).get("sections") or {}
        items.append({
            "id": p["id"],
            "parsed": p["pred_target"] is not None,  # a video and an audio explanation
            "ref": q["gt_target"]["sections"],
            "pred": {part: sections.get(part, "") for part in PARTS},
            "cell": f"{q['gt_label']}_{q['meta']['video_method']}",
            "meta": q["meta"],
            "usage": p.get("usage") or {},
        })
    text = {part: text_scores([i["ref"][part] for i in items], [i["pred"][part] for i in items]) for part in PARTS}
    text["both"] = text_scores([" ".join(i["ref"][p] for p in PARTS) for i in items],
                               [" ".join(i["pred"][p] for p in PARTS).strip() for i in items])
    return {
        "questions": len(questions),
        "answered": len(items),
        "missing": len(questions) - len(items),
        "unreadable": sum(not i["parsed"] for i in items),
        "parse_rate": parse_rate([i["parsed"] for i in items]),
        "text": text,
        **judge_scores(items, {r["id"]: r for r in judged or []}),
        "per_cell": per_group(items, lambda i: i["cell"]),
        "per_frames": per_group(items, lambda i: i["meta"]["frames"]),
        "tokens": {k: sum(i["usage"].get(k, 0) for i in items)
                   for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
    }
