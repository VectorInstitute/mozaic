"""Annotate each clip, one call per kind of part it has (its fake parts, its real parts): annotator -> checks ->
verifier (up to `tries` rounds), then rewriter -> checks -> verifier. Replies, settings and stats go to build/ (see
README.md)."""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from mozaic.annotation.t4_explanation.facts import claims as claims_module
from mozaic.annotation.t4_explanation.facts import metadata_facts
from mozaic.annotation.t4_explanation.prompts import annotator as annotator_prompt
from mozaic.annotation.t4_explanation.prompts import rewriter as rewriter_prompt
from mozaic.annotation.t4_explanation.prompts import verifier as verifier_prompt
from mozaic.annotation.t4_explanation.models.annotator import Annotator
from mozaic.annotation.t4_explanation.facts.artifacts import CATEGORIES
from mozaic.annotation.t4_explanation.build_questions import (CONFIG, annotation_order, build_dir, is_failed,
                                                               kind_of, rewrite_failed, saved_replies, view)
from mozaic.annotation.t4_explanation.checks.rules import check_reply, check_rewrite
from mozaic.annotation.t4_explanation.facts.claims import KINDS, claims_of, parts_of
from mozaic.annotation.t4_explanation.models.client import cut_off, parse_json
from mozaic.annotation.t4_explanation.facts.metadata_facts import facts
from mozaic.annotation.t4_explanation.models.rewriter import Rewriter
from mozaic.annotation.t4_explanation.models.verifier import Verifier
from mozaic.common.io import append_jsonl, read_jsonl, write_json, write_jsonl

Row = dict[str, Any]
log = logging.getLogger(__name__)


def sha256(value: Any) -> str:
    """Hash of a string, or of any JSON value."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def settings() -> Row:
    """Everything that shapes what the models are sent: a run with other settings is refused."""
    a, v, w = CONFIG["annotator"], CONFIG["verifier"], CONFIG["rewriter"]
    return {
        "annotator": {k: a[k] for k in ("model_version", "generation", "frames", "audio_sample_rate")},
        "verifier": {k: v[k] for k in ("model_version", "generation")},
        "rewriter": {k: w[k] for k in ("model_version", "generation")},
        "tries": CONFIG["tries"],
        "rewrite_tries": CONFIG["rewrite_tries"],
        "annotator_prompt_sha256": sha256(annotator_prompt.PROMPT + annotator_prompt.RETRY),
        "verifier_prompt_sha256": sha256(verifier_prompt.PROMPT + verifier_prompt.REWRITE_PROMPT),
        "rewriter_prompt_sha256": sha256(rewriter_prompt.SYSTEM),
        "claims_sha256": sha256({"real": claims_module.REAL, "whole_clip": claims_module.WHOLE_CLIP,
                                 "scene": claims_module.SCENE}),
        "real_annotator_prompt_sha256": sha256(annotator_prompt.REAL_PROMPT + annotator_prompt.RETRY),
        "real_verifier_prompt_sha256": sha256(verifier_prompt.REAL_PROMPT),
        "real_claims_sha256": sha256({"verdict": claims_module.REAL_VERDICT, "scene": claims_module.REAL_SCENE,
                                      "extra": annotator_prompt.EXTRA}),
        "artifacts_sha256": sha256(CATEGORIES),
        "facts_sha256": sha256({"video": metadata_facts.VIDEO, "audio": metadata_facts.AUDIO,
                                "signs": metadata_facts.SIGNS, "off": metadata_facts.OFF}),
    }


def check_settings(out_dir: Path, current: Row) -> None:
    """Save the run's settings once. Refuse to mix a run with other settings. A setting the saved ones lack (a folder
    annotated before it existed, e.g. the real parts') is added once: the replies made before it never used it."""
    path = out_dir / "settings.json"
    if not path.exists():
        write_json(path, current)
        return
    old = json.loads(path.read_text())
    differ = sorted(k for k in set(old) | set(current) if k in old and old.get(k) != current.get(k))
    if differ:
        raise SystemExit(f"{out_dir} was annotated with other settings ({', '.join(differ)}): "
                         "the replies would mix two setups. Restore the settings, or start a new build folder.")
    if (new := sorted(current.keys() - old.keys())):
        log.info("[t4] settings: %s added to %s", ", ".join(new), path)
        write_json(path, {**old, **{k: current[k] for k in new}})


def rewrite(reply: Row, claims: list[Row], parts: list[str], rewriter: Rewriter, verifier: Verifier,
            kind: str = "fake") -> Row:
    """The rewrite of checked sentences: rewriter -> checks -> verifier (against the sentences). problems is []
    when it passes. A real part's verdict is not rewritten: the rewriter gets its scene sentences, and the verdict
    sentence (claims.REAL_SENTENCE) ends the paragraph as it is."""
    verdicts = {c["id"] for c in claims if c["kind"] == "verdict"}
    scene = {p: [s for s in reply[p] if not verdicts & set(s.get("claims") or [])] or reply[p] for p in parts} \
        if kind == "real" else reply
    gen = rewriter.generate(rewriter_prompt.build_prompt(scene, parts, kind))
    text = parse_json(gen.raw)
    if kind == "real" and isinstance(text, dict):
        text = {p: f"{v.strip()} {claims_module.REAL_SENTENCE[p]}" if isinstance(v, str) and v.strip() and p in parts
                else v for p, v in text.items()}
    problems = check_rewrite(text, reply, claims, parts, kind)
    if cut_off(gen):
        problems.append(f"rewriter: cut off ({gen.finish_reason})")
    out: Row = {"raw": gen.raw, "finish_reason": gen.finish_reason, "usage": gen.usage,
                "text": text if isinstance(text, dict) else None}
    if not problems:
        # a real part's verdict as it ends the paragraph, not as the annotator worded it ("the part is ...")
        checked = {p: [{**s, "text": claims_module.REAL_SENTENCE[p]} if verdicts & set(s.get("claims") or []) else s
                       for s in reply[p]] for p in parts} if kind == "real" else reply
        question, texts = verifier_prompt.build_rewrite_prompt(checked, text, parts)
        check = verifier.generate(question)
        problems = ["rewriter: " + p for p in verifier_prompt.problems(parse_json(check.raw), texts)]
        if cut_off(check):
            problems.append(f"rewriter: verifier: cut off ({check.finish_reason})")
        out["verifier"] = {"raw": check.raw, "finish_reason": check.finish_reason, "usage": check.usage}
    return {**out, "problems": problems}


def explain(clip_file: str, clip_facts: Row, gender: str, annotator: Annotator, verifier: Verifier,
            rewriter: Rewriter, tries: int, kind: str = "fake") -> Row:
    """One clip's rounds of annotator -> checks -> verifier for its parts of one kind, until one passes or `tries`
    are used, then its rewrite. An API error is raised (the clip is not saved); a reply that fails every round is
    returned as failed, with no rewrite."""
    parts = parts_of(clip_facts, kind)
    claims = claims_of(clip_facts, kind)
    span = annotator_prompt.lengths(claims, parts, kind)
    template = verifier_prompt.PROMPT if kind == "fake" else verifier_prompt.REAL_PROMPT
    rounds: list[Row] = []
    for _ in range(tries):
        last = rounds[-1]["problems"] if rounds else None
        gen = annotator.generate(clip_file, annotator_prompt.build_prompt(claims, parts, gender, last, kind))
        reply = parse_json(gen.raw)
        problems = check_reply(reply, claims, parts, span, kind)                   # 2. code checks
        if cut_off(gen):
            problems.append(f"annotator: cut off ({gen.finish_reason})")
        done: Row = {"annotator": {"raw": gen.raw, "finish_reason": gen.finish_reason, "usage": gen.usage}}
        if not problems:                                                           # 3. the verifier
            question, texts = verifier_prompt.build_prompt(claims, reply, parts, template)
            check = verifier.generate(question)
            problems = verifier_prompt.problems(parse_json(check.raw), texts)
            if cut_off(check):
                problems.append(f"verifier: cut off ({check.finish_reason})")
            done["verifier"] = {"raw": check.raw, "finish_reason": check.finish_reason, "usage": check.usage}
        rounds.append({**done, "problems": problems})
        if not problems:
            break
    reply, ok = parse_json(rounds[-1]["annotator"]["raw"]), not rounds[-1]["problems"]
    out = rewrite(reply, claims, parts, rewriter, verifier, kind) if ok else None  # 4. rewriter -> checks -> verifier
    return {"kind": kind, "claims": claims, "reply": reply, "ok": ok, "rounds": rounds, "rewrite": out, "rewrites_failed": []}


def problem_kind(problem: str) -> str:
    """A problem without its sentence and details: 'video 3: says flicker, which ...' -> 'says flicker'."""
    if problem.startswith("verifier: S"):
        return "verifier: adds"
    if problem.startswith("rewriter: "):
        return "rewriter: " + problem_kind(problem.removeprefix("rewriter: "))
    if "cut off" in problem:
        return problem.split(":")[0] + ": cut off"
    text = problem.split(": ", 1)[1] if re.match(r"(video|audio)( \d+)?: ", problem) else problem
    text = re.sub(r"^\d+ sentences, asked.*", "sentence count", text)
    return re.split(r" \(|, which|: \[|: '|: \"", text)[0]


def stats(out_dir: Path) -> Row:
    """What the saved replies show, for the record: build/stats.json."""
    saved = list(saved_replies(out_dir / "replies.jsonl").values())
    dropped = out_dir / "replies_dropped.jsonl"
    rounds = [x for r in saved for x in r["rounds"]]
    rewrites = [w for r in saved for w in [*r.get("rewrites_failed", []), r.get("rewrite")] if w]  # every attempt
    rewrite_by = collections.Counter(
        "cut off" if any("cut off" in p for p in w["problems"]) else "verifier" if "verifier" in w else "checks"
        for w in rewrites if w["problems"])
    failed_by = collections.Counter(
        "cut off" if any("cut off" in p for p in x["problems"]) else "verifier" if "verifier" in x else "checks"
        for x in rounds if x["problems"])
    out = {
        "clips": {"explained": len(saved), "passed": sum(r["ok"] for r in saved),
                  "failed": sum(not r["ok"] for r in saved)},
        "by_kind": {k: {"explained": len(mine), "passed": sum(r["ok"] for r in mine),
                        "failed": sum(not r["ok"] for r in mine)}
                    for k in KINDS if (mine := [r for r in saved if kind_of(r) == k])},
        "passed_in_round": {str(k): sum(r["ok"] and len(r["rounds"]) == k for r in saved)
                            for k in range(1, CONFIG["tries"] + 1)},
        "rounds_failed_by": {k: failed_by[k] for k in ("checks", "verifier", "cut off")},
        "problems": dict(collections.Counter(problem_kind(p) for x in rounds for p in x["problems"]).most_common()),
        "rewrites": {"passed": sum(bool(r.get("rewrite")) and not r["rewrite"]["problems"] for r in saved),
                     "failed": sum(bool(r.get("rewrite")) and bool(r["rewrite"]["problems"]) for r in saved),
                     "fixed_by_a_retry": sum(bool(r.get("rewrites_failed")) and not r["rewrite"]["problems"]
                                             for r in saved if r.get("rewrite")),
                     "attempts_failed_by": {k: rewrite_by[k] for k in ("checks", "verifier", "cut off")},
                     "problems": dict(collections.Counter(problem_kind(p) for w in rewrites
                                                          for p in w["problems"]).most_common())},
        "calls": {**{m: sum(m in x for x in rounds) for m in ("annotator", "verifier")}, "rewriter": len(rewrites),
                  "rewrite verifier": sum("verifier" in w for w in rewrites)},
        "tokens": {**{m: sum(x[m]["usage"].get("total_tokens", 0) for x in rounds if m in x)
                      for m in ("annotator", "verifier")},
                   "rewriter": sum(w["usage"].get("total_tokens", 0) for w in rewrites),
                   "rewrite verifier": sum(w["verifier"]["usage"].get("total_tokens", 0) for w in rewrites
                                           if "verifier" in w)},
        "asked_again": len(read_jsonl(dropped)) if dropped.exists() else 0,
    }
    write_json(out_dir / "stats.json", out)
    return out


def annotate(golden_only: bool = False, limit: int | None = None, retry_failed: bool = False,
             annotator: Annotator | None = None, verifier: Verifier | None = None,
             rewriter: Rewriter | None = None) -> Path:
    """Explain the selected clips (golden first): one call per clip and kind of part (claims.KINDS) it has, for each
    with no saved reply yet. Fake and real calls are independent: a clip's two calls can come in either order."""
    rows = annotation_order(golden_only)[:limit]
    clips = view()
    out_dir = build_dir()
    check_settings(out_dir, settings())

    path = out_dir / "replies.jsonl"
    saved = saved_replies(path)
    if retry_failed and saved:
        # sentences that failed, or a rewrite that failed rewrite_tries times: the annotator again, from round 1
        stuck = lambda r, f: rewrite_failed(r, facts(clips[f])) and \
            len(r.get("rewrites_failed", [])) + 1 >= CONFIG["rewrite_tries"]
        keep = {key: r for key, r in saved.items() if not is_failed(r, facts(clips[key[0]])) and not stuck(r, key[0])}
        for key in saved.keys() - keep.keys():
            append_jsonl(out_dir / "replies_dropped.jsonl", saved[key])
        write_jsonl(path, list(keep.values()))
        log.info("[t4] asking again %d replies that failed the checks or the verifier, or every rewrite",
                 len(saved) - len(keep))
        saved = keep
    selected = {r["clips"][0] for r in rows}
    todo = [(r["clips"][0], kind) for r in rows for kind in KINDS
            if parts_of(facts(clips[r["clips"][0]]), kind) and (r["clips"][0], kind) not in saved]
    # a failed rewrite is asked again alone: its sentences passed, so no annotator call
    again = [key for key, r in saved.items()
             if retry_failed and key[0] in selected and rewrite_failed(r, facts(clips[key[0]]))]
    log.info("[t4] %s: %d clips, %d calls to make (%s), %d to rewrite again", "golden" if golden_only else "every set",
             len(rows), len(todo), ", ".join(f"{k} {sum(x[1] == k for x in todo)}" for k in KINDS), len(again))
    if not todo and not again:
        stats(out_dir)
        return path

    if annotator is None:
        annotator = Annotator({**CONFIG["annotator"], "retry": CONFIG["retry"]})
        annotator.load()
    if verifier is None:
        verifier = Verifier({**CONFIG["verifier"], "retry": CONFIG["retry"]})
        verifier.load()
    if rewriter is None:
        rewriter = Rewriter({**CONFIG["rewriter"], "retry": CONFIG["retry"]})
        rewriter.load()
    for file, kind in again:
        r = saved[file, kind]
        out = rewrite(r["reply"], r["claims"], parts_of(facts(clips[file]), kind), rewriter, verifier, kind)
        append_jsonl(path, {**r, "rewrite": out, "rewrites_failed": [*r.get("rewrites_failed", []), r["rewrite"]],
                            "time": dt.datetime.now().isoformat(timespec="seconds")})  # the last row of a clip wins
        log.info("rewrite again %s (%s)  %s", Path(file).stem, kind, "ok" if not out["problems"] else "FAILED")
    failed_in_a_row = 0
    for i, (file, kind) in enumerate(todo, 1):
        try:
            out = explain(file, facts(clips[file]), clips[file]["gender"], annotator, verifier, rewriter,
                          CONFIG["tries"], kind)
        except Exception as e:  # noqa: BLE001  any failed call is asked again on the next run
            failed_in_a_row += 1
            log.warning("%s (%s): failed: %s", file, kind, e)
            if failed_in_a_row >= CONFIG["stop_after_failures"]:
                log.error("STOPPED after %d failures in a row. Run the same command to continue.", failed_in_a_row)
                break
            continue
        failed_in_a_row = 0
        append_jsonl(path, {"clip": file, **out, "time": dt.datetime.now().isoformat(timespec="seconds")})
        log.info("[%d/%d] %s (%s)  %s (%d rounds), rewrite %s", i, len(todo), Path(file).stem, kind,
                 "ok" if out["ok"] else "FAILED", len(out["rounds"]),
                 "-" if not out["rewrite"] else "ok" if not out["rewrite"]["problems"] else "FAILED")
    s = stats(out_dir)
    log.info("[t4] %s; by kind %s; failed rounds by %s; rewrites %s; replies in %s", s["clips"], s["by_kind"],
             s["rounds_failed_by"], {k: s["rewrites"][k] for k in ("passed", "failed")}, path)
    return path
