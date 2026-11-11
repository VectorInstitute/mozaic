"""T4 question files: select() picks the clips from metadata, build_final() adds the saved explanations (fake and
real parts). No model calls."""

from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any

import yaml

from mozaic.annotation.sampling import balanced_sample
from mozaic.annotation.t4_explanation.checks.rules import check_rewrite
from mozaic.annotation.t4_explanation.facts.claims import KINDS, SECTIONS, claims_of, parts_of
from mozaic.annotation.t4_explanation.facts.metadata_facts import REGIONS, facts
from mozaic.common.io import iter_jsonl, read_jsonl, write_jsonl
from mozaic.common.metadata import FRAMES, SPLITS, check_label, load_metadata
from mozaic.common.paths import qas_dir, subset

Row = dict[str, Any]
log = logging.getLogger(__name__)
CONFIG = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())

TASK = "task4_explanation"
QUESTION = (
    "Which parts of this audio-visual clip, if any, were generated or altered by AI, where, "
    "and what in the clip shows it? Normal processing such as compression does not count."
)
GOLDEN = "golden"
SETS = (*SPLITS, GOLDEN)


# Another folder for every step (--out), None for qas_small/t4_explanation/. Set once per run by use_out.
_out: Path | None = None


def use_out(folder: str | None) -> None:
    """Run every step in this folder instead of qas_small/t4_explanation/ (experiments, e.g. a prompt trial)."""
    global _out
    _out = Path(folder) if folder else None


def t4_dir() -> Path:
    """qas_small/t4_explanation/, or the --out folder. Refuses any other subset, or the full dataset."""
    if subset() != CONFIG["subset"]:
        raise SystemExit(f"T4 is built on the {CONFIG['subset']} subset only: add --subset {CONFIG['subset']}")
    return _out or qas_dir("t4")


def build_dir() -> Path:
    """qas_small/t4_explanation/build/: what the final files are built from (selection, settings, replies)."""
    return t4_dir() / "build"


def selection_path(name: str) -> Path:
    """The clips of one set (train, val, test or golden), facts only."""
    return build_dir() / "selection" / f"{name}.jsonl"


def final_path(name: str) -> Path:
    """The final question file of one set, e.g. qas_small/t4_explanation/t4_golden.jsonl."""
    return t4_dir() / f"t4_{name}.jsonl"


def in_view(clip: Row) -> bool:
    """The vocoded view: one arm per label (config arms)."""
    return clip["arm"] == CONFIG["arms"][clip["label"]]


def cell(row: Row) -> str:
    """Golden cell: label x video generator (8 cells), e.g. FVFA_s2v."""
    return f"{row['gt_label']}_{row['meta']['video_method']}"


def make_row(clip: Row) -> Row:
    """The T4 row for one clip: the T1 to T3 keys, the facts, no evidence yet."""
    return {
        "id": "t4_" + Path(clip["file_name"]).stem,
        "task": TASK,
        "split": clip["split"],
        "clips": [clip["file_name"]],
        "question": QUESTION,
        "options": list(REGIONS),  # the region list shown to the model
        "gt_answer": None,
        "gt_label": clip["label"],
        "gt_target": {**facts(clip), "evidence": None},
        "meta": {
            "video_method": clip["video_generator"],
            "audio_method": clip["audio_generator"],
            "frames": FRAMES[clip["length"]],
            "arm": clip["arm"],
            "source_id": clip["source_id"],
            "target_id": clip["target_id"],
            "pair_id": clip["pair_id"],
        },
    }


def view() -> dict[str, Row]:
    """The clips in the vocoded view, keyed by file_name, with their labels checked. Small only."""
    t4_dir()  # refuses the full dataset before its metadata is read
    clips = {f: c for f, c in load_metadata().items() if in_view(c)}
    problems = [p for c in clips.values() if (p := check_label(c))]
    if problems:
        raise ValueError(f"{len(problems)} clips with a wrong label, nothing written. First: {problems[0]}")
    return clips


def select() -> list[Path]:
    """Write the selection: every clip of the view per split, and the golden clips drawn from test."""
    clips = view()
    rows = [make_row(c) for c in clips.values()]
    sets = {split: [r for r in rows if r["split"] == split] for split in SPLITS}

    golden = CONFIG["golden"]
    gender = {c["file_name"]: c["gender"] for c in clips.values()}  # not stored in the row: it is in the metadata
    sets[GOLDEN] = balanced_sample(sets[golden["split"]], golden["size"], golden["seed"], answer=cell,
                                   kind=lambda r: (cell(r), r["meta"]["frames"], gender[r["clips"][0]]))

    changed = [n for n, part in sets.items() if selection_path(n).exists()
               and [r["id"] for r in iter_jsonl(selection_path(n))] != [r["id"] for r in part]]
    if changed:
        raise SystemExit(f"the selection of {', '.join(changed)} in {build_dir()} holds other clips (metadata or "
                         "config changed): the saved replies were made for it. Move build/ away to start over.")
    paths = []
    for name, part in sets.items():
        path = selection_path(name)
        write_jsonl(path, part)
        paths.append(path)
        log.info("[t4] %s: %s clips -> %s", name, f"{len(part):,}", path)
    log.info("[t4] golden per cell: %s, pairs: %d", dict(collections.Counter(cell(r) for r in sets[GOLDEN])),
             len({r["meta"]["pair_id"] for r in sets[GOLDEN]}))
    return paths


def annotation_order(golden_only: bool) -> list[Row]:
    """The selected rows to annotate, golden first, each clip once."""
    names = [GOLDEN] if golden_only else [GOLDEN, *SPLITS]
    missing = [n for n in names if not selection_path(n).exists()]
    if missing:
        raise SystemExit(f"no selection for {', '.join(missing)} in {build_dir()}: run --select first")
    rows = {r["clips"][0]: r for n in names for r in iter_jsonl(selection_path(n))}
    return list(rows.values())


def kind_of(saved: Row) -> str:
    """The kind of part a saved reply explains (claims.KINDS): rows saved before real parts were explained have none,
    and are fake."""
    return saved.get("kind", "fake")


def saved_replies(path: Path) -> dict[tuple[str, str], Row]:
    """The saved replies by (clip, kind): the last row of each wins."""
    return {(r["clip"], kind_of(r)): r for r in iter_jsonl(path)} if path.exists() else {}


def is_failed(saved: Row, clip_facts: Row) -> bool:
    """A saved reply whose sentences need the annotator again: they failed the checks or the verifier when they were
    written, or the clip's claims (of its kind) have changed since. The sentences are checked when they are written;
    what ships is the rewrite, checked again by rewrite_failed."""
    return not saved["ok"] or saved["claims"] != claims_of(clip_facts, kind_of(saved))


def rewrite_failed(saved: Row, clip_facts: Row) -> bool:
    """A reply whose sentences passed but whose rewrite did not, or no longer passes the current code checks (a
    stricter rule needs no new call to apply): --retry-failed asks the rewriter again, not the annotator."""
    w, kind = saved.get("rewrite"), kind_of(saved)
    return not w or bool(w["problems"]) or w["text"] is None or bool(
        check_rewrite(w["text"], saved["reply"], saved["claims"], parts_of(clip_facts, kind), kind))


def evidence(clip_facts: Row) -> list[Row]:
    """The evidence items of a clip: its signs, from metadata, never from a reply."""
    return [{"modality": name, "artifact": s["artifact"], "meaning": s["meaning"]}
            for name in SECTIONS for s in clip_facts["parts"][name].get("signs") or []]


def sections(clip_facts: Row, saved: dict[str, Row | None]) -> Row | None:
    """The text per part: the rewrite of the reply of its kind (saved: kind -> reply). None when a part has no passing
    reply or no passing rewrite (--retry-failed asks again): never another text in its place."""
    for kind in KINDS:
        r = saved.get(kind)
        if parts_of(clip_facts, kind) and (r is None or is_failed(r, clip_facts) or rewrite_failed(r, clip_facts)):
            return None
    kind = {name: "fake" if clip_facts["parts"][name]["fake"] else "real" for name in SECTIONS}
    return {name: saved[kind[name]]["rewrite"]["text"][name].strip() for name in SECTIONS}


def build_final() -> list[Path]:
    """Selection + passing replies -> one final file per set. Clips with a part (fake or real) and no passing reply
    or rewrite are left out (--annotate --retry-failed asks them again). human_checked: the golden clips, in
    t4_golden.jsonl and in t4_test.jsonl alike (the same row in both)."""
    clips = view()
    saved = saved_replies(build_dir() / "replies.jsonl")
    golden = {r["clips"][0] for r in iter_jsonl(selection_path(GOLDEN))} if selection_path(GOLDEN).exists() else set()
    paths = []
    for name in SETS:
        if not selection_path(name).exists():
            raise SystemExit(f"no selection for {name} in {build_dir()}: run --select first")
        selected = read_jsonl(selection_path(name))
        rows = []
        for row in selected:
            file = row["clips"][0]
            clip_facts = facts(clips[file])  # from the metadata now: a cutoff change needs no new --select
            text = sections(clip_facts, {kind: saved.get((file, kind)) for kind in KINDS})
            if text is None:
                continue
            rows.append({
                **row,
                "gt_answer": " ".join(text.values()),
                "gt_target": {**clip_facts, "sections": text, "evidence": evidence(clip_facts)},
                "meta": {**row["meta"], "annotator": CONFIG["annotator"]["model_version"],
                         "verifier": CONFIG["verifier"]["model_version"],
                         "rewriter": CONFIG["rewriter"]["model_version"], "human_checked": file in golden},
            })
        write_jsonl(final_path(name), rows)
        paths.append(final_path(name))
        log.info("[t4] %s: %s of %s clips explained -> %s", name, f"{len(rows):,}", f"{len(selected):,}", final_path(name))
    return paths
