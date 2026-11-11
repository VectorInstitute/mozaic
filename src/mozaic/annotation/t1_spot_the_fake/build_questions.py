"""Build the T1 question file from metadata.jsonl.

One question per clip. The answer comes from the clip's label, so no model is used.
Writes one file per split, or with --size a balanced sample of the test split.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from mozaic.annotation.sampling import balanced_sample, shuffled_options
from mozaic.annotation.writer import write_questions
from mozaic.common.metadata import FRAMES, check_label, filter_clips, load_metadata

Row = dict[str, Any]
log = logging.getLogger(__name__)
CONFIG = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())

TASK = "task1_spot_the_fake"
QUESTION = (
    "Which part of this audio-visual clip, if any, is AI-generated or manipulated? "
    "A real recording that was only processed, without changing who is seen or heard, still counts as real."
)
OPTIONS = ["the video", "the audio", "both", "neither"]
# Vocoded clips keep their label: a vocoded real voice is still real.
ANSWERS = {"RVRA": "neither", "RVFA": "the audio", "FVRA": "the video", "FVFA": "both"}


def make_row(clip: Row) -> Row:
    """The T1 question for one clip."""
    clip_id = "t1_" + Path(clip["file_name"]).stem
    return {
        "id": clip_id,
        "task": TASK,
        "split": clip["split"],
        "clips": [clip["file_name"]],
        "question": QUESTION,
        "options": shuffled_options(OPTIONS, CONFIG["options_seed"], clip_id),
        "gt_answer": ANSWERS[clip["label"]],
        "gt_label": clip["label"],
        "gt_target": {"video_fake": int(not clip["face_real"]), "audio_fake": int(not clip["voice_real"])},
        "meta": {
            "video_method": clip["video_generator"],
            "frames": FRAMES[clip["length"]],
            "arm": clip["arm"],
            "source_id": clip["source_id"],
            "target_id": clip["target_id"],
            "pair_id": clip["pair_id"],
        },
    }


def kind(row: Row) -> tuple:
    """Clip kind: label, video method, frames, arm."""
    m = row["meta"]
    return row["gt_label"], m["video_method"], m["frames"], m["arm"]


def build(size: int | None = None, seed: int | None = None, out: str | None = None) -> list[Path]:
    clips = filter_clips(load_metadata().values(), **{k: v for k, v in CONFIG["filters"].items() if v})
    clips = [c for c in clips if c["arm"] == CONFIG["arms"][c["label"]]]  # the vocoded view

    problems = [p for c in clips if (p := check_label(c))]
    if problems:
        raise ValueError(f"{len(problems)} clips with a wrong label, nothing written. First: {problems[0]}")

    rows = [make_row(c) for c in clips]
    default_seed = CONFIG["sample"]["seed"]
    if size:
        seed = seed if seed is not None else default_seed
        pool = [r for r in rows if r["split"] == CONFIG["sample"]["split"]]
        rows = balanced_sample(pool, size, seed, answer=lambda r: r["gt_label"], kind=kind)

    paths = write_questions("t1", rows, size=size, seed=seed, default_seed=default_seed, out=out, answer_field="gt_answer")
    if size:
        log.info("kinds: %d, pairs: %d", len({kind(r) for r in rows}), len({r["meta"]["pair_id"] for r in rows}))
    return paths