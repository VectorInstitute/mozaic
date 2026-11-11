"""Build the T2 question file from metadata.jsonl.

One question per clip, only about the video. The answer comes from how the video was
made (video_generator), so no model is used. Writes one file per split, or with
--size a balanced sample of the test split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mozaic.annotation.sampling import balanced_sample, shuffled_options
from mozaic.annotation.writer import write_questions
from mozaic.common.metadata import FRAMES, filter_clips, load_metadata

Row = dict[str, Any]
CONFIG = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())

TASK = "task2_manipulation_type"
QUESTION = (
    "Which part of this video, if any, was generated or altered by AI? "
    "Ignore normal processing such as brightness, colour or compression."
)
# Index = gt_target. What each option means is explained in the prompt.
OPTIONS = ["real", "face swap", "fully generated", "full-body swap"]
TARGET_NAMES = ["real", "face_swap", "fully_generated", "full_body_swap"]  # OPTIONS as ids, same order
TARGETS = {"real": 0, "facefusion": 1, "s2v": 2, "animate": 3}  # video_generator -> gt_target


def make_row(clip: Row) -> Row:
    """The T2 question for one clip."""
    target = TARGETS[clip["video_generator"]]
    clip_id = "t2_" + Path(clip["file_name"]).stem
    return {
        "id": clip_id,
        "task": TASK,
        "split": clip["split"],
        "clips": [clip["file_name"]],
        "question": QUESTION,
        "options": shuffled_options(OPTIONS, CONFIG["options_seed"], clip_id),
        "gt_answer": OPTIONS[target],
        "gt_label": clip["label"],
        "gt_target": target,
        "gt_target_name": TARGET_NAMES[target],
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


def check(clip: Row) -> str | None:
    """The video is real exactly when face_real is true and the label starts with R."""
    real = clip["video_generator"] == "real"
    if real != bool(clip["face_real"]) or real != (clip["label"][0] == "R"):
        return (
            f"{clip['file_name']}: video_generator {clip['video_generator']}, "
            f"face_real={clip['face_real']}, label {clip['label']}"
        )
    return None


def kind(row: Row) -> tuple:
    """Clip kind: answer, frames, arm."""
    return row["gt_target"], row["meta"]["frames"], row["meta"]["arm"]


def selected(filters: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (filters or {}).items() if v}


def build(size: int | None = None, seed: int | None = None, out: str | None = None) -> list[Path]:
    clips = filter_clips(load_metadata().values(), **selected(CONFIG["filters"]))

    problems = [p for c in clips if (p := check(c))]
    if problems:
        raise ValueError(f"{len(problems)} clips disagree with face_real or the label, nothing written. First: {problems[0]}")

    rows = [make_row(c) for c in clips]
    default_seed = CONFIG["sample"]["seed"]
    if size:
        seed = seed if seed is not None else default_seed
        sample = CONFIG["sample"]
        pool = filter_clips(clips, splits=[sample["split"]], **selected(sample.get("filters")))
        rows = balanced_sample([make_row(c) for c in pool], size, seed, answer=lambda r: r["gt_target"], kind=kind)

    paths = write_questions("t2", rows, size=size, seed=seed, default_seed=default_seed, out=out, answer_field="gt_target_name")
    if size:
        print(f"  kinds: {len({kind(r) for r in rows})}, pairs: {len({r['meta']['pair_id'] for r in rows})}")
    return paths