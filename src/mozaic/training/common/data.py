"""Question files → clips → the model's input, the same sampling as evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import Dataset

from mozaic.annotation.t1_spot_the_fake.build_questions import ANSWERS as T1_ANSWERS
from mozaic.annotation.t2_manipulation_type.build_questions import (
    TARGET_NAMES as T2_TARGET_NAMES,
)
from mozaic.annotation.t3_identity_check.build_questions import (
    TARGET_NAMES as T3_TARGET_NAMES,
)
from mozaic.common.frame_sampler import load_audio, read_frames
from mozaic.common.io import read_jsonl
from mozaic.common.paths import qas_dir, resolve
from mozaic.common.questions import select

# Class order per task: the head's output index, as evaluation's from_probs and metrics key them.
# Taken from the task's annotation code, never a copy.
CLASSES = {
    "t1": list(T1_ANSWERS),     # RVRA, RVFA, FVRA, FVFA
    "t2": list(T2_TARGET_NAMES),  # real, face_swap, fully_generated, full_body_swap
    "t3": list(T3_TARGET_NAMES),  # real_same_person, real_different_person, fake_this_person, fake_different_person
}

# What a model gets per task, as in evaluation: t2 is about the video only, so no sound.
# t3 has two clips per question, frames (face) or sound (voice): common/pairs.py.
MODALITIES = {
    "t1": ("video", "audio"),
    "t2": ("video",),
}


def label(task: str, question: dict[str, Any]) -> int:
    """The question's class index in CLASSES[task]."""
    if task == "t1":
        return CLASSES["t1"].index(question["gt_label"])
    if task == "t2":
        return CLASSES["t2"].index(question["gt_target_name"])
    if task == "t3":
        return CLASSES["t3"].index(question["gt_target_name"])
    raise ValueError(f"unknown task {task}")


def load_questions(task: str, entry: str | dict[str, Any], limit: int | None) -> tuple[str, list[dict[str, Any]]]:
    """A split's question file and its rows: `t1_train.jsonl` or {file, filters}.

    limit keeps about limit rows, the same number of each class in file order: the files are
    sorted by label, so their first rows would all be one class.
    """
    file, filters = (entry, None) if isinstance(entry, str) else (entry["file"], entry.get("filters"))
    rows = select(read_jsonl(qas_dir(task) / file), filters)
    if not limit:
        return file, rows
    per_class = -(-limit // len(CLASSES[task]))
    by_class: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_class.setdefault(label(task, row), []).append(row)
    return file, [row for c in sorted(by_class) for row in by_class[c][:per_class]]


class ClipDataset(Dataset):
    """One item per question: the processor's tensors, the class index, and the question's position."""

    def __init__(self, task: str, questions: list[dict[str, Any]], processor, config: dict[str, Any]):
        self.task, self.questions, self.processor = task, questions, processor
        self.frames = config["frames"]
        self.rate = config["audio_sample_rate"] if "audio" in MODALITIES[task] else None

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, i: int) -> dict[str, Any]:
        q = self.questions[i]
        clip = resolve(q["clips"][0])
        frames, indices, total = read_frames(clip, self.frames)
        audio = np.frombuffer(load_audio(clip, self.rate), dtype=np.int16).astype(np.float32) if self.rate else None
        item = self.processor(frames, indices, total, audio)
        return {**item, "label": label(self.task, q), "index": i}
