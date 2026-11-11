"""T3: two clips per question, compared. The dataset, the batches and the pair head, for any classifier.

A T3 question has two clips and one part, as evaluation sends them (t3_prompts.clip_inputs): a face
question is two videos without sound (their frames), a voice question two sounds without frames. Each
clip goes through the model's encoder for its part on its own (as T2's video, or a voice clip's audio
alone), its features are pooled over time (masked mean) and layer-normed, and the head compares the two:
[a, b, |a - b|, a * b] to the four answers. One head per part: the parts' encoders differ.

A model's T3 version (classifiers/<model>/pair.py) puts PairProcessor before its Processor (face, voice:
one clip's input) and PairClassifier before its Model (encode_part: one part's features). A micro-batch
may hold both parts: collate keeps them apart and the model gives the logits back in the batch's order.
With a part unused in a micro-batch, the loop shares the gradients every micro-step for t3 (common/loop.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from mozaic.common.frame_sampler import load_audio, read_frames
from mozaic.common.paths import resolve
from mozaic.training.common import data
from mozaic.training.common.data import label
from mozaic.training.common.heads import make_head, masked_mean

PARTS = ("face", "voice")


def load_questions(task: str, entry: str | dict[str, Any], limit: int | None) -> tuple[str, list[dict[str, Any]]]:
    """data.load_questions, with limit keeping the same number of each class of each part: the files are
    sorted by part, so the first rows of each class would all be face questions."""
    file, rows = data.load_questions(task, entry, None)
    if not limit:
        return file, rows
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["part"], label(task, row)), []).append(row)
    per_group = -(-limit // len(groups))
    return file, [row for key in sorted(groups) for row in groups[key][:per_group]]


class PairDataset(Dataset):
    """One item per question: both clips as the model's input for the question's part, the class index, and
    the question's position. Each clip is read as in ClipDataset: its sampled frames, or its complete audio."""

    def __init__(self, task: str, questions: list[dict[str, Any]], processor, config: dict[str, Any]):
        self.task, self.questions, self.processor = task, questions, processor
        self.frames, self.rate = config["frames"], config["audio_sample_rate"]

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, i: int) -> dict[str, Any]:
        q = self.questions[i]
        clips = []
        for path in q["clips"]:
            clip = resolve(path)
            if q["part"] == "face":  # the frames, no sound
                clips.append(self.processor.face(*read_frames(clip, self.frames)))
            else:  # the sound, no frames
                audio = np.frombuffer(load_audio(clip, self.rate), dtype=np.int16).astype(np.float32)
                clips.append(self.processor.voice(audio))
        return {"part": q["part"], "clips": clips, "label": label(self.task, q), "index": i}


def clip_batch(batch: dict[str, torch.Tensor], part: str, k: int) -> dict[str, torch.Tensor]:
    """Clip k's tensors of the part's questions, as the model's own collate made them (valid included)."""
    prefix = f"{part}.{k}."
    return {name[len(prefix):]: value for name, value in batch.items() if name.startswith(prefix)}


class PairProcessor:
    """Before a model's Processor. face and voice give one clip's input; collate batches the questions."""

    def face(self, frames: list, indices: list[int], total: int) -> dict[str, Any]:
        """A face clip: its frames as the model's video-only input (T2's), no sound."""
        return self(frames, indices, total, None)

    def voice(self, audio: np.ndarray) -> dict[str, Any]:
        """A voice clip: its complete audio as the model's audio input, no frames."""
        raise NotImplementedError

    def collate(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Per part in the batch: each clip batched by the model's own collate (padding, valid), as
        face.0.video, face.0.valid, face.1.video, ..., and the part's positions in the batch (face.pos).
        label and index in the batch's order."""
        batch = {"label": torch.tensor([item["label"] for item in items]),
                 "index": torch.tensor([item["index"] for item in items])}
        for part in PARTS:
            positions = [j for j, item in enumerate(items) if item["part"] == part]
            if not positions:
                continue
            batch[f"{part}.pos"] = torch.tensor(positions)
            for k in (0, 1):
                for name, value in super().collate([items[j]["clips"][k] for j in positions]).items():
                    batch[f"{part}.{k}.{name}"] = value
        return batch


class PairClassifier:
    """Before a model's Model: its backbone, a head per part on both clips' pooled features.

    A micro-batch with one part leaves the other part's encoder and head unused (dynamic_graph).
    """

    def __init__(self, config: dict[str, Any], n_classes: int):
        super().__init__(config, n_classes)
        head = config.get("head") or {}
        self.head = nn.ModuleDict({part: make_head(head, 4 * self.dim, n_classes) for part in PARTS})
        self.dynamic_graph = True

    def encode_part(self, part: str, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """One clip's per-step features (B, T, D) and valid steps (B, T), through the part's encoder."""
        raise NotImplementedError

    def pooled(self, part: str, batch: dict[str, torch.Tensor], k: int) -> torch.Tensor:
        """Clip k's features (B, D), pooled over time and layer-normed (no weights): the encoders' feature sizes
        differ (some of PE-AV's are near 100), and a * b would square them."""
        x = masked_mean(*self.encode_part(part, clip_batch(batch, part, k)))
        return F.layer_norm(x, x.shape[-1:])

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """The questions' logits (B, C), in the batch's order."""
        logits, positions = [], []
        for part in PARTS:
            if f"{part}.pos" not in batch:
                continue
            a, b = (self.pooled(part, batch, k) for k in (0, 1))
            logits.append(self.head[part](torch.cat([a, b, (a - b).abs(), a * b], dim=-1)))
            positions.append(batch[f"{part}.pos"])
        return torch.cat(logits)[torch.cat(positions).argsort()], None
