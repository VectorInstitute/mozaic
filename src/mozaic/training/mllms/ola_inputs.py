"""Ola's training inputs and validation answers, through its test code (models/mllms/ola): the same frames, sound and
prompt by construction. The authors' model takes its clips as lists; the one loaded model is shared with validation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from mozaic.evaluation.base import ClipInput
from mozaic.training.mllms.lora_merge import merge as merge_lora

PLACEHOLDERS, END_OF_TURN = "<video><audio>", "<|im_end|>\n"
TESTED = None  # the test code's Model, set by ola_train.run


def question(row: dict[str, Any]) -> tuple[list[ClipInput], str]:
    clip = ClipInput(path=row["videos"][0], send_frames=True, send_audio=True)
    return [clip], row["messages"][0]["content"].replace(PLACEHOLDERS, "", 1)


def load_template(config: dict[str, Any]):
    return None, TESTED.tokenizer, None


def training_inputs(tested, row: dict[str, Any]) -> dict[str, Any]:
    """A row's prompt, as the test run builds it, then its reply; only the reply is learned. On the GPU, in bfloat16."""
    ids, media, _ = tested.inputs(*question(row))
    reply = tested.tokenizer(row["messages"][1]["content"] + END_OF_TURN, add_special_tokens=False, return_tensors="pt").input_ids
    labels = torch.cat([torch.full_like(ids, -100), reply], dim=1)
    ids = torch.cat([ids, reply], dim=1).cuda()
    for key in ("images", "speech"):
        media[key] = [t.to(device="cuda", dtype=torch.bfloat16) for t in media[key]]
    media["images_highres"] = media["images"]
    for key in ("speech_lengths", "speech_chunks", "speech_wav"):
        media[key] = [t.to("cuda") for t in media[key]]
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels.cuda(), **media}


def answer(model, template, tokenizer, processor, example: dict[str, Any], generate_args: dict[str, Any]) -> str:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return TESTED.generate_clips(*question(example)).raw


def merge(weights, adapter, dest) -> None:
    merge_lora(weights, adapter, dest)
    record = Path("config.json.metadata")  # the test code's settings read the weights' hub revision from it
    source = Path(weights) / ".cache" / "huggingface" / "download" / record
    target = Path(dest) / ".cache" / "huggingface" / "download" / record
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
