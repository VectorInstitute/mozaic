"""OmniVinci's training inputs and validation answers, through its test code (models/mllms/omnivinci): the same frames,
sound and prompt by construction. Its forward takes a different path in training (it re-extracts sound and packs
batches), so the clips go through _embed in eval, as in the test run, and the language model is called directly."""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from mozaic.evaluation.base import ClipInput
from mozaic.training.mllms.lora_merge import merge as merge_lora

PLACEHOLDERS, END_OF_TURN = "<video><audio>", "<|im_end|>\n"
LAYERS = r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
DTYPE, SCALE_LOSS = torch.float16, True
TESTED = None  # the test code's Model, set by native_train.run


def question(row: dict[str, Any]) -> tuple[list[ClipInput], str]:
    clip = ClipInput(path=row["videos"][0], send_frames=True, send_audio=True)
    return [clip], row["messages"][0]["content"].replace(PLACEHOLDERS, "", 1)


def load_template(config: dict[str, Any]):
    return None, TESTED.processor.tokenizer, None


def language_model(tested):
    return tested.model.llm


def train_layers(tested) -> None:
    tested.model.eval()  # its self.training picks the training path in _embed
    tested.model.llm.model.layers.train()
    tested.model.llm.model.training = True  # checkpointing reads the module's own flag


def loss(tested, row: dict[str, Any]) -> torch.Tensor:
    ids, media, _ = tested.inputs(*question(row))
    reply = tested.processor.tokenizer(row["messages"][1]["content"] + END_OF_TURN, add_special_tokens=False,
                                       return_tensors="pt").input_ids
    labels = torch.cat([torch.full_like(ids, -100), reply], dim=1).cuda()
    ids = torch.cat([ids, reply], dim=1).cuda()
    embeds, labels, mask = tested.model._embed(ids, media, defaultdict(dict), labels, torch.ones_like(ids, dtype=torch.bool))
    return tested.model.llm(inputs_embeds=embeds, attention_mask=mask, labels=labels, use_cache=False).loss


def answer(model, template, tokenizer, processor, example: dict[str, Any], generate_args: dict[str, Any]) -> str:
    with torch.autocast("cuda", dtype=DTYPE):
        return TESTED.generate_clips(*question(example)).raw


def merge(weights, adapter, dest) -> None:
    """Its language model's files, in llm/, take the LoRA; the rest is copied as released."""
    weights, dest = Path(weights), Path(dest)
    tmp = dest.with_name(dest.name + ".parts")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for path in weights.iterdir():
        if path.name in ("llm", ".cache"):
            continue
        if path.is_dir():
            shutil.copytree(path, tmp / path.name)
        else:
            shutil.copy(path, tmp / path.name)
    merge_lora(str(weights / "llm"), adapter, tmp / "llm")
    record = Path(".cache/huggingface/download/config.json.metadata")  # the test code's settings read the hub revision
    (tmp / record).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(weights / record, tmp / record)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
