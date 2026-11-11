"""VITA-1.5's training inputs and validation answers, through its test code (models/mllms/vita_1_5): the same frames,
sound and prompt by construction. Its answers start with a mark (☜ for a text question), kept in the targets. The loss
reads the reply's positions only: logits over its 8,900-piece input would not fit beside the model."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from mozaic.evaluation.base import ClipInput
from mozaic.training.mllms.lora_merge import merge as merge_lora

PLACEHOLDERS, END_OF_TURN, MARK = "<video><audio>", "<|im_end|>\n", "☜"
LAYERS = r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
DTYPE, SCALE_LOSS = torch.float16, True
TESTED = None  # the test code's Model, set by native_train.run
_questions = 0


def question(row: dict[str, Any]) -> tuple[list[ClipInput], str]:
    clip = ClipInput(path=row["videos"][0], send_frames=True, send_audio=True)
    return [clip], row["messages"][0]["content"].replace(PLACEHOLDERS, "", 1)


def load_template(config: dict[str, Any]):
    return None, TESTED.tokenizer, None


def loss(tested, row: dict[str, Any]) -> torch.Tensor:
    global _questions
    ids, images, audios, _ = tested.inputs(*question(row))  # seeds torch for its sound step's noise, as the test run does
    _questions += 1
    torch.manual_seed(tested.config["seed"] + 1000 * (torch.distributed.get_rank() + 1) + _questions)  # LoRA dropout varies
    reply = tested.tokenizer(MARK + row["messages"][1]["content"] + END_OF_TURN, add_special_tokens=False,
                             return_tensors="pt").input_ids
    labels = torch.cat([torch.full_like(ids, -100), reply], dim=1).cuda()
    ids = torch.cat([ids, reply], dim=1).cuda()
    audios = {"audios": audios["audios"].half().cuda(), "lengths": audios["lengths"].half().cuda(),
              "lengths_for_llm": audios["lengths_for_llm"].cuda()}
    _, position, mask, _, embeds, labels = tested.model.prepare_inputs_labels_for_multimodal(
        ids, None, torch.ones_like(ids), None, labels, images.half().cuda(), audios, None)
    hidden = tested.model.get_model()(inputs_embeds=embeds, attention_mask=mask, position_ids=position, use_cache=False)[0]
    shifted, targets = hidden[:, :-1], labels[:, 1:]
    keep = targets != -100
    return torch.nn.functional.cross_entropy(tested.model.lm_head(shifted[keep]).float(), targets[keep])


def answer(model, template, tokenizer, processor, example: dict[str, Any], generate_args: dict[str, Any]) -> str:
    with torch.autocast("cuda", dtype=DTYPE):
        return TESTED.generate_clips(*question(example)).raw


def merge(weights, adapter, dest) -> None:
    merge_lora(weights, adapter, dest)
    root = Path(weights) / ".cache" / "huggingface" / "download"  # the test code's settings read the hub revision from it
    record = sorted(root.rglob("*.metadata"))[0]
    target = Path(dest) / ".cache" / "huggingface" / "download" / record.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(record, target)
