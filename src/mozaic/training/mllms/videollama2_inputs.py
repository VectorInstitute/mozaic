"""VideoLLaMA2-AV's training inputs and validation answers, through its test code (models/mllms/videollama2_av): the
same frames, soundtrack and prompt by construction. It answers in float16, so it trains its LoRA under float16 autocast."""

from __future__ import annotations

from typing import Any

import torch

from mozaic.evaluation.prompts import t4_prompts
from mozaic.evaluation.run_inference import read_clips
from mozaic.training.mllms.lora_merge import merge  # noqa: F401  (merge.py takes it from here)

PLACEHOLDERS, END_OF_TURN = "<video><audio>", "<|im_end|>\n"
LAYERS = r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
DTYPE, SCALE_LOSS = torch.float16, True
TESTED = None  # the test code's Model, set by native_train.run


def question(tested, row: dict[str, Any]):
    clips = read_clips({"clips": [row["videos"][0]]}, t4_prompts, tested.config)
    return clips, row["messages"][0]["content"].replace(PLACEHOLDERS, "", 1)


def load_template(config: dict[str, Any]):
    return None, TESTED.tokenizer, None


def training_inputs(tested, row: dict[str, Any]) -> dict[str, Any]:
    """A row's prompt, as the test run builds it, then its reply; only the reply is learned."""
    from videollama2.mm_utils import tokenizer_multimodal_token

    from mozaic.evaluation.models.mllms.videollama2_av.model import message

    clips, prompt = question(tested, row)
    text, tag = message(clips, prompt)
    data, modal = tested.clip_input(clips[0])
    images = [({k: v.half().cuda() for k, v in data.items()}, modal)]
    chat = tested.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer_multimodal_token(chat, tested.tokenizer, tag, return_tensors="pt").unsqueeze(0)
    reply = tested.tokenizer(row["messages"][1]["content"] + END_OF_TURN, add_special_tokens=False, return_tensors="pt").input_ids
    labels = torch.cat([torch.full_like(ids, -100), reply], dim=1).cuda()
    ids = torch.cat([ids, reply], dim=1).cuda()
    return {"input_ids": ids, "attention_mask": ids.ne(tested.tokenizer.pad_token_id).long(), "labels": labels, "images": images}


def answer(model, template, tokenizer, processor, example: dict[str, Any], generate_args: dict[str, Any]) -> str:
    with torch.autocast("cuda", dtype=DTYPE):
        return TESTED.generate_clips(*question(TESTED, example)).raw
