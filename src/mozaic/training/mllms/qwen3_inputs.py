"""Qwen3-Omni's inputs, built with its own processor from the clips its test code reads (qwen_omni_vllm), not through
LLaMA-Factory: its video-with-sound layout interleaves audio tokens between frames, which LLaMA-Factory's plugin does not."""

from __future__ import annotations

from typing import Any

import torch

from mozaic.evaluation.base import ClipInput
from mozaic.evaluation.models.mllms.qwen_omni_vllm.model import message, use_qwen_omni_utils
from mozaic.training.mllms.lora_merge import merge as merge_lora

PLACEHOLDERS, END_OF_TURN = "<video><audio>", "<|im_end|>\n"
_read = {}


def load_template(config: dict[str, Any]):
    from transformers import AutoProcessor

    _read["info"] = use_qwen_omni_utils(config)
    _read["config"] = config
    processor = AutoProcessor.from_pretrained(config["weights"])
    return None, processor.tokenizer, processor


def encode(processor, row: dict[str, Any], reply: str = "") -> dict[str, torch.Tensor]:
    """A row's model inputs as the test run builds them, with the reply after the prompt if given (batch of one)."""
    from transformers import ProcessorMixin

    config = _read["config"]
    clip = ClipInput(path=row["videos"][0], send_frames=True, send_audio=True)
    msgs, sound_in_video = message([clip], row["messages"][0]["content"].replace(PLACEHOLDERS, "", 1), config["frame_size"])
    text = ProcessorMixin.apply_chat_template(processor, msgs, add_generation_prompt=True, tokenize=False) + reply
    audios, _, videos, kwargs = _read["info"](msgs, use_audio_in_video=sound_in_video, return_video_kwargs=True)
    pixels = config["frame_size"] ** 2  # keeps the clips' own size, as the test run's size setting does
    return dict(processor(text=[text], audio=audios, videos=videos, return_tensors="pt", padding=True, fps=kwargs["fps"][0],
                          use_audio_in_video=sound_in_video, size={"shortest_edge": pixels, "longest_edge": pixels}))


def training_example(processor, row: dict[str, Any]) -> dict[str, torch.Tensor]:
    reply = row["messages"][1]["content"] + END_OF_TURN
    inputs = encode(processor, row, reply)
    labels = inputs["input_ids"].clone()
    count = len(processor.tokenizer(reply, add_special_tokens=False)["input_ids"])
    labels[:, :-count] = -100
    return {**inputs, "labels": labels}


def answer(model, template, tokenizer, processor, example: dict[str, Any], generate_args: dict[str, Any]) -> str:
    inputs = {k: v.to(model.device) for k, v in encode(processor, example).items()}
    args = {k: v for k, v in generate_args.items() if k != "use_audio_in_video"}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(**inputs, use_audio_in_video=True, use_cache=True, **args)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def merge(weights, adapter, dest) -> None:
    merge_lora(weights, adapter, dest, prefix="thinker.")
