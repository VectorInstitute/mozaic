"""MiniCPM-o 4.5's inputs in LLaMA-Factory, built as its T4 test run builds them (minicpm_o_vllm).

The same 32 frames (OpenCV, evenly spaced), the same prompt layout (frames, newline, sound, newline, prompt),
no system prompt, thinking off. One difference on purpose: the sound is in [-1, 1]. vLLM 0.28's reader gives this
data's 16-bit sound unscaled, which shifts every audio feature; the trained test run (normalize_audio) corrects it too.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from llamafactory.data import template as lf_template
from llamafactory.data.formatter import StringFormatter
from llamafactory.data.mm_plugin import MiniCPMVPlugin, register_mm_plugin

TEMPLATE = "mozaic_minicpm_o"  # LLaMA-Factory's minicpm_o with our plugin
USER = "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"  # enable_thinking: false
PLACEHOLDERS, LAID_OUT = "<video><audio>", "<video>\n<audio>\n"
SAMPLE_RATE = 16000
AUDIO, AUDIO_CHUNK = re.compile(r"<\|audio_start\|>((?:<unk>)+)<\|audio_end\|>"), 10  # sound tokens per pair of marks in the test run


def in_chunks(match: re.Match) -> str:
    """One clip's sound tokens as the test run writes them: every 10 between their own start and end marks."""
    count = match.group(1).count("<unk>")
    return "".join(f"<|audio_start|>{'<unk>' * min(AUDIO_CHUNK, count - i)}<|audio_end|>" for i in range(0, count, AUDIO_CHUNK))


def read_frames(path: str, frames: int) -> list:
    """vLLM's OpenCV reader: frames evenly spaced over the clip, as RGB."""
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted, out, i = set(np.linspace(0, total - 1, frames, dtype=int).tolist()), [], 0
    while capture.grab():
        if i in wanted:
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError(f"{path}: frame {i} could not be read")
            out.append(Image.fromarray(frame[:, :, ::-1].copy()))
        i += 1
    capture.release()
    if len(out) != frames:
        raise ValueError(f"{path}: {len(out)} frames read, not {frames}")
    return out


def read_sound(path: str) -> np.ndarray:
    import librosa

    return librosa.load(path, sr=SAMPLE_RATE, mono=True)[0]


class EvalReaderPlugin(MiniCPMVPlugin):
    frames = None  # set by register()

    def _regularize_videos(self, videos: list, **kwargs) -> dict[str, list]:
        return {"videos": [read_frames(v, self.frames) for v in videos]}

    def _regularize_audios(self, audios: list, sampling_rate: float, **kwargs) -> dict[str, list]:
        return {"audios": [a if isinstance(a, np.ndarray) else read_sound(a) for a in audios],
                "sampling_rates": [sampling_rate] * len(audios)}

    def process_messages(self, messages, images, videos, audios, processor):
        messages = [{**m, "content": m["content"].replace(PLACEHOLDERS, LAID_OUT, 1)} for m in messages]
        out = super().process_messages(messages, images, videos, audios, processor)
        return [{**m, "content": AUDIO.sub(in_chunks, m["content"])} for m in out]


def register(config: dict[str, Any]) -> str:
    """Adds our template to LLaMA-Factory (once per process); returns its name."""
    if TEMPLATE in lf_template.TEMPLATES:
        return TEMPLATE
    EvalReaderPlugin.frames = config["frames"]
    register_mm_plugin(TEMPLATE, EvalReaderPlugin)
    base = lf_template.TEMPLATES["minicpm_o"]
    plugin = EvalReaderPlugin(**{f.name: getattr(base.mm_plugin, f.name) for f in fields(base.mm_plugin) if f.init})
    lf_template.TEMPLATES[TEMPLATE] = replace(base, mm_plugin=plugin, default_system="",
                                              format_user=StringFormatter(slots=[USER]))
    return TEMPLATE


def patch(config: dict[str, Any]) -> None:
    """The model's own code on transformers 4.57.6: what its Whisper layers and forward expect."""
    from transformers.cache_utils import DynamicCache
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.whisper.modeling_whisper import WhisperAttention

    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())  # its generate reads the attribute 4.57 removed
    attention = WhisperAttention.forward

    def attention_forward(self, *args, past_key_value=None, **kwargs):
        if past_key_value is not None:
            kwargs["past_key_values"] = past_key_value
        out = attention(self, *args, **kwargs)
        return (*out, past_key_value) if len(out) == 2 else out

    WhisperAttention.forward = attention_forward
    model = get_class_from_dynamic_module("modeling_minicpmo.MiniCPMO", config["weights"])
    forward = model.forward

    def model_forward(self, data=None, **kwargs):  # data holds the inputs; the collator's copies would clash
        if not getattr(self, "_embeddings_cloned", False):  # after LLaMA-Factory's own hook for gradient checkpointing,
            # which makes the embeddings a leaf that requires grad: the model writes the frames into them in place
            self.llm.model.embed_tokens.register_forward_hook(lambda module, inputs, output: output.clone())
            self._embeddings_cloned = True
        for key in ("input_ids", "inputs_embeds", "position_ids"):
            kwargs.pop(key, None)
        if "attention_mask" not in kwargs and data is not None and "attention_mask" in data:
            kwargs["attention_mask"] = data["attention_mask"]
        return forward(self, data, **kwargs)

    model.forward = model_forward


def load_template(config: dict[str, Any]):
    """(template, tokenizer, processor), as LLaMA-Factory's SFT loads them."""
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer

    model_args, data_args, _, _ = get_infer_args({"model_name_or_path": config["weights"], "template": register(config),
                                                  "trust_remote_code": True})
    loaded = load_tokenizer(model_args)
    return get_template_and_fix_tokenizer(loaded["tokenizer"], data_args), loaded["tokenizer"], loaded["processor"]


def model_inputs(template, tokenizer, processor, example: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    """A dataset row's prompt token ids (up to the empty think block) and its clip's inputs, as in training."""
    videos, audios = example["videos"], example["audios"]
    msgs = template.mm_plugin.process_messages(example["messages"], [], videos, audios, processor)
    prompt_ids, _ = template.encode_oneturn(tokenizer, msgs)
    return prompt_ids, template.mm_plugin.get_mm_inputs([], videos, audios, [0], [len(videos)], [len(audios)],
                                                        [prompt_ids], processor)


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return [to_device(v, device) for v in value] if isinstance(value, list) else value


def answer(model, template, tokenizer, processor, example: dict[str, Any], generate_args: dict[str, Any]) -> str:
    """The reply to one question: the model's own generate, which takes its clips as lists per question."""
    prompt_ids, mm = model_inputs(template, tokenizer, processor, example)
    ids = torch.tensor([prompt_ids], device=model.device)
    mm = {k: to_device(v, model.device) for k, v in mm.items() if k != "image_sizes"}
    args = {k: v for k, v in generate_args.items() if k != "use_audio_in_video"}
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, out = base.generate(input_ids=ids, attention_mask=torch.ones_like(ids), tokenizer=tokenizer, use_cache=True,
                              **mm, **args)
    return tokenizer.decode(out.sequences[0], skip_special_tokens=True)


def merge(weights: str, adapter: Path, dest: Path) -> None:
    """The LoRA written into a copy of the released files, so its own code and TTS weights stay as released."""
    from safetensors.torch import load_file, save_file

    settings = json.loads((adapter / "adapter_config.json").read_text())
    scale = settings["lora_alpha"] / settings["r"]
    lora = load_file(adapter / "adapter_model.safetensors")
    pairs = {k[len("base_model.model."):-len(".lora_A.weight")]: k for k in lora if k.endswith(".lora_A.weight")}
    if not pairs or len(lora) != 2 * len(pairs):
        raise ValueError(f"{adapter}: {len(lora)} tensors, {len(pairs)} LoRA pairs")
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    done = set()
    for path in Path(weights).iterdir():
        if path.name.startswith("."):  # the download's own cache
            continue
        if path.suffix == ".safetensors":
            shard = load_file(path)
            for module in set(pairs) & {k[:-len(".weight")] for k in shard if k.endswith(".weight")}:
                a, b = lora[pairs[module]], lora[pairs[module].replace("lora_A", "lora_B")]
                name = f"{module}.weight"
                shard[name] = (shard[name].float() + scale * (b.float() @ a.float())).to(shard[name].dtype)
                done.add(module)
            save_file(shard, tmp / path.name, metadata={"format": "pt"})
        elif path.is_dir():
            shutil.copytree(path, tmp / path.name)
        else:
            shutil.copy(path, tmp / path.name)
    if done != set(pairs):
        raise ValueError(f"{len(set(pairs) - done)} LoRA layers not in the weights, e.g. {sorted(set(pairs) - done)[:3]}")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
