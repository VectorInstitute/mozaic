"""Qwen2.5-Omni's inputs in LLaMA-Factory, built exactly as evaluation builds them.

LLaMA-Factory's own video reader floors the frame indices and gives every clip one fps, which places
the sound next to the wrong frames for one of the two clip lengths. Our plugin reads each clip with
evaluation's reader instead (qwen_omni_vllm, qwen_omni_utils 0.0.9). check_inputs.py verifies the result.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import numpy as np
from llamafactory.data import template as lf_template
from llamafactory.data.mm_plugin import Qwen2OmniPlugin, register_mm_plugin

from mozaic.evaluation.base import ClipInput
from mozaic.evaluation.models.mllms.qwen_omni_vllm.model import message, use_qwen_omni_utils
from mozaic.training.mllms.run_config import with_sound

TEMPLATE = "mozaic_qwen2_omni"  # LLaMA-Factory's qwen2_omni with our plugin


class EvalReaderPlugin(Qwen2OmniPlugin):
    """Qwen2OmniPlugin as released, but clips read by evaluation's reader."""

    process_mm_info = None  # set by register()
    frame_size = None
    sound = None

    def read(self, path: str) -> tuple[Any, float, Any]:
        """(frames, their frame rate, the sound or None), as evaluation reads the clip for this task."""
        clip = ClipInput(path=str(path), send_frames=True, send_audio=self.sound)
        msgs, sound_in_video = message([clip], "", self.frame_size)
        audios, _, videos, kwargs = self.process_mm_info(msgs, use_audio_in_video=sound_in_video,
                                                         return_video_kwargs=True)
        return videos[0], kwargs["fps"][0], audios[0] if audios else None

    def read_sound(self, path: str) -> Any:
        """The sound evaluation sends of this clip: from inside its video, or alone (t3's voice questions)."""
        if self.sound:
            return self.read(path)[2]
        msgs, _ = message([ClipInput(path=str(path), send_audio=True)], "", self.frame_size)
        return self.process_mm_info(msgs, use_audio_in_video=False)[0][0]

    def _regularize_videos(self, videos: list, **kwargs) -> dict[str, list]:
        read = [self.read(v) for v in videos]
        frames, fps = [r[0] for r in read], [r[1] for r in read]
        return {"videos": frames, "fps_per_video": fps, "durations": [len(v) / f for v, f in zip(frames, fps)],
                "frames_indices": [list(range(len(v))) for v in frames]}

    def _regularize_audios(self, audios: list, sampling_rate: float, **kwargs) -> dict[str, list]:
        # An array is the collator's masked dummy sound for a batch without any (t2, t3's face): passed as it is
        results = [a if isinstance(a, np.ndarray) else self.read_sound(a) for a in audios]
        return {"audios": results, "sampling_rates": [sampling_rate] * len(results)}


def register(config: dict[str, Any]) -> str:
    """Adds our template to LLaMA-Factory (once per process); returns its name."""
    if TEMPLATE in lf_template.TEMPLATES:
        return TEMPLATE
    EvalReaderPlugin.process_mm_info = staticmethod(use_qwen_omni_utils(config))
    EvalReaderPlugin.frame_size = config["frame_size"]
    EvalReaderPlugin.sound = with_sound(config["task"])
    register_mm_plugin(TEMPLATE, EvalReaderPlugin)
    base = lf_template.TEMPLATES[config.get("base_template", "qwen2_omni")]  # Qwen3-Omni: qwen3_omni_nothink
    plugin = EvalReaderPlugin(**{f.name: getattr(base.mm_plugin, f.name) for f in fields(base.mm_plugin) if f.init})
    lf_template.TEMPLATES[TEMPLATE] = replace(base, mm_plugin=plugin)
    return TEMPLATE


def load_template(config: dict[str, Any]):
    """(template, tokenizer, processor), as LLaMA-Factory's SFT loads them."""
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer

    model_args, data_args, _, _ = get_infer_args({"model_name_or_path": config["weights"],
                                                  "template": register(config),
                                                  "use_audio_in_video": with_sound(config["task"])})
    loaded = load_tokenizer(model_args)
    return get_template_and_fix_tokenizer(loaded["tokenizer"], data_args), loaded["tokenizer"], loaded["processor"]


def model_inputs(template, tokenizer, processor, example: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    """A dataset row's prompt token ids (up to "assistant\\n") and its clip tensors, as in training."""
    audios = example.get("audios", [])
    msgs = template.mm_plugin.process_messages(example["messages"], [], example["videos"], audios, processor)
    prompt_ids, _ = template.encode_oneturn(tokenizer, msgs)
    return prompt_ids, template.mm_plugin._get_mm_inputs([], example["videos"], audios, processor)
