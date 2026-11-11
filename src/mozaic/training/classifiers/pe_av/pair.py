"""PE-AV for T3: two clips per question, each through one of its encoders, compared by our pair head
(common/pairs.py). model.py and preprocess.py as they are, for T1 and T2.

Taken as released
    face questions   its video encoder alone, as T2 (PE-L per frame and a Transformer over the frames): per-frame output
    voice questions  its audio encoder alone (the DAC-VAE tokens, 25 per second, and a Transformer): per-token
                     output, the class token dropped as in the audio-video encoder. The same weights as inside
                     the audio-video encoder
    input format     preprocess.py: the frames as T2's; the 48 kHz waveform as T1's, without the frames
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from mozaic.training.classifiers.pe_av import model, preprocess
from mozaic.training.classifiers.pe_av.preprocess import HOP, audio_input
from mozaic.training.common.pairs import PairClassifier, PairProcessor


class Processor(PairProcessor, preprocess.Processor):
    def voice(self, audio: np.ndarray) -> dict[str, Any]:
        steps = audio_input(audio)
        return {"audio": steps, "length": len(steps)}


class Model(PairClassifier, model.Model):
    processor = Processor

    def build_backbone(self, config: dict[str, Any]) -> int:
        from transformers import PeAudioVideoModel

        released = PeAudioVideoModel.from_pretrained(config["backbone"]["weights"], dtype=torch.float32)
        embedder = released.audio_video_encoder.embedder
        self.backbone = nn.ModuleDict({"face": embedder.video_encoder, "voice": embedder.audio_encoder})
        if config["backbone"].get("gradient_checkpointing"):  # memory only, as model.py
            for encoder in self.backbone.values():
                encoder.gradient_checkpointing_enable({"use_reentrant": False})
            for m in self.backbone.modules():
                if hasattr(m, "set_grad_checkpointing"):  # timm (PE-L)
                    m.set_grad_checkpointing(True)
        dims = {part: encoder.config.hidden_size for part, encoder in self.backbone.items()}
        if len(set(dims.values())) != 1:
            raise ValueError(f"the video and audio encoders' feature sizes differ: {dims}")
        return dims["face"]

    def encode_part(self, part: str, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if part == "face":
            return self.backbone["face"](pixel_values_videos=batch["video"]).last_hidden_state, batch["valid"]
        b, steps, _ = batch["audio"].shape
        out = self.backbone["voice"](input_values=batch["audio"].reshape(b, 1, steps * HOP),
                                     padding_mask=batch["valid"].repeat_interleave(HOP, dim=1))
        return out.last_hidden_state, batch["valid"]

    def tunable(self) -> dict[str, Any]:
        return {"backbone": self.backbone, "new": []}
