"""PE-AV (facebookresearch/perception_models) as a T1 and T2 backbone: per-step features, our head on top.

Loaded through transformers (PeAudioVideoModel, models/pe_audio_video).

Taken as released
    model, weights   PeAudioVideoModel and facebook/pe-av-large, through from_pretrained
    t1 features      its audio-video encoder (audio_video_encoder): the audio encoder (DAC-VAE
                     tokens, 25 per second, and a Transformer), the video encoder (PE-L per frame
                     and a Transformer over the frames), the frames aligned to the audio tokens
                     (nearest), joined per token, and a joint Transformer. Its per-token output
    t2 features      its video encoder alone (the same weights as the one inside the audio-video
                     encoder): per-frame output
    input format     preprocess.py, as its processor: RGB 336, mean 0.5, std 0.5; 48 kHz waveform
Dropped
    text, heads      the text encoder and the contrastive heads: our head on the mean feature
"""

from __future__ import annotations

from typing import Any

import torch

from mozaic.training.base import BaseClassifier
from mozaic.training.classifiers.pe_av.preprocess import HOP, Processor
from mozaic.training.common.data import MODALITIES


class Model(BaseClassifier):
    processor = Processor

    def build_backbone(self, config: dict[str, Any]) -> int:
        from transformers import PeAudioVideoModel

        released = PeAudioVideoModel.from_pretrained(config["backbone"]["weights"], dtype=torch.float32)
        self.sound = "audio" in MODALITIES[config["task"]]
        encoder = released.audio_video_encoder
        self.backbone = encoder if self.sound else encoder.embedder.video_encoder
        # The DAC-VAE runs under no_grad as released, so fully trained its weights get no gradient
        self.dynamic_graph = config["tuning"]["mode"] == "full"
        if config["backbone"].get("gradient_checkpointing"):  # memory only: the Transformers' layers and PE-L's blocks
            self.backbone.gradient_checkpointing_enable({"use_reentrant": False})
            for m in self.backbone.modules():
                if hasattr(m, "set_grad_checkpointing"):  # timm (PE-L)
                    m.set_grad_checkpointing(True)
        return self.backbone.config.hidden_size

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.sound:
            return self.backbone(pixel_values_videos=batch["video"]).last_hidden_state, batch["valid"]
        b, steps, _ = batch["audio"].shape
        out = self.backbone(input_values=batch["audio"].reshape(b, 1, steps * HOP),
                            pixel_values_videos=batch["video"],
                            padding_mask=batch["valid"].repeat_interleave(HOP, dim=1))
        return out.last_hidden_state, batch["valid"]

    def tunable(self) -> dict[str, Any]:
        return {"backbone": self.backbone, "new": []}
