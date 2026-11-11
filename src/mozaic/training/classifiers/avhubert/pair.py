"""AV-HuBERT for T3: two clips per question, each through one of its branches, compared by our pair head
(common/pairs.py). model.py and preprocess.py as they are, for T1 and T2.

Taken as released
    face questions   extract_finetune with the video alone, as T2: the audio features zero, as when
                     pretraining dropped the audio
    voice questions  extract_finetune with the audio alone: the video features zero, as when pretraining
                     dropped the video
    input format     preprocess.py: the frames as T2's; the filterbank as T1's, one step per 4 filterbank
                     frames (25 per second). With no frames to fit it to, every step of the audio is kept
                     (T1 fits it to the clip's frame count, which is the same at 25 fps)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mozaic.training.classifiers.avhubert import model, preprocess
from mozaic.training.classifiers.avhubert.preprocess import audio_input
from mozaic.training.common.pairs import PairClassifier, PairProcessor

STEPS_PER_SECOND = 25  # AV-HuBERT's audio steps: 100 filterbank frames per second, 4 per step


class Processor(PairProcessor, preprocess.Processor):
    def voice(self, audio: np.ndarray) -> dict[str, Any]:
        rate = self.config["audio_sample_rate"]
        total = -(-len(audio) * STEPS_PER_SECOND // rate)  # the audio's own steps
        feats = torch.from_numpy(audio_input(audio, rate, total))
        return {"audio": F.layer_norm(feats, feats.shape[1:]).T.contiguous(), "length": total}  # (104, T), as T1's


class Model(PairClassifier, model.Model):
    processor = Processor

    def encode_part(self, part: str, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        source = {"audio": None, "video": batch["video"]} if part == "face" else {"audio": batch["audio"], "video": None}
        x, padding = self.backbone.extract_finetune(source, padding_mask=~batch["valid"], mask=False)
        return x, batch["valid"] if padding is None else ~padding
