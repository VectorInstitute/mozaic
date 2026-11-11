"""The protocol's clip (32 frames, complete audio) in AV-HuBERT's input format.

AV-HuBERT joins audio and video step by step, one step per frame of the clip. So the input has
one step per frame: the 32 sampled frames sit at their own indices, the other steps are empty
(0 after normalisation), and the complete audio fills every step. A task without sound (t2)
gets the video only.

    video  grayscale (its first conv takes 1 channel), resized to 88 x 88, scaled to [0, 1],
           normalised by mean 0.421 and std 0.165 (hubert_pretraining.py)
    audio  log filterbank, 26 bins at 100 per second, 4 stacked per step (104), trimmed or padded
           to the clip's frame count, then layer norm per step (hubert_dataset.py)
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import logfbank

from mozaic.training.base import BaseProcessor

SIZE = 88
MEAN, STD = 0.421, 0.165
STACK = 4


def video_input(frames: list, indices: list[int], total: int) -> np.ndarray:
    """(total, 88, 88): each sampled frame at its index, the rest empty."""
    video = np.zeros((total, SIZE, SIZE), dtype=np.float32)
    for frame, i in zip(frames, indices, strict=True):
        gray = np.asarray(frame.convert("L").resize((SIZE, SIZE)), dtype=np.float32) / 255.0
        video[i] = (gray - MEAN) / STD
    return video


def audio_input(audio: np.ndarray, rate: int, total: int) -> np.ndarray:
    """(total, 104): the clip's filterbank, 4 steps stacked per frame."""
    feats = logfbank(audio, samplerate=rate).astype(np.float32)
    if len(feats) % STACK:
        feats = np.concatenate([feats, np.zeros((STACK - len(feats) % STACK, feats.shape[1]), np.float32)])
    feats = feats.reshape(-1, STACK * feats.shape[1])
    if len(feats) < total:
        feats = np.concatenate([feats, np.zeros((total - len(feats), feats.shape[1]), np.float32)])
    return feats[:total]


class Processor(BaseProcessor):
    time_dims: ClassVar[dict[str, int]] = {"video": 1, "audio": 1}

    def __call__(self, frames: list, indices: list[int], total: int, audio: np.ndarray | None) -> dict[str, Any]:
        item = {"video": torch.from_numpy(video_input(frames, indices, total)).unsqueeze(0),  # (1, T, 88, 88)
                "length": total}
        if audio is not None:
            feats = torch.from_numpy(audio_input(audio, self.config["audio_sample_rate"], total))
            item["audio"] = F.layer_norm(feats, feats.shape[1:]).T.contiguous()  # (104, T)
        return item
