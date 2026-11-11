"""The protocol's clip (32 frames, full frame; the complete audio) in PE-AV's input format.

    video  RGB, resized to 336 x 336 (bilinear), scaled to [0, 1], normalised by mean 0.5 and
           std 0.5 (video_preprocessor_config.json)
    audio  the complete audio at 48 kHz (the loader resamples the clips' 16 kHz), scaled to
           [-1, 1], reflect-padded to a multiple of 1920 samples (PeAudioFeatureExtractor), one
           step per 1920 samples: 25 per second, the rate of its DAC-VAE tokens
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image

from mozaic.training.base import BaseProcessor

SIZE = 336
MEAN, STD = 0.5, 0.5
RATE = 48000
HOP = 1920  # samples per audio token


def video_input(frames: list) -> torch.Tensor:
    """(T, 3, 336, 336), normalised."""
    x = np.stack([np.asarray(f.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)) for f in frames])
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.0)
    return (x - MEAN) / STD


def audio_input(audio: np.ndarray) -> torch.Tensor:
    """(steps, 1920): the clip's waveform, one row per audio token."""
    wave = audio / 32768.0
    if len(wave) % HOP:  # as PeAudioFeatureExtractor._reflect_pad
        wave = np.pad(wave, (0, HOP - len(wave) % HOP), mode="reflect")
    return torch.from_numpy(wave.astype(np.float32)).view(-1, HOP)


class Processor(BaseProcessor):
    time_dims: ClassVar[dict[str, int]] = {"audio": 0}

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        if config["audio_sample_rate"] != RATE:
            raise ValueError(f"PE-AV takes {RATE} Hz audio (its DAC-VAE), set audio_sample_rate: {RATE}")

    def __call__(self, frames: list, indices: list[int], total: int, audio: np.ndarray | None) -> dict[str, Any]:
        item = {"video": video_input(frames), "length": len(frames)}
        if audio is not None:
            item["audio"] = audio_input(audio)
            item["length"] = len(item["audio"])
        return item
