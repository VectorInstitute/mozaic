"""The protocol's clip (32 frames, full frame; the complete audio) in HAVIC's input format.

    video  RGB, resized to 224 x 224, scaled to [0, 1], no mean or std (HAVIC's dataloader)
    audio  the complete 16 kHz audio, mean removed, kaldi fbank (128 mel bins, 10 ms hop, hanning),
           resized along time to 1024 steps, normalised by HAVIC's dataset mean -6.996 and std 3.1205
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image

from mozaic.training.base import BaseProcessor

SIZE = 224
AUDIO_STEPS = 1024
MEL_BINS = 128
AUDIO_MEAN, AUDIO_STD = -6.9960, 3.1205


def video_input(frames: list) -> torch.Tensor:
    """(T, 3, 224, 224) in [0, 1]."""
    x = np.stack([np.asarray(f.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)) for f in frames])
    return torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.0)


def audio_input(audio: np.ndarray, rate: int) -> torch.Tensor:
    """(1024, 128): the clip's fbank, resized to 1024 steps whatever its length, normalised."""
    wave = torch.from_numpy(audio / 32768.0).float()[None]
    wave = wave - wave.mean()
    fbank = torchaudio.compliance.kaldi.fbank(wave, htk_compat=True, sample_frequency=rate, use_energy=False,
                                              window_type="hanning", num_mel_bins=MEL_BINS, dither=0.0, frame_shift=10)
    fbank = F.interpolate(fbank[None].transpose(1, 2), size=AUDIO_STEPS, mode="linear", align_corners=False)
    return (fbank.transpose(1, 2)[0] - AUDIO_MEAN) / AUDIO_STD


class Processor(BaseProcessor):
    time_dims: ClassVar[dict[str, int]] = {"frames": 0}

    def __call__(self, frames: list, indices: list[int], total: int, audio: np.ndarray | None) -> dict[str, Any]:
        return {"frames": video_input(frames), "audio": audio_input(audio, self.config["audio_sample_rate"]),
                "length": len(frames)}
