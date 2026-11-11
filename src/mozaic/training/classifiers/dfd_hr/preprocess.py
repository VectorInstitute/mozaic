"""The protocol's clip (32 frames, full frame) in DFD-HR's input format: (32, 3, 448, 448) in [-1, 1].

448 is one 224 global view and 2 x 2 tiles of 224. Resized as in deepfake-moe (bicubic, antialias).
Video only: DFD-HR takes no sound.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from mozaic.training.base import BaseProcessor

SIZE = 448


class Processor(BaseProcessor):
    time_dims: ClassVar[dict[str, int]] = {"frames": 0}

    def __call__(self, frames: list, indices: list[int], total: int, audio: np.ndarray | None) -> dict[str, Any]:
        x = torch.from_numpy(np.stack([np.asarray(f.convert("RGB")) for f in frames]))
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bicubic", align_corners=False, antialias=True).clamp_(0.0, 1.0)
        return {"frames": x.sub_(0.5).div_(0.5), "length": len(frames)}
