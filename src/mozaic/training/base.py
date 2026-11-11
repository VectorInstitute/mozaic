"""The contract every trained classifier follows."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from mozaic.training.common.heads import make_head, masked_mean


class BaseProcessor:
    """Turns one clip into the model's input, in the DataLoader workers. No weights here.

    The clip comes as every model gets it (common/frame_sampler.py): the sampled frames, their
    indices in the clip, the clip's frame count, and the complete audio (None when the task
    has no sound, t2). time_dims names each tensor's time axis, so collate pads it to the
    longest clip in the batch; a name missing from an item is not padded.
    """

    time_dims: ClassVar[dict[str, int]] = {}

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def __call__(self, frames: list, indices: list[int], total: int, audio: np.ndarray | None) -> dict[str, Any]:
        """{tensor name: tensor, ..., "length": time steps}."""
        raise NotImplementedError

    def collate(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pad every time axis to the batch's longest clip. valid (B, T) marks the real steps."""
        lengths = torch.tensor([item["length"] for item in items])
        longest = int(lengths.max())
        batch = {"valid": torch.arange(longest)[None, :] < lengths[:, None]}
        for name, dim in self.time_dims.items():
            if name not in items[0]:
                continue
            padded = []
            for item in items:
                x = item[name]
                shape = list(x.shape)
                shape[dim] = longest - x.shape[dim]
                padded.append(torch.cat([x, x.new_zeros(shape)], dim=dim))
            batch[name] = torch.stack(padded)
        for name, value in items[0].items():
            if name not in self.time_dims and name != "length":
                values = [item[name] for item in items]
                batch[name] = torch.stack(values) if torch.is_tensor(value) else torch.as_tensor(np.array(values))
        return batch


class BaseClassifier(nn.Module):
    """A backbone and our head.

    A model builds its backbone, encodes a batch into per-step features, and says which of
    its parts training may unfreeze (tunable). By default the features are pooled over time
    (masked mean), then the head gives the clip's logits. A model with per_step scores every
    step instead: the head runs on each step, the clip's logits are the mean of the steps'
    logits, and the loss is taken on every valid step with the clip's label (DFD-HR).
    """

    processor: type[BaseProcessor] = BaseProcessor
    dynamic_graph = False  # True when a step may skip trainable parts (DDP must then look for them)
    per_step = False  # True: the head and the loss on every step, the clip's logits their mean

    def __init__(self, config: dict[str, Any], n_classes: int):
        super().__init__()
        self.config = config
        self.dim = self.build_backbone(config)
        self.head = make_head(config.get("head") or {}, self.dim, n_classes)

    def build_backbone(self, config: dict[str, Any]) -> int:
        """Load the pretrained backbone; return its feature size."""
        raise NotImplementedError

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-step features (B, T, D) and the valid steps (B, T)."""
        raise NotImplementedError

    def tunable(self) -> dict[str, Any]:
        """{"backbone": the pretrained part, "new": parts not in the checkpoint (always trained)}."""
        raise NotImplementedError

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The clip's logits (B, C), and each step's (B, T, C) when per_step."""
        x, valid = self.encode(batch)
        if not self.per_step:
            return self.head(masked_mean(x, valid)), None
        steps = self.head(x)
        return masked_mean(steps, valid), steps
