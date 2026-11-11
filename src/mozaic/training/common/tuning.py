"""Which parameters train, set by the config's tuning mode.

    frozen      head only
    layernorm   head + every LayerNorm (or RMSNorm, its place in newer Transformers) of the backbone
    full        everything
    lora        not yet

The head and the model's new parts (not in the checkpoint) train in every mode. A backbone
that is not fully trained stays in eval mode, so BatchNorm statistics, dropout and layer drop
do not change during training. A fully trained one updates its BatchNorm statistics, so they are saved too.
"""

from __future__ import annotations

from typing import Any

from torch import nn

MODES = ("frozen", "layernorm", "full", "lora")


def _params(parts: list[Any]):
    for part in parts:
        if isinstance(part, nn.Parameter):
            yield part
        else:
            yield from part.parameters()


def is_norm(m: nn.Module) -> bool:
    """A LayerNorm, or an RMSNorm: torch's, or a model's own class (transformers defines one per model)."""
    return isinstance(m, nn.LayerNorm) or type(m).__name__.endswith("RMSNorm")


def apply(model: nn.Module, mode: str) -> dict[str, int]:
    """Set requires_grad by mode. Returns the trainable and total parameter counts."""
    if mode not in MODES:
        raise ValueError(f"unknown tuning mode {mode}, expected one of {MODES}")
    if mode == "lora":
        raise NotImplementedError("tuning mode lora is not implemented yet")
    parts = model.tunable()
    backbone = parts["backbone"]
    for p in backbone.parameters():
        p.requires_grad = mode == "full"
    if mode == "layernorm":
        for m in backbone.modules():
            if is_norm(m):
                for p in m.parameters():
                    p.requires_grad = True
    for p in _params(parts.get("new", [])):
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    return {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters())}


def train_mode(model: nn.Module, mode: str) -> None:
    """model.train(), with the backbone in eval mode unless it is fully trained."""
    model.train()
    if mode != "full":
        model.tunable()["backbone"].eval()


def trainable_state(model: nn.Module, mode: str) -> dict[str, Any]:
    """What best.pt holds next to the original checkpoint: the trained weights and, when the backbone is fully
    trained (in training mode), the buffers it updates (BatchNorm statistics)."""
    state = model.state_dict()
    keep = {name for name, p in model.named_parameters() if p.requires_grad}
    if mode == "full":
        keep |= {name for name, _ in model.named_buffers()} & state.keys()  # persistent ones only
    return {name: state[name].detach().cpu() for name in keep}
