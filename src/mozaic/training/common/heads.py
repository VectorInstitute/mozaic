"""Pooling and the classification head, the same for every model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over the valid time steps only: (B, T, D), (B, T) -> (B, D)."""
    w = valid.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(1) / w.sum(1).clamp(min=1)


def make_head(config: dict[str, Any], dim: int, n_classes: int) -> nn.Module:
    """mlp (default): one hidden layer of hidden units (default: the feature size). Or linear."""
    kind = config.get("type", "mlp")
    dropout = config.get("dropout", 0.0)
    if kind == "linear":
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))
    if kind == "mlp":
        hidden = config.get("hidden", dim)
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(),
                             nn.Dropout(dropout), nn.Linear(hidden, n_classes))
    raise ValueError(f"unknown head {kind}, expected linear or mlp")
