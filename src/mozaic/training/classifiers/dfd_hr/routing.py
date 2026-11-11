"""DFD-HR building blocks: the MoE adapter, the token and layer routers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MoEAdapter(nn.Module):
    """Residual mixture of bottleneck experts, gated per token over the top k."""

    def __init__(self, width: int, n_experts: int, top_k: int, ratio: float) -> None:
        super().__init__()
        hidden = int(width * ratio)
        self.top_k = top_k
        self.gate = nn.Linear(width, n_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width)) for _ in range(n_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return x plus the gate-weighted sum of the chosen experts' outputs."""
        probs = F.softmax(self.gate(x), dim=-1)
        values, indices = probs.topk(self.top_k, dim=-1)
        gates = torch.zeros_like(probs).scatter(-1, indices, values / values.sum(dim=-1, keepdim=True))
        mixed = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            mixed = mixed + gates[..., index : index + 1] * expert(x)
        return x + mixed


class TokenRouter(nn.Module):
    """Linear importance score per token, computed in float32."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight_predictor = nn.Linear(width, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, L, D) tokens to (N, L) scores."""
        return self.weight_predictor(x.float()).squeeze(-1)


class LayerRouter(nn.Module):
    """Per-sample logit for continuing through a block, from the class token."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, 1, bias=False)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        """(N, D) class tokens to (N,) logits."""
        return self.fc2(F.relu(self.fc1(cls))).squeeze(-1)


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, eps: float = 1e-20) -> torch.Tensor:
    """Sigmoid of logits plus Gumbel noise, divided by tau."""
    uniform = torch.rand_like(logits)
    return torch.sigmoid((logits - torch.log(-torch.log(uniform + eps) + eps)) / tau)
