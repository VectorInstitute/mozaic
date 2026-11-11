"""OpenAI CLIP visual tower whose attention takes an additive bias.

Mirrors CLIP's VisionTransformer, parameter names included, so the released
ViT-L/14 weights load unchanged. Tokens are kept batch-first, (N, L, D). The
blocks expose their sublayers, so a method can insert adapters between them.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class QuickGELU(nn.Module):
    """The GELU approximation CLIP was trained with."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply x * sigmoid(1.702 x)."""
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Pre-norm self-attention and MLP; attention accepts an additive bias."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        # Holds the released projection weights; attention itself runs in attend.
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [("c_fc", nn.Linear(width, width * 4)), ("gelu", QuickGELU()), ("c_proj", nn.Linear(width * 4, width))]
            )
        )
        self.ln_2 = nn.LayerNorm(width)

    def attend(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        """Multi-head attention over (N, L, D); bias is (N, heads, L, L) or None."""
        batch, length, width = x.shape
        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.reshape(batch, length, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        if bias is None:
            attended = F.scaled_dot_product_attention(q, k, v)
        else:
            # Written out: the fused kernels cannot back-propagate into a per-head bias.
            weights = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5 + bias.to(q.dtype)
            attended = weights.softmax(dim=-1).to(v.dtype) @ v
        return self.attn.out_proj(attended.transpose(1, 2).reshape(batch, length, width))

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Apply attention and MLP with residual connections."""
        x = x + self.attend(self.ln_1(x), bias)
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    """The block stack, named as in the released weights."""

    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(ResidualAttentionBlock(width, heads) for _ in range(layers))


class CLIPVisionTransformer(nn.Module):
    """CLIP ViT: patch embedding, class token, blocks, final norm and projection."""

    def __init__(self, image_size: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int) -> None:
        super().__init__()
        scale = width**-0.5
        self.heads = heads
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((image_size // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @property
    def blocks(self) -> nn.ModuleList:
        """The residual attention blocks."""
        return self.transformer.resblocks

    def embed(self, pixels: torch.Tensor) -> torch.Tensor:
        """Patch and class tokens with positions, after ln_pre, (N, 1 + L, D)."""
        x = self.conv1(pixels).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        return self.ln_pre(torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype))


def load_openai_weights(visual: CLIPVisionTransformer, path: str) -> None:
    """Load the released ViT-L/14 visual weights (ViT-L-14.pt, a TorchScript archive)."""
    archive = torch.jit.load(path, map_location="cpu")  # type: ignore[no-untyped-call]
    state = {k.removeprefix("visual."): v.float() for k, v in archive.state_dict().items() if k.startswith("visual.")}
    visual.load_state_dict(state)
