"""DFD-HR (Sun et al., CVPR 2026) as a T2 backbone: per-frame features, our head on top.

Each frame is seen twice: resized to one 224 global view, and cut into 2 x 2 tiles of 224. Both
pass through frozen CLIP ViT-L/14 with an MoE adapter after every attention and MLP. In the
routed blocks, a layer router lets a view stop early and a token router sends only the top-scoring
tokens through, re-weighted by their scores. Per frame, the projected global class token and a
query-attention summary of the four tile class tokens are the feature (1536). As released, each
frame is classified (our mlp head on its feature) and trained with its video's label; the clip's
logits are the mean of its frames' logits.

Ported from deepfake-moe, src/deepfake_moe/models/dfd_hr/ (DFDHR in model.py), which re-implements
Disguiser15/DFD-HR.

Taken as released
    clip.py, routing.py   CLIP ViT-L/14 and its weights loader; MoEAdapter, TokenRouter, LayerRouter,
                          gumbel_sigmoid (unchanged but for the weights download, done once instead)
    network               DFDHR.__init__ (adapters on all 24 blocks, routers on 21-23, layer routers
                          frozen, query token and attention), adapted, routed; DFDHR.encode is view_cls
    frame feature         DFDHR.frame_logits up to its features: global view, 2 x 2 tiles, query summary
    clip prediction       the mean of the frames' logits over the genuine frames (DFDHR.forward)
    loss                  CE on every genuine frame with its video's label (objective.py)
Dropped
    head                  DFDHR.head (linear 1536 → 2): our mlp on each frame's feature, 4 classes
    ranking term          0.1 x (1 - Spearman), with its guide argument and spearman_corr: in the
                          released code it comes from argsort ranks and has no gradient, so it changes
                          no weight
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from mozaic.training.base import BaseClassifier
from mozaic.training.classifiers.dfd_hr.clip import (
    CLIPVisionTransformer,
    ResidualAttentionBlock,
    load_openai_weights,
)
from mozaic.training.classifiers.dfd_hr.preprocess import Processor
from mozaic.training.classifiers.dfd_hr.routing import (
    LayerRouter,
    MoEAdapter,
    TokenRouter,
    gumbel_sigmoid,
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# DFD-HR's settings (deepfake-moe configs/models/dfd_hr/config.yaml); the config may override any.
DEFAULTS = {
    "clip_size": 224, "clip_patch": 14, "clip_width": 1024, "clip_layers": 24, "clip_heads": 16, "clip_dim": 768,
    "n_experts": 4, "top_k": 4, "expert_ratio": 0.25, "routed_blocks": [21, 22, 23], "capacity": 0.75,
    "router_hidden": 64, "query_heads": 4, "encoder_chunk": 32,
}


def tiles(x: torch.Tensor, n: int) -> torch.Tensor:
    """Cut (N, C, H, W) into n x n tiles, stacked tile-major along the batch."""
    batch, channels, height, width = x.shape
    x = x.reshape(batch, channels, n, height // n, n, width // n)
    return x.permute(2, 4, 0, 1, 3, 5).reshape(n * n * batch, channels, height // n, width // n)


class Model(BaseClassifier):
    processor = Processor
    dynamic_graph = True  # a layer router may stop a view, so some adapters get no gradient in a step
    per_step = True  # as released: CE on every frame with its video's label; clip = mean of frame logits

    def build_backbone(self, config: dict[str, Any]) -> int:
        self.cfg = c = {**DEFAULTS, **config["backbone"]}
        self.encoder = CLIPVisionTransformer(c["clip_size"], c["clip_patch"], c["clip_width"], c["clip_layers"],
                                             c["clip_heads"], c["clip_dim"])
        load_openai_weights(self.encoder, c["checkpoint"])
        width = c["clip_width"]
        self.adapters_attn = nn.ModuleList(
            MoEAdapter(width, c["n_experts"], c["top_k"], c["expert_ratio"]) for _ in range(c["clip_layers"]))
        self.adapters_mlp = nn.ModuleList(
            MoEAdapter(width, c["n_experts"], c["top_k"], c["expert_ratio"]) for _ in range(c["clip_layers"]))
        routed = [str(block) for block in c["routed_blocks"]]
        self.token_routers = nn.ModuleDict({block: TokenRouter(width) for block in routed})
        self.layer_routers = nn.ModuleDict({block: LayerRouter(width, c["router_hidden"]) for block in routed})
        # As released, the exit decision reaches the loss only through indexing, so the layer
        # routers never get a gradient and keep their initial weights.
        self.layer_routers.requires_grad_(False)
        self.query_token = nn.Parameter(torch.randn(1, 1, c["clip_dim"]))
        self.query_attn = nn.MultiheadAttention(c["clip_dim"], c["query_heads"], batch_first=True)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)
        return 2 * c["clip_dim"]

    def adapted(self, index: int, block: ResidualAttentionBlock, x: torch.Tensor) -> torch.Tensor:
        """One CLIP block with an MoE adapter after its attention and after its MLP."""
        x = x + self.adapters_attn[index](block.attend(block.ln_1(x), None))
        return x + self.adapters_mlp[index](block.mlp(block.ln_2(x)))

    def routed(self, index: int, block: ResidualAttentionBlock, x: torch.Tensor) -> torch.Tensor:
        """Run the adapted block on the top-scoring tokens, re-weighted by score."""
        scores = self.token_routers[str(index + 1)](x)
        k = max(1, int(self.cfg["capacity"] * x.shape[1]))
        chosen = scores.topk(k, dim=1).indices.sort(dim=1).values
        gather = chosen[..., None].expand(-1, -1, x.shape[-1])
        updated = self.adapted(index, block, x.gather(1, gather))
        weight = F.softmax(scores.gather(1, chosen), dim=1)[..., None]
        return x.scatter(1, gather, (updated * (1 + weight)).to(x.dtype))

    def view_cls(self, pixels: torch.Tensor) -> torch.Tensor:
        """CLIP-normalised (N, 3, 224, 224) views → their last class token (N, width)."""
        x = self.encoder.embed(pixels)
        active = torch.ones(len(x), dtype=torch.bool, device=x.device)
        for index, block in enumerate(self.encoder.blocks):
            if str(index + 1) not in self.layer_routers:
                x = checkpoint(self.adapted, index, block, x, use_reentrant=False)
                continue
            logits = self.layer_routers[str(index + 1)](x[:, 0])
            go = gumbel_sigmoid(logits) > 0.5 if self.training else torch.sigmoid(logits) > 0.5
            active = active & go
            if not active.any():
                continue
            rows = active.nonzero(as_tuple=True)[0]
            x = x.index_put((rows,), checkpoint(self.routed, index, block, x[rows], use_reentrant=False))
        return x[:, 0]

    def frame_features(self, pixels: torch.Tensor) -> torch.Tensor:
        """(N, 3, 448, 448) frames in [-1, 1] → (N, 1536): global view and tile summary."""
        pixels = (pixels * 0.5 + 0.5 - self.mean) / self.std
        size = self.cfg["clip_size"]
        global_view = F.interpolate(pixels.float(), size=(size, size), mode="bicubic").to(pixels.dtype)
        global_cls = self.view_cls(global_view)
        tile_cls = self.view_cls(tiles(pixels, 2))

        def project(cls: torch.Tensor) -> torch.Tensor:
            return self.encoder.ln_post(cls) @ self.encoder.proj

        batch = len(pixels)
        tile_tokens = project(tile_cls).reshape(4, batch, -1).transpose(0, 1)
        summary, _ = self.query_attn(self.query_token.expand(batch, -1, -1), tile_tokens, tile_tokens)
        return torch.cat([project(global_cls), summary[:, 0]], dim=-1)

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        frames, valid = batch["frames"], batch["valid"]
        genuine = frames[valid]
        chunk = self.cfg["encoder_chunk"]
        features = torch.cat([self.frame_features(genuine[i:i + chunk]) for i in range(0, len(genuine), chunk)])
        out = features.new_zeros(*valid.shape, features.shape[-1])
        out[valid] = features
        return out, valid

    def tunable(self) -> dict[str, Any]:
        return {"backbone": self.encoder,
                "new": [self.adapters_attn, self.adapters_mlp, self.token_routers, self.query_token, self.query_attn]}
