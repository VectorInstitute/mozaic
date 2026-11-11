"""HAVIC (tuffy-studio/HAVIC) as a T1 backbone: its encoders and aggregation, our head on top.

This is HAVIC's fine-tuning model, HAVIC_FT in src/models/HAVIC.py of tuffy-studio/HAVIC at 66f4908
(lines 477-649), with its classifiers replaced by ours. Module code in vendor/ (MIT).

Taken as released
    encoders        AudioEncoder, VisualEncoder: HAVIC_FT.__init__ (503-530), the same arguments
    interaction     AudioVisualInteractionModule(num_layers=1) (550)
    aggregation     TokenWise_TokenReducer on layers 3, 6, 9, 12 of each encoder and on each
                    interaction output, LearnableWeightedPool over the 4 layers (552-566), the
                    3072-d concat [audio, video, audio interaction, video interaction] (607-638)
    weights         the encoders and interaction module of pt_model.200.pth, the same filter as
                    weights/initialize_finetune_weights.py (line 21)
Changed
    video segments  16 for our 32 frames, not 8: vendor/visual_modules.py and the interaction
                    module take the segment count from their input (marked "mozaic:"); the three
                    video tensors sized by segment (pos_embedding, cls_tokens, cls_pos_embed) are
                    interpolated along time from 8 to 16 on load. Every other weight loads unchanged.
                    The audio is unchanged: its 8 segments are mel bands (patches are ordered
                    band-major), and the complete audio is resized to 1024 steps as released.
Dropped
    classifiers     classifier, classifier_audio, classifier_visual (532-548): our head takes the
                    3072-d feature
    losses          the three BCE terms (src/traintest_finetune.py, line 163): our CE on 4 classes
    decoders, A2V, V2A   pretraining only (not in HAVIC_FT either)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mozaic.training.base import BaseClassifier
from mozaic.training.classifiers.havic.preprocess import AUDIO_STEPS, Processor
from mozaic.training.classifiers.havic.vendor.audio_modules import AudioEncoder
from mozaic.training.classifiers.havic.vendor.classification_modules import (
    TokenWise_TokenReducer,
)
from mozaic.training.classifiers.havic.vendor.interaction_modules import (
    AudioVisualInteractionModule,
)
from mozaic.training.classifiers.havic.vendor.visual_modules import VisualEncoder

DIM = 768
PRETRAINED = ("audio_encoder", "visual_encoder", "AudioVisualInteractionModule")
LAYERS = (3, 6, 9, 12)


class WeightedPool(nn.Module):
    """HAVIC's LearnableWeightedPool: softmax-weighted sum over the encoder layers."""

    def __init__(self, n: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n) / n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, layers, D) -> (B, D)
        return (F.softmax(self.weights, dim=0)[None, :, None] * x).sum(1)


def stretch_time(x: torch.Tensor, segments: int, to: int) -> torch.Tensor:
    """(1, segments * P, D) → (1, to * P, D): linear interpolation along the segment axis."""
    _, n, d = x.shape
    grid = x.reshape(segments, n // segments, d).permute(1, 2, 0)  # (P, D, segments)
    grid = F.interpolate(grid, size=to, mode="linear", align_corners=True)
    return grid.permute(2, 0, 1).reshape(1, -1, d)


def split_cls(emb: torch.Tensor, segments: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, segments * (1 + P), D) → cls tokens (B, segments, D) and patch tokens (B, segments * P, D)."""
    b, n, d = emb.shape
    emb = emb.view(b, segments, n // segments, d)
    return emb[:, :, 0], emb[:, :, 1:].reshape(b, -1, d)


class Model(BaseClassifier):
    processor = Processor
    # As released, each interaction block also runs a self-attention on video and audio whose output
    # is never returned: its LayerNorms train but get no gradient, so DDP must look for them.
    dynamic_graph = True

    def build_backbone(self, config: dict[str, Any]) -> int:
        """HAVIC_FT.__init__ (HAVIC.py 478-566) without its three classifiers."""
        frames = config["frames"]
        self.video_segments = frames // 2
        self.backbone = nn.ModuleDict({
            "audio_encoder": AudioEncoder(audio_length=AUDIO_STEPS, mel_bins=128, patch_size=16, embed_dim=DIM,
                                          num_heads=12, encoder_depth=12, qkv_bias=True),
            "visual_encoder": VisualEncoder(img_size=224, patch_size=16, n_frames=frames, embed_dim=DIM, depth=12,
                                            num_heads=12, qkv_bias=True, tubelet_size=2),
            "AudioVisualInteractionModule": AudioVisualInteractionModule(num_layers=1),
        })
        self.load_pretrained(config["backbone"]["checkpoint"])
        self.audio_reducers = nn.ModuleList(TokenWise_TokenReducer() for _ in LAYERS)
        self.video_reducers = nn.ModuleList(TokenWise_TokenReducer() for _ in LAYERS)
        self.audio_inter_reducer = TokenWise_TokenReducer()
        self.video_inter_reducer = TokenWise_TokenReducer()
        self.pool_a = WeightedPool(len(LAYERS))
        self.pool_v = WeightedPool(len(LAYERS))
        return 4 * DIM

    def load_pretrained(self, path: str) -> None:
        """The encoders and interaction module from pt_model.200.pth (initialize_finetune_weights.py),
        the video's segment-sized tensors stretched to our segments, then a strict load."""
        state = torch.load(path, map_location="cpu")
        state = {k.removeprefix("module."): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if k.split(".")[0] in PRETRAINED}
        saved = state["visual_encoder.cls_tokens"].shape[1]
        if saved != self.video_segments:
            for key in ("visual_encoder.pos_embedding.emb", "visual_encoder.cls_tokens", "visual_encoder.cls_pos_embed"):
                state[key] = stretch_time(state[key], saved, self.video_segments)
        self.backbone.load_state_dict(state, strict=True)

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """HAVIC_FT.forward (HAVIC.py 578-638), up to final_feat; its classifiers (641-647) are ours."""
        video = batch["frames"].permute(0, 2, 1, 3, 4)  # (B, 3, T, 224, 224)
        enc = self.backbone
        # 583-584: layers 3, 6, 9, 12 of each encoder (use_mask is not an argument of the encoders' forward)
        v_layers = enc["visual_encoder"](video, ids_keep=None, apply_cls_tokens=True)
        a_layers = enc["audio_encoder"](batch["audio"], ids_keep=None, apply_cls_tokens=True)
        # 587-596: split each layer into its cls tokens and patch tokens (8 segments there; 16 video here)
        a_segments = enc["audio_encoder"].cls_tokens.shape[1]
        v_split = [split_cls(x, self.video_segments) for x in v_layers]
        a_split = [split_cls(x, a_segments) for x in a_layers]
        v_cls, v_last = v_split[-1]
        a_cls, a_last = a_split[-1]
        # 601-604: the interaction module on layer 12, with its cls tokens; its cls tokens are dropped
        v_inter, a_inter = enc["AudioVisualInteractionModule"](
            v_last, a_last, video_cls_tokens=v_cls[:, :, None], audio_cls_tokens=a_cls[:, :, None])
        _, v_inter = split_cls(v_inter, self.video_segments)
        _, a_inter = split_cls(a_inter, a_segments)
        # 607-635: a token reducer per layer, then the weighted pool over the 4 layers
        audio = self.pool_a(torch.stack([r(x) for r, (_, x) in zip(self.audio_reducers, a_split, strict=True)], 1))
        video = self.pool_v(torch.stack([r(x) for r, (_, x) in zip(self.video_reducers, v_split, strict=True)], 1))
        # 638: final_feat = [audio, video, audio interaction, video interaction], 3072-d
        feature = torch.cat([audio, video, self.audio_inter_reducer(a_inter), self.video_inter_reducer(v_inter)], -1)
        return feature[:, None], torch.ones(len(feature), 1, dtype=torch.bool, device=feature.device)

    def tunable(self) -> dict[str, Any]:
        visual = self.backbone["visual_encoder"]
        new = [self.audio_reducers, self.video_reducers, self.audio_inter_reducer, self.video_inter_reducer,
               self.pool_a, self.pool_v]
        if self.video_segments != 8:  # stretched from the checkpoint's 8: they learn their new length
            new += [visual.pos_embedding.emb, visual.cls_tokens, visual.cls_pos_embed]
        return {"backbone": self.backbone, "new": new}
