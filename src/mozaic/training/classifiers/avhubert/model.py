"""AV-HuBERT (facebookresearch/av_hubert) as a T1 and T2 backbone: per-step features, our head on top.

Loaded through the authors' fairseq fork (facebookresearch/av_hubert, avhubert/hubert.py).

Taken as released
    model, weights   AVHubertModel and large_vox_iter5.pt, through fairseq's checkpoint loader
    features         extract_finetune: the per-step output of the last layer. It never drops a
                     modality (forward() drops one at random in training mode)
    input format     preprocess.py: grayscale 88 x 88 normalised as hubert_pretraining.py, log
                     filterbank stacked 4 per step and layer-normed as hubert_dataset.py
Dropped
    decoders         their fine-tuning heads (ASR, VSR) are not loaded: our head on the mean feature

Without audio (t2) it runs the video branch alone: the audio features are zeros, as when
pretraining dropped the audio.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from mozaic.common.config import REPO_ROOT
from mozaic.training.base import BaseClassifier
from mozaic.training.classifiers.avhubert.preprocess import Processor


class Model(BaseClassifier):
    processor = Processor

    def build_backbone(self, config: dict[str, Any]) -> int:
        # A checkpoint saved before the move names notes/external_repo/av_hubert: the same clone, in external-repos/
        code = Path(str(config["backbone"]["code"]).replace("notes/external_repo/", "external-repos/"))
        from fairseq import (
            checkpoint_utils,  # first: fairseq must finish importing before avhubert
        )

        # Appended, so the installed fairseq wins over the repo's fairseq/ folder. Imported as a
        # package: run as plain modules, hubert.py picks its imports by sys.argv.
        sys.path.append(str(code if code.is_absolute() else REPO_ROOT / code))
        import avhubert  # noqa: F401  registers the model and task

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([config["backbone"]["checkpoint"]])
        self.backbone = models[0]
        # Fully trained, the backbone is in training mode: layer drop skips layers, and parts that
        # extract_finetune never runs (final_proj, label_embs; the audio branch in t2) get no gradient.
        self.dynamic_graph = config["tuning"]["mode"] == "full"
        return self.backbone.encoder_embed_dim

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, padding = self.backbone.extract_finetune(
            {"audio": batch.get("audio"), "video": batch["video"]}, padding_mask=~batch["valid"], mask=False)
        return x, batch["valid"] if padding is None else ~padding

    def tunable(self) -> dict[str, Any]:
        return {"backbone": self.backbone, "new": []}
