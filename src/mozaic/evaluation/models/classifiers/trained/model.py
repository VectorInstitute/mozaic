"""A classifier trained on MOZAIC (training/), tested on a task's test split.

best.pt holds the trained weights and the training config. The network is rebuilt from that config
with training's own code (its original checkpoint, then the trained weights on top), and the clips
are read and preprocessed by training's dataset and processor: the same input as in training.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from mozaic.evaluation.base import BaseClassifier
from mozaic.training.common.data import CLASSES, ClipDataset
from mozaic.training.common.pairs import PairDataset
from mozaic.training.run_train import load_model


class Model(BaseClassifier):
    def load(self) -> None:
        state = torch.load(self.config["weights"], map_location="cpu", weights_only=False)
        self.train_config, self.epoch = state["config"], state["epoch"]
        self.task = self.train_config["task"]
        if state["classes"] != CLASSES[self.task]:
            raise ValueError(f"{self.config['weights']} has classes {state['classes']}, not {CLASSES[self.task]}")
        Net = load_model(self.train_config)
        self.net = Net(self.train_config, len(state["classes"]))
        unexpected = self.net.load_state_dict(state["model"], strict=False).unexpected_keys
        if unexpected:
            raise ValueError(f"{self.config['weights']} has weights the model lacks: {unexpected[:5]}")
        self.device = torch.device("cuda")
        self.net.to(self.device).eval()
        self.processor = Net.processor(self.train_config)

    def settings(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "frames": self.train_config["frames"],
                "audio_sample_rate": self.train_config["audio_sample_rate"]}

    @torch.no_grad()
    def predict_proba(self, questions: list[dict[str, Any]]) -> list[dict[str, float]]:
        Dataset = PairDataset if self.task == "t3" else ClipDataset  # t3: two clips per question, as in training
        dataset = Dataset(self.task, questions, self.processor, self.train_config)
        loader = DataLoader(dataset, batch_size=self.config["micro_batch_size"], num_workers=self.config["workers"],
                            collate_fn=self.processor.collate, pin_memory=True)
        probs = []
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):  # as in training
                logits = self.net(batch)[0].float()
            probs += logits.softmax(-1).tolist()
        return [dict(zip(CLASSES[self.task], p, strict=True)) for p in probs]

    def unload(self) -> None:
        self.net = None
        torch.cuda.empty_cache()
