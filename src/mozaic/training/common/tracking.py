"""Weights & Biases tracking, on the first GPU's process only.

One W&B run per training run, named <model>/<experiment_name>/<task> and grouped by
<model>/<task>. Its id is kept in the run folder, so a resumed run logs into the same W&B run.
Without internet, set WANDB_MODE=offline, then `wandb sync` the run's wandb/ folder.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ID_FILE = "wandb_id.txt"


class Tracker:
    def __init__(self, config: dict[str, Any], settings: dict[str, Any], out: Path, rank: int):
        self.run = None
        options = config.get("log") or {}
        if rank != 0 or not options.get("wandb"):
            return
        import wandb

        id_path = out / ID_FILE
        run_id = id_path.read_text().strip() if id_path.exists() else uuid.uuid4().hex[:8]
        self.run = wandb.init(
            project=options.get("project", "mozaic"), entity=options.get("entity"),
            name=f"{config['model']}/{config['experiment_name']}/{config['task']}",
            group=f"{config['model']}/{config['task']}", dir=str(out), config=settings,
            id=run_id, resume="allow")
        if not id_path.exists():
            id_path.write_text(run_id + "\n")
        log.info("wandb run %s (%s)", run_id, wandb.run.settings.mode)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def epoch_metrics(scores: dict[str, Any], val_loss: float, train_loss: float, classes: list[str]) -> dict[str, Any]:
    """The val scores worth a chart: headline numbers, the three binary views, F1 per class."""
    row = {"train/epoch_loss": train_loss, "val/loss": val_loss, "val/accuracy": scores["accuracy"],
           "val/macro_f1": scores["macro"]["f1"], "val/macro_auroc": scores["macro"]["auroc"],
           "val/balanced_accuracy": scores["balanced_accuracy"], "val/cohen_kappa": scores["cohen_kappa"]}
    for view in ("real_vs_fake", "video", "audio"):
        if view in scores:
            for key in ("auroc", "eer", "f1", "balanced_accuracy"):
                row[f"val/{view}/{key}"] = scores[view][key]
    for c in classes:
        row[f"val/f1/{c}"] = scores["per_class"][c]["f1"]
    return row
