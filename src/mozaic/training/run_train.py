"""Train a classifier on a task's train split and validate it on its val split. No test split here.

    torchrun --nproc_per_node=4 -m mozaic.training.run_train --config avhubert/t1_ln
    python -m mozaic.training.run_train --config avhubert/t1_ln --limit 64        # one GPU, a few clips
    torchrun --nproc_per_node=4 -m mozaic.training.run_train --config avhubert/t1_ln --subset small

A rerun with the same settings resumes after the last finished epoch (last.pt); a finished run is not trained
again (its last.pt is removed once every epoch is done).
--overwrite deletes the run's files first, then trains fresh.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml

from mozaic.common.paths import subset, use_subset
from mozaic.training.common import store, tuning
from mozaic.training.common.data import CLASSES, load_questions
from mozaic.training.common.loop import (
    class_weights,
    seed_everything,
    start_distributed,
    train,
)
from mozaic.training.common.pairs import load_questions as load_pairs
from mozaic.training.common.tracking import Tracker

CONFIG_DIR = Path(__file__).with_name("configs")

log = logging.getLogger(__name__)


def load_train_config(name_or_path: str) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    return yaml.safe_load(path.read_text())


def load_model(config: dict[str, Any]):
    """Imported here, not at the top: each model has its own environment and packages."""
    if config["task"] == "t3":  # two clips per question: the model's pair version (common/pairs.py)
        if config["model"] == "avhubert":
            from mozaic.training.classifiers.avhubert.pair import Model
        elif config["model"] == "pe_av":
            from mozaic.training.classifiers.pe_av.pair import Model
        else:
            raise ValueError(f"{config['model']} has no t3 version (classifiers/<model>/pair.py)")
        return Model
    if config["model"] == "avhubert":
        from mozaic.training.classifiers.avhubert.model import Model
    elif config["model"] == "dfd_hr":
        from mozaic.training.classifiers.dfd_hr.model import Model
    elif config["model"] == "havic":
        from mozaic.training.classifiers.havic.model import Model
    elif config["model"] == "pe_av":
        from mozaic.training.classifiers.pe_av.model import Model
    else:
        raise ValueError(f"unknown model {config['model']}")
    return Model


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Train a classifier and validate it.")
    p.add_argument("--config", required=True, help="config in training/configs (e.g. avhubert/t1_ln), or a path")
    p.add_argument("--limit", type=int, help="about N train and val questions, the same number of each class")
    p.add_argument("--overwrite", action="store_true", help="delete the run's files, then train fresh")
    p.add_argument("--subset", help="train on a subset, e.g. small (default: the full dataset)")
    args = p.parse_args()
    use_subset(args.subset)
    rank, world, device = start_distributed()
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    config = load_train_config(args.config)
    task = config["task"]
    if task not in CLASSES:
        raise ValueError(f"unknown task {task}, expected one of {sorted(CLASSES)}")
    load = load_pairs if task == "t3" else load_questions  # t3: --limit takes each class of each part
    train_file, train_qs = load(task, config["questions"]["train"], args.limit)
    val_file, val_qs = load(task, config["questions"]["val"], args.limit)
    weights = class_weights(task, train_qs, config["train"].get("class_weights", "balanced"))

    seed_everything(config["train"]["seed"])
    Model = load_model(config)
    model = Model(config, len(CLASSES[task]))
    counts = tuning.apply(model, config["tuning"]["mode"])

    out = store.run_dir(config)
    settings = {**config, "train_file": train_file, "val_file": val_file, "limit": args.limit,
                "subset": subset(), "train_questions": len(train_qs), "val_questions": len(val_qs),
                "class_weights": weights, "trainable_parameters": counts["trainable"]}
    if rank == 0:
        if args.overwrite:
            store.remove_run(out)
        store.check_settings(out, settings)
        log.info("%s on %s: %d train, %d val questions, %d GPU(s); trainable %.2fM of %.1fM parameters; out %s",
                 config["model"], task, len(train_qs), len(val_qs), world,
                 counts["trainable"] / 1e6, counts["total"] / 1e6, out)
        log.info("git commit %s", git_commit())
    if world > 1:
        import torch.distributed as dist
        dist.barrier()

    tracker = Tracker(config, settings, out, rank)
    train(model, Model.processor(config), config, train_qs, val_qs, weights, out, tracker, rank, world, device)


if __name__ == "__main__":
    main()
