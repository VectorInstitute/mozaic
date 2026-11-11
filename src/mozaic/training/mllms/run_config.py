"""The flags, config and run folder every step shares, and the task's evaluation prompt."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mozaic.common.paths import use_subset
from mozaic.training.common import store
from mozaic.training.run_train import load_train_config


def parse_args(description: str) -> argparse.Namespace:
    """The same flags for every step: train_mllm.sh passes its arguments to each."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="config in training/configs (e.g. qwen2_5_omni/t1_lora), or a path")
    p.add_argument("--limit", type=int, help="about N train and val questions, the same number per class")
    p.add_argument("--overwrite", action="store_true", help="delete the run's files, then train fresh")
    p.add_argument("--subset", help="train on a subset, e.g. small (default: the full dataset)")
    p.add_argument("--epoch", type=int, help="merge: this epoch as the final model (default: the best on val)")
    return p.parse_args()


def load(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """The config and its run folder: training[_small]/<model>/<experiment_name>/<task>/."""
    use_subset(args.subset)
    config = load_train_config(args.config)
    return config, store.run_dir(config)


def prompts(task: str):
    """evaluation/prompts/<task>_prompts.py: the prompt, its modalities and the reply parser."""
    from mozaic.evaluation.run_inference import load_prompts

    return load_prompts(task)


def with_sound(task: str) -> bool:
    """t1, t4: the clip's sound inside the video; t2: frames only; t3: frames or sound per question, never together."""
    return task != "t3" and "audio" in prompts(task).MODALITIES


def run_name(config: dict[str, Any]) -> str:
    return f"{config['model']}/{config['experiment_name']}/{config['task']}"
