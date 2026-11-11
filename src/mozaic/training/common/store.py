"""The run's folder: settings lock, history, scores, checkpoints. All group-writable."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from mozaic.common.io import FILE_MODE, make_dir, write_json, write_jsonl
from mozaic.common.paths import training_dir

FILES = ("train_settings.json", "history.jsonl", "val_scores.json", "best.pt", "last.pt", "wandb_id.txt")


def run_dir(config: dict[str, Any]) -> Path:
    return training_dir(config["model"], config["experiment_name"]) / config["task"]


def remove_run(out: Path) -> None:
    for name in FILES:
        (out / name).unlink(missing_ok=True)
    shutil.rmtree(out / "val_scores", ignore_errors=True)
    shutil.rmtree(out / "wandb", ignore_errors=True)


def check_settings(out: Path, settings: dict[str, Any]) -> None:
    """Save the run's settings once. Refuse to resume a run with other settings."""
    path = out / "train_settings.json"
    if not path.exists():
        write_json(path, settings)
        return
    old = json.loads(path.read_text())
    differ = sorted(k for k in set(old) | set(settings) if old.get(k) != settings.get(k))
    if differ:
        raise SystemExit(f"{out} was trained with other settings ({', '.join(differ)}). "
                         "Use another experiment_name, or --overwrite.")


def finished(out: Path, epochs: int) -> bool:
    """Whether the run has trained all its epochs (history.jsonl holds one line per epoch)."""
    path = out / "history.jsonl"
    return path.exists() and len(path.read_text().splitlines()) >= epochs


def save_epoch(out: Path, epoch: int, history: list[dict[str, Any]], scores: dict[str, Any]) -> None:
    write_jsonl(out / "history.jsonl", history)
    write_json(out / "val_scores" / f"epoch_{epoch}.json", scores)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    make_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.chmod(tmp, FILE_MODE)
    os.replace(tmp, path)
