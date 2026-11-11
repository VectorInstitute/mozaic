"""Paths inside the shared dataset root.

The repo holds code only. Videos, metadata, qas, checkpoints and results all live
under one dataset root, set in config.yaml.

A subset (e.g. small, set with --subset small) is a choice of rows: its files sit next to
the full ones with the subset's name added (metadata_small.jsonl, qas_small/, results_small/,
training_small/), and it shares videos/ with the full dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

from mozaic.common.config import load_config


def data_root() -> Path:
    """Dataset root from config.yaml, or the MOZAIC_ROOT env var if set."""
    root = Path(os.environ.get("MOZAIC_ROOT") or load_config()["data_root"]).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    return root


# The subset in use, None for the full dataset. Set once per run by use_subset (the --subset flag).
_subset: str | None = None


def subset() -> str | None:
    """The subset in use (--subset), or None for the full dataset."""
    return _subset


def use_subset(name: str | None) -> None:
    """Set the subset for this run. Entry points call it first, before any path is read."""
    global _subset
    if name and name not in load_config()["subsets"]:
        raise ValueError(f"unknown subset {name}, expected one of {load_config()['subsets']}")
    _subset = name or None


def suffix() -> str:
    """"_small" on a subset, "" on the full dataset."""
    name = subset()
    return f"_{name}" if name else ""


def resolve(relative: str | Path) -> Path:
    """Absolute path of a clip path stored in a question file."""
    return data_root() / relative


def metadata_path(full: bool = False) -> Path:
    """metadata.jsonl, or the subset's metadata_small.jsonl; full=True always the full file."""
    return data_root() / f"metadata{'' if full else suffix()}.jsonl"


def summary_path(full: bool = False) -> Path:
    """summary.json, or the subset's summary_small.json; full=True always the full file."""
    return data_root() / f"summary{'' if full else suffix()}.json"


def qas_dir(task: str) -> Path:
    """Folder of a task's question files, e.g. qas/t1_spot_the_fake or qas_small/t1_spot_the_fake."""
    tasks = load_config()["tasks"]
    if task not in tasks:
        raise ValueError(f"unknown task {task}, expected one of {sorted(tasks)}")
    return data_root() / f"qas{suffix()}" / tasks[task]


def training_dir(model: str, experiment: str, on: str | None | bool = True) -> Path:
    """A training run's folder: on the subset in use (on=True, the default), or on the one named
    (on="small"; None: the full dataset), e.g. a run trained on Small tested on the full test split."""
    if on is True:
        return data_root() / f"training{suffix()}" / model / experiment
    if on and on not in load_config()["subsets"]:
        raise ValueError(f"unknown subset {on}, expected one of {load_config()['subsets']}")
    return data_root() / f"training{f'_{on}' if on else ''}" / model / experiment


def results_dir(model: str, experiment: str, task: str) -> Path:
    return data_root() / f"results{suffix()}" / model / experiment / task
