"""Shared by seeds/, lr_sweep/, epoch_sweep/ and unseen_generators/: every variant run's test metrics next to its original run's."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from mozaic.common.paths import data_root

# The metrics read from scores.json: T1-T3 (classification) and T4 (explanation, judge until judged: blank)
CLASSIFICATION = ["n", "unreadable", "accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "cohen_kappa"]
EXPLANATION = ["n", "unreadable", "judge", "judge_video", "judge_audio", "rouge_l"]
METRICS = list(dict.fromkeys(CLASSIFICATION + EXPLANATION))


def scores(model: str, experiment: str, task: str, small: bool) -> dict | None:
    """A run's scores.json, None when it has no scores yet. T4 is on Small only."""
    path = data_root() / f"results{'_small' if small or task == 't4' else ''}" / model / experiment / task / "scores.json"
    return json.loads(path.read_text()) if path.exists() else None


def metrics(model: str, experiment: str, task: str, small: bool) -> dict[str, float] | None:
    """A run's test metrics, None when it has no scores yet."""
    s = scores(model, experiment, task, small)
    if s is None:
        return None
    if task == "t4":
        judge = s.get("judge") or {}
        return {"n": s["questions"], "unreadable": s["unreadable"], "judge": judge.get("both", {}).get("mean"),
                "judge_video": judge.get("video", {}).get("mean"), "judge_audio": judge.get("audio", {}).get("mean"),
                "rouge_l": s["text"]["both"]["rouge_l"]}
    return {"n": s["n"], "unreadable": s["unreadable"], "accuracy": s["accuracy"],
            "balanced_accuracy": s["balanced_accuracy"], "macro_f1": s["macro"]["f1"],
            "macro_auroc": s["macro"].get("auroc"), "cohen_kappa": s["cohen_kappa"]}


def runs(folder: Path, small: bool) -> dict[tuple[str, str, str], dict[str, dict[str, float] | None]]:
    """(model, the original's experiment, task) -> {variant: metrics}, the original under "base"."""
    out: dict[tuple[str, str, str], dict[str, dict[str, float] | None]] = {}
    for path in sorted((folder / "configs").glob("*_eval.yaml")):
        config = yaml.safe_load(path.read_text())
        variant = path.stem.removesuffix("_eval").rsplit("_", 1)[1]
        model, task = config["model"], next(iter(config["tasks"]))
        base = config["experiment_name"].removesuffix(f"_{variant}")
        out.setdefault((model, base, task), {"base": metrics(model, base, task, small)})[variant] = metrics(
            model, config["experiment_name"], task, small)
    return out


def cells(values: dict[str, float] | None) -> str:
    """The METRICS columns of one run as CSV cells, blank when not scored."""
    values = values or {}
    return ",".join("" if values.get(m) is None else f"{values[m]:.4f}".rstrip("0").rstrip(".") for m in METRICS)
