"""Pick a subset of a task's question records by filters, the same way in every stage.

A task in a model config is either `t1: t1_test.jsonl` or
`t1: {file: t1_test.jsonl, filters: {arms: [vocoded]}, modalities: [video]}`. A missing or null filter keeps all;
missing modalities send what the task sends.
"""

from __future__ import annotations

from typing import Any

Row = dict[str, Any]


def _arm(row: Row, values: set) -> bool:
    """Arms apply to real voices only. Always kept: fake-voice clips, which have one arm, and
    T3 face questions, which send no sound."""
    return row.get("part") == "face" or row["gt_label"][2] != "R" or row["meta"]["arm"] in values


FILTERS = {
    "arms": _arm,
    "frames": lambda row, values: row["meta"]["frames"] in values,
    "labels": lambda row, values: row["gt_label"] in values,
    "video_methods": lambda row, values: row["meta"]["video_method"] in values,
}


def select(rows: list[Row], filters: dict[str, Any] | None) -> list[Row]:
    """Keep rows that pass every filter given."""
    filters = filters or {}
    unknown = sorted(set(filters) - set(FILTERS))
    if unknown:
        raise ValueError(f"unknown filters {unknown}, known: {sorted(FILTERS)}")
    active = {k: set(v) for k, v in filters.items() if v is not None}
    return [r for r in rows if all(FILTERS[k](r, v) for k, v in active.items())]


def task_entry(config: dict[str, Any], task: str) -> tuple[str, dict[str, Any] | None]:
    """The task's question file and filters from a model config."""
    entry = config["tasks"][task]
    if isinstance(entry, str):
        return entry, None
    return entry["file"], entry.get("filters")


def task_modalities(config: dict[str, Any], task: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """What the model gets of each clip: the task's modalities in a model config, else default (the task's own)."""
    entry = config["tasks"][task]
    modalities = entry.get("modalities") if isinstance(entry, dict) else None
    return tuple(modalities) if modalities else default
