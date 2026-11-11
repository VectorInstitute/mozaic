"""Write a task's question files, shared by every task."""

from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any

from mozaic.common.io import write_jsonl
from mozaic.common.metadata import SPLITS
from mozaic.common.paths import qas_dir

Row = dict[str, Any]

log = logging.getLogger(__name__)


def write_questions(
    task: str,
    rows: list[Row],
    *,
    size: int | None,
    seed: int | None,
    default_seed: int,
    out: str | None,
    answer_field: str,
) -> list[Path]:
    """Full build: one file per split (t1_train.jsonl, ...). Sample: one file (t1_200.jsonl)."""
    folder = Path(out) if out else qas_dir(task)
    if size:
        name = f"{task}_{size}.jsonl" if seed == default_seed else f"{task}_{size}_s{seed}.jsonl"
        files = {name: rows}
    else:
        files = {f"{task}_{split}.jsonl": [r for r in rows if r["split"] == split] for split in SPLITS}

    paths = []
    for name, part in files.items():
        path = folder / name
        write_jsonl(path, part)
        paths.append(path)
        log.info("wrote %s questions to %s", f"{len(part):,}", path)
    log.info("per answer: %s", dict(collections.Counter(r[answer_field] for r in rows)))
    return paths
