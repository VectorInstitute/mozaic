"""Read and write json and jsonl in the shared dataset root.

Folders get mode 2775 and files 664, so the whole group can read and rerun.
Always write through these helpers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

Row = dict[str, Any]

DIR_MODE = 0o2775
FILE_MODE = 0o664


def make_dir(path: str | Path) -> Path:
    """Create a folder and its parents, group-writable."""
    path = Path(path)
    missing = []
    p = path
    while not p.exists():
        missing.append(p)
        p = p.parent
    path.mkdir(parents=True, exist_ok=True)
    for p in reversed(missing):
        os.chmod(p, DIR_MODE)
    return path


def share_tree(path: str | Path) -> None:
    """Give a folder written by another library (LLaMA-Factory, transformers) the modes above, all the way down:
    safetensors saves its files as 600, readable by their owner only."""
    path = Path(path)
    for p in [path, *path.rglob("*")]:
        os.chmod(p, DIR_MODE if p.is_dir() else FILE_MODE)


def iter_jsonl(path: str | Path) -> Iterator[Row]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[Row]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[Row]) -> int:
    """Write rows through a temp file, then rename. Returns the row count."""
    path = Path(path)
    make_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    os.chmod(tmp, FILE_MODE)
    os.replace(tmp, path)
    return n


def append_jsonl(path: str | Path, row: Row) -> None:
    """Append one row and flush, so a crash never loses a finished row."""
    path = Path(path)
    make_dir(path.parent)
    new = not path.exists()
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
    if new:
        os.chmod(path, FILE_MODE)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    make_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.chmod(tmp, FILE_MODE)
    os.replace(tmp, path)


def write_text(path: str | Path, text: str) -> None:
    """Write text through a temp file, then rename."""
    path = Path(path)
    make_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.chmod(tmp, FILE_MODE)
    os.replace(tmp, path)
