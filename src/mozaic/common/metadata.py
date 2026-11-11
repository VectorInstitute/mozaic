"""Load metadata.jsonl, one row per clip, keyed by file_name."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mozaic.common.io import iter_jsonl
from mozaic.common.paths import metadata_path

Row = dict[str, Any]

LABELS = ("RVRA", "RVFA", "FVRA", "FVFA")
SPLITS = ("train", "val", "test")
FRAMES = {"77f": 77, "153f": 153}


def load_metadata(path: str | Path | None = None) -> dict[str, Row]:
    """Every clip, keyed by file_name."""
    out: dict[str, Row] = {}
    for row in iter_jsonl(path or metadata_path()):
        name = row["file_name"]
        if name in out:
            raise ValueError(f"duplicate file_name in metadata: {name}")
        out[name] = row
    return out


def filter_clips(
    clips: Iterable[Row],
    *,
    labels: Iterable[str] | None = None,
    splits: Iterable[str] | None = None,
    arms: Iterable[str] | None = None,
    lengths: Iterable[str] | None = None,
    video_generators: Iterable[str] | None = None,
) -> list[Row]:
    """Keep clips that match every filter given. None keeps everything."""
    wanted = {
        "label": labels,
        "split": splits,
        "arm": arms,
        "length": lengths,
        "video_generator": video_generators,
    }
    wanted = {k: set(v) for k, v in wanted.items() if v is not None}
    return [c for c in clips if all(c[k] in v for k, v in wanted.items())]


def check_label(clip: Row) -> str | None:
    """Check the label against face_real and voice_real. Returns a problem or None."""
    expected = ("R" if clip["face_real"] else "F") + "V" + ("R" if clip["voice_real"] else "F") + "A"
    if clip["label"] != expected:
        return f"{clip['file_name']}: label {clip['label']}, face_real={clip['face_real']}, voice_real={clip['voice_real']}"
    return None
