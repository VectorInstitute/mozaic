"""A made-up MOZAIC release for the tests: metadata.jsonl and summary.json, no videos."""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

from mozaic.annotation.t1_spot_the_fake import build_questions as t1
from mozaic.annotation.t3_identity_check import build_questions as t3

Row = dict[str, Any]


def clip(
    label: str,
    generator: str,
    arm: str,
    split: str,
    source_clip: str,
    target_id: str,
    length: str,
    kept_by: list[str] | None = None,
    voice: tuple[float, float] = (0.2, 0.6),
) -> Row:
    """One metadata row, with the fields the tasks use.

    The target's face picture comes from its video fff and its voice reference from vvv;
    voice is the pair's (voice_source, voice_target).
    """
    folder = {"RVRA": "real", "RVFA": "seed_vc"}.get(label, generator)
    pair_id = f"{source_clip}__{target_id}"
    source_id = source_clip[:7]
    return {
        "file_name": f"videos/{label}/{folder}/{length}/{arm}/{label}__{folder}__{length}__{arm}__{pair_id}.mov",
        "label": label,
        "video_generator": "real" if label in ("RVRA", "RVFA") else generator,
        "audio_generator": "bigvgan" if arm == "vocoded" else "seed_vc" if label[2] == "F" else "none",
        "length": length,
        "arm": arm,
        "split": split,
        "face_id": target_id if label[0] == "F" else source_id,
        "voice_id": target_id if label[2] == "F" else source_id,
        "source_id": source_id,
        "target_id": target_id,
        "pair_id": pair_id,
        "source_clip": source_clip,
        "target_face_clip": f"{target_id}_fff_00009",
        "target_voice_clip": f"{target_id}_vvv_00008",
        "kept_by": kept_by or [],
        "face_real": label[0] == "R",
        "voice_real": label[2] == "R",
        "voice_source": voice[0],
        "voice_target": voice[1],
    }


def pair_clips(
    source_clip: str,
    target_id: str,
    split: str,
    length: str,
    generators: list[str],
    voice: tuple[float, float] = (0.2, 0.6),
) -> list[Row]:
    """One pair's clips: three real-face ones, then three per generator."""
    args = (split, source_clip, target_id, length, generators, voice)
    rows = [clip("RVRA", "real", "main", *args), clip("RVRA", "real", "vocoded", *args), clip("RVFA", "real", "main", *args)]
    for g in generators:
        rows += [clip("FVRA", g, "main", *args), clip("FVRA", g, "vocoded", *args), clip("FVFA", g, "main", *args)]
    return rows


def real_clip(source_clip: str, target_id: str, length: str) -> str:
    """The file name of a pair's RVRA main clip."""
    return str(clip("RVRA", "real", "main", "train", source_clip, target_id, length)["file_name"])


# The pairs of the made-up release. For T3, only the train pairs at 77 frames whose two
# people have a real clip from another video (P1, P4, P6) give questions:
#   P5  left out: id00002's only other 77f clip comes from fff, the video of its
#       target's face picture
#   P7  left out: id00001's other clips are 77 frames, P7 is 153
#   P6  converted voice closer to the source (0.6 >= 0.5): asked all the same
P1 = ("id00001_aaa_00001", "id00002", "train", "77f", ["facefusion", "animate"])
P2 = ("id00003_bbb_00002", "id00004", "val", "153f", ["s2v"])
P3 = ("id00005_ccc_00003", "id00006", "test", "77f", ["facefusion"])
P4 = ("id00001_ddd_00004", "id00002", "train", "77f", ["s2v"])
P5 = ("id00002_eee_00005", "id00001", "train", "77f", ["facefusion"])
P6 = ("id00002_fff_00006", "id00001", "train", "77f", ["facefusion"])
P7 = ("id00001_ggg_00007", "id00002", "train", "153f", ["facefusion"])


def made_up_release(root: Path) -> list[Row]:
    """Write metadata.jsonl and summary.json for the pairs above; return the clips."""
    rows = [c for p in (P1, P2, P3, P4, P5, P7) for c in pair_clips(*p)] + pair_clips(*P6, voice=(0.6, 0.5))
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_folder: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for r in rows:
        by_folder[r["file_name"].rsplit("/", 1)[0]][r["split"]] += 1
    counts = {
        "by_split": dict(collections.Counter(r["split"] for r in rows)),
        "by_label": dict(collections.Counter(r["label"] for r in rows)),
        "by_folder": {folder: dict(n) for folder, n in by_folder.items()},
        "pairs": dict(collections.Counter(length for length, _ in {(r["length"], r["pair_id"]) for r in rows})),
    }
    (root / "summary.json").write_text(json.dumps({"counts": counts}))
    return rows


def t1_rows() -> list[Row]:
    """T1 rows for two test pairs kept by facefusion (answers: neither, neither, the audio, the video, ...)."""
    clips = (pair_clips("id00001_aaa_00001", "id00002", "test", "77f", ["facefusion"])
             + pair_clips("id00003_bbb_00002", "id00004", "test", "77f", ["facefusion"]))
    return [t1.make_row(c) for c in clips]


def task3_test_rows(pairs: int) -> list[Row]:
    """T3 rows for this many test pairs at 77 frames, kept by facefusion (clip 2 made up)."""
    rows = []
    for i in range(pairs):
        clips = pair_clips(f"id{i:05d}_aaa_00001", f"id{90000 + i}", "test", "77f", ["facefusion"])
        rows += [t3.make_row(c, part, person, "videos/clip2.mov")
                 for part, c in t3.asked_clips(clips) for person in t3.PEOPLE]
    return rows


# One question per kind; each part has each of the four answers once, 2 with clip 2 = A
# and 2 with B.
SMALL_COUNTS = {
    ("face", "real", 77, "A"): 1, ("face", "real", 77, "B"): 1, ("face", "facefusion", 77, "A"): 1,
    ("face", "facefusion", 77, "B"): 1, ("voice", "vocoded", 77, "A"): 1, ("voice", "vocoded", 77, "B"): 1,
    ("voice", "converted", 77, "A"): 1, ("voice", "converted", 77, "B"): 1,
}
