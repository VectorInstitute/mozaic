"""Tests for data_generation/inputs: the pair draw and the voice clip list, on made-up clip lists.

Nothing here reads VoxCeleb2. Importing settings needs configs/paths.yaml or MOZAIC_GENERATION_PATHS.
The draw tests pin what every build of one draw relies on:
  * a folder counts only if it can give each length its own clips;
  * the two lengths' driver clips are different by construction;
  * the share-out depends only on (seed, key, the folder's clips), so the 77-frame and the
    153-frame builds of one draw agree without reading each other;
  * one sample is cut into built and reserve rings, which therefore never overlap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mozaic.data_generation import settings as S
from mozaic.data_generation.inputs import make_audio_extract_list as MAL
from mozaic.data_generation.inputs import draw_pairs as MM

SIX_S = S.min_driver_clip_duration_s(153)
THREE_S = S.min_driver_clip_duration_s(77)


def _clips(*durations: float) -> list[tuple[str, float]]:
    return [(f"clip{i:02d}", d) for i, d in enumerate(durations)]


def _dual(frames: tuple[int, ...] | None = (153, 77)) -> None:
    MM.ELIGIBILITY_FRAMES = frames


def teardown_function() -> None:
    """Never leak the module-level draw state into another test."""
    MM.ELIGIBILITY_FRAMES = None
    MM.ACTIVE_CROP_FRAMES = S.CROP_FRAMES


def test_dual_eligibility_needs_distinct_clips_per_stratum() -> None:
    """Two 6 s clips and one 3 s clip is NOT enough: the lengths would share a clip."""
    _dual()
    # Both long clips go to 153 frames, leaving one clip for 77 frames, which needs two.
    assert MM.folder_is_usable(_clips(SIX_S, SIX_S, THREE_S)) is False
    assert MM.folder_is_usable(_clips(SIX_S, SIX_S, THREE_S, THREE_S)) is True
    # Long clips are long enough for 77 frames too, so four long clips pass.
    assert MM.folder_is_usable(_clips(SIX_S, SIX_S, SIX_S, SIX_S)) is True
    # ... but a folder without two 6 s clips fails, whatever else it holds.
    assert MM.folder_is_usable(_clips(SIX_S, THREE_S, THREE_S, THREE_S)) is False


def test_single_stratum_eligibility_is_unchanged() -> None:
    """With no --eligibility-frames, a folder needs CLIPS_PER_FOLDER clips of the build's length."""
    MM.ELIGIBILITY_FRAMES = None
    MM.ACTIVE_CROP_FRAMES = 77
    assert MM.folder_is_usable(_clips(THREE_S, THREE_S)) is True
    assert MM.folder_is_usable(_clips(THREE_S)) is False


def test_strata_allocations_are_disjoint_and_correct_length() -> None:
    """Each length gets its own clips, and every clip is long enough for it."""
    _dual()
    clips = _clips(SIX_S + 5, SIX_S + 1, THREE_S, THREE_S + 1, SIX_S + 2)
    out = MM.allocate_stratum_clips(clips, 42, ("clips", "train", "f", "id1", "fold1"))
    long_ids = {c[0] for c in out[153]}
    short_ids = {c[0] for c in out[77]}
    assert len(long_ids) == S.CLIPS_PER_FOLDER
    assert len(short_ids) == S.CLIPS_PER_FOLDER
    assert long_ids & short_ids == set()
    assert all(d >= SIX_S for _, d in out[153])
    assert all(d >= THREE_S for _, d in out[77])


def test_allocation_does_not_depend_on_the_build_being_written() -> None:
    """The 77-frame and the 153-frame builds compute the same share-out without reading each other."""
    _dual()
    clips = _clips(SIX_S + 5, SIX_S + 1, THREE_S, THREE_S + 1, SIX_S + 2)
    key = ("clips", "train", "f", "id1", "fold1")
    first = MM.allocate_stratum_clips(clips, 42, key)
    second = MM.allocate_stratum_clips(list(reversed(clips)), 42, key)
    assert first == second
    assert MM.allocate_stratum_clips(clips, 43, key) != first


def test_parse_per_split_defaults_and_rejects_unknown_splits() -> None:
    assert MM.parse_per_split(["train=70", "val=15", "test=15"], 1) == {"train": 70, "val": 15, "test": 15}
    assert MM.parse_per_split([], 0) == {"train": 0, "val": 0, "test": 0}
    assert MM.parse_per_split(["train=14"], 0)["val"] == 0
    with pytest.raises(SystemExit):
        MM.parse_per_split(["dev=3"], 0)
    with pytest.raises(SystemExit):
        MM.parse_per_split(["train"], 0)


def test_pinned_rings_read_the_legacy_single_ring_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An older summary has no "draw" and records one bare ring per cell."""
    summary = {"rings": {"train": {"m": ["a", "b", "c", "d", "e"]}}}
    path = tmp_path / "manifest_summary.json"
    path.write_text(json.dumps(summary))
    monkeypatch.setattr(S, "new_gen_manifest_summary_path", lambda _build_id: path)
    pinned = MM.load_pinned_rings("legacy_v1")
    assert pinned["primary_rings"]["train"]["m"] == [["a", "b", "c", "d", "e"]]
    assert pinned["reserve_rings"] == {}


def test_primary_and_reserve_rings_come_from_one_partition() -> None:
    """A reserve ring can never also be a built ring, whatever the counts."""
    candidates = [f"id{i:04d}" for i in range(500)]
    ring_size, n_primary, n_reserve = 5, 70, 14
    need = ring_size * (n_primary + n_reserve)
    drawn = MM.substream(42, "ring", "train", "m").sample(candidates, need)
    rings = [drawn[i : i + ring_size] for i in range(0, need, ring_size)]
    primary, reserve = rings[:n_primary], rings[n_primary:]
    flat_primary = {i for ring in primary for i in ring}
    flat_reserve = {i for ring in reserve for i in ring}
    assert len(primary) == n_primary
    assert len(reserve) == n_reserve
    assert flat_primary & flat_reserve == set()
    assert len(flat_primary | flat_reserve) == need


def _manifest(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_voice_clip_list_keeps_to_the_given_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only rows that need a conversion count, and --pair-ids-file narrows them to its pairs."""
    rows = [
        {"pair_id": "p1", "needs_conversion": True, "voice_ref_clip_id": "r1"},
        {"pair_id": "p2", "needs_conversion": True, "voice_ref_clip_id": "r2"},
        {"pair_id": "p3", "needs_conversion": False, "voice_ref_clip_id": "r3"},
    ]
    path = _manifest(tmp_path, rows)
    monkeypatch.setattr(S, "new_gen_manifest_path", lambda _b: path)
    assert MAL.voice_refs("b") == ["r1", "r2"]
    assert MAL.voice_refs("b", {"p1"}) == ["r1"]
