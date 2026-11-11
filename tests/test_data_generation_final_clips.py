"""Tests for final_clips/: the mux's drops and redo rules, and withdrawing half-shipped pairs.

Made-up files only, nothing is encoded. The settings read the two folders on import, so set
MOZAIC_GENERATION_PATHS to a paths.yaml (any two folders) or create configs/paths.yaml.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from mozaic.data_generation import settings as S
from mozaic.data_generation.final_clips import mux as T
from mozaic.data_generation.final_clips import withdraw_incomplete_pairs as W

CLASSES = ("RVRA", "RVFA", "FVRA", "FVFA")


# --------------------------------------------------------------------------- the mux: drops and failures
def _job(tmp_path: Path, sample_id: str) -> dict[str, object]:
    return {
        "video": "v.mp4",
        "audio": "a.wav",
        "out": str(tmp_path / f"{sample_id}.mov"),
        "target_frames": 77,
        "sample_id": sample_id,
        "class_name": "RVRA",
        "pair_id": sample_id.split("__", maxsplit=1)[0],
    }


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcomes: list[T.TerminalResult]) -> tuple[int, Path]:
    jobs = tmp_path / "terminal_jobs_pairs2_abcd1234.jsonl"
    jobs.write_text("\n".join(json.dumps(_job(tmp_path, f"p{i}__RVRA")) for i in range(len(outcomes))) + "\n")
    queue = iter(outcomes)
    monkeypatch.setattr(T, "process", lambda *a, **k: next(queue))
    monkeypatch.setattr(T, "write_provenance", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["mux.py", "--jobs", str(jobs)])
    return T.main(), jobs.with_name("terminal_jobs_pairs2_abcd1234_dropped.jsonl")


def _ok() -> T.TerminalResult:
    return T.TerminalResult(ok=True, out_path=Path("x"), target_frames=77)


def test_a_drop_is_recorded_beside_the_job_list_and_the_step_exits_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clip that misses the picture target is dropped by design, not a stoppage."""
    drop = T.TerminalResult(
        ok=False, out_path=Path("x"), target_frames=77, error="output chroma 12.0 misses the 17.0 target", dropped=True
    )
    rc, dropped_path = _run(tmp_path, monkeypatch, [_ok(), drop])
    assert rc == 0
    rows = [json.loads(line) for line in dropped_path.read_text().splitlines()]
    assert [r["sample_id"] for r in rows] == ["p1__RVRA"]
    assert rows[0]["pair_id"] == "p1"
    assert "chroma" in rows[0]["reason"]


def test_a_hard_failure_still_fails_the_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fail = T.TerminalResult(ok=False, out_path=Path("x"), target_frames=77, error="OSError: ffmpeg not found")
    rc, dropped_path = _run(tmp_path, monkeypatch, [_ok(), fail])
    assert rc == 1
    assert dropped_path.read_text() == "", "a failure is not a drop"


def test_a_tolerance_miss_is_a_drop_and_a_contract_breach_is_not() -> None:
    """Picture misses, the peak ceiling and a short RENDER are drops; all else fails."""
    assert issubclass(T.TerminalDropError, T.TerminalError)
    source = Path(T.__file__).read_text()
    assert source.count("raise TerminalDropError(") == 4
    short = source.split("input video decodes", 1)[1].split("in_rate = ", 1)[0]
    assert "video_in.parent.name == RENDER_DIR_NAME" in short, "only a short GENERATED input is a drop"
    after_drop = short.split("raise TerminalDropError(", 1)[1]
    assert "raise TerminalError(msg)" in after_drop, "a short CROP is our cropper's breach: still a failure"
    peak = source.split("dBTP exceeds the", 1)[1].split("\n", 3)
    assert "raise TerminalDropError(msg)" in "\n".join(peak[:3]), (
        "a loud converted voice must drop its pair, not stop the step"
    )
    assert "raise TerminalError(msg)" in source.split("audio onset moved", 1)[1][:200], (
        "a moved onset is still a failure"
    )


def test_an_existing_clip_that_cannot_be_read_is_made_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip left without its index by a cut-off write must not count as done."""
    out = tmp_path / "p0__RVRA.mov"
    out.write_bytes(b"half a clip")
    monkeypatch.setattr(T, "_readable", lambda path: False)
    monkeypatch.setattr(T, "_assert_inputs", lambda *a: (_ for _ in ()).throw(T.TerminalError("stop here")))
    result = T.process(tmp_path / "v.mp4", tmp_path / "a.wav", out, 77)
    assert not result.ok and result.error == "stop here", "the clip went back through the chain"
    assert not out.exists(), "the unreadable clip was removed before the redo"


def test_a_readable_existing_clip_is_still_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "p0__RVRA.mov"
    out.write_bytes(b"a whole clip")
    monkeypatch.setattr(T, "_readable", lambda path: True)
    result = T.process(tmp_path / "v.mp4", tmp_path / "a.wav", out, 77)
    assert result.ok and result.checks == {"skipped": "exists"}


def test_each_writer_has_its_own_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two muxes over one job list must never write the same temporary path."""
    seen: list[Path] = []
    monkeypatch.setattr(T, "_readable", lambda path: True)
    monkeypatch.setattr(T, "_assert_inputs", lambda *a: {})
    monkeypatch.setattr(T, "measure_loudness", lambda *a: {})
    monkeypatch.setattr(T, "compute_gain_db", lambda *a: 0.0)
    monkeypatch.setattr(T, "solve_corrections", lambda *a: (_ for _ in ()).throw(T.TerminalError("stop")))
    real_unlink = Path.unlink

    def spy(self: Path, missing_ok: bool = False) -> None:
        seen.append(self)
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", spy)
    T.process(tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "p0__RVRA.mov", 77)
    assert len(seen) == 1
    name = seen[0].name
    assert name.startswith("p0__RVRA.mov.") and name.endswith(".tmp") and name != "p0__RVRA.mov.tmp"
    assert str(os.getpid()) in name


# --------------------------------------------------------------------------- withdrawing half-shipped pairs
def _rows(pair_id: str) -> list[dict[str, str]]:
    return [{"sample_id": f"{pair_id}__{c}", "class_name": c} for c in CLASSES]


def _ship(root: Path, pair_id: str, classes: tuple[str, ...]) -> None:
    for c in classes:
        path = S.final_sample_path(c, f"{pair_id}__{c}", root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("clip")
        path.with_suffix(".json").write_text('{"checks": {}}')


def test_a_complete_pair_is_left_alone(tmp_path: Path) -> None:
    _ship(tmp_path, "p_full", CLASSES)
    incomplete, complete, absent = W.classify({"p_full": _rows("p_full")}, tmp_path)
    assert (incomplete, complete, absent) == ({}, 1, 0)


def test_a_pair_that_shipped_nothing_is_not_withdrawn(tmp_path: Path) -> None:
    """A refused render drops the whole pair upstream; it is not an incomplete group."""
    incomplete, complete, absent = W.classify({"p_none": _rows("p_none")}, tmp_path)
    assert (incomplete, complete, absent) == ({}, 0, 1)


def test_a_half_shipped_pair_is_found_and_moved(tmp_path: Path) -> None:
    """The real-classes-only case: exactly the shape that leaks the label."""
    _ship(tmp_path, "p_half", ("RVRA", "RVFA"))
    expected = {"p_half": _rows("p_half")}
    incomplete, complete, absent = W.classify(expected, tmp_path)
    assert complete == 0
    assert absent == 0
    assert set(incomplete) == {"p_half"}
    assert {r["class_name"] for r in incomplete["p_half"]["missing"]} == {"FVRA", "FVFA"}

    # A dry run must not touch anything.
    W.withdraw("p_half", incomplete["p_half"], tmp_path, apply=False)
    assert S.final_sample_path("RVRA", "p_half__RVRA", tmp_path).exists()

    moved = W.withdraw("p_half", incomplete["p_half"], tmp_path, apply=True)
    assert not S.final_sample_path("RVRA", "p_half__RVRA", tmp_path).exists()
    assert not S.final_sample_path("RVFA", "p_half__RVFA", tmp_path).exists()
    dest = tmp_path / "_withdrawn" / "p_half"
    # Clip AND its record travel together, or the record is orphaned.
    assert len(moved) == 4
    assert {p.name for p in dest.iterdir()} == set(moved) | {"reason.json"}
    reason = json.loads((dest / "reason.json").read_text())
    assert reason["reason"] == "incomplete_group"
    assert sorted(reason["missing"]) == ["p_half__FVFA", "p_half__FVRA"]


def test_withdrawing_is_idempotent(tmp_path: Path) -> None:
    """Re-running after a mux must not fail on a pair already moved aside."""
    _ship(tmp_path, "p_half", ("RVRA",))
    expected = {"p_half": _rows("p_half")}
    incomplete, _, _ = W.classify(expected, tmp_path)
    W.withdraw("p_half", incomplete["p_half"], tmp_path, apply=True)
    # Second pass: nothing is on disk any more, so the pair reads as never shipped.
    again, complete, absent = W.classify(expected, tmp_path)
    assert again == {}
    assert (complete, absent) == (0, 1)
