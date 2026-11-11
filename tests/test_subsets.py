"""Tests for MOZAIC-Small: the draw, on made-up pairs; its files and the --subset switch, on a made-up release."""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Any

import pytest
from made_up import P1, made_up_release, pair_clips, real_clip

from mozaic.annotation.t1_spot_the_fake import build_questions as t1
from mozaic.annotation.t3_identity_check import build_questions as t3
from mozaic.common import paths, subsets
from mozaic.common.io import read_jsonl
from mozaic.common.metadata import SPLITS
from mozaic.common.paths import (
    metadata_path,
    qas_dir,
    results_dir,
    summary_path,
    training_dir,
    use_subset,
)


def made_up_pairs() -> list[dict[str, Any]]:
    """12 people, 5 pairs each as the source, over two lengths, three generator sets, two genders."""
    kinds = [("77f", ["animate", "facefusion", "s2v"]), ("77f", ["facefusion"]), ("153f", ["facefusion", "s2v"]),
             ("153f", ["facefusion", "s2v"]), ("77f", ["animate", "facefusion", "s2v"])]
    return [{"pair_id": f"id{person:05d}_v{i}_00001__id{90000 + person}", "source_id": f"id{person:05d}",
             "target_id": f"id{90000 + person}", "length": length, "kept_by": kept_by,
             "gender": "m" if person < 6 else "f"}
            for person in range(12) for i, (length, kept_by) in enumerate(kinds)]


def test_quotas_follow_the_full_mix_and_add_up() -> None:
    pairs = made_up_pairs()
    q = subsets.quotas(pairs, 20)
    assert sum(q.values()) == 20
    full = collections.Counter(subsets.stratum(p) for p in pairs)
    assert all(abs(q[s] - 20 * n / len(pairs)) < 1 for s, n in full.items())  # within one pair of the exact share


def test_draw_covers_every_person_keeps_the_mix_and_the_limit() -> None:
    pairs = made_up_pairs()
    picked = subsets.draw(pairs, 20, "7:test")
    assert len(picked) == len({p["pair_id"] for p in picked}) == 20
    per_person = collections.Counter(p["source_id"] for p in picked)
    assert set(per_person) == {p["source_id"] for p in pairs}  # every person is a source
    assert max(per_person.values()) <= math.ceil(20 / 12)
    assert collections.Counter(subsets.stratum(p) for p in picked) == {
        s: n for s, n in subsets.quotas(pairs, 20).items() if n}
    assert subsets.draw(pairs, 20, "7:test") == picked  # same seed, same pairs
    assert subsets.draw(pairs, 20, "8:test") != picked


def test_draw_refuses_fewer_pairs_than_people() -> None:
    with pytest.raises(ValueError, match="cannot cover"):
        subsets.draw(made_up_pairs(), 11, "7:test")


# --------------------------------------------------------------------------- #
# the files: a made-up full release of four pairs, one drawn per split
# --------------------------------------------------------------------------- #
Q1 = ("id00001_aaa_00001", "id00002", "train", "77f", ["facefusion"])
Q2 = ("id00001_ddd_00004", "id00002", "train", "77f", ["facefusion"])  # same source as Q1: train gets one of the two
Q3 = ("id00003_bbb_00002", "id00004", "val", "153f", ["s2v"])
Q4 = ("id00005_ccc_00003", "id00006", "test", "77f", ["facefusion"])


def lines(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


@pytest.fixture
def full_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """metadata, pairs, splits, summary and the parquet table of four pairs; about 6 clips per Small split."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "mozaic"
    root.mkdir()
    monkeypatch.setenv("MOZAIC_ROOT", str(root))
    monkeypatch.setattr(paths, "_subset", None)
    monkeypatch.setattr(subsets, "TARGET_CLIPS", dict.fromkeys(SPLITS, 6))
    rows = sorted(({**c, "gender": "m"} for p in (Q1, Q2, Q3, Q4) for c in pair_clips(*p)), key=lambda r: r["file_name"])
    (root / "metadata.jsonl").write_text(lines(rows))
    pq.write_table(pa.Table.from_pylist(rows), root / "metadata_table.parquet")
    pairs = [{"pair_id": f"{p[0]}__{p[1]}", "length": p[3], "split": p[2], "source_id": p[0][:7], "target_id": p[1]}
             for p in (Q1, Q2, Q3, Q4)]
    (root / "pairs.jsonl").write_text(lines(pairs))
    splits = {person: p["split"] for p in pairs for person in (p["source_id"], p["target_id"])}
    (root / "splits.json").write_text(json.dumps(splits))
    folders: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for r in rows:
        folders[r["file_name"].rsplit("/", 1)[0]][r["split"]] += 1
    summary = {"name": "MOZAIC", "version": "1.0", "schema_version": "1.0", "source": {"people": len(splits)},
               "counts": {"clips": len(rows), "pairs": dict(collections.Counter(p["length"] for p in pairs)),
                          "by_label": dict(collections.Counter(r["label"] for r in rows)),
                          "by_split": dict(collections.Counter(r["split"] for r in rows)),
                          "by_folder": {f: dict(n) for f, n in sorted(folders.items())}},
               "people_per_split": dict(collections.Counter(splits.values()))}  # fmt: skip
    (root / "summary.json").write_text(json.dumps(summary))
    return root


def test_write_small_writes_four_files_of_unchanged_rows_next_to_the_full_ones(full_release: Path) -> None:
    import pyarrow.parquet as pq

    full_files = {p.name: p.read_bytes() for p in full_release.iterdir()}
    paths = subsets.write_small()
    assert [p.name for p in paths] == ["metadata_small.jsonl", "pairs_small.jsonl", "summary_small.json",
                                       "metadata_table_small.parquet"]
    assert {p.name: p.read_bytes() for p in full_release.iterdir() if p.name in full_files} == full_files  # untouched
    assert not (full_release / "subsets").exists() and not (full_release / "videos").exists()  # nothing copied

    full_lines = (full_release / "metadata.jsonl").read_text().splitlines()
    small_lines = (full_release / "metadata_small.jsonl").read_text().splitlines()
    assert len(small_lines) == 18 and set(small_lines) <= set(full_lines)  # 3 whole pairs, rows unchanged
    rows = [json.loads(x) for x in small_lines]
    assert collections.Counter(r["split"] for r in rows) == dict.fromkeys(SPLITS, 6)
    assert pq.read_table(paths[3]).to_pylist() == rows

    summary = json.loads(paths[2].read_text())
    assert summary["name"] == "MOZAIC-Small" and summary["counts"]["clips"] == 18
    assert summary["counts"]["pairs"] == {"77f": 2, "153f": 1}
    assert list(summary)[:4] == ["name", "version", "schema_version", "subset"]
    assert summary["subset"]["seed"] == 7 and summary["subset"]["splits"]["train"]["full_counts"]["pairs"] == 2
    assert subsets.write_small() == paths and (full_release / "metadata_small.jsonl").read_text().splitlines() == small_lines


def test_paths_follow_the_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOZAIC_ROOT", str(tmp_path))
    monkeypatch.setattr(paths, "_subset", None)
    full = [metadata_path(), summary_path(), qas_dir("t1"), results_dir("m", "e", "t1"), training_dir("m", "e")]
    assert [p.relative_to(tmp_path).as_posix() for p in full] == [
        "metadata.jsonl", "summary.json", "qas/t1_spot_the_fake", "results/m/e/t1", "training/m/e"]
    use_subset("small")
    small = [metadata_path(), summary_path(), qas_dir("t1"), results_dir("m", "e", "t1"), training_dir("m", "e")]
    assert [p.relative_to(tmp_path).as_posix() for p in small] == [
        "metadata_small.jsonl", "summary_small.json", "qas_small/t1_spot_the_fake", "results_small/m/e/t1",
        "training_small/m/e"]
    assert (metadata_path(full=True), summary_path(full=True)) == (full[0], full[1])
    with pytest.raises(ValueError, match="unknown subset tiny"):
        use_subset("tiny")
    use_subset(None)
    assert qas_dir("t1") == full[2]


def test_small_question_files_are_the_full_rows_of_small_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Built on Small, T1 and T3 hold exactly the full build's rows of Small's pairs. P1's clip 2 for A
    is P4's real clip, outside Small: T3 still draws it, from the full metadata."""
    root = tmp_path / "mozaic"
    monkeypatch.setenv("MOZAIC_ROOT", str(root))
    monkeypatch.setattr(paths, "_subset", None)
    release = made_up_release(root)
    p1 = f"{P1[0]}__{P1[1]}"
    (root / "metadata_small.jsonl").write_text(lines([c for c in release if c["pair_id"] == p1]))
    (root / "summary_small.json").write_text(json.dumps({"counts": {"pairs": {"77f": 1}}}))

    t1.build()
    t3.build()
    full = {task: read_jsonl(qas_dir(task) / f"{task}_train.jsonl") for task in ("t1", "t3")}
    use_subset("small")
    t1.build()
    t3.build()
    for task in ("t1", "t3"):
        small = read_jsonl(root / "qas_small" / qas_dir(task).name / f"{task}_train.jsonl")
        assert small and small == [r for r in full[task] if r["meta"]["pair_id"] == p1]
    assert real_clip("id00001_ddd_00004", "id00002", "77f") in {r["clips"][1] for r in small}
