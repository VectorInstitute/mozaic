"""Tests for the annotation tasks, on the made-up release (no videos)."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

import pytest
from made_up import (
    P1,
    P4,
    P5,
    P6,
    SMALL_COUNTS,
    clip,
    made_up_release,
    real_clip,
    task3_test_rows,
)

from mozaic.annotation.sampling import counted_sample
from mozaic.annotation.t1_spot_the_fake import build_questions as t1
from mozaic.annotation.t2_manipulation_type import build_questions as t2
from mozaic.annotation.t3_identity_check import build_questions as t3
from mozaic.common.io import read_jsonl
from mozaic.common.metadata import SPLITS


@pytest.fixture
def release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The made-up release as the dataset root (MOZAIC_ROOT)."""
    monkeypatch.setenv("MOZAIC_ROOT", str(tmp_path / "mozaic"))
    return made_up_release(tmp_path / "mozaic")


def pair_id(p: tuple) -> str:
    return f"{p[0]}__{p[1]}"


# --------------------------------------------------------------------------- #
# T1 and T2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "arm", "answer", "video_fake", "audio_fake"),
    [("RVRA", "main", "neither", 0, 0), ("RVRA", "vocoded", "neither", 0, 0), ("RVFA", "main", "the audio", 0, 1),
     ("FVRA", "vocoded", "the video", 1, 0), ("FVFA", "main", "both", 1, 1)],
)
def test_t1_answer_follows_the_label_and_the_option_order_is_fixed_per_clip(
    label: str, arm: str, answer: str, video_fake: int, audio_fake: int
) -> None:
    c = clip(label, "s2v", arm, "train", "id00001_aaa_00001", "id00002", "153f")
    row = t1.make_row(c)
    assert (row["gt_answer"], row["gt_label"]) == (answer, label)
    assert row["gt_target"] == {"video_fake": video_fake, "audio_fake": audio_fake}
    assert sorted(row["options"]) == sorted(t1.OPTIONS)
    assert row == t1.make_row(c)  # the same option order in every build
    assert row["meta"] == {"video_method": c["video_generator"], "frames": 153, "arm": arm, "source_id": "id00001",
                           "target_id": "id00002", "pair_id": "id00001_aaa_00001__id00002"}


def test_t2_asks_each_video_once(release: list[dict[str, Any]], tmp_path: Path) -> None:
    t2.build(out=str(tmp_path / "t2"))
    rows = [r for s in SPLITS for r in read_jsonl(tmp_path / "t2" / f"t2_{s}.jsonl")]
    videos = [c for c in release if (c["label"], c["arm"]) in (("RVRA", "main"), ("FVRA", "main"))]
    assert sorted(r["clips"][0] for r in rows) == sorted(c["file_name"] for c in videos)
    names = {r["meta"]["video_method"]: r["gt_target_name"] for r in rows}
    assert names == {"real": "real", "facefusion": "face_swap", "s2v": "fully_generated",
                     "animate": "full_body_swap"}


# --------------------------------------------------------------------------- #
# T3
# --------------------------------------------------------------------------- #
def test_t3_asks_each_picture_and_sound_once_with_both_people(release: list[dict[str, Any]]) -> None:
    rows = t3.make_rows(release)
    per_pair = collections.Counter((r["meta"]["pair_id"], r["part"]) for r in rows)
    p1, p4, p6 = pair_id(P1), pair_id(P4), pair_id(P6)
    # P1: 3 pictures, 2 sounds (vocoded real, converted); P4: 2 and 2; P6: 2 and 2 (its unclear
    # converted voice too); x A, B. The raw real voice is not asked (the vocoded view).
    assert per_pair == {(p1, "face"): 6, (p1, "voice"): 4, (p4, "face"): 4, (p4, "voice"): 4, (p6, "face"): 4,
                        (p6, "voice"): 4}
    assert collections.Counter(r["gt_answer"] for r in rows) == {
        "real, same person": 6, "real, different person": 6, "fake of this person": 7, "fake of a different person": 7}
    assert collections.Counter((r["part"], r["meta"]["clip2_person"]) for r in rows) == {
        ("face", "A"): 7, ("face", "B"): 7, ("voice", "A"): 6, ("voice", "B"): 6}
    answer = {(r["part"], r["gt_label"], r["meta"]["arm"], r["meta"]["clip2_person"]): r["gt_answer"]
              for r in rows if r["meta"]["pair_id"] == p1}
    assert answer == {
        ("face", "RVRA", "main", "A"): "real, same person",  # A's real face
        ("face", "RVRA", "main", "B"): "real, different person",
        ("face", "FVRA", "main", "A"): "fake of a different person",  # the fakes show B's face
        ("face", "FVRA", "main", "B"): "fake of this person",
        ("voice", "RVRA", "vocoded", "A"): "real, same person",  # only processed: still A's real voice
        ("voice", "RVRA", "vocoded", "B"): "real, different person",
        ("voice", "RVFA", "main", "A"): "fake of a different person",  # converted to B's voice
        ("voice", "RVFA", "main", "B"): "fake of this person",
    }
    names = dict(zip(t3.OPTIONS, t3.TARGET_NAMES, strict=True))
    for r in rows:
        assert r["id"].startswith("t3_") and r["task"] == t3.TASK
        assert r["question"] == t3.QUESTIONS[r["part"]]
        fake, same = (r["gt_target"]["fake"], r["gt_target"]["same_person"])
        assert t3.OPTIONS[t3.TARGETS.index((fake, same))] == r["gt_answer"]
        assert r["gt_target_name"] == names[r["gt_answer"]]
        assert fake == (r["gt_label"][0 if r["part"] == "face" else 2] == "F")


def test_t3_shuffles_the_options_per_question_the_same_in_every_build(release: list[dict[str, Any]]) -> None:
    rows = t3.make_rows(release)
    assert all(sorted(r["options"]) == sorted(t3.OPTIONS) for r in rows)
    assert len({tuple(r["options"]) for r in rows}) > 1  # not one order for every question
    assert len({r["options"].index(r["gt_answer"]) for r in rows}) > 1  # the right answer moves
    assert t3.make_rows(release) == rows  # the same order in every build


def test_t3_clip2_is_a_real_clip_from_a_video_the_pair_never_used(release: list[dict[str, Any]]) -> None:
    rows = t3.make_rows(release)
    pool = t3.real_clips(release)
    one = {c["pair_id"]: c for c in release}  # any clip of a pair carries the pair's fields
    p1, p5 = one[pair_id(P1)], one[pair_id(P5)]
    # A: not P1's own video aaa, and not the 153-frame clip from ggg
    assert t3.clip2_choices(p1, "A", pool) == [real_clip(*P4[:2], "77f")]
    # B: not P6's clip, whose video fff is the one B's face picture came from
    assert t3.clip2_choices(p1, "B", pool) == [real_clip(*P5[:2], "77f")]
    assert t3.clip2_choices(p5, "A", pool) == []  # so P5 is left out, with P2, P3 and P7
    assert {r["meta"]["pair_id"] for r in rows} == {pair_id(p) for p in (P1, P4, P6)}
    clip2: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for r in rows:
        clip2[r["meta"]["pair_id"], r["meta"]["clip2_person"]].add(r["clips"][1])
    assert len(clip2) == 6 and all(len(names) == 1 for names in clip2.values())  # one per pair and person


def test_t3_rows_do_not_depend_on_the_order_of_the_clips(release: list[dict[str, Any]]) -> None:
    rows = sorted(t3.make_rows(release), key=lambda r: r["id"])
    assert sorted(t3.make_rows(list(reversed(release))), key=lambda r: r["id"]) == rows


def test_t3_build_writes_sorted_files_and_its_guard_stops_a_wrong_answer(
    release: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t3.build(out=str(tmp_path / "t3"))
    train = read_jsonl(tmp_path / "t3" / "t3_train.jsonl")
    assert len(train) == 26 and [r["id"] for r in train] == sorted(r["id"] for r in train)
    assert read_jsonl(tmp_path / "t3" / "t3_test.jsonl") == []  # the test pair P3 has no clip 2

    right = t3.make_row

    def wrong(c: dict[str, Any], part: str, person: str, clip2: str) -> dict[str, Any]:
        row = right(c, part, person, clip2)
        return {**row, "gt_answer": "real, same person"} if row["gt_label"] == "RVFA" and person == "B" else row

    monkeypatch.setattr(t3, "make_row", wrong)
    with pytest.raises(ValueError, match="nothing written"):
        t3.build(out=str(tmp_path / "t3_wrong"))
    assert not (tmp_path / "t3_wrong").exists()


# --------------------------------------------------------------------------- #
# samples
# --------------------------------------------------------------------------- #
def test_counted_sample_picks_each_kind_from_different_pairs() -> None:
    rows = task3_test_rows(10)
    picks = counted_sample(rows, SMALL_COUNTS, 1, t3.kind)
    assert collections.Counter(t3.kind(r) for r in picks) == SMALL_COUNTS
    assert len({r["meta"]["pair_id"] for r in picks}) == 8
    assert t3.sample_problems(picks, SMALL_COUNTS, "test") == []
    assert picks == counted_sample(rows, SMALL_COUNTS, 1, t3.kind)  # same seed, same picks
    assert t3.sample_problems(picks[:-1], SMALL_COUNTS, "test") != []  # one question missing
    with pytest.raises(ValueError, match="not enough"):
        counted_sample(task3_test_rows(3), SMALL_COUNTS, 1, t3.kind)  # 8 questions need 8 pairs


def test_counted_sample_spreads_every_kind_over_the_list() -> None:
    rows = [{"kind": k, "meta": {"pair_id": f"{k}{i}"}} for k in "ab" for i in range(3)]
    picks = counted_sample(rows, {"a": 3, "b": 3}, 1, lambda r: r["kind"])
    assert [sorted(r["kind"] for r in picks[i : i + 2]) for i in (0, 2, 4)] == [["a", "b"]] * 3


def test_sample_follows_the_counts_and_refuses_a_wrong_size_or_an_unbalanced_table() -> None:
    rows = sorted(task3_test_rows(10), key=lambda r: r["id"])
    settings = {"split": "test", "counts": [[*k, n] for k, n in SMALL_COUNTS.items()]}
    assert collections.Counter(t3.kind(r) for r in t3.sample(rows, 8, 1, settings)) == SMALL_COUNTS
    with pytest.raises(ValueError, match="--size must be 8"):
        t3.sample(rows, 10, 1, settings)
    with pytest.raises(ValueError, match="not balanced"):
        t3.sample(rows, 1, 1, {"split": "test", "counts": [["face", "real", 77, "A", 1]]})


def test_sample_counts_in_config_are_balanced() -> None:
    counts = {tuple(k): n for *k, n in t3.CONFIG["sample"]["counts"]}
    size = sum(counts.values())
    assert size == 40
    totals: collections.Counter[Any] = collections.Counter()
    assert not [k for k in counts if k[:2] == ("voice", "real")]  # the vocoded view: real voices are vocoded ones
    for (part, shows, frames, person), n in counts.items():
        fake = shows not in ("real", "vocoded")  # a fake shows B; a real face or voice is A's
        same = person == ("B" if fake else "A")
        totals[part, t3.OPTIONS[t3.TARGETS.index((int(fake), int(same)))]] += n
        totals[part, person] += n
        assert not (shows == "animate" and frames == 153)  # Animate has 77 frames only
    for part in ("face", "voice"):
        assert [totals[part, o] for o in t3.OPTIONS] == [size // 8] * 4
        assert totals[part, "A"] == totals[part, "B"] == size // 4
