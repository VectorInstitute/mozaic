"""Build the T3 question file from metadata.jsonl.

Two clips per question. Clip 1 is a clip of a pair; clip 2 is a real recording of one
of the pair's two people: A (source_id) or B (target_id). A face question asks whether
the face in clip 1 is real or fake, and whether it is clip 2's person; the model gets the
frames of both clips, no sound. A voice question asks the same about the voice; the
model gets the sound of both clips, no frames. Four options: real or fake, times same
person or not. The answer comes from the label and from whose face and whose voice clip 1
was made to show, so no model is used. Writes one file per split, or with --size the
sample set in config.yaml.

On a subset (--subset small) clip 1 comes from the subset, clip 2 from the full metadata:
clip 2 is drawn per pair and person, so every question is the same as in the full build.

Before anything is written, every row is worked out again another way (check below).
If any row disagrees, nothing is written.
"""

from __future__ import annotations

import collections
import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml

from mozaic.annotation.sampling import counted_sample, shuffled_options
from mozaic.annotation.writer import write_questions
from mozaic.common.metadata import FRAMES, SPLITS, load_metadata
from mozaic.common.paths import metadata_path, summary_path

Row = dict[str, Any]
Pool = dict[tuple[str, str], list[Row]]  # (person, length) -> that person's RVRA main clips
log = logging.getLogger(__name__)
CONFIG = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())

TASK = "task3_identity_check"
SEED = CONFIG["clip2_seed"]
PARTS = ("face", "voice")
PEOPLE = ("A", "B")  # whose recording clip 2 is: A = source_id, B = target_id
QUESTIONS = {
    "face": (
        "Clip 2 is a real recording of one person. Look only at the faces (the inner face: eyes, nose and mouth). "
        "Is the face in clip 1 real or fake, and is it the same person as in clip 2? A real recording that was only "
        "processed, without changing whose face it is, still counts as real."
    ),
    "voice": (
        "Clip 2 is a real recording of one person. Listen only to the voices. Is the voice in clip 1 real or fake, "
        "and is it the same person as in clip 2? A real recording that was only processed, without changing whose "
        "voice it is, still counts as real."
    ),
}
# The four options. What each one means is explained in the prompt (evaluation/prompts/t3_prompts.py).
OPTIONS = ["real, same person", "real, different person", "fake of this person", "fake of a different person"]
TARGETS = [(0, 1), (0, 0), (1, 1), (1, 0)]  # (fake, same_person) of each option
TARGET_NAMES = ["real_same_person", "real_different_person", "fake_this_person", "fake_different_person"]
# The clips that ask about each picture and each sound of a pair, as (label, arm). FVRA
# main is one clip per generator that kept the pair. Voices are the vocoded view, as in
# T1 and T4: the real voice is its vocoded copy only, so every voice in clip 1 has been
# through a vocoder. The pair's other clips repeat one of these pictures or sounds, so
# they are not asked again (no FVFA clip appears).
ASKED = {
    "face": (("RVRA", "main"), ("FVRA", "main")),
    "voice": (("RVRA", "vocoded"), ("RVFA", "main")),
}
VOICE_KIND = {"none": "real", "bigvgan": "vocoded", "seed_vc": "converted"}  # audio_generator -> the sound


def answer_fields(fake: int, same_person: int) -> dict[str, Any]:
    """gt_answer, gt_target and gt_target_name for one answer."""
    i = TARGETS.index((fake, same_person))
    return {"gt_answer": OPTIONS[i], "gt_target": {"fake": fake, "same_person": same_person},
            "gt_target_name": TARGET_NAMES[i]}


# --------------------------------------------------------------------------- #
# build: per pair, its pictures and sounds, each with clip 2 = A and clip 2 = B
# --------------------------------------------------------------------------- #
def video_id(clip_id: str) -> str:
    """A clip id's YouTube video: id00012_C_FAL9gv8bo_00031 -> C_FAL9gv8bo."""
    return clip_id[8:-6]  # the person is the first 7 characters, the segment the last 5


def person_id(clip: Row, person: str) -> str:
    """The id of A (the source) or B (the target) of a clip's pair."""
    return str(clip["source_id"] if person == "A" else clip["target_id"])


def real_clips(clips: list[Row]) -> Pool:
    """Every RVRA main clip by (person, length): the candidates for clip 2."""
    pool: Pool = collections.defaultdict(list)
    for clip in clips:
        if (clip["label"], clip["arm"]) == ("RVRA", "main"):
            pool[clip["source_id"], clip["length"]].append(clip)
    return pool


def clip2_choices(clip: Row, person: str, pool: Pool) -> list[str]:
    """The possible clip 2s for one person of a clip's pair, sorted.

    Each is an RVRA main clip of that person, of the pair's length, from a YouTube video
    the pair never used: not the source clip's, nor those of the target's face picture
    and voice reference.
    """
    used = {video_id(clip[k]) for k in ("source_clip", "target_face_clip", "target_voice_clip")}
    real = pool.get((person_id(clip, person), clip["length"]), [])
    return sorted(c["file_name"] for c in real if video_id(c["source_clip"]) not in used)


def draw(choices: list[str], pair_id: str, person: str) -> str | None:
    """Draw clip 2 from the choices; None if there is none.

    Seeded by SEED, the pair and the person, so each draw stands alone: the same in every
    build, whatever the order of the clips.
    """
    return random.Random(f"{SEED}:{pair_id}:{person}").choice(choices) if choices else None


def asked_clips(pair: list[Row]) -> list[tuple[str, Row]]:
    """(part, clip 1) for every picture and every sound of one pair, once each."""
    face = [("face", c) for c in pair if (c["label"], c["arm"]) in ASKED["face"]]
    return face + [("voice", c) for c in pair if (c["label"], c["arm"]) in ASKED["voice"]]


def make_row(clip: Row, part: str, person: str, clip2: str) -> Row:
    """The question for one clip 1, one part and one clip 2.

    From the label: a real face (RV..) or a real voice (..RA, main or vocoded) is A's; a
    fake one (FV.., ..FA) was made to show B. The answer: real or fake, and whether that
    person is clip 2's.
    """
    fake = int(clip["label"][0 if part == "face" else 2] == "F")
    shown = "B" if fake else "A"
    question_id = f"t3_{part}_{person}_" + Path(clip["file_name"]).stem
    answer = answer_fields(fake, int(shown == person))
    return {
        "id": question_id,
        "task": TASK,
        "split": clip["split"],
        "clips": [clip["file_name"], clip2],
        "question": QUESTIONS[part],
        "options": shuffled_options(OPTIONS, CONFIG["options_seed"], question_id),
        "part": part,
        "gt_answer": answer["gt_answer"],
        "gt_label": clip["label"],
        "gt_target": answer["gt_target"],
        "gt_target_name": answer["gt_target_name"],
        "meta": {
            "video_method": clip["video_generator"],
            "audio_method": clip["audio_generator"],
            "frames": FRAMES[clip["length"]],
            "arm": clip["arm"],
            "source_id": clip["source_id"],
            "target_id": clip["target_id"],
            "pair_id": clip["pair_id"],
            "clip2_person": person,
        },
    }


def make_rows(clips: list[Row], recordings: list[Row] | None = None) -> list[Row]:
    """Every row; logs the pairs left out. Clip 2 is drawn from recordings (default: clips)."""
    pool = real_clips(recordings or clips)
    pairs: dict[str, list[Row]] = collections.defaultdict(list)
    for clip in clips:
        pairs[clip["pair_id"]].append(clip)
    rows: list[Row] = []
    left_out: list[str] = []
    for pair_id, pair in sorted(pairs.items()):
        first = pair[0]  # every clip of a pair carries the pair's fields
        drawn = {person: draw(clip2_choices(first, person, pool), pair_id, person) for person in PEOPLE}
        clip2 = {person: c for person, c in drawn.items() if c is not None}
        if len(clip2) < len(PEOPLE):
            missing = " or ".join(sorted(drawn.keys() - clip2.keys()))
            left_out.append(f"{pair_id} ({first['split']}, no clip 2 for {missing})")
            continue
        rows += [make_row(clip, part, person, clip2[person]) for part, clip in asked_clips(pair) for person in PEOPLE]
    log.info("%s: %s pairs; %d left out, A or B has no clip 2: %s", TASK, f"{len(pairs):,}", len(left_out), left_out[:20])
    return rows


# --------------------------------------------------------------------------- #
# check: every row worked out again another way, before anything is written
# --------------------------------------------------------------------------- #
def expected_clip2(pair_id: str, clip: Row, pool: Pool) -> dict[str, str] | None:
    """Clip 2 worked out again for A and B of a pair; None if either has none."""
    used = {video_id(clip["source_clip"]), video_id(clip["target_face_clip"]), video_id(clip["target_voice_clip"])}
    picks: dict[str, str] = {}
    for person, who in (("A", clip["source_id"]), ("B", clip["target_id"])):
        names = sorted(c["file_name"] for c in pool.get((who, clip["length"]), [])
                       if video_id(c["source_clip"]) not in used)
        if not names:
            return None
        picks[person] = random.Random(f"{SEED}:{pair_id}:{person}").choice(names)
    return picks


def expected_clip1s(pair: list[Row]) -> tuple[list[tuple[str, Row]], list[str]]:
    """A pair's clip 1s found again from face_real, voice_real, arm and kept_by.

    Face: the real-face, real-voice main clip, then for each generator in kept_by its
    fake-face, real-voice main clip. Voice: every real-face clip but the raw real voice
    (real-voice main). Returns (part, clip) pairs and problems.
    """
    first = pair[0]
    real_voice_main = [c for c in pair if c["voice_real"] and c["arm"] == "main"]
    face = [c for c in real_voice_main if c["face_real"]]
    problems = [] if len(face) == 1 else [f"{first['pair_id']}: {len(face)} real-face real-voice main clips"]
    for generator in first["kept_by"]:
        made = [c for c in real_voice_main if not c["face_real"] and c["video_generator"] == generator]
        face += made
        if len(made) != 1:
            problems.append(f"{first['pair_id']}: {len(made)} fake-face real-voice main clips by {generator}")
    voice = [c for c in pair if c["face_real"] and not (c["voice_real"] and c["arm"] == "main")]
    return [("face", c) for c in face] + [("voice", c) for c in voice], problems


def expected_row(clip: Row, part: str, person: str, clip2: str) -> Row:
    """One row worked out again: the answer from face_real / voice_real and face_id / voice_id."""
    who = clip["source_id"] if person == "A" else clip["target_id"]
    real = clip["face_real" if part == "face" else "voice_real"]
    same = clip["face_id" if part == "face" else "voice_id"] == who
    question_id = f"t3_{part}_{person}_" + clip["file_name"].rsplit("/", 1)[1].removesuffix(".mov")
    answer = answer_fields(int(not real), int(same))
    return {
        "id": question_id,
        "task": TASK,
        "split": clip["split"],
        "clips": [clip["file_name"], clip2],
        "question": QUESTIONS[part],
        "options": shuffled_options(OPTIONS, CONFIG["options_seed"], question_id),
        "part": part,
        "gt_answer": answer["gt_answer"],
        "gt_label": ("R" if clip["face_real"] else "F") + "V" + ("R" if clip["voice_real"] else "F") + "A",
        "gt_target": answer["gt_target"],
        "gt_target_name": answer["gt_target_name"],
        "meta": {
            "video_method": clip["video_generator"],
            "audio_method": clip["audio_generator"],
            "frames": int(clip["length"].removesuffix("f")),
            "arm": clip["arm"],
            "source_id": clip["source_id"],
            "target_id": clip["target_id"],
            "pair_id": f"{clip['source_clip']}__{clip['target_id']}",
            "clip2_person": person,
        },
    }


def expected_rows(clips: list[Row], recordings: list[Row]) -> tuple[dict[str, Row], list[str]]:
    """Every row worked out again; returns (rows by id, problems)."""
    pool: Pool = collections.defaultdict(list)
    pairs: dict[str, list[Row]] = collections.defaultdict(list)
    for clip in clips:
        pairs[clip["pair_id"]].append(clip)
    for clip in recordings:
        if clip["face_real"] and clip["voice_real"] and clip["arm"] == "main":
            pool[clip["face_id"], clip["length"]].append(clip)
    rows: dict[str, Row] = {}
    problems: list[str] = []
    for pair_id, pair in pairs.items():
        clip2 = expected_clip2(pair_id, pair[0], pool)
        if clip2 is None:
            continue
        clip1s, found = expected_clip1s(pair)
        problems += found
        for part, clip in clip1s:
            for person in PEOPLE:
                row = expected_row(clip, part, person, clip2[person])
                rows[row["id"]] = row
    return rows, problems


def compare(rows: list[Row], expected: dict[str, Row]) -> list[str]:
    """Rows that are missing, not expected, or differ from the rows worked out again."""
    by_id = {r["id"]: r for r in rows}
    problems = [] if len(by_id) == len(rows) else ["ids are not unique"]
    problems += [f"{i}: missing" for i in sorted(expected.keys() - by_id.keys())]
    problems += [f"{i}: not expected" for i in sorted(by_id.keys() - expected.keys())]
    for i in sorted(by_id.keys() & expected.keys()):
        if by_id[i] != expected[i]:
            differs = sorted(k for k in by_id[i].keys() | expected[i].keys() if by_id[i].get(k) != expected[i].get(k))
            problems.append(f"{i}: {differs} differ when worked out again")
    return problems


def people_problems(rows: list[Row]) -> list[str]:
    """Each split and part must have as many clip 2 = A as clip 2 = B."""
    people = collections.Counter((r["split"], r["part"], r["meta"]["clip2_person"]) for r in rows)
    return [f"{s} {part}: clip 2 of A {people[s, part, 'A']:,}, of B {people[s, part, 'B']:,}"
            for s in SPLITS for part in PARTS if people[s, part, "A"] != people[s, part, "B"]]


def pair_problems(clips: list[Row]) -> list[str]:
    """The pairs per length must match summary.json (summary_small.json on Small)."""
    path = summary_path()
    summary = json.loads(path.read_text())
    pairs = collections.Counter(length for length, _ in {(c["length"], c["pair_id"]) for c in clips})
    if pairs != summary["counts"]["pairs"]:
        return [f"pairs per length {dict(pairs)}, {path.name} says {summary['counts']['pairs']}"]
    return []


def check(rows: list[Row], clips: list[Row], recordings: list[Row]) -> list[str]:
    """Every row worked out again another way, the A/B balance, the pairs."""
    expected, problems = expected_rows(clips, recordings)
    return problems + compare(rows, expected) + people_problems(rows) + pair_problems(clips)


# --------------------------------------------------------------------------- #
# samples: an exact number of questions per kind, set in config.yaml
# --------------------------------------------------------------------------- #
def kind(row: Row) -> tuple:
    """Question kind: part, what clip 1 shows, frames, whose recording clip 2 is."""
    meta = row["meta"]
    shows = meta["video_method"] if row["part"] == "face" else VOICE_KIND[meta["audio_method"]]
    return row["part"], shows, meta["frames"], meta["clip2_person"]


def sample_problems(picks: list[Row], counts: dict[tuple, int], split: str) -> list[str]:
    """What is wrong with a sample: kinds, pairs, split, or balance."""
    found = collections.Counter(kind(r) for r in picks)
    problems = [] if found == collections.Counter(counts) else [f"kinds {dict(found)} differ from {counts}"]
    if len({r["meta"]["pair_id"] for r in picks}) != len(picks):
        problems.append("two questions come from the same pair")
    problems += [f"{r['id']} is not in the {split} split" for r in picks if r["split"] != split]
    for part in PARTS:
        answers = collections.Counter(r["gt_answer"] for r in picks if r["part"] == part)
        people = collections.Counter(r["meta"]["clip2_person"] for r in picks if r["part"] == part)
        if len({answers[o] for o in OPTIONS}) != 1 or people["A"] != people["B"]:
            problems.append(f"{part}: answers {dict(answers)}, clip 2 of {dict(people)}")
    return problems


def sample(rows: list[Row], size: int, seed: int, settings: dict[str, Any]) -> list[Row]:
    """The sample set by the config's sample: settings; size must be its total."""
    counts = {tuple(k): n for *k, n in settings["counts"]}
    if size != sum(counts.values()):
        raise ValueError(f"this sample is set by the counts in config.yaml: --size must be {sum(counts.values())}")
    picks = counted_sample([r for r in rows if r["split"] == settings["split"]], counts, seed, kind)
    problems = sample_problems(picks, counts, settings["split"])
    if problems:
        raise ValueError(f"the sample is not balanced, nothing written: {problems}")
    return picks


def build(size: int | None = None, seed: int | None = None, out: str | None = None) -> list[Path]:
    clips = list(load_metadata().values())
    full = metadata_path(full=True)  # clip 2's pool: always the full set
    recordings = clips if metadata_path() == full else list(load_metadata(full).values())
    rows = sorted(make_rows(clips, recordings), key=lambda r: r["id"])

    problems = check(rows, clips, recordings)
    if problems:
        raise ValueError(f"{len(problems)} problems when the rows are worked out again, nothing written. "
                         f"First: {problems[:5]}")

    default_seed = CONFIG["sample"]["seed"]
    if size:
        seed = seed if seed is not None else default_seed
        rows = sample(rows, size, seed, CONFIG["sample"])

    paths = write_questions("t3", rows, size=size, seed=seed, default_seed=default_seed, out=out, answer_field="gt_answer")
    if size:
        log.info("kinds: %d, pairs: %d", len({kind(r) for r in rows}), len({r["meta"]["pair_id"] for r in rows}))
    return paths
