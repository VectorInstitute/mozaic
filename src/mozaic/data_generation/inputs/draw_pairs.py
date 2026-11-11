"""Draw the pairs of one build and write its pair list (manifest): one row per clip to make.

People are drawn in rings of 5 people of one gender, inside one split of split_map.json, so
every face and voice a clip borrows comes from the clip's own split. In a ring, person i's
folder f borrows from ring-mate (i + f + 1) % 5: each person borrows from each of the other
four once, never from themselves. Same gender only, or pitch alone would tell the
mismatched classes apart. Each person gives 4 folders (recording sessions) x 2 clips, so a
build has 8 pairs per person. One pair (a driver clip + the borrowed person) gives:

    class  face from  voice from  made by
    RVRA   driver     driver      nothing (real)
    RVFA   driver     borrowed    Seed-VC
    FVRA   borrowed   driver      the build's video generator
    FVFA   borrowed   borrowed    both (the same render and the same voice as above)

plus vocoded twins of RVRA and FVRA (every pair), a same-speaker RVFA and a self-drive FVRA
(10% of pairs each).

Per (split, gender), ONE seeded sample gives the built rings and the reserve rings, cut in
that order; only the built rings become pairs, and the summary records both. Keep a draw's
reserve count: random.sample picks its method from the sample size, so another count can
change the built rings too. Every choice has its own seed key (split, gender, person,
folder), never one shared stream.
--eligibility-frames 153 77 draws people who can give both lengths from different clips,
and shares each folder's clips out between the lengths, so the 77-frame and 153-frame builds
of one draw never use the same clip. --rings-from takes the rings of an earlier build, so
every build of one draw has the same people. The release commands are in README.md.

    python -m mozaic.data_generation.inputs.draw_pairs --build-id trial --splits train      # a trial draw
    python -m mozaic.data_generation.inputs.draw_pairs --build-id trial_ff --splits train \
        --video-method facefusion --rings-from trial                                         # same people

Reads  voxceleb2/split_map.json, voxceleb2/_reports/inventory.parquet
Writes new_generated/manifests/<build-id>/manifest.jsonl + manifest_summary.json (a written
       manifest is never overwritten: a change is a new build id)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from mozaic.data_generation import settings as S

# split -> identity -> folder (video_id) -> [(clip_id, duration_s), ...]
Pools = dict[str, dict[str, dict[str, list[tuple[str, float]]]]]

MAX_VOICE_REF_CLIPS = 40  # voice clips listed per row; Seed-VC uses the first

# Set once in main() from --crop-frames, before any pool is read.
ACTIVE_CROP_FRAMES = S.CROP_FRAMES

# Set once in main() from --eligibility-frames. None = one length. A value means one person
# serves several lengths from different clips, which changes eligibility and clip choice.
ELIGIBILITY_FRAMES: tuple[int, ...] | None = None


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Draw the pairs of one build and write its pair list.")
    p.add_argument("--build-id", required=True, help="build id, e.g. bulk_v1_wan77 (written into every row)")
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=S.SPLIT_SEED)
    p.add_argument("--ring-size", type=int, default=S.RING_SIZE)
    p.add_argument("--same-speaker-frac", type=float, default=S.SAME_SPEAKER_ARM_FRAC)
    p.add_argument("--self-drive-frac", type=float, default=S.SELF_DRIVE_CONTROL_FRAC)
    p.add_argument("--vocoded-real-frac", type=float, default=S.VOCODED_REAL_FRAC)
    p.add_argument(
        "--video-method",
        choices=list(S.VIDEO_METHODS),
        default=S.VIDEO_METHOD_ANIMATE,
        help="the video generator of this build (default: animate); it sets the render_key prefix",
    )
    p.add_argument(
        "--crop-frames",
        type=int,
        default=S.CROP_FRAMES,
        help="frames per clip in this build (default: run.yaml clips.crop_frames); "
        "another length adds __f<N> to crop_key",
    )
    p.add_argument(
        "--eligibility-frames",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="every length this draw serves, e.g. 153 77: a folder counts only if it can give "
        "CLIPS_PER_FOLDER different clips to each length (default: one length)",
    )
    p.add_argument(
        "--rings-per-split-gender",
        nargs="*",
        default=[],
        metavar="SPLIT=N",
        help="rings to build per (split, gender), e.g. train=70 val=15 test=15 (default: 1 everywhere)",
    )
    p.add_argument(
        "--reserve-rings-per-split-gender",
        nargs="*",
        default=[],
        metavar="SPLIT=N",
        help="reserve rings per (split, gender): drawn in the same sample as the built rings, "
        "recorded in the summary, never built (default: 0)",
    )
    p.add_argument(
        "--rings-from",
        default=None,
        metavar="BUILD_ID",
        help="take the rings from this build's summary (a build id or a .json path) instead of "
        "drawing them; every other build of one draw uses it",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="default: new_generated/manifests/<build-id>")
    return p.parse_args()


def substream(seed: int, *parts: str) -> random.Random:
    """A random generator seeded by the seed and a key, e.g. (split, gender, person)."""
    return random.Random(f"{seed}|" + "|".join(parts))


def load_pools() -> tuple[Pools, dict[str, str]]:
    """Read the split map + clip list into (pools, identity -> gender)."""
    split_map: dict[str, str] = json.loads(S.split_map_path().read_text())["groups"]
    inv = pd.read_parquet(S.inventory_parquet_path())
    usable = inv[inv["drop_reason"].isna()]

    pools: Pools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gender: dict[str, str] = {}
    for identity_id, video_id, clip_id, duration_s, g in zip(
        usable["identity_id"],
        usable["video_id"],
        usable["clip_id"],
        usable["duration_s"],
        usable["gender"],
        strict=True,
    ):
        split = split_map.get(identity_id)
        if split is None:
            continue  # a person with no usable clip when the split was made
        pools[split][identity_id][video_id].append((clip_id, float(duration_s)))
        gender[identity_id] = g if isinstance(g, str) and g else "unknown"
    return pools, gender


def clips_at_least(clips: list[tuple[str, float]], frames: int) -> list[tuple[str, float]]:
    """Clips long enough for a crop of *frames*."""
    return [c for c in clips if c[1] >= S.min_driver_clip_duration_s(frames)]


def long_clips(clips: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Clips long enough for a crop of this build's length."""
    return clips_at_least(clips, ACTIVE_CROP_FRAMES)


def folder_is_usable(clips: list[tuple[str, float]]) -> bool:
    """Whether this folder can give every length its own clips.

    One length: CLIPS_PER_FOLDER clips long enough. Several lengths: longest first, the k
    longest lengths together need k * CLIPS_PER_FOLDER different clips at least that long.
    With (153, 77): 2 clips >= 6.62 s and 4 clips >= 3.58 s. A folder of [7 s, 7 s, 3.9 s]
    fails: its two long clips go to 153 frames, leaving one clip for 77 frames.
    """
    if not ELIGIBILITY_FRAMES:
        return len(long_clips(clips)) >= S.CLIPS_PER_FOLDER
    return all(
        len(clips_at_least(clips, frames)) >= S.CLIPS_PER_FOLDER * rank
        for rank, frames in enumerate(sorted(ELIGIBILITY_FRAMES, reverse=True), start=1)
    )


def usable_folders(folders: dict[str, list[tuple[str, float]]]) -> list[str]:
    """Folders that can serve every length of this draw, sorted."""
    return sorted(vid for vid, clips in folders.items() if folder_is_usable(clips))


def allocate_stratum_clips(
    clips: list[tuple[str, float]], seed: int, key: tuple[str, ...]
) -> dict[int, list[tuple[str, float]]]:
    """Share one folder's clips out between the lengths: different clips, longest length first.

    The result depends only on (seed, key, the folder's whole clip list), never on the build
    being written, so the 77-frame and 153-frame builds of one draw get the same answer.
    """
    remaining = sorted(clips)
    out: dict[int, list[tuple[str, float]]] = {}
    for frames in sorted(ELIGIBILITY_FRAMES or (), reverse=True):
        pool = sorted(clips_at_least(remaining, frames))
        rng = substream(seed, *key, f"f{frames}")
        picked = sorted(rng.sample(pool, S.CLIPS_PER_FOLDER))
        out[frames] = picked
        taken = set(picked)
        remaining = [c for c in remaining if c not in taken]
    return out


def eligible_identities(
    pools: Pools, gender: dict[str, str], split: str, want_gender: str, folders_needed: int
) -> list[str]:
    """People of one gender in one split with enough usable folders, sorted."""
    out = [
        identity
        for identity, folders in pools[split].items()
        if gender.get(identity) == want_gender and len(usable_folders(folders)) >= folders_needed
    ]
    return sorted(out)


def pick_voice_refs(
    folders: dict[str, list[tuple[str, float]]], exclude_folders: set[str], rng: random.Random
) -> tuple[list[str], float]:
    """Voice clips from every folder except the excluded ones, and their total seconds.

    The face folder is left out whole, so the face and the voice never come from one
    recording session.
    """
    pool: list[tuple[str, float]] = []
    for vid in sorted(folders):
        if vid in exclude_folders:
            continue
        pool.extend(folders[vid])
    rng.shuffle(pool)
    picked = pool[:MAX_VOICE_REF_CLIPS]
    return [c[0] for c in picked], round(sum(c[1] for c in picked), 2)


def base_row(ctx: dict[str, Any]) -> dict[str, Any]:
    """Fields shared by every row of one driver clip."""
    return {
        "build_id": ctx["build_id"],
        "pair_id": ctx["pair_id"],
        "split": ctx["split"],
        "ring_gender": ctx["ring_gender"],
        "ring_index": ctx["ring_index"],
        "driver_identity_id": ctx["driver_identity"],
        "driver_folder": ctx["driver_folder"],
        "driver_clip_id": ctx["driver_clip_id"],
        "pair_source_identity_id": ctx["source_identity"],
        "face_ref_folder": ctx["face_ref_folder"],
        "face_ref_clip_id": ctx["face_ref_clip_id"],
        "gender_driver": ctx["gender_driver"],
        "gender_source": ctx["gender_source"],
        "gender_match": ctx["gender_driver"] == ctx["gender_source"],
        "crop_key": ctx["crop_key"],
        "crop_start_s": S.CROP_START_S,
        "crop_frames": ctx["crop_frames"],
        "crop_duration_s": S.crop_duration_s(ctx["crop_frames"]),
        "duration_s": S.crop_duration_s(ctx["crop_frames"]),  # one length per build: length cannot tell the label
        "generator_version": S.generator_version(ctx["video_method"]),
        "seed": ctx["seed"],
        "protocols": ["A"],  # the one protocol of every build
    }


def make_main_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """The four classes of one driver clip."""
    driver, source = ctx["driver_identity"], ctx["source_identity"]
    # (class, face identity, voice identity, label, needs_render, needs_conversion)
    spec = [
        (S.CLASS_NAMES[0], driver, driver, S.LABEL_REAL, False, False),  # RVRA
        (S.CLASS_NAMES[1], driver, source, S.LABEL_FAKE, False, True),  # RVFA
        (S.CLASS_NAMES[2], source, driver, S.LABEL_FAKE, True, False),  # FVRA
        (S.CLASS_NAMES[3], source, source, S.LABEL_FAKE, True, True),  # FVFA
    ]
    rows: list[dict[str, Any]] = []
    for class_name, face_id, voice_id, label, needs_render, needs_conv in spec:
        row = base_row(ctx)
        row.update(
            {
                "sample_id": S.sample_id_for(ctx["pair_id"], class_name, ctx["video_method"]),
                "class_name": class_name,
                "label": label,
                "arm": "main",
                "face_identity_id": face_id,
                "voice_identity_id": voice_id,
                "identity_coherent": face_id == voice_id,
                "same_speaker_arm": False,
                "video_status": "generated" if needs_render else "real",
                "video_method": ctx["video_method"] if needs_render else S.VIDEO_METHOD_NONE,
                "video_driver": S.VIDEO_DRIVER_CROSS if needs_render else S.VIDEO_DRIVER_NONE,
                "audio_status": "converted" if needs_conv else "real",
                "audio_method": S.AUDIO_METHOD_SEEDVC if needs_conv else S.AUDIO_METHOD_NONE,
                "audio_variant": S.AUDIO_VARIANT_CONVERTED if needs_conv else S.AUDIO_VARIANT_NATURAL,
                "voice_ref_clip_id": ctx["voice_ref_clip_id"] if needs_conv else None,
                "voice_ref_clip_ids": ctx["voice_ref_clip_ids"] if needs_conv else [],
                "voice_ref_total_s": ctx["voice_ref_total_s"] if needs_conv else 0.0,
                "needs_render": needs_render,
                "needs_conversion": needs_conv,
                "needs_vocode": False,
                "render_key": ctx["render_key_cross"] if needs_render else None,
                "conversion_key": ctx["conversion_key_cross"] if needs_conv else None,
                "parent_sample_id": None,
            }
        )
        rows.append(row)
    return rows


def make_same_speaker_row(
    ctx: dict[str, Any], voice_folder: str, voice_refs: list[str], voice_total_s: float
) -> dict[str, Any]:
    """RVFA arm with the driver's OWN voice from another recording: Seed-VC runs, nobody changes."""
    row = base_row(ctx)
    driver = ctx["driver_identity"]
    row.update(
        {
            "sample_id": S.sample_id_for(ctx["pair_id"], S.CLASS_NAMES[1], ctx["video_method"], "__same_speaker"),
            "class_name": S.CLASS_NAMES[1],
            "label": S.LABEL_FAKE,
            "arm": "same_speaker",
            "face_identity_id": driver,
            "voice_identity_id": driver,
            "identity_coherent": True,
            "same_speaker_arm": True,
            "pair_source_identity_id": driver,  # no borrow: the arm stays with one person
            "gender_source": ctx["gender_driver"],
            "gender_match": True,
            "video_status": "real",
            "video_method": S.VIDEO_METHOD_NONE,
            "video_driver": S.VIDEO_DRIVER_NONE,
            "audio_status": "converted",
            "audio_method": S.AUDIO_METHOD_SEEDVC,
            "audio_variant": S.AUDIO_VARIANT_CONVERTED,
            "voice_ref_folder": voice_folder,
            "voice_ref_clip_id": voice_refs[0],
            "voice_ref_clip_ids": voice_refs,
            "voice_ref_total_s": voice_total_s,
            "needs_render": False,
            "needs_conversion": True,
            "needs_vocode": False,
            "render_key": None,
            "conversion_key": f"conv__{ctx['crop_key']}__ownvoice_{voice_folder}",
            "parent_sample_id": S.sample_id_for(ctx["pair_id"], S.CLASS_NAMES[0], ctx["video_method"]),
        }
    )
    return row


def make_vocoded_row(ctx: dict[str, Any], class_name: str) -> dict[str, Any]:
    """The twin of an RVRA or FVRA row whose real voice went through the vocoder.

    So "went through a vocoder" is not a sign of "fake": RVRA's twin is real, FVRA's twin
    is fake. Both reuse what the pair already makes (the vocoded wav is per crop).
    """
    generated = class_name == S.CLASS_NAMES[2]
    row = base_row(ctx)
    driver = ctx["driver_identity"]
    face_id = ctx["source_identity"] if generated else driver
    row.update(
        {
            "sample_id": S.sample_id_for(ctx["pair_id"], class_name, ctx["video_method"], "__vocoded"),
            "class_name": class_name,
            "label": S.LABEL_FAKE if generated else S.LABEL_REAL,
            "arm": "vocoded_fake" if generated else "vocoded_real",
            "face_identity_id": face_id,
            "voice_identity_id": driver,
            "identity_coherent": face_id == driver,
            "same_speaker_arm": False,
            "pair_source_identity_id": ctx["source_identity"] if generated else driver,
            "gender_source": ctx["gender_source"] if generated else ctx["gender_driver"],
            "gender_match": ctx["gender_driver"] == ctx["gender_source"] if generated else True,
            "video_status": "generated" if generated else "real",
            "video_method": ctx["video_method"] if generated else S.VIDEO_METHOD_NONE,
            "video_driver": S.VIDEO_DRIVER_CROSS if generated else S.VIDEO_DRIVER_NONE,
            "audio_status": "real",
            "audio_method": S.AUDIO_METHOD_NONE,
            "audio_variant": S.AUDIO_VARIANT_VOCODED,
            "voice_ref_clip_id": None,
            "voice_ref_clip_ids": [],
            "voice_ref_total_s": 0.0,
            "needs_render": generated,
            "needs_conversion": False,
            "needs_vocode": True,
            "render_key": ctx["render_key_cross"] if generated else None,
            "conversion_key": None,
            "parent_sample_id": S.sample_id_for(ctx["pair_id"], class_name, ctx["video_method"]),
        }
    )
    return row


def tag_test_subsets(rows: list[dict[str, Any]]) -> None:
    """Tag each row with the test subset(s) it ships in, from its audio alone.

    The test split ships twice over the same pairs: natural_real audio is in the natural
    subset only, vocoded_real in the vocoded subset only, converted_fake in both. Rows of
    train and val are "both".
    """
    by_variant = {
        S.AUDIO_VARIANT_NATURAL: S.TEST_SUBSET_NATURAL,
        S.AUDIO_VARIANT_VOCODED: S.TEST_SUBSET_VOCODED,
    }
    for row in rows:
        row["test_subset"] = (
            by_variant.get(row["audio_variant"], S.TEST_SUBSET_BOTH) if row["split"] == "test" else S.TEST_SUBSET_BOTH
        )


def make_self_drive_row(ctx: dict[str, Any]) -> dict[str, Any]:
    """FVRA arm where the generator re-makes the driver as themselves: the render runs, nobody changes."""
    row = base_row(ctx)
    driver = ctx["driver_identity"]
    row.update(
        {
            "sample_id": S.sample_id_for(ctx["pair_id"], S.CLASS_NAMES[2], ctx["video_method"], "__self_drive"),
            "class_name": S.CLASS_NAMES[2],
            "label": S.LABEL_FAKE,
            "arm": "self_drive",
            "face_identity_id": driver,
            "voice_identity_id": driver,
            "identity_coherent": True,
            "same_speaker_arm": False,
            "pair_source_identity_id": driver,
            "gender_source": ctx["gender_driver"],
            "gender_match": True,
            "face_ref_folder": ctx["driver_folder"],
            "face_ref_clip_id": ctx["driver_clip_id"],
            "video_status": "generated",
            "video_method": ctx["video_method"],
            "video_driver": S.VIDEO_DRIVER_SELF,
            "audio_status": "real",
            "audio_method": S.AUDIO_METHOD_NONE,
            "audio_variant": S.AUDIO_VARIANT_NATURAL,
            "voice_ref_clip_id": None,
            "voice_ref_clip_ids": [],
            "voice_ref_total_s": 0.0,
            "needs_render": True,
            "needs_conversion": False,
            "needs_vocode": False,
            "render_key": S.render_key_self(ctx["crop_key"], ctx["video_method"]),
            "conversion_key": None,
            "parent_sample_id": S.sample_id_for(ctx["pair_id"], S.CLASS_NAMES[0], ctx["video_method"]),
        }
    )
    return row


def build_pair_contexts(
    pools: Pools,
    gender: dict[str, str],
    split: str,
    ring_gender: str,
    ring: list[str],
    seed: int,
    build_id: str,
    video_method: str = S.VIDEO_METHOD_ANIMATE,
    crop_frames: int = S.CROP_FRAMES,
) -> list[dict[str, Any]]:
    """Every driver clip of one ring with what it borrows, in a fixed order."""
    ring_size = len(ring)
    chosen_folders: dict[str, list[str]] = {}
    for identity in ring:
        folders = usable_folders(pools[split][identity])
        rng = substream(seed, "folders", split, ring_gender, identity)
        chosen_folders[identity] = sorted(rng.sample(folders, S.FOLDERS_PER_CELEB))

    contexts: list[dict[str, Any]] = []
    for i, driver_identity in enumerate(ring):
        for f, driver_folder in enumerate(chosen_folders[driver_identity]):
            # +1 so the shift is never zero: nobody borrows their own face.
            source_identity = ring[(i + f + 1) % ring_size]
            face_ref_folder = chosen_folders[source_identity][f]

            if ELIGIBILITY_FRAMES:
                picked = allocate_stratum_clips(
                    pools[split][driver_identity][driver_folder],
                    seed,
                    ("clips", split, ring_gender, driver_identity, driver_folder),
                )[crop_frames]
                # Own seed key, not the clips one: every length of one draw then picks the
                # same face clip and voice clips for this folder.
                rng = substream(seed, "refs", split, ring_gender, driver_identity, driver_folder)
            else:
                rng = substream(seed, "clips", split, ring_gender, driver_identity, driver_folder)
                driver_clips = sorted(long_clips(pools[split][driver_identity][driver_folder]))
                picked = rng.sample(driver_clips, S.CLIPS_PER_FOLDER)

            face_clips = sorted(pools[split][source_identity][face_ref_folder])
            face_ref_clip_id = rng.choice(face_clips)[0]

            voice_refs, voice_total_s = pick_voice_refs(
                pools[split][source_identity], exclude_folders={face_ref_folder}, rng=rng
            )
            if not voice_refs:
                # Said out loud: the ring is then short of driver clips (check_pairs C10).
                print(
                    f"[manifest] WARNING: skipping {driver_identity}/{driver_folder} — source {source_identity} "
                    f"has no voice material outside its face folder {face_ref_folder}",
                    file=sys.stderr,
                )
                continue

            for driver_clip_id, _duration in picked:
                crop_key = S.crop_key_for(driver_clip_id, crop_frames)
                contexts.append(
                    {
                        "build_id": build_id,
                        "seed": seed,
                        "split": split,
                        "ring_gender": ring_gender,
                        "ring_index": i,
                        "driver_identity": driver_identity,
                        "driver_folder": driver_folder,
                        "driver_clip_id": driver_clip_id,
                        "source_identity": source_identity,
                        "face_ref_folder": face_ref_folder,
                        "face_ref_clip_id": face_ref_clip_id,
                        "gender_driver": gender.get(driver_identity, "unknown"),
                        "gender_source": gender.get(source_identity, "unknown"),
                        "crop_key": crop_key,
                        "pair_id": f"{crop_key}__x__{source_identity}",
                        "video_method": video_method,
                        "crop_frames": crop_frames,
                        "render_key_cross": S.render_key_cross(crop_key, source_identity, video_method),
                        "conversion_key_cross": f"conv__{crop_key}__voice_{source_identity}",
                        "voice_ref_clip_id": voice_refs[0],
                        "voice_ref_clip_ids": voice_refs,
                        "voice_ref_total_s": voice_total_s,
                    }
                )
    return contexts


def add_arms(
    contexts: list[dict[str, Any]], pools: Pools, split: str, args: argparse.Namespace
) -> list[dict[str, Any]]:
    """The four main rows of every context, plus the sampled extra arms."""
    rows: list[dict[str, Any]] = []
    n = len(contexts)
    rng = substream(args.seed, "arms", split)
    idx = list(range(n))

    same_speaker_idx = set(rng.sample(idx, round(n * args.same_speaker_frac)))
    self_drive_idx = set(rng.sample(idx, round(n * args.self_drive_frac)))
    # The vocoded arm runs in every split at one fraction, test included, so "vocoded" is
    # never a sign of "fake" in any split.
    vocoded_idx = set(rng.sample(idx, round(n * args.vocoded_real_frac)))

    for k, ctx in enumerate(contexts):
        rows.extend(make_main_rows(ctx))
        if k in same_speaker_idx:
            driver_folders = sorted(pools[split][ctx["driver_identity"]])
            other = [v for v in driver_folders if v != ctx["driver_folder"]]
            if other:
                arm_rng = substream(args.seed, "ownvoice", ctx["driver_clip_id"])
                # Every other folder of the driver, not one: so this arm's voice clips are
                # as long as the main arm's.
                refs, total_s = pick_voice_refs(
                    pools[split][ctx["driver_identity"]],
                    exclude_folders={ctx["driver_folder"]},
                    rng=arm_rng,
                )
                if refs:
                    voice_folder = refs[0].rsplit("/", 2)[1]
                    rows.append(make_same_speaker_row(ctx, voice_folder, refs, total_s))
        if k in vocoded_idx:
            # Both classes with natural real audio, so the vocoded test subset is complete.
            rows.append(make_vocoded_row(ctx, S.CLASS_NAMES[0]))
            rows.append(make_vocoded_row(ctx, S.CLASS_NAMES[2]))
        if k in self_drive_idx:
            rows.append(make_self_drive_row(ctx))
    return rows


def vocoded_implies_fake(rows: list[dict[str, Any]]) -> dict[str, float]:
    """How often "the voice went through a vocoder" means "fake", per split and test subset.

    Converted and vocoded voices both went through the vocoder; natural ones did not.
    """
    processed = {S.AUDIO_VARIANT_CONVERTED, S.AUDIO_VARIANT_VOCODED}

    def rate(subset: list[dict[str, Any]]) -> float:
        hits = [r for r in subset if r["audio_variant"] in processed]
        if not hits:
            return 0.0
        return round(sum(1 for r in hits if r["label"] == S.LABEL_FAKE) / len(hits), 4)

    out = {split: rate([r for r in rows if r["split"] == split]) for split in sorted({r["split"] for r in rows})}
    for subset in (S.TEST_SUBSET_NATURAL, S.TEST_SUBSET_VOCODED):
        selected = test_subset_rows(rows, subset)
        if selected:
            out[f"test/{subset}"] = rate(selected)
    return out


def test_subset_rows(rows: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    """The rows one test subset ships: its own, plus the ones in both."""
    return [r for r in rows if r["split"] == "test" and r["test_subset"] in (subset, S.TEST_SUBSET_BOTH)]


def vocoded_cue_auc(rows: list[dict[str, Any]]) -> dict[str, float]:
    """The AUC of a detector that reads only "went through a vocoder": 0.5 = no help, 1.0 = all it needs.

    For a yes/no cue the AUC is (P(cue | fake) + P(no cue | real)) / 2.
    """

    def auc(subset: list[dict[str, Any]]) -> float:
        processed = {S.AUDIO_VARIANT_CONVERTED, S.AUDIO_VARIANT_VOCODED}
        fake = [r for r in subset if r["label"] == S.LABEL_FAKE]
        real = [r for r in subset if r["label"] == S.LABEL_REAL]
        if not fake or not real:
            return 0.0
        tpr = sum(1 for r in fake if r["audio_variant"] in processed) / len(fake)
        tnr = sum(1 for r in real if r["audio_variant"] not in processed) / len(real)
        return round(0.5 * (tpr + tnr), 4)

    out = {split: auc([r for r in rows if r["split"] == split]) for split in sorted({r["split"] for r in rows})}
    for subset in (S.TEST_SUBSET_NATURAL, S.TEST_SUBSET_VOCODED):
        selected = test_subset_rows(rows, subset)
        if selected:
            out[f"test/{subset}"] = auc(selected)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts to read before any GPU time is spent."""
    renders = {r["render_key"] for r in rows if r["needs_render"]}
    conversions = {r["conversion_key"] for r in rows if r["needs_conversion"]}
    crops = {r["crop_key"] for r in rows}
    # Wan-Animate cost follows chunks, not renders (it pads to 1 + 76k frames); minutes measured at 480p.
    chunks = max(1, -(-(S.CROP_FRAMES - S.ANIMATE_REFERT_NUM) // (S.ANIMATE_FRAME_NUM - S.ANIMATE_REFERT_NUM)))
    animate_min_per_render = 2.16 + 3.89 * chunks
    return {
        "n_samples": len(rows),
        "n_pairs": len({r["pair_id"] for r in rows}),
        "n_crops": len(crops),
        "n_renders": len(renders),
        "n_conversions": len(conversions),
        "n_vocoded": sum(1 for r in rows if r["needs_vocode"]),
        "est_animate_a100_hours": round(len(renders) * animate_min_per_render / 60.0, 1),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_class": dict(Counter(r["class_name"] for r in rows)),
        "by_label": dict(Counter(r["label"] for r in rows)),
        "by_arm": dict(Counter(r["arm"] for r in rows)),
        "by_audio_variant": dict(Counter(r["audio_variant"] for r in rows)),
        "by_test_subset": dict(Counter(r["test_subset"] for r in rows if r["split"] == "test")),
        "vocoded_implies_fake": vocoded_implies_fake(rows),
        "vocoded_cue_auc": vocoded_cue_auc(rows),
        "by_video_driver": dict(Counter(r["video_driver"] for r in rows)),
        "main_rows_by_class": dict(Counter(r["class_name"] for r in rows if r["arm"] == "main")),
        "identities_per_split": {
            s: len(
                {i for r in rows if r["split"] == s for i in (r["driver_identity_id"], r["pair_source_identity_id"])}
            )
            for s in sorted({r["split"] for r in rows})
        },
        "gender_match_all_true": all(r["gender_match"] for r in rows),
        "effective_sessions": len({(r["driver_identity_id"], r["driver_folder"]) for r in rows}),
    }


def parse_per_split(pairs: list[str], default: int) -> dict[str, int]:
    """``["train=70", "val=15"]`` -> ``{"train": 70, "val": 15, "test": default}``."""
    out = dict.fromkeys(("train", "val", "test"), default)
    for item in pairs:
        if "=" not in item:
            msg = f"expected SPLIT=N, got {item!r}"
            raise SystemExit(msg)
        split, _, value = item.partition("=")
        if split not in out:
            msg = f"unknown split {split!r} (train, val or test)"
            raise SystemExit(msg)
        out[split] = int(value)
    return out


def load_pinned_rings(build_id: str) -> dict[str, Any]:
    """The draw of another build, so this build gets the same rings.

    Also reads the older summary shape with no "draw", where "rings" held one bare ring per
    (split, gender).
    """
    candidate = Path(build_id)  # a build id, or the path of a manifest_summary.json
    path = candidate if candidate.suffix == ".json" else S.new_gen_manifest_summary_path(build_id)
    summary = json.loads(path.read_text())
    draw = summary.get("draw")
    if draw:
        return draw
    legacy: dict[str, dict[str, list[list[str]]]] = {}
    for split, by_gender in (summary.get("rings") or {}).items():
        legacy[split] = {g: [list(ring)] for g, ring in by_gender.items()}
    return {
        "primary_rings": legacy,
        "reserve_rings": {},
        "eligibility_frames": [],
        "ring_size": summary.get("ring_size", S.RING_SIZE),
    }


def main() -> int:
    """Draw the rings of the requested splits, write the pair list and its summary."""
    args = parse_args()
    global ACTIVE_CROP_FRAMES, ELIGIBILITY_FRAMES
    ACTIVE_CROP_FRAMES = args.crop_frames
    ELIGIBILITY_FRAMES = tuple(sorted(set(args.eligibility_frames), reverse=True)) if args.eligibility_frames else None
    rings_per_cell = parse_per_split(args.rings_per_split_gender, 1)
    reserve_per_cell = parse_per_split(args.reserve_rings_per_split_gender, 0)
    if ELIGIBILITY_FRAMES and args.crop_frames not in ELIGIBILITY_FRAMES:
        print(
            f"[manifest] REFUSING: --crop-frames {args.crop_frames} is not one of --eligibility-frames "
            f"{list(ELIGIBILITY_FRAMES)}; this build would have no stratum to draw from",
            file=sys.stderr,
        )
        return 1
    pools, gender = load_pools()

    out_dir = args.out_dir or S.new_gen_manifest_dir(args.build_id)
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        print(
            f"[manifest] REFUSING to overwrite frozen manifest {manifest_path} — use a new --build-id", file=sys.stderr
        )
        return 1

    all_rows: list[dict[str, Any]] = []
    rings_report: dict[str, dict[str, list[list[str]]]] = {}
    pinned = load_pinned_rings(args.rings_from) if args.rings_from else None
    if pinned is not None and pinned.get("eligibility_frames") != list(ELIGIBILITY_FRAMES or []):
        print(
            f"[manifest] REFUSING: --rings-from {args.rings_from} was drawn with eligibility_frames="
            f"{pinned.get('eligibility_frames')}, this build has {list(ELIGIBILITY_FRAMES or [])}; "
            "a sibling of one draw must use the same eligibility",
            file=sys.stderr,
        )
        return 1
    primary_rings: dict[str, dict[str, list[list[str]]]] = {}
    reserve_rings: dict[str, dict[str, list[list[str]]]] = {}
    for split in args.splits:
        primary_rings[split], reserve_rings[split] = {}, {}
        for ring_gender in S.RING_GENDERS:
            n_primary = rings_per_cell.get(split, 1)
            n_reserve = reserve_per_cell.get(split, 0)
            if pinned is not None:
                got_primary = pinned["primary_rings"].get(split, {}).get(ring_gender, [])
                got_reserve = pinned["reserve_rings"].get(split, {}).get(ring_gender, [])
            else:
                candidates = eligible_identities(pools, gender, split, ring_gender, S.FOLDERS_PER_CELEB)
                need = args.ring_size * (n_primary + n_reserve)
                if len(candidates) < need:
                    print(
                        f"[manifest] ERROR: split={split} gender={ring_gender} has only {len(candidates)} "
                        f"eligible identities, need {need} "
                        f"({n_primary} primary + {n_reserve} reserve rings of {args.ring_size})",
                        file=sys.stderr,
                    )
                    return 1
                # ONE sample for both roles: a reserve ring can never be a built ring.
                drawn = substream(args.seed, "ring", split, ring_gender).sample(candidates, need)
                rings = [drawn[i : i + args.ring_size] for i in range(0, need, args.ring_size)]
                got_primary, got_reserve = rings[:n_primary], rings[n_primary:]
            primary_rings[split][ring_gender] = got_primary
            reserve_rings[split][ring_gender] = got_reserve
            rings_report.setdefault(split, {})[ring_gender] = got_primary
            for ring in got_primary:
                contexts = build_pair_contexts(
                    pools,
                    gender,
                    split,
                    ring_gender,
                    ring,
                    args.seed,
                    args.build_id,
                    args.video_method,
                    args.crop_frames,
                )
                all_rows.extend(add_arms(contexts, pools, split, args))
    draw = {
        "role": "primary",  # the built rings; kept so the summary keeps its format
        "ring_size": args.ring_size,
        "rings_per_split_gender": rings_per_cell,
        "reserve_rings_per_split_gender": reserve_per_cell,
        "eligibility_frames": list(ELIGIBILITY_FRAMES or []),
        "primary_rings": primary_rings,
        "reserve_rings": reserve_rings,
        "rings_from": args.rings_from,
    }

    tag_test_subsets(all_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = summarize(all_rows)
    summary.update(
        {
            "build_id": args.build_id,
            "seed": args.seed,
            "splits": args.splits,
            "rings": rings_report,
            "draw": draw,
            "video_method": args.video_method,
            "crop_frames": args.crop_frames,
            # Always empty here (they recorded build modes left out of this code); kept so
            # the summary has the same keys as the release builds' summaries.
            "excluded_driver_clips_of": [],
            "reserve_file": None,
            "reserve_sha256": None,
            "parent_build": None,
            "pairs_from": None,
        }
    )
    # Beside the manifest, so --out-dir never touches a written build's summary.
    (out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"[manifest] wrote {len(all_rows)} samples -> {manifest_path}")
    print(
        f"[manifest] pairs={summary['n_pairs']} crops={summary['n_crops']} renders={summary['n_renders']} conversions={summary['n_conversions']}"
    )
    # The cost model is Wan-Animate's; other generators get their own estimate.
    if args.video_method == S.VIDEO_METHOD_ANIMATE:
        print(
            f"[manifest] est Animate cost: {summary['est_animate_a100_hours']} A100-hours ({summary['n_renders']} renders)"
        )
    else:
        est_min = round(summary["n_renders"] * S.FACEFUSION_SEC_PER_RENDER / 60.0, 1)
        print(f"[manifest] video_method={args.video_method}: ~{est_min} A40-minutes ({summary['n_renders']} renders)")
    print(f"[manifest] by class: {summary['by_class']}")
    print(f"[manifest] by arm: {summary['by_arm']}")
    print(f"[manifest] next: python -m mozaic.data_generation.inputs.check_pairs --build-id {args.build_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
