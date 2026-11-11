"""Write a build's Wan2.2-Animate job list: one render per render_key.

One job per render, not per clip: FVRA and FVFA differ only in the sound put on later, so they share one render
(the manifest's render_key); this step groups the rows by render_key, records the sample_ids each render
serves, and stops if a render_key's rows disagree on what to render. Two modes of Wan-Animate:

    cross (the main path)  replace mode: the other person is put into the driver's crop
                           (sam2 segmentation + the relighting LoRA)
    self  (the control)    animate mode: the driver re-made as themselves, so a quality drop is the
                           generator's, not the swap's

The driving video is the crop (S.crop_video_path), so a render has exactly the crop's frames. The face still is
not written here: run.py extracts it (S.ref_image_path) at the time recorded on the job, picked here by
reference_frame.best_reference_time(). Every input is checked here, so a GPU job never starts on a missing file.

The list goes to <build's manifest folder>/animate_jobs<suffix>/jobs.jsonl (suffix: S.batch_suffix_for, when
--pair-ids-file names a batch of pairs; _micro<N> with --limit; then _<jobs-suffix>). Run in the core environment:

    python -m mozaic.data_generation.generators.animate.make_jobs --build-id <build>
    python -m mozaic.data_generation.generators.animate.make_jobs --build-id <build> --pair-ids-file <pair_ids.txt>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S

MODE_BY_DRIVER = {S.VIDEO_DRIVER_CROSS: "replace", S.VIDEO_DRIVER_SELF: "animate"}

# Cost model (planning only, written on each job as est_minutes), fitted on measured wall times on an 80 GB A100
# at --resolution_area "832 480", preprocess and the ~2.5 min model load included:
#   replace  112f (2 chunks): 9.92, 9.85, 9.80 min | 151f (2 chunks): 10.2 min | 423f (6 chunks): 25.5 min
#   animate  112f (2 chunks): 8.90, 8.80, 8.70 min
# Cost follows chunks, not frames (upstream pads to 1 + 76k frames). Animate mode is cheaper by a constant
# (no sam2, no relighting LoRA).
ANIMATE_MIN_PER_CHUNK = 3.89
ANIMATE_FIXED_MIN = {"replace": 2.16, "animate": 1.02}


def render_chunks(frames: int) -> int:
    """How many 77-frame chunks upstream renders for `frames` (it pads to 1 + 76k, then cuts back)."""
    overlap = S.ANIMATE_REFERT_NUM
    stride = S.ANIMATE_FRAME_NUM - overlap
    return max(1, -(-(frames - overlap) // stride))


def estimate_render_minutes(frames: int, mode: str) -> float:
    """Wall-clock minutes of one render, from the measured chunk model."""
    return ANIMATE_FIXED_MIN.get(mode, ANIMATE_FIXED_MIN["replace"]) + ANIMATE_MIN_PER_CHUNK * render_chunks(frames)


def default_jobs_dir(build_id: str, suffix: str) -> Path:
    """<build's manifest folder>/animate_jobs<suffix>: a batch of pairs gets its own folder."""
    return S.new_gen_manifest_dir(build_id) / f"animate_jobs{suffix}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a build's Wan2.2-Animate job list.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument(
        "--out-dir", type=Path, default=None, help="default: <build's manifest folder>/animate_jobs<suffix>"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="N renders spread across modes (a first small test); writes to animate_jobs_micro<N>",
    )
    p.add_argument("--only-mode", choices=sorted(set(MODE_BY_DRIVER.values())), default=None)
    p.add_argument("--prompt", default=None, help="override S.ANIMATE_PROMPT (tests)")
    p.add_argument("--sample-steps", type=int, default=None, help="override S.ANIMATE_SAMPLE_STEPS")
    p.add_argument(
        "--refert-num",
        type=int,
        choices=[1, 5],
        default=None,
        help="frames of context carried between 77-frame chunks (upstream allows 1 or 5); only matters for "
        "crops longer than one chunk. Default S.ANIMATE_REFERT_NUM.",
    )
    p.add_argument("--force-mode", choices=["replace", "animate"], default=None, help="override the per-row mode")
    p.add_argument(
        "--ref-mode",
        choices=["best", "fixed"],
        default="best",
        help="'best' scores frames with reference_frame (default); 'fixed' uses S.FACE_REF_AT_S",
    )
    p.add_argument("--jobs-suffix", default=None, help="append to the jobs folder name (keeps test variants apart)")
    p.add_argument(
        "--no-relighting",
        action="store_true",
        help="replace mode without the relighting LoRA (an experiment; MOZAIC kept it on); recorded on every job",
    )
    p.add_argument(
        "--pair-ids-file",
        type=Path,
        default=None,
        help="newline-separated pair_ids; keep only the renders serving them",
    )
    p.add_argument("--allow-missing-crops", action="store_true", help="write the jobs anyway (not for real runs)")
    return p.parse_args()


def group_by_render(manifest_path: Path, pair_ids: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The manifest rows that need a render, keyed by render_key (only those serving `pair_ids`, when given)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if pair_ids is not None and row.get("pair_id") not in pair_ids:
            continue
        if row.get("needs_render"):
            groups[row["render_key"]].append(row)
    return groups


def render_seed(render_key: str) -> int:
    """The render's own seed, the same on every run: blake2b of "<seed>:<render_key>".

    Kept below 2**31: generate.py reads --base_seed as a C int and draws a random seed for a negative one.
    """
    digest = hashlib.blake2b(f"{S.SPLIT_SEED}:{render_key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def build_job(render_key: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """One render job, with the sample_ids that will use its output."""
    head = rows[0]
    driver = head["video_driver"]
    mode = args.force_mode or MODE_BY_DRIVER[driver]

    # The still fixes face, clothes, framing and background for every rendered frame, so it is scored, not
    # taken at a fixed time.
    ref_at_s = S.FACE_REF_AT_S
    ref_parts: dict[str, float] = {}
    if args.ref_mode == "best":
        from mozaic.data_generation.generators.reference_frame import best_reference_time

        ref_at_s, ref_parts = best_reference_time(S.VOXCELEB2_MP4_ROOT / f"{head['face_ref_clip_id']}.mp4")
    return {
        "job_id": render_key,
        "render_key": render_key,
        "mode": mode,
        "relighting": not args.no_relighting,
        "video_driver": driver,
        "split": head["split"],
        # The driving video is the crop, so the render has the crop's frames.
        "driving_video": str(S.crop_video_path(head["crop_key"])),
        "crop_key": head["crop_key"],
        # run.py imports nothing from this repo, so every value it needs travels on the job,
        # with the frame count and fps it must give the render.
        "expected_frames": S.row_crop_frames(head),
        "target_fps": S.TARGET_FPS,
        "est_minutes": round(estimate_render_minutes(S.row_crop_frames(head), mode), 2),
        # The face still: one frame of the face donor's clip (run.py extracts it).
        "face_ref_clip_id": head["face_ref_clip_id"],
        "face_ref_source_mp4": str(S.VOXCELEB2_MP4_ROOT / f"{head['face_ref_clip_id']}.mp4"),
        "face_ref_at_s": ref_at_s,
        "face_ref_pick": args.ref_mode,
        "face_ref_score": ref_parts,
        # Passed on purpose: left out, generate.py uses its generic demo prompt.
        "prompt": args.prompt or S.ANIMATE_PROMPT,
        "ref_image": str(S.ref_image_path(render_key)),
        "out_video": str(S.render_path(render_key)),
        "resolution_area": S.ANIMATE_RESOLUTION_AREA,
        # Every sampling setting is passed, none left to upstream's defaults; left out, --base_seed would be
        # a random draw per render.
        "base_seed": render_seed(render_key),
        "sample_steps": args.sample_steps or S.ANIMATE_SAMPLE_STEPS,
        "sample_shift": S.ANIMATE_SAMPLE_SHIFT,
        "sample_guide_scale": S.ANIMATE_SAMPLE_GUIDE_SCALE,
        "frame_num": S.ANIMATE_FRAME_NUM,
        "refert_num": args.refert_num or S.ANIMATE_REFERT_NUM,
        "face_identity_id": head["face_identity_id"],
        "driver_identity_id": head["driver_identity_id"],
        # Provenance: the shipped samples this render serves.
        "consumer_sample_ids": sorted(r["sample_id"] for r in rows),
        "consumer_classes": sorted({r["class_name"] for r in rows}),
        "generator_version": S.GENERATOR_VERSION,
    }


def validate_jobs(groups: dict[str, list[dict[str, Any]]], jobs: list[dict[str, Any]]) -> list[str]:
    """Problems that would make a render ambiguous or a GPU job fail late."""
    problems: list[str] = []

    # One render_key is one render: the same crop, face donor and mode in every row.
    for render_key, rows in sorted(groups.items()):
        for field in ("crop_key", "face_ref_clip_id", "video_driver", "face_identity_id", "split"):
            values = {r[field] for r in rows}
            if len(values) != 1:
                problems.append(f"{render_key}: rows disagree on {field}: {sorted(values)}")

    # A cross render must change the person; a self render must not.
    for job in jobs:
        same = job["face_identity_id"] == job["driver_identity_id"]
        if job["video_driver"] == S.VIDEO_DRIVER_CROSS and same:
            problems.append(f"{job['job_id']}: cross render but face == driver identity")
        if job["video_driver"] == S.VIDEO_DRIVER_SELF and not same:
            problems.append(f"{job['job_id']}: self render but face != driver identity")

    # FVRA and FVFA share the cross render: a render with one user only is flagged here.
    for job in jobs:
        if job["video_driver"] == S.VIDEO_DRIVER_CROSS and sorted(job["consumer_classes"]) != ["FVFA", "FVRA"]:
            problems.append(f"{job['job_id']}: cross render consumed by {job['consumer_classes']}, expected FVRA+FVFA")

    return problems


def stratified_limit(jobs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """`limit` jobs taken in turn from each mode, not the first N: render_keys sort every cross__* before every
    self__*, so the first N would never test animate mode."""
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        by_mode[job["mode"]].append(job)
    picked: list[dict[str, Any]] = []
    modes = sorted(by_mode)
    while len(picked) < limit and any(by_mode[m] for m in modes):
        for mode in modes:
            if by_mode[mode] and len(picked) < limit:
                picked.append(by_mode[mode].pop(0))
    return picked


def check_inputs(jobs: list[dict[str, Any]]) -> list[str]:
    """Every input file that is missing now."""
    missing: list[str] = []
    for job in jobs:
        for key in ("driving_video", "face_ref_source_mp4"):
            if not Path(job[key]).exists():
                missing.append(f"{job['job_id']}: missing {key} -> {job[key]}")
    return missing


def main() -> int:
    """Group the manifest into render jobs, check them, and write jobs.jsonl."""
    args = parse_args()
    manifest_path = args.manifest or S.new_gen_manifest_path(args.build_id)
    if not manifest_path.exists():
        print(f"[wa-jobs] manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    report = S.new_gen_validate_report_path(args.build_id)
    verdict = json.loads(report.read_text()).get("verdict") if report.exists() else None
    if verdict != "PASS":
        print(
            f"[wa-jobs] REFUSING: {args.build_id} has no PASSing validate_report.json (verdict={verdict}). "
            f"Validate the manifest first: renders are the costliest step.",
            file=sys.stderr,
        )
        return 1

    pair_ids = None
    if args.pair_ids_file:
        pair_ids = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
    groups = group_by_render(manifest_path, pair_ids)
    if pair_ids is not None and not groups:
        print(f"[wa-jobs] REFUSING: no renders serve the {len(pair_ids)} requested pair_id(s)", file=sys.stderr)
        return 1
    jobs = [build_job(k, rows, args) for k, rows in sorted(groups.items())]
    if args.only_mode:
        jobs = [j for j in jobs if j["mode"] == args.only_mode]

    problems = validate_jobs(groups, jobs)
    if problems:
        print(f"[wa-jobs] {len(problems)} manifest/render inconsistency(ies):", file=sys.stderr)
        for p in problems[:10]:
            print(f"[wa-jobs]   {p}", file=sys.stderr)
        return 1

    # A batch of pairs (or a --limit test) gets its own folder: a short list at the build's own path would look
    # like the whole build.
    suffix = ""
    if pair_ids is not None:
        suffix = S.batch_suffix_for(pair_ids)
    if args.limit is not None:
        jobs = stratified_limit(jobs, args.limit)
        suffix = f"_micro{args.limit}"
    if args.jobs_suffix:
        suffix = f"{suffix}_{args.jobs_suffix}"

    missing = check_inputs(jobs)
    if missing:
        print(f"[wa-jobs] {len(missing)} missing input(s) — run the crop step first:", file=sys.stderr)
        for m in missing[:10]:
            print(f"[wa-jobs]   {m}", file=sys.stderr)
        if not args.allow_missing_crops:
            return 1

    out_dir = args.out_dir or default_jobs_dir(args.build_id, suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = out_dir / "jobs.jsonl"
    with jobs_file.open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")

    by_mode: dict[str, int] = defaultdict(int)
    for job in jobs:
        by_mode[job["mode"]] += 1
    est_hours = round(sum(j["est_minutes"] for j in jobs) / 60.0, 1)
    chunks = render_chunks(S.CROP_FRAMES)
    print(f"[wa-jobs] wrote {len(jobs)} render jobs -> {jobs_file}")
    print(f"[wa-jobs] by mode: {dict(by_mode)}")
    print(f"[wa-jobs] renders serve {sum(len(j['consumer_sample_ids']) for j in jobs)} shipped samples")
    print(f"[wa-jobs] ESTIMATED COST: {est_hours} A100-hours ({S.CROP_FRAMES}f = {chunks} chunks; measured model)")
    print(f"[wa-jobs] render with: sbatch --array=0-<shards-1> scripts/data_generation/render_animate.sh {jobs_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
