"""Write a build's FaceFusion job list: one face swap per render, and each swap's face photo.

- One job per render, not per clip: FVRA and FVFA of a pair differ only in the sound put on later, so they share
  one render (the manifest's render_key).
- The video comes from the crop step (S.crop_video_path), so the swap runs over exactly the crop's frames.
- FaceFusion takes the other person's face as a picture. This step writes it (S.ref_image_path): one frame of
  that person's own VoxCeleb2 clip, at the time reference_frame.best_reference_time() picks, the same picker as
  the other two generators. Never a still from another dataset: that would add a cross-dataset fingerprint to
  the fake classes only.
- Every input is checked here, so a GPU job never starts on a missing file.

The list goes to <build's manifest folder>/facefusion_jobs<suffix>/jobs.jsonl (suffix: S.batch_suffix_for, when
--pair-ids-file names a batch of pairs). Run in the core environment, with FF_UPSTREAM set to the FaceFusion
clone (its commit goes into every job):

    python -m mozaic.data_generation.generators.facefusion.make_jobs --build-id <build>
    python -m mozaic.data_generation.generators.facefusion.make_jobs --build-id <build> --pair-ids-file <pair_ids.txt>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a build's FaceFusion job list.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument(
        "--out-dir", type=Path, default=None, help="default: <build's manifest folder>/facefusion_jobs<suffix>"
    )
    p.add_argument(
        "--pair-ids-file",
        type=Path,
        default=None,
        help="newline-separated pair_ids; keep only the renders serving them",
    )
    p.add_argument(
        "--ref-mode",
        choices=["best", "fixed"],
        default="best",
        help="'best' scores frames with reference_frame (default); 'fixed' uses S.FACE_REF_AT_S",
    )
    p.add_argument("--allow-missing-crops", action="store_true", help="write the jobs anyway (not for real runs)")
    return p.parse_args()


def default_jobs_dir(build_id: str, suffix: str) -> Path:
    """<build's manifest folder>/facefusion_jobs<suffix>: a batch of pairs gets its own folder."""
    return S.new_gen_manifest_dir(build_id) / f"facefusion_jobs{suffix}"


def upstream_commit() -> str | None:
    """The commit of the FaceFusion clone in FF_UPSTREAM, recorded on every job (None when unset)."""
    upstream = os.environ.get("FF_UPSTREAM")
    if not upstream or not Path(upstream).is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", upstream, "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def group_by_render(manifest_path: Path, pair_ids: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The manifest rows that need a render, keyed by render_key."""
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


def extract_reference(source_mp4: Path, at_seconds: float, dest: Path) -> bool:
    """Write one frame of the face donor's clip as the swap's source picture.

    Always overwrites: a file left by an earlier --ref-mode could hold another frame, and every later check
    would miss it (the render would show the right person from the wrong frame).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{at_seconds:.3f}",
            "-i",
            str(source_mp4),
            "-frames:v",
            "1",
            str(dest),
        ],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0 and dest.exists()


def build_job(
    render_key: str, rows: list[dict[str, Any]], args: argparse.Namespace, commit: str | None
) -> dict[str, Any]:
    """One swap job, with the sample_ids that will use its output."""
    head = rows[0]
    ref_source = S.VOXCELEB2_MP4_ROOT / f"{head['face_ref_clip_id']}.mp4"

    # The same still picker as Wan-Animate and S2V, so a quality gap between families is the generator's.
    ref_at_s = S.FACE_REF_AT_S
    ref_parts: dict[str, float] = {}
    if args.ref_mode == "best":
        from mozaic.data_generation.generators.reference_frame import best_reference_time

        ref_at_s, ref_parts = best_reference_time(ref_source)

    return {
        # what run.py reads
        "job_id": render_key,
        "source_image": str(S.ref_image_path(render_key)),
        "target_video": str(S.crop_video_path(head["crop_key"])),
        "face_swapper_model": S.FACEFUSION_SWAPPER_MODEL,
        "face_detector_model": S.FACEFUSION_DETECTOR_MODEL,
        "detector_score": S.FACEFUSION_DETECTOR_SCORE,
        "face_enhancer": S.FACEFUSION_FACE_ENHANCER,
        "pixel_boost": S.FACEFUSION_PIXEL_BOOST,
        "output_video_quality": S.FACEFUSION_OUTPUT_VIDEO_QUALITY,
        # provenance, copied as it is into the render's sidecar
        "provenance_extra": {
            "render_key": render_key,
            "build_id": args.build_id,
            "video_method": S.VIDEO_METHOD_FACEFUSION,
            "video_driver": head["video_driver"],
            "split": head["split"],
            "crop_key": head["crop_key"],
            "expected_frames": S.row_crop_frames(head),
            "target_fps": S.TARGET_FPS,
            "face_identity_id": head["face_identity_id"],
            "driver_identity_id": head["driver_identity_id"],
            "face_ref_clip_id": head["face_ref_clip_id"],
            "face_ref_source_mp4": str(ref_source),
            "face_ref_at_s": ref_at_s,
            "face_ref_pick": args.ref_mode,
            "face_ref_score": ref_parts,
            "pixel_boost": S.FACEFUSION_PIXEL_BOOST,
            "facefusion_upstream_commit": commit,
            "consumer_sample_ids": sorted(r["sample_id"] for r in rows),
            "consumer_classes": sorted({r["class_name"] for r in rows}),
            "generator_version": S.generator_version(S.VIDEO_METHOD_FACEFUSION),
        },
    }


def prepare_inputs(jobs: list[dict[str, Any]]) -> list[str]:
    """Check every job's inputs exist now, and write each job's face picture. Returns what is missing."""
    missing: list[str] = []
    for job in jobs:
        target = Path(job["target_video"])
        if not target.exists():
            missing.append(f"{job['job_id']}: missing crop -> {target}")
            continue
        extra = job["provenance_extra"]
        ref_source = Path(extra["face_ref_source_mp4"])
        if not ref_source.exists():
            missing.append(f"{job['job_id']}: missing face-ref source -> {ref_source}")
            continue
        if not extract_reference(ref_source, extra["face_ref_at_s"], Path(job["source_image"])):
            missing.append(f"{job['job_id']}: could not extract reference frame -> {job['source_image']}")
    return missing


def validate_jobs(groups: dict[str, list[dict[str, Any]]], jobs: list[dict[str, Any]]) -> list[str]:
    """Problems that would make a render ambiguous: one render_key must be one render, a cross render must change
    the person, and both FVRA and FVFA must use a cross render (they share it)."""
    problems: list[str] = []

    for render_key, rows in sorted(groups.items()):
        for field in ("crop_key", "face_ref_clip_id", "video_driver", "face_identity_id", "split"):
            values = {r[field] for r in rows}
            if len(values) != 1:
                problems.append(f"{render_key}: rows disagree on {field}: {sorted(values)}")
        methods = {r.get("video_method") for r in rows}
        if methods != {S.VIDEO_METHOD_FACEFUSION}:
            problems.append(f"{render_key}: video_method is {sorted(methods)}, expected facefusion")

    for job in jobs:
        extra = job["provenance_extra"]
        same = extra["face_identity_id"] == extra["driver_identity_id"]
        if extra["video_driver"] == S.VIDEO_DRIVER_CROSS and same:
            problems.append(f"{job['job_id']}: cross render but face == driver identity")
        if extra["video_driver"] == S.VIDEO_DRIVER_SELF and not same:
            problems.append(f"{job['job_id']}: self render but face != driver identity")
        if extra["video_driver"] == S.VIDEO_DRIVER_CROSS and extra["consumer_classes"] != ["FVFA", "FVRA"]:
            problems.append(
                f"{job['job_id']}: cross render consumed by {extra['consumer_classes']}, expected FVRA+FVFA"
            )

    return problems


def main() -> int:
    """Group the manifest into swap jobs, write the face pictures, then jobs.jsonl."""
    args = parse_args()
    manifest_path = args.manifest or S.new_gen_manifest_path(args.build_id)
    if not manifest_path.exists():
        print(f"[ff-jobs] manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    # Renders are the costly step: only a manifest that passed its check gets them.
    report = S.new_gen_validate_report_path(args.build_id)
    verdict = json.loads(report.read_text()).get("verdict") if report.exists() else None
    if verdict != "PASS":
        print(
            f"[ff-jobs] REFUSING: {args.build_id} has no PASSing validate_report.json (verdict={verdict}). "
            f"Validate the manifest first.",
            file=sys.stderr,
        )
        return 1

    pair_ids = None
    if args.pair_ids_file:
        pair_ids = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
    groups = group_by_render(manifest_path, pair_ids)
    if not groups:
        scope = f"the {len(pair_ids)} requested pair_id(s)" if pair_ids is not None else "this manifest"
        print(f"[ff-jobs] REFUSING: no renders serve {scope}", file=sys.stderr)
        return 1

    commit = upstream_commit()
    jobs = [build_job(k, rows, args, commit) for k, rows in sorted(groups.items())]

    problems = validate_jobs(groups, jobs)
    if problems:
        print(f"[ff-jobs] {len(problems)} manifest/render inconsistency(ies):", file=sys.stderr)
        for p in problems[:10]:
            print(f"[ff-jobs]   {p}", file=sys.stderr)
        return 1

    missing = prepare_inputs(jobs)
    if missing:
        print(f"[ff-jobs] {len(missing)} missing input(s) — run the crop step first:", file=sys.stderr)
        for m in missing[:10]:
            print(f"[ff-jobs]   {m}", file=sys.stderr)
        if not args.allow_missing_crops:
            return 1

    # A batch of pairs gets its own folder: a short list at the build's own path would look like the whole build.
    out_dir = args.out_dir or default_jobs_dir(args.build_id, S.batch_suffix_for(pair_ids))
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = out_dir / "jobs.jsonl"
    with jobs_file.open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")

    by_driver: dict[str, int] = defaultdict(int)
    for job in jobs:
        by_driver[job["provenance_extra"]["video_driver"]] += 1
    est_minutes = round(len(jobs) * S.FACEFUSION_SEC_PER_RENDER / 60.0, 1)
    served = sum(len(j["provenance_extra"]["consumer_sample_ids"]) for j in jobs)
    print(f"[ff-jobs] wrote {len(jobs)} swap jobs -> {jobs_file}")
    print(f"[ff-jobs] by driver: {dict(by_driver)}")
    print(f"[ff-jobs] renders serve {served} shipped samples")
    print(f"[ff-jobs] upstream commit: {commit or 'UNKNOWN (FF_UPSTREAM unset or not a clone)'}")
    print(f"[ff-jobs] ESTIMATED COST: {est_minutes} GPU-minutes at {S.FACEFUSION_SEC_PER_RENDER}s/render")
    print(f"[ff-jobs] render with: sbatch --array=0-<shards-1> scripts/data_generation/render_facefusion.sh "
          f"{jobs_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
