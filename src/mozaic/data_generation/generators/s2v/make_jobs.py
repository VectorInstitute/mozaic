"""Write a build's Wan2.2-S2V job list: one render per render_key, and each render's face photo.

S2V has no motion driver: the driver's crop gives only the sound (the crop wav, real speech) and the length;
the other person gives one still, picked by reference_frame.best_reference_time() as for the other two
generators. This step writes that still (S.ref_image_path) and checks every input, so a GPU job never starts
on a missing file. One job per render: FVRA and FVFA share it, the voice is put on later by the mux.

S2V makes 80-frame chunks at 16 fps (S.S2V_SECONDS_PER_CLIP = 5 s); a job asks for enough chunks to cover the
crop, and run.py retimes the result to 25 fps (S.S2V_RETIME_METHOD).

The list goes to <build's manifest folder>/s2v_jobs<suffix>/jobs.jsonl (suffix: S.batch_suffix_for, when
--pair-ids-file names a batch of pairs). Run in the core environment:

    python -m mozaic.data_generation.generators.s2v.make_jobs --build-id <build>
    python -m mozaic.data_generation.generators.s2v.make_jobs --build-id <build> --pair-ids-file <pair_ids.txt>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a build's Wan2.2-S2V job list.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <build's manifest folder>/s2v_jobs<suffix>")
    p.add_argument("--pair-ids-file", type=Path, default=None, help="newline-separated pair_ids to keep")
    p.add_argument("--ref-mode", choices=["best", "fixed"], default="best")
    p.add_argument("--limit", type=int, default=None, help="keep only the first N jobs (to render one first)")
    return p.parse_args()


def default_jobs_dir(build_id: str, suffix: str) -> Path:
    """<build's manifest folder>/s2v_jobs<suffix>: a batch of pairs gets its own folder."""
    return S.new_gen_manifest_dir(build_id) / f"s2v_jobs{suffix}"


def render_seed(render_key: str) -> int:
    """The render's own seed, the same on every run: blake2b of "<seed>:<render_key>", as for Wan-Animate."""
    digest = hashlib.blake2b(f"{S.SPLIT_SEED}:{render_key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def extract_reference(source_mp4: Path, at_seconds: float, dest: Path) -> bool:
    """Write one frame of the face donor's clip as the still S2V animates (always overwrites)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
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
    return r.returncode == 0 and dest.exists()


def build_job(render_key: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """One S2V job, driven by the real sound, used by FVRA and FVFA."""
    head = rows[0]
    frames = S.row_crop_frames(head)
    ref_source = S.VOXCELEB2_MP4_ROOT / f"{head['face_ref_clip_id']}.mp4"
    ref_at_s, ref_parts = S.FACE_REF_AT_S, {}
    if args.ref_mode == "best":
        from mozaic.data_generation.generators.reference_frame import best_reference_time

        ref_at_s, ref_parts = best_reference_time(ref_source)
    # Enough 5 s chunks to cover the crop: ceil, not round (6.12 s needs two).
    num_clip = max(1, math.ceil(S.crop_duration_s(frames) / S.S2V_SECONDS_PER_CLIP))
    return {
        "job_id": render_key,
        "image": str(S.ref_image_path(render_key)),
        "audio": str(S.crop_audio_path(head["crop_key"])),
        "prompt": S.S2V_PROMPT,
        "num_clip": num_clip,
        "pose_video": None,
        "size": S.S2V_SIZE,
        "infer_frames": S.S2V_INFER_FRAMES,
        "sample_steps": S.S2V_SAMPLE_STEPS,
        "guide_scale": S.S2V_GUIDE_SCALE,
        "start_from_ref": S.S2V_START_FROM_REF,
        "base_seed": render_seed(render_key),
        "est_minutes": round(num_clip * S.S2V_MIN_PER_CHUNK, 1),
        "provenance_extra": {
            "render_key": render_key,
            "build_id": args.build_id,
            "video_method": S.VIDEO_METHOD_S2V,
            "video_driver": head["video_driver"],
            "split": head["split"],
            "crop_key": head["crop_key"],
            "expected_frames": frames,
            "target_fps": S.TARGET_FPS,
            "native_fps": S.S2V_NATIVE_FPS,
            "retime_method": S.S2V_RETIME_METHOD,
            "audio_source": "crop wav (driver's real speech); converted audio is muxed on for FVFA",
            "face_identity_id": head["face_identity_id"],
            "driver_identity_id": head["driver_identity_id"],
            "face_ref_clip_id": head["face_ref_clip_id"],
            "face_ref_source_mp4": str(ref_source),
            "face_ref_at_s": ref_at_s,
            "face_ref_pick": args.ref_mode,
            "face_ref_score": ref_parts,
            "consumer_sample_ids": sorted(r["sample_id"] for r in rows),
            "consumer_classes": sorted({r["class_name"] for r in rows}),
            "generator_version": S.generator_version(S.VIDEO_METHOD_S2V),
        },
    }


def main() -> int:
    """Group the manifest into S2V jobs, write the stills, then jobs.jsonl."""
    args = parse_args()
    manifest_path = args.manifest or S.new_gen_manifest_path(args.build_id)
    report = S.new_gen_validate_report_path(args.build_id)
    verdict = json.loads(report.read_text()).get("verdict") if report.exists() else None
    if verdict != "PASS":
        print(f"[s2v-jobs] REFUSING: {args.build_id} has no PASSing validate_report.json", file=sys.stderr)
        return 1
    pair_ids = None
    if args.pair_ids_file:
        pair_ids = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if pair_ids is not None and row.get("pair_id") not in pair_ids:
            continue
        if row.get("needs_render"):
            if row.get("video_method") != S.VIDEO_METHOD_S2V:
                print(f"[s2v-jobs] REFUSING: {args.build_id} is a {row.get('video_method')} build", file=sys.stderr)
                return 1
            groups[row["render_key"]].append(row)
    if not groups:
        print("[s2v-jobs] REFUSING: no renders to build", file=sys.stderr)
        return 1
    jobs = [build_job(k, rows, args) for k, rows in sorted(groups.items())]
    if args.limit:
        jobs = jobs[: args.limit]
    # Both checks run for every job, so every job's still is written even when its wav is missing.
    missing = [
        f"{j['job_id']}: {what}"
        for j in jobs
        for what, ok in (
            ("crop wav", Path(j["audio"]).exists()),
            (
                "reference still",
                extract_reference(
                    Path(j["provenance_extra"]["face_ref_source_mp4"]),
                    j["provenance_extra"]["face_ref_at_s"],
                    Path(j["image"]),
                ),
            ),
        )
        if not ok
    ]
    if missing:
        print(f"[s2v-jobs] {len(missing)} missing input(s):", file=sys.stderr)
        for m in missing[:10]:
            print(f"[s2v-jobs]   {m}", file=sys.stderr)
        return 1
    suffix = S.batch_suffix_for(pair_ids)
    if args.limit:
        suffix = f"{suffix}_first{args.limit}"
    out_dir = args.out_dir or default_jobs_dir(args.build_id, suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = out_dir / "jobs.jsonl"
    with jobs_file.open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")
    hours = round(sum(j["est_minutes"] for j in jobs) / 60.0, 1)
    print(f"[s2v-jobs] wrote {len(jobs)} jobs -> {jobs_file}")
    print(
        f"[s2v-jobs] chunks per job: {sorted({j['num_clip'] for j in jobs})}; "
        f"size {S.S2V_SIZE}; steps {S.S2V_SAMPLE_STEPS}"
    )
    print(f"[s2v-jobs] ESTIMATED COST: {hours} A100-hours ({S.S2V_MIN_PER_CHUNK} min/chunk)")
    print(f"[s2v-jobs] render with: sbatch --array=0-<shards-1> scripts/data_generation/render_s2v.sh {jobs_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
