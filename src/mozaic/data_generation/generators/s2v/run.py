"""Run Wan2.2-S2V-14B over a job list (make_jobs.py writes it), one shard per array task.

Each job line is one render: one face still (`image`) and one sound (`audio`, the driver's real speech) become a
talking video, through Wan2.2's generate.py. S2V makes 16 fps; for a job that names the target (expected_frames
and target_fps in its provenance), the 16 fps file is kept as <job_id>__native16.mp4 and the render is retimed to
25 fps by repeating frames (ffmpeg fps=25, nothing invented), trimmed to expected_frames, video only.

Outputs in the output folder (WAN_OUT, the build's renders folder), per job: <job_id>.mp4, its sidecar
<job_id>.json (how it was made) and <job_id>.done, written last, so a killed or requeued shard resumes where it
stopped; a finished generation whose retime failed only redoes the retime. A failure is retried once, then gets
<job_id>.failed.json, which drops the pair; the shard still exits 0.

Runs in S2V's own environment and imports nothing from this repo:

    python src/mozaic/data_generation/generators/s2v/run.py --jobs jobs.jsonl --shard-index 0 --num-shards 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Wan2.2's defaults (wan/configs/wan_s2v_14B.py), so a job that leaves a setting out is still recorded truly.
DEFAULT_SIZE = "1024*704"
DEFAULT_INFER_FRAMES = 80
DEFAULT_SAMPLE_STEPS = 40
DEFAULT_GUIDE_SCALE = 4.5
OUTPUT_FPS = 16  # wan_shared_cfg.sample_fps; S2V does not change it


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Wan2.2-S2V-14B over a JSONL job list.")
    p.add_argument("--jobs", type=Path, required=True, help="JSONL job file")
    p.add_argument("--shard-index", type=int, default=0, help="0-based index of this shard")
    p.add_argument("--num-shards", type=int, default=1, help="total number of shards")
    p.add_argument("--upstream", type=Path, default=None, help="Wan2.2 clone (default $WAN_UPSTREAM)")
    p.add_argument("--ckpt-dir", type=Path, default=None, help="checkpoint dir (default $WAN_CKPT)")
    p.add_argument("--out-dir", type=Path, default=None, help="output dir (default $WAN_OUT)")
    p.add_argument("--size", default=DEFAULT_SIZE, help=f"generation area (default {DEFAULT_SIZE})")
    p.add_argument("--infer-frames", type=int, default=DEFAULT_INFER_FRAMES, help="frames per clip, multiple of 4")
    p.add_argument("--sample-steps", type=int, default=DEFAULT_SAMPLE_STEPS, help="denoising steps")
    p.add_argument("--guide-scale", type=float, default=DEFAULT_GUIDE_SCALE, help="classifier-free guidance scale")
    p.add_argument("--start-from-ref", action="store_true", help="use the reference image as the first frame")
    p.add_argument("--no-offload", action="store_true", help="keep every module resident on the GPU")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    return p.parse_args()


def model_env() -> dict[str, str]:
    """This environment without internet use: every weight is local."""
    env = dict(os.environ)
    for key, value in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("TOKENIZERS_PARALLELISM", "false")):
        env.setdefault(key, value)
    return env


def resolve_dir(explicit: Path | None, env_var: str) -> Path:
    """`explicit` if given, else the folder named by `env_var`."""
    if explicit is not None:
        return explicit
    value = os.environ.get(env_var)
    if not value:
        msg = f"neither the CLI flag nor ${env_var} is set"
        raise SystemExit(msg)
    return Path(value)


def load_shard(jobs_file: Path, shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    """The job lines of this shard: every num_shards-th line, from line shard_index."""
    if not 0 <= shard_index < num_shards:
        msg = f"shard-index {shard_index} out of range for {num_shards} shards"
        raise SystemExit(msg)
    jobs: list[dict[str, Any]] = []
    with jobs_file.open() as fh:
        for i, raw in enumerate(fh):
            line = raw.strip()
            if line and i % num_shards == shard_index:
                jobs.append(json.loads(line))
    return jobs


def effective_settings(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """A job's settings: its own size, sample_steps, guide_scale, infer_frames and start_from_ref, else the
    command-line values (MOZAIC's jobs carry all five)."""
    return {
        "size": job.get("size", args.size),
        "sample_steps": int(job.get("sample_steps", args.sample_steps)),
        "guide_scale": float(job.get("guide_scale", args.guide_scale)),
        "infer_frames": int(job.get("infer_frames", args.infer_frames)),
        "start_from_ref": bool(job.get("start_from_ref", args.start_from_ref)),
    }


def build_command(
    job: dict[str, Any],
    args: argparse.Namespace,
    ckpt_dir: Path,
    out_file: Path,
    settings: dict[str, Any],
) -> list[str]:
    """The generate.py command of one job."""
    cmd = [
        sys.executable,
        "generate.py",
        "--task",
        "s2v-14B",
        "--ckpt_dir",
        str(ckpt_dir),
        "--size",
        settings["size"],
        "--image",
        str(job["image"]),
        "--audio",
        str(job["audio"]),
        "--prompt",
        str(job["prompt"]),
        "--save_file",
        str(out_file),
        "--infer_frames",
        str(settings["infer_frames"]),
        "--sample_steps",
        str(settings["sample_steps"]),
        "--sample_guide_scale",
        str(settings["guide_scale"]),
        "--convert_model_dtype",
    ]
    if not args.no_offload:
        # The 14B model, the 10.6 GB T5 and the VAE take turns on the GPU: slower, about 30 GB less memory.
        cmd += ["--offload_model", "True"]
    if job.get("num_clip"):
        cmd += ["--num_clip", str(job["num_clip"])]
    if job.get("pose_video"):
        cmd += ["--pose_video", str(job["pose_video"])]
    if settings["start_from_ref"]:
        cmd += ["--start_from_ref"]
    # generate.py draws a random seed when --base_seed is left out; MOZAIC's jobs carry their own.
    if job.get("base_seed") is not None:
        cmd += ["--base_seed", str(int(job["base_seed"]))]
    return cmd


def _probe_frames(path: Path) -> tuple[int | None, str]:
    """(frame count, r_frame_rate) of a video's first stream, read by key (ffprobe's csv order is its own)."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    kv = dict(line.split("=", 1) for line in probe.stdout.splitlines() if "=" in line)
    frames = kv.get("nb_read_packets", "")
    return (int(frames) if frames.isdigit() else None), kv.get("r_frame_rate", "")


def retime_to_contract(native: Path, out_file: Path, target_fps: int, expected_frames: int) -> dict[str, Any]:
    """Turn S2V's 16 fps output into target_fps and expected_frames, and say how.

    Frame repetition (ffmpeg fps=N), never interpolation: nothing is invented, the result is deterministic, and
    the cadence it leaves is written in the sidecar. Video only.
    """
    native_frames, native_rate = _probe_frames(native)
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(native),
            "-an",
            "-vf",
            f"fps={target_fps}",
            "-frames:v",
            str(expected_frames),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(out_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    got, _ = _probe_frames(out_file) if out_file.exists() else (None, "")
    return {
        "native_file": str(native),
        "native_frames": native_frames,
        "native_rate": native_rate,
        "target_fps": target_fps,
        "expected_frames": expected_frames,
        "output_frames": got,
        "method": "ffmpeg fps filter (frame duplication), trim to expected_frames, video only, x264 crf 10",
        "ok": result.returncode == 0 and got == expected_frames,
    }


def write_provenance(
    path: Path,
    job: dict[str, Any],
    cmd: list[str],
    elapsed: float,
    settings: dict[str, Any],
) -> None:
    """Write how one render was made, next to it."""
    record = {
        "job_id": job["job_id"],
        "generator": "Wan-AI/Wan2.2-S2V-14B",
        "reference_image": str(job["image"]),
        "driving_audio": str(job["audio"]),
        "prompt": job["prompt"],
        "pose_video": job.get("pose_video"),
        "num_clip": job.get("num_clip"),
        "size": settings["size"],
        "infer_frames": settings["infer_frames"],
        "sample_steps": settings["sample_steps"],
        "guide_scale": settings["guide_scale"],
        "start_from_ref": settings["start_from_ref"],
        "output_fps": OUTPUT_FPS,
        "experiment": job.get("experiment"),
        "elapsed_seconds": round(elapsed, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": cmd,
        "base_seed": job.get("base_seed"),
    }
    # The job maker's provenance (render_key, build_id, the face still, the samples this render serves) and
    # the retime facts are copied as they are.
    record.update(job.get("provenance_extra") or {})
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def write_failure_sidecar(out_dir: Path, job_id: str, refused: bool, rc: int, elapsed: float, log_tail: str) -> Path:
    """Write why a render did not happen, where it would have been.

    <job_id>.refused.json: the generator declined the input; <job_id>.failed.json: any other failure. Both
    keep the end of the log.
    """
    kind = "refused" if refused else "failed"
    path = out_dir / f"{job_id}.{kind}.json"
    path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "kind": kind,
                "reason": "content_analyser" if refused else f"rc={rc}",
                "returncode": rc,
                "elapsed_s": round(elapsed, 1),
                "log_tail": log_tail[-1500:],
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
        + "\n"
    )
    return path


def main() -> int:
    """Render every job of this shard, skipping the finished ones."""
    args = parse_args()
    upstream = resolve_dir(args.upstream, "WAN_UPSTREAM")
    ckpt_dir = resolve_dir(args.ckpt_dir, "WAN_CKPT")
    out_dir = resolve_dir(args.out_dir, "WAN_OUT")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_shard(args.jobs, args.shard_index, args.num_shards)
    print(f"[s2v] shard {args.shard_index}/{args.num_shards}: {len(jobs)} jobs", flush=True)

    failures = 0
    for n, job in enumerate(jobs, start=1):
        job_id = job["job_id"]
        out_file = out_dir / f"{job_id}.mp4"
        done_marker = out_dir / f"{job_id}.done"
        if done_marker.exists():
            print(f"[s2v] {n}/{len(jobs)} {job_id} — skip (done)", flush=True)
            continue
        extra = job.get("provenance_extra") or {}
        native = out_dir / f"{job_id}__native{OUTPUT_FPS}.mp4"
        if extra.get("expected_frames") and extra.get("target_fps") and native.exists() and not out_file.exists():
            # A finished generation whose retime failed: redo only the retime (the GPU work is on disk).
            print(f"[s2v] {n}/{len(jobs)} {job_id} — native render present, retiming only", flush=True)
            retime = retime_to_contract(native, out_file, int(extra["target_fps"]), int(extra["expected_frames"]))
            extra["retime"] = retime
            if not retime["ok"]:
                print(f"[s2v] {job_id} RETIME FAILED: {retime}", file=sys.stderr, flush=True)
                failures += 1
                continue
            settings = effective_settings(job, args)
            cmd = build_command(job, args, ckpt_dir, out_file, settings)
            write_provenance(out_dir / f"{job_id}.json", job, cmd, 0.0, settings)
            done_marker.touch()
            print(f"[s2v] {job_id} ok (retime only) -> {out_file}", flush=True)
            continue

        settings = effective_settings(job, args)
        cmd = build_command(job, args, ckpt_dir, out_file, settings)
        if args.dry_run:
            print(" ".join(cmd))
            continue

        print(f"[s2v] {n}/{len(jobs)} {job_id} — generating", flush=True)
        started = time.time()
        result = subprocess.run(cmd, cwd=upstream, env=model_env(), check=False)
        if result.returncode != 0 or not out_file.exists():
            # One retry in the shard: a passing GPU or library error is the usual cause.
            print(f"[s2v] {job_id} failed once (rc={result.returncode}); retrying", file=sys.stderr, flush=True)
            result = subprocess.run(cmd, cwd=upstream, env=model_env(), check=False)
        elapsed = time.time() - started

        if result.returncode != 0 or not out_file.exists():
            sidecar = write_failure_sidecar(out_dir, job_id, False, result.returncode, elapsed, "")
            print(
                f"[s2v] {job_id} FAILED rc={result.returncode} after {elapsed:.0f}s -> {sidecar.name}",
                file=sys.stderr,
                flush=True,
            )
            failures += 1
            continue
        (out_dir / f"{job_id}.failed.json").unlink(missing_ok=True)

        extra = job.get("provenance_extra") or {}
        if extra.get("expected_frames") and extra.get("target_fps"):
            native = out_dir / f"{job_id}__native{OUTPUT_FPS}.mp4"
            out_file.replace(native)
            retime = retime_to_contract(native, out_file, int(extra["target_fps"]), int(extra["expected_frames"]))
            extra["retime"] = retime
            if not retime["ok"]:
                print(f"[s2v] {job_id} RETIME FAILED: {retime}", file=sys.stderr, flush=True)
                failures += 1
                continue
        write_provenance(out_dir / f"{job_id}.json", job, cmd, elapsed, settings)
        done_marker.touch()
        print(f"[s2v] {job_id} ok in {elapsed:.0f}s -> {out_file}", flush=True)

    if failures:
        print(
            f"[s2v] shard finished with {failures} documented failure(s) after a retry; the mux skips those pairs",
            file=sys.stderr,
        )
    # A failure is a recorded drop of the pair: exit 0 so the steps queued after this shard can run.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
