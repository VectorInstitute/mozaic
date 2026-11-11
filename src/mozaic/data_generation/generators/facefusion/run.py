"""Run FaceFusion face swaps over a job list (make_jobs.py writes it), one shard per array task.

Each job line is one swap: the face in `source_image` is put on the person in `target_video` (the driver's
crop); the motion, expression and sound of the crop stay. Only the picture is fake.

Outputs in the output folder (FF_OUT, the build's renders folder), per job: <job_id>.mp4, its sidecar
<job_id>.json (how it was made) and <job_id>.done, written last, so a killed or requeued shard resumes where it
stopped. A job FaceFusion refuses (its content check) gets <job_id>.refused.json and is never tried again; any
other failure is retried once, then gets <job_id>.failed.json. Both drop the pair, and the shard still exits 0.

Runs in FaceFusion's own environment and imports nothing from this repo:

    python src/mozaic/data_generation/generators/facefusion/run.py --jobs jobs.jsonl --shard-index 0 --num-shards 1
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

# FaceFusion's own defaults for the swap, so a job that leaves a setting out is still recorded truly.
DEFAULT_SWAPPER_MODEL = "inswapper_128_fp16"
DEFAULT_DETECTOR_MODEL = "yolo_face"
DEFAULT_DETECTOR_SCORE = 0.5
# Opens a model with onnxruntime on the GPU; exit 0 only when the GPU is really used.
GPU_PROBE = ("import sys, onnxruntime as o; "
             "s = o.InferenceSession(sys.argv[1], providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); "
             "sys.exit(0 if s.get_providers()[0] == 'CUDAExecutionProvider' else 1)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FaceFusion face swaps over a JSONL job list.")
    p.add_argument("--jobs", type=Path, required=True, help="JSONL job file")
    p.add_argument("--shard-index", type=int, default=0, help="0-based index of this shard")
    p.add_argument("--num-shards", type=int, default=1, help="total number of shards")
    p.add_argument("--upstream", type=Path, default=None, help="FaceFusion clone (default $FF_UPSTREAM)")
    p.add_argument("--out-dir", type=Path, default=None, help="output dir (default $FF_OUT)")
    p.add_argument("--swapper-model", default=DEFAULT_SWAPPER_MODEL, help="default face-swapper model")
    p.add_argument("--detector-score", type=float, default=DEFAULT_DETECTOR_SCORE, help="default detector score")
    p.add_argument("--threads", type=int, default=4, help="execution thread count")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    return p.parse_args()


def resolve_dir(explicit: Path | None, env_var: str) -> Path:
    """`explicit` if given, else the folder named by `env_var`."""
    if explicit is not None:
        return explicit
    value = os.environ.get(env_var)
    if not value:
        msg = f"neither the CLI flag nor ${env_var} is set"
        raise SystemExit(msg)
    return Path(value)


def gpu_env() -> dict[str, str]:
    """This environment, with the venv's CUDA 12 and cuDNN 9 libraries (its nvidia-*-cu12 wheels) first on
    LD_LIBRARY_PATH: onnxruntime-gpu does not find them by itself."""
    env = dict(os.environ)
    libs = [str(p) for p in sorted(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/*/lib")) if p.is_dir()]
    if env.get("LD_LIBRARY_PATH"):
        libs.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(libs)
    return env


def require_gpu(model: Path, env: dict[str, str]) -> None:
    """Stop unless onnxruntime opens the swapper model on the GPU: without CUDA, FaceFusion would swap on the
    CPU without saying so (slower, and other bytes than the release)."""
    if not model.exists():
        raise SystemExit(f"{model} not found: download FaceFusion's models first (README.md)")
    probe = subprocess.run([sys.executable, "-c", GPU_PROBE, str(model)], env=env, capture_output=True, text=True,
                           check=False)  # fmt: skip
    if probe.returncode != 0:
        why = next((line[line.index(key):] for line in (probe.stdout + probe.stderr).splitlines()
                    for key in ("CUDA failure", "Failed to create CUDAExecutionProvider") if key in line), "")
        raise SystemExit("no usable GPU for FaceFusion (none visible, or CUDA 12 / cuDNN 9 not found)"
                         + (f": {why.strip()[:200]}" if why else ""))


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
    """A job's settings: its own values, else the command-line defaults."""
    return {
        "face_swapper_model": job.get("face_swapper_model", args.swapper_model),
        "detector_model": job.get("face_detector_model", DEFAULT_DETECTOR_MODEL),
        "detector_score": float(job.get("detector_score", args.detector_score)),
        "face_enhancer": job.get("face_enhancer"),  # None = off
        # Optional quality settings; unset means FaceFusion's defaults. pixel_boost runs the 128 px swapper
        # on a larger face, split into 128 px images of every n-th pixel.
        "pixel_boost": job.get("pixel_boost"),  # e.g. "256x256"
        "enhancer_blend": job.get("face_enhancer_blend"),  # 0..100
        "video_quality": job.get("output_video_quality"),  # 0..100
    }


def build_command(job: dict[str, Any], args: argparse.Namespace, out_file: Path, settings: dict[str, Any]) -> list[str]:
    """The facefusion.py headless-run command of one swap."""
    processors = ["face_swapper"]
    if settings["face_enhancer"]:
        processors.append("face_enhancer")
    # Own jobs and temp folders per job: FaceFusion's defaults are shared (.jobs in the clone, /tmp keyed by
    # the target's name), so shards running at once would collide.
    work = out_file.parent / "_work" / job["job_id"]
    cmd = [
        sys.executable,
        "facefusion.py",
        "headless-run",
        "--jobs-path",
        str(work / "jobs"),
        "--temp-path",
        str(work / "temp"),
        "--processors",
        *processors,
        "--source-paths",
        str(job["source_image"]),
        "--target-path",
        str(job["target_video"]),
        "--output-path",
        str(out_file),
        "--execution-providers",
        "cuda",
        "--execution-thread-count",
        str(args.threads),
        "--face-swapper-model",
        settings["face_swapper_model"],
        "--face-detector-model",
        settings["detector_model"],
        "--face-detector-score",
        str(settings["detector_score"]),
        # FaceFusion cannot copy the sound stream, so it re-encodes it; the mux replaces it later anyway.
        "--output-audio-encoder",
        "aac",
    ]
    if settings["face_enhancer"]:
        cmd += ["--face-enhancer-model", str(settings["face_enhancer"])]
        if settings["enhancer_blend"] is not None:
            cmd += ["--face-enhancer-blend", str(settings["enhancer_blend"])]
    if settings["pixel_boost"]:
        cmd += ["--face-swapper-pixel-boost", str(settings["pixel_boost"])]
    if settings["video_quality"] is not None:
        cmd += ["--output-video-quality", str(settings["video_quality"])]
    return cmd


def write_provenance(path: Path, job: dict[str, Any], cmd: list[str], elapsed: float, settings: dict[str, Any]) -> None:
    """Write how one render was made, next to it."""
    record = {
        "job_id": job["job_id"],
        "generator": "facefusion/inswapper",
        "manipulation": "face_swap",
        "streams_faked": ["video"],  # the sound is the crop's own
        "source_identity_image": str(job["source_image"]),
        "target_video": str(job["target_video"]),
        "source_clip_id": job.get("source_clip_id"),
        "source_dataset": job.get("source_dataset"),
        "face_swapper_model": settings["face_swapper_model"],
        "face_detector_model": settings["detector_model"],
        "detector_score": settings["detector_score"],
        "face_enhancer": settings["face_enhancer"],
        "pixel_boost": settings["pixel_boost"],
        "face_enhancer_blend": settings["enhancer_blend"],
        "output_video_quality": settings["video_quality"],
        "experiment": job.get("experiment"),
        "elapsed_seconds": round(elapsed, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": cmd,
    }
    # The job maker's provenance (render_key, build_id, the face still, the upstream commit, the samples this
    # render serves) is copied as it is.
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
    """Swap every job of this shard, skipping the finished ones."""
    args = parse_args()
    upstream = resolve_dir(args.upstream, "FF_UPSTREAM")
    out_dir = resolve_dir(args.out_dir, "FF_OUT")
    env = gpu_env()
    if not args.dry_run:
        require_gpu(upstream / ".assets" / "models" / f"{args.swapper_model}.onnx", env)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_shard(args.jobs, args.shard_index, args.num_shards)
    print(f"[ff] shard {args.shard_index}/{args.num_shards}: {len(jobs)} jobs", flush=True)

    failures = 0
    refusals = 0
    for n, job in enumerate(jobs, start=1):
        job_id = job["job_id"]
        out_file = out_dir / f"{job_id}.mp4"
        done_marker = out_dir / f"{job_id}.done"
        if done_marker.exists():
            print(f"[ff] {n}/{len(jobs)} {job_id} — skip (done)", flush=True)
            continue
        if (out_dir / f"{job_id}.refused.json").exists():
            # A refusal is final: running the job again only refuses again.
            print(f"[ff] {n}/{len(jobs)} {job_id} — skip (refused by the content analyser earlier)", flush=True)
            refusals += 1
            continue

        settings = effective_settings(job, args)
        cmd = build_command(job, args, out_file, settings)
        if args.dry_run:
            print(" ".join(cmd))
            continue

        print(f"[ff] {n}/{len(jobs)} {job_id} — swapping", flush=True)
        started = time.time()
        result = subprocess.run(cmd, cwd=upstream, env=env, check=False, capture_output=True, text=True)
        log = (result.stdout or "") + (result.stderr or "")
        # FaceFusion's content analyser exits 1 right after "analysing", before any frame is extracted;
        # every other failure gets past that point.
        refused = "analysing" in log and "extracting frames" not in log
        if (result.returncode != 0 or not out_file.exists()) and not refused:
            # One retry in the shard: a passing GPU or read error is the usual cause.
            print(f"[ff] {job_id} failed once (rc={result.returncode}); retrying", file=sys.stderr, flush=True)
            result = subprocess.run(cmd, cwd=upstream, env=env, check=False, capture_output=True, text=True)
            log = (result.stdout or "") + (result.stderr or "")
        elapsed = time.time() - started

        if result.returncode != 0 or not out_file.exists():
            sidecar = write_failure_sidecar(out_dir, job_id, refused, result.returncode, elapsed, log)
            print(f"[ff] {job_id} {'REFUSED by the content analyser' if refused else 'FAILED'} "
                  f"rc={result.returncode} after {elapsed:.0f}s -> {sidecar.name}", file=sys.stderr, flush=True)
            print(log[-800:], file=sys.stderr, flush=True)
            if refused:
                refusals += 1
            else:
                failures += 1
            continue
        for stale in (out_dir / f"{job_id}.refused.json", out_dir / f"{job_id}.failed.json"):
            stale.unlink(missing_ok=True)

        write_provenance(out_dir / f"{job_id}.json", job, cmd, elapsed, settings)
        done_marker.touch()
        print(f"[ff] {job_id} ok in {elapsed:.0f}s -> {out_file}", flush=True)

    if refusals:
        print(f"[ff] shard finished with {refusals} documented refusal(s); the mux skips those pairs", file=sys.stderr)
    if failures:
        print(
            f"[ff] shard finished with {failures} documented failure(s) after a retry; the mux skips those pairs",
            file=sys.stderr,
        )
    # Refusals and failures both exit 0: each is a recorded drop of the pair, and a non-zero exit would keep
    # the steps queued after this shard waiting forever.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
