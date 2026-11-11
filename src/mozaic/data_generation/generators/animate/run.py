"""Run Wan2.2-Animate-14B over a job list (make_jobs.py writes it), one shard per array task.

Each job line is one render (FVRA and FVFA share it). Per job, four steps:

  1. face still: one frame of the face donor's clip becomes the character picture (job's ref_image).
  2. preprocess: pose and face tracking (+ sam2 segmentation in replace mode) into a materials folder per
     render; a failed or stopped job reuses them when it runs again (several GPU-minutes, same output).
  3. generate: the 14B model renders from those materials.
  4. re-stamp fps: Animate stamps its output 30 fps, so the timestamps are rescaled to target_fps without
     re-encoding, and the frames that decode are counted against expected_frames.

Two frame counts must agree, and both are checked. Animate returns the frame count of the preprocessed driver,
and preprocess_data.py defaults to --fps 30, which would turn a 25 fps driver into more frames (and the render
into a longer video than its sound). So step 2 passes --fps target_fps, checks src_pose.mp4 before the render
is paid for, and does not reuse materials whose frame count no longer matches. A mismatch left at step 4 fails
the job and writes no render (allow_frame_mismatch on a job, for length tests, only records it).

Outputs: the render at the job's out_video (the build's renders folder), its sidecar <render_key>.json and
<render_key>.done, written last, so a killed or requeued shard resumes where it stopped. A failure (generate is
retried once) gets <render_key>.failed.json, which drops the pair; the shard still exits 0. Preprocess
materials go to --proc-root (default $WA_OUT/proc), deleted after a render unless --keep-proc.

Runs in Animate's own environment and imports nothing from this repo: every value it needs is on the job.

    python src/mozaic/data_generation/generators/animate/run.py --jobs jobs.jsonl --shard-index 0 --num-shards 1
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

TIMEOUT_FFMPEG_S = 300
MIN_GPU_GIB = 70  # Animate-14B on one GPU needs an 80 GB card


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Wan2.2-Animate over a JSONL job list.")
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--upstream", type=Path, default=None, help="default $WA_UPSTREAM")
    p.add_argument("--ckpt-dir", type=Path, default=None, help="default $WA_CKPT")
    p.add_argument("--proc-root", type=Path, default=None, help="preprocess materials root (default $WA_OUT/proc)")
    p.add_argument("--keep-proc", action="store_true", help="keep preprocess materials even on success")
    p.add_argument("--skip-gpu-check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
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


def assert_big_gpu() -> None:
    """Stop before the first render rather than run out of memory deep into one."""
    import torch

    if not torch.cuda.is_available():
        msg = "no CUDA device visible"
        raise SystemExit(msg)
    name = torch.cuda.get_device_name(0)
    gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[wa] device: {name} ({gib:.0f} GiB)", flush=True)
    if gib < MIN_GPU_GIB:
        msg = f"animate-14B needs an 80GB card; got {name} at {gib:.0f} GiB"
        raise SystemExit(msg)


@functools.cache
def model_env() -> dict[str, str]:
    """The environment of the preprocessing and the render: the venv's CUDA 12 libraries, torch's own and the
    CUDA 11 ones (WA_ONNX_CUDA11_LIBS, for a CUDA 11 onnxruntime-gpu) first on LD_LIBRARY_PATH, so the ONNX
    pose and detector models run on the GPU; and no internet use (every weight is local)."""
    env = dict(os.environ)
    site = sorted(Path(sys.prefix).glob("lib/python*/site-packages"))
    libs = [str(p) for s in site for p in sorted(s.glob("nvidia/*/lib")) if p.is_dir()]
    libs += [str(s / "torch" / "lib") for s in site if (s / "torch" / "lib").is_dir()][:1]
    if env.get("WA_ONNX_CUDA11_LIBS"):
        cuda11 = Path(env["WA_ONNX_CUDA11_LIBS"]).expanduser().resolve()
        libs += [str(p) for p in sorted(cuda11.glob("nvidia/*/lib")) if p.is_dir()]
    if env.get("LD_LIBRARY_PATH"):
        libs.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(libs)
    for key, value in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("TOKENIZERS_PARALLELISM", "false")):
        env.setdefault(key, value)
    return env


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None,
        env: dict[str, str] | None = None) -> tuple[int, str]:  # fmt: skip
    """Run a command; (return code, end of its stderr)."""
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    return r.returncode, r.stderr.decode(errors="replace")[-500:]


class VramSampler:
    """Sample the GPU memory of this process and its children in the background; keep the peak, in MiB.

    Reads nvidia-smi, since generate.py runs as a child process (torch counters here would read zero), and
    counts only our own processes, so another job on the card is not counted.
    """

    def __init__(self, interval_s: float = 5.0) -> None:
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> int | None:
        cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        mine = {os.getpid(), *(_descendant_pids())}
        total = 0
        for line in out.stdout.decode().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                pid, mib = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if pid in mine:
                total += mib
        return total or None

    def _loop(self) -> None:
        while not self._stop.is_set():
            mib = self._sample_once()
            if mib is not None and (self.peak_mib is None or mib > self.peak_mib):
                self.peak_mib = mib
            self._stop.wait(self.interval_s)

    def __enter__(self) -> VramSampler:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5)


def _descendant_pids() -> set[int]:
    """The PIDs of this process's children, grandchildren, and so on."""
    try:
        out = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, timeout=30, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()
    children: dict[int, list[int]] = {}
    for line in out.stdout.decode().splitlines():
        try:
            pid, ppid = (int(x) for x in line.split())
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    seen: set[int] = set()
    stack = [os.getpid()]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def probe_frames(path: Path) -> int | None:
    """A video's frame count, by counting packets."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        return int(out.stdout.decode().strip())
    except (ValueError, subprocess.TimeoutExpired):
        return None


def probe_fps(path: Path) -> str | None:
    """The video stream's r_frame_rate, as its raw 'num/den' string."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        return out.stdout.decode().strip() or None
    except subprocess.TimeoutExpired:
        return None


def extract_ref_frame(job: dict[str, Any]) -> str | None:
    """Step 1: write the face still from the face donor's clip (kept if it exists). An error string, or None."""
    dst = Path(job["ref_image"])
    if dst.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".png.tmp")
    rc, err = run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            job["face_ref_source_mp4"],
            "-ss",
            str(job["face_ref_at_s"]),
            "-frames:v",
            "1",
            "-f",
            "image2",
            str(tmp),
        ],
        timeout=TIMEOUT_FFMPEG_S,
    )
    if rc != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return f"ref-frame extract failed rc={rc}: {err}"
    tmp.replace(dst)
    return None


def preprocess(job: dict[str, Any], upstream: Path, ckpt_dir: Path, proc_dir: Path) -> str | None:
    """Step 2: pose and face materials (+ segmentation in replace mode), reused when already made."""
    pose, ref = proc_dir / "src_pose.mp4", proc_dir / "src_ref.png"
    if pose.exists() and ref.exists():
        # Reuse only materials with the frame count expected now.
        cached = probe_frames(pose)
        if cached == int(job["expected_frames"]):
            print(f"[wa] {job['job_id']}: preprocess materials present ({cached}f) — reusing", flush=True)
            return None
        print(
            f"[wa] {job['job_id']}: cached materials have {cached}f, expected "
            f"{job['expected_frames']}f — discarding and re-running preprocess",
            flush=True,
        )
    shutil.rmtree(proc_dir, ignore_errors=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    if job["mode"] == "replace":
        sam2 = ckpt_dir / "process_checkpoint" / "sam2" / "sam2_hiera_large.pt"
        if not sam2.exists():
            return f"replace mode needs the sam2 checkpoint at {sam2}"
        mode_flags = ["--replace_flag", "--iterations", "3", "--k", "7", "--w_len", "1", "--h_len", "1"]
    else:
        mode_flags = ["--retarget_flag"]

    # --fps is passed: its default (30) would retime a 25 fps driver into more frames, and the render (which
    # takes src_pose's frame count) would be longer than its sound. The target fps keeps one count end to end.
    cmd = [
        sys.executable, "./wan/modules/animate/preprocess/preprocess_data.py",
        "--ckpt_path", str(ckpt_dir / "process_checkpoint"),
        "--video_path", job["driving_video"],
        "--refer_path", job["ref_image"],
        "--save_path", str(proc_dir),
        "--resolution_area", *str(job["resolution_area"]).split(),
        "--fps", str(job["target_fps"]),
        *mode_flags,
    ]
    rc, err = run(cmd, cwd=upstream, env=model_env())
    if rc != 0:
        return f"preprocess rc={rc}: {err}"
    if not ((proc_dir / "src_pose.mp4").exists() and (proc_dir / "src_ref.png").exists()):
        return "preprocess produced no src_pose.mp4/src_ref.png"

    # generate returns src_pose's frame count: check it before paying for the render.
    pose_frames = probe_frames(proc_dir / "src_pose.mp4")
    if pose_frames is not None and pose_frames != int(job["expected_frames"]):
        return (
            f"src_pose.mp4 has {pose_frames} frames, expected {job['expected_frames']} — "
            f"preprocess retimed the driver (check --fps {job['target_fps']})"
        )
    return None


def generate(job: dict[str, Any], upstream: Path, ckpt_dir: Path, proc_dir: Path, raw_out: Path) -> str | None:
    """Step 3: render.

    --size stays the 720p placeholder (animate-14B accepts only the 720p sizes); the real size came from
    --resolution_area at preprocess.
    """
    # generate is ~90% of a render's cost: a raw render that decodes to the expected frame count is reused.
    if raw_out.exists():
        have = decoded_frames(raw_out)
        if have == int(job["expected_frames"]):
            print(f"[wa] {job['job_id']}: raw render present ({have}f) — reusing, skipping generate", flush=True)
            return None
        print(
            f"[wa] {job['job_id']}: raw render decodes {have}f, expected {job['expected_frames']}f — regenerating",
            flush=True,
        )
        raw_out.unlink(missing_ok=True)

    # Replace mode re-lights the face with a LoRA on every frame; a job may carry relighting=false to render
    # without it (an experiment; MOZAIC's renders all have it).
    gen_flags = []
    if job["mode"] == "replace":
        gen_flags = ["--replace_flag"] + (["--use_relighting_lora"] if job.get("relighting", True) else [])
    # Every sampling setting is passed, none left to upstream's defaults: generate.py turns a missing
    # --base_seed (-1) into a random seed, so a re-run would give a different video.
    cmd = [
        sys.executable, "generate.py",
        "--task", "animate-14B",
        "--ckpt_dir", str(ckpt_dir),
        "--src_root_path", str(proc_dir),
        "--refert_num", str(job["refert_num"]),
        "--frame_num", str(job["frame_num"]),
        "--base_seed", str(job["base_seed"]),
        "--sample_steps", str(job["sample_steps"]),
        "--sample_shift", str(job["sample_shift"]),
        "--sample_guide_scale", str(job["sample_guide_scale"]),
        "--prompt", str(job["prompt"]),
        "--size", "1280*720",
        "--save_file", str(raw_out),
        "--convert_model_dtype",
        "--offload_model", "True",
        *gen_flags,
    ]
    rc, err = run(cmd, cwd=upstream, env=model_env())
    if rc != 0:
        return f"generate rc={rc}: {err}"
    if not raw_out.exists():
        return "generate produced no output file"
    return None


def decoded_frames(path: Path) -> int | None:
    """The number of frames that decode (not packets): broken timestamps can keep every packet and still lose
    frames. -map 0:v:0 -an counts the video stream only."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-f", "framemd5", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_FFMPEG_S, check=False)
    except subprocess.TimeoutExpired:
        return None
    return sum(1 for line in out.stdout.decode(errors="replace").splitlines() if line and not line.startswith("#"))


def restamp_fps(
    raw_out: Path, final_out: Path, target_fps: int, expected_frames: int, allow_mismatch: bool = False
) -> tuple[str | None, dict[str, Any]]:
    """Step 4: rewrite the render's timestamps to target_fps without re-encoding.

    Animate stamps its output at its own rate (30/1) while carrying the preprocessed driver's frame count. The
    container timestamps are scaled by source_fps / target_fps with -itsscale, which keeps the packet order
    and the timestamps B-frames need (a raw Annex-B round trip would lose frames). Every frame that went in
    must still decode; a final count other than expected_frames fails the job and writes nothing, unless
    allow_mismatch (length tests only).
    """
    raw_fps_str = probe_fps(raw_out)
    info: dict[str, Any] = {
        "raw_frames": probe_frames(raw_out),
        "raw_decoded_frames": decoded_frames(raw_out),
        "raw_fps": raw_fps_str,
        "final_frames": None,
        "final_decoded_frames": None,
        "final_fps": None,
    }
    try:
        num, den = raw_fps_str.split("/") if raw_fps_str else ("0", "1")
        source_fps = int(num) / int(den)
    except (ValueError, ZeroDivisionError, AttributeError):
        return f"cannot read the render's frame rate (got {raw_fps_str!r})", info
    if source_fps <= 0:
        return f"render reports a non-positive frame rate ({raw_fps_str})", info

    final_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_out.with_suffix(".mp4.tmp")
    scale = source_fps / target_fps  # > 1 slows the timeline: 30 -> 25 is 1.2

    rc, err = run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-itsscale",
            f"{scale:.10f}",
            "-i",
            str(raw_out),
            "-an",
            "-c:v",
            "copy",
            "-f",
            "mp4",
            str(tmp),
        ],
        timeout=TIMEOUT_FFMPEG_S,
    )
    if rc != 0:
        tmp.unlink(missing_ok=True)
        return f"restamp rc={rc}: {err}", info

    info["final_frames"] = probe_frames(tmp)
    info["final_decoded_frames"] = decoded_frames(tmp)
    info["final_fps"] = probe_fps(tmp)
    info["frames_match_expected"] = info["final_decoded_frames"] == expected_frames

    # Every frame that went in must still decode out (a packet count would not catch a lost frame).
    if info["raw_decoded_frames"] is not None and info["final_decoded_frames"] != info["raw_decoded_frames"]:
        tmp.unlink(missing_ok=True)
        return (
            f"re-stamp changed the DECODED frame count: {info['raw_decoded_frames']} -> "
            f"{info['final_decoded_frames']} (packets {info['raw_frames']} -> {info['final_frames']})",
            info,
        )
    if info["final_decoded_frames"] != expected_frames and not allow_mismatch:
        tmp.unlink(missing_ok=True)
        return (
            f"render decodes {info['final_decoded_frames']} frames, expected {expected_frames} "
            f"(raw was {info['raw_frames']} packets @ {info['raw_fps']})",
            info,
        )
    if info["final_fps"] != f"{target_fps}/1":
        tmp.unlink(missing_ok=True)
        return f"re-stamp produced fps {info['final_fps']}, expected {target_fps}/1", info

    tmp.replace(final_out)
    return None, info


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


def write_provenance(path: Path, job: dict[str, Any], elapsed: float, info: dict[str, Any]) -> None:
    """Write how one render was made, next to it."""
    record = {
        "render_key": job["render_key"],
        "relighting": job.get("relighting", True),
        "generator": "Wan-AI/Wan2.2-Animate-14B",
        "generator_version": job.get("generator_version"),
        "mode": job["mode"],
        "video_driver": job["video_driver"],
        "driving_video": job["driving_video"],
        "ref_image": job["ref_image"],
        "face_ref_clip_id": job["face_ref_clip_id"],
        "face_identity_id": job["face_identity_id"],
        "driver_identity_id": job["driver_identity_id"],
        "resolution_area": job["resolution_area"],
        # The settings, recorded so a shipped render names what made it.
        "base_seed": job["base_seed"],
        "prompt": job.get("prompt"),
        "face_ref_at_s": job.get("face_ref_at_s"),
        "face_ref_pick": job.get("face_ref_pick"),
        "sample_steps": job["sample_steps"],
        "sample_shift": job["sample_shift"],
        "sample_guide_scale": job["sample_guide_scale"],
        "frame_num": job["frame_num"],
        "refert_num": job["refert_num"],
        "expected_frames": job["expected_frames"],
        "target_fps": job["target_fps"],
        "raw_frames": info.get("raw_frames"),
        "raw_fps": info.get("raw_fps"),
        "final_frames": info.get("final_frames"),
        "raw_decoded_frames": info.get("raw_decoded_frames"),
        "final_decoded_frames": info.get("final_decoded_frames"),
        "final_fps": info.get("final_fps"),
        "frames_match_expected": info.get("frames_match_expected"),
        "peak_vram_mib": info.get("peak_vram_mib"),
        "stage_seconds": info.get("stage_seconds"),
        "allow_frame_mismatch": bool(job.get("allow_frame_mismatch", False)),
        "native_length_probe": bool(job.get("native_length_probe", False)),
        "consumer_sample_ids": job.get("consumer_sample_ids"),
        "elapsed_seconds": round(elapsed, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    """Render every job of this shard, skipping the finished ones."""
    args = parse_args()
    upstream = resolve_dir(args.upstream, "WA_UPSTREAM")
    ckpt_dir = resolve_dir(args.ckpt_dir, "WA_CKPT")
    proc_root = args.proc_root or (resolve_dir(None, "WA_OUT") / "proc")

    jobs = load_shard(args.jobs, args.shard_index, args.num_shards)
    print(f"[wa] shard {args.shard_index}/{args.num_shards}: {len(jobs)} render(s)", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"{job['job_id']}  mode={job['mode']}  -> {job['out_video']}")
        return 0
    if not jobs:
        return 0
    if not args.skip_gpu_check:
        assert_big_gpu()

    failures = 0
    for n, job in enumerate(jobs, start=1):
        render_key = job["render_key"]
        final_out = Path(job["out_video"])
        done_marker = final_out.with_suffix(".done")
        if done_marker.exists():
            # Trust a .done marker only if its render was made to the same frame count (render_key does not
            # name it): compare with the sidecar.
            sidecar = final_out.with_suffix(".json")
            stale = None
            try:
                prev = json.loads(sidecar.read_text())
                if int(prev.get("expected_frames", -1)) != int(job["expected_frames"]):
                    stale = f"{prev.get('expected_frames')}f, now expecting {job['expected_frames']}f"
            except (OSError, ValueError, json.JSONDecodeError):
                stale = "provenance sidecar missing or unreadable"
            if stale is None:
                print(f"[wa] {n}/{len(jobs)} {render_key} — skip (done)", flush=True)
                continue
            print(f"[wa] {n}/{len(jobs)} {render_key} — stale ({stale}); re-rendering", flush=True)
            done_marker.unlink(missing_ok=True)

        proc_dir = proc_root / render_key
        raw_out = proc_dir / "raw_render.mp4"
        allow_mismatch = bool(job.get("allow_frame_mismatch", False))
        note = "  [PROBE: frame mismatch tolerated]" if allow_mismatch else ""
        print(
            f"[wa] {n}/{len(jobs)} {render_key} — mode={job['mode']} expect={job['expected_frames']}f{note}",
            flush=True,
        )
        started = time.time()
        stage_s: dict[str, float] = {}

        with VramSampler() as vram:
            t0 = time.time()
            err = extract_ref_frame(job)
            stage_s["ref_frame"] = round(time.time() - t0, 1)
            if err is None:
                t0 = time.time()
                err = preprocess(job, upstream, ckpt_dir, proc_dir)
                stage_s["preprocess"] = round(time.time() - t0, 1)
            if err is None:
                t0 = time.time()
                err = generate(job, upstream, ckpt_dir, proc_dir, raw_out)
                if err is not None:
                    # One retry in the shard: a passing GPU or library error is the usual cause.
                    print(f"[wa] {render_key} generate failed once ({err}); retrying", file=sys.stderr, flush=True)
                    err = generate(job, upstream, ckpt_dir, proc_dir, raw_out)
                stage_s["generate"] = round(time.time() - t0, 1)
        info: dict[str, Any] = {}
        if err is None:
            t0 = time.time()
            err, info = restamp_fps(
                raw_out, final_out, int(job["target_fps"]), int(job["expected_frames"]), allow_mismatch
            )
            stage_s["restamp"] = round(time.time() - t0, 1)
        info["peak_vram_mib"] = vram.peak_mib
        info["stage_seconds"] = stage_s

        elapsed = time.time() - started
        if err is not None:
            # Materials are kept on failure: they are the costly part and the evidence of what went wrong.
            sidecar = write_failure_sidecar(final_out.parent, render_key, False, 1, elapsed, err)
            print(
                f"[wa] {render_key} FAILED after {elapsed:.0f}s: {err} -> {sidecar.name}", file=sys.stderr, flush=True
            )
            failures += 1
            continue

        write_provenance(final_out.with_suffix(".json"), job, elapsed, info)
        done_marker.touch()
        if not args.keep_proc:
            shutil.rmtree(proc_dir, ignore_errors=True)
        print(
            f"[wa] {render_key} ok in {elapsed / 60:.1f}min -> "
            f"{info['final_frames']}f @ {info['final_fps']} "
            f"(expected {job['expected_frames']}f, match={info.get('frames_match_expected')}) "
            f"peak_vram={info.get('peak_vram_mib')}MiB stages={info.get('stage_seconds')}",
            flush=True,
        )

    if failures:
        print(
            f"[wa] shard finished with {failures} documented failure(s) after a retry; the mux skips those pairs",
            file=sys.stderr,
        )
    # A failure is a recorded drop of the pair: exit 0 so the steps queued after this shard can run.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
