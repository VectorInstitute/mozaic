"""Cut every driver clip of a build to its fixed window: the first step of every class.

Two files per clip, video and audio apart, so the one mux (final_clips/) joins them the same way
for every class:

    new_generated/crops/<crop_key>.mp4   video only, exactly crop_frames frames, lossless H.264
    new_generated/crops/<crop_key>.wav   audio only, exactly crop_frames x 640 samples, 16 kHz mono

Both streams are cut by count, never by a float duration, and both outputs are probed again
before they are kept. A crop that exists and probes right is skipped, so a run can be repeated.

    python -m mozaic.data_generation.crop.crop_clips --build-id <build_id>
    python -m mozaic.data_generation.crop.crop_clips --build-id <build_id> --limit 5
    python -m mozaic.data_generation.crop.crop_clips --build-id <build_id> --shard-index 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S

TIMEOUT_FFMPEG_S = 120  # a hang guard: one cut takes well under a second


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cut each driver clip to the fixed crop window.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--manifest", type=Path, default=None, help="pair list (default: the build's manifest.jsonl)")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="only the first N clips")
    p.add_argument("--pair-ids-file", type=Path, default=None,
                   help="only the driver clips of these pair_ids, one per line (default: the whole build)")
    p.add_argument("--concurrency", type=int, default=S.CROP_CONCURRENCY)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def driver_clips(manifest_path: Path, pair_ids: set[str] | None = None) -> list[tuple[str, tuple[str, int]]]:
    """The unique crop_key -> (driver_clip_id, crop_frames) of a pair list, sorted.

    The length comes from each row (crop_frames; rows without it use the default).
    """
    seen: dict[str, tuple[str, int]] = {}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if pair_ids is not None and row.get("pair_id") not in pair_ids:
            continue
        seen.setdefault(row["crop_key"], (row["driver_clip_id"], S.row_crop_frames(row)))
    return sorted(seen.items())


def shard(
    items: list[tuple[str, tuple[str, int]]], shard_index: int, num_shards: int
) -> list[tuple[str, tuple[str, int]]]:
    """Round-robin share, so shards stay balanced for any list length."""
    if not 0 <= shard_index < num_shards:
        msg = f"shard-index {shard_index} out of range for {num_shards} shards"
        raise SystemExit(msg)
    return [it for i, it in enumerate(items) if i % num_shards == shard_index]


def probe_source_fps(path: Path) -> float | None:
    """Frame rate of a source clip.

    Every usable VoxCeleb2 clip is 25 fps. Another rate is refused, never resampled:
    resampling would add or drop frames and break the frame/sample match.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path),
    ]  # fmt: skip
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=S.TIMEOUT_FFPROBE, check=False)
        num, den = out.stdout.decode().strip().split("/")
        return int(num) / int(den)
    except (ValueError, ZeroDivisionError, subprocess.TimeoutExpired):
        return None


def probe_video_frames(path: Path) -> int | None:
    """Frame count, by counting packets (the container's nb_frames can be missing or stale)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", str(path),
    ]  # fmt: skip
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=S.TIMEOUT_FFPROBE, check=False)
        return int(out.stdout.decode().strip())
    except (ValueError, subprocess.TimeoutExpired):
        return None


def probe_audio(path: Path) -> dict[str, Any]:
    """Sample count, rate and channel count of a wav."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=duration_ts,sample_rate,channels",
        "-of", "json", str(path),
    ]  # fmt: skip
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=S.TIMEOUT_FFPROBE, check=False)
        stream = json.loads(out.stdout)["streams"][0]
        return {
            "samples": int(stream["duration_ts"]),
            "sample_rate_hz": int(stream["sample_rate"]),
            "channels": int(stream["channels"]),
        }
    except (KeyError, IndexError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return {"samples": None, "sample_rate_hz": None, "channels": None}


def outputs_valid(video: Path, audio: Path, frames: int) -> bool:
    """Both crops exist and probe to the exact counts (a crop cut short by a killed job is redone)."""
    if not (video.exists() and audio.exists()):
        return False
    if probe_video_frames(video) != frames:
        return False
    a = probe_audio(audio)
    return (
        a["samples"] == S.crop_audio_samples(frames)
        and a["sample_rate_hz"] == S.AUDIO_SAMPLE_RATE_HZ
        and a["channels"] == S.AUDIO_CHANNELS
    )


def run_ffmpeg(cmd: list[str]) -> str | None:
    """Run ffmpeg; None on success, else a short error."""
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_FFMPEG_S, check=False)
    except subprocess.TimeoutExpired:
        return "ffmpeg timeout"
    if r.returncode != 0:
        return f"ffmpeg rc={r.returncode}: {r.stderr.decode(errors='replace')[:300]}"
    return None


def cut_video(src: Path, dst: Path, frames: int) -> str | None:
    """Video only: `frames` frames from crop_start_s, lossless (the mux encodes the final clip)."""
    # -ss after -i seeks exactly; -frames:v sets the count. No -r: the source is checked to be 25 fps.
    return run_ffmpeg([
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(src), "-ss", f"{S.CROP_START_S:.6f}", "-frames:v", str(frames),
        "-an", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-f", "mp4", str(dst),
    ])  # fmt: skip


def cut_audio(src: Path, dst: Path, samples: int) -> str | None:
    """Audio only: `samples` samples from crop_start_s, cut by sample index (640 per frame)."""
    start_sample = round(S.CROP_START_S * S.AUDIO_SAMPLE_RATE_HZ)
    return run_ffmpeg([
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-vn",
        "-af", f"aresample={S.AUDIO_SAMPLE_RATE_HZ},"
               f"atrim=start_sample={start_sample}:end_sample={start_sample + samples},"
               "asetpts=PTS-STARTPTS",
        "-ac", str(S.AUDIO_CHANNELS), "-ar", str(S.AUDIO_SAMPLE_RATE_HZ),
        "-c:a", "pcm_s16le", "-f", "wav", str(dst),
    ])  # fmt: skip


def crop_one(crop_key: str, driver_clip_id: str, overwrite: bool, frames: int = S.CROP_FRAMES) -> dict[str, Any]:
    """Cut one driver clip's video and audio, check both; returns its log row."""
    samples = S.crop_audio_samples(frames)
    src = S.VOXCELEB2_MP4_ROOT / f"{driver_clip_id}.mp4"
    video_dst = S.crop_video_path(crop_key)
    audio_dst = S.crop_audio_path(crop_key)
    row: dict[str, Any] = {
        "crop_key": crop_key,
        "driver_clip_id": driver_clip_id,
        "ok": False,
        "source_fps": None,
        "video_frames": None,
        "audio_samples": None,
        "error": None,
    }

    if not src.exists():
        row["error"] = f"source missing: {src}"
        return row

    fps = probe_source_fps(src)
    row["source_fps"] = fps
    if fps is None or abs(fps - S.TARGET_FPS) > 1e-6:
        row["error"] = f"source fps {fps} != TARGET_FPS {S.TARGET_FPS}; refusing to resample frames"
        return row

    if not overwrite and outputs_valid(video_dst, audio_dst, frames):
        row.update({"ok": True, "video_frames": frames, "audio_samples": samples})
        return row

    video_dst.parent.mkdir(parents=True, exist_ok=True)
    # The pid is in the temp names, so two processes on one build never write the same file.
    video_tmp = video_dst.with_suffix(f".mp4.{os.getpid()}.tmp")
    err = cut_video(src, video_tmp, frames)
    if err:
        video_tmp.unlink(missing_ok=True)
        row["error"] = f"video: {err}"
        return row

    audio_tmp = audio_dst.with_suffix(f".wav.{os.getpid()}.tmp")
    err = cut_audio(src, audio_tmp, samples)
    if err:
        audio_tmp.unlink(missing_ok=True)
        video_tmp.unlink(missing_ok=True)
        row["error"] = f"audio: {err}"
        return row

    got_frames = probe_video_frames(video_tmp)
    audio = probe_audio(audio_tmp)
    row["video_frames"] = got_frames
    row["audio_samples"] = audio["samples"]
    if got_frames != frames:
        row["error"] = f"expected {frames} video frames, got {got_frames}"
    elif audio["samples"] != samples:
        row["error"] = f"expected {samples} audio samples, got {audio['samples']}"
    if row["error"]:
        video_tmp.unlink(missing_ok=True)
        audio_tmp.unlink(missing_ok=True)
        return row

    video_tmp.replace(video_dst)  # atomic: a killed job never leaves a crop that looks finished
    audio_tmp.replace(audio_dst)
    row["ok"] = True
    return row


def main() -> int:
    """Crop this shard's driver clips, log every result, mark the build done."""
    args = parse_args()
    manifest_path = args.manifest or S.new_gen_manifest_path(args.build_id)
    if not manifest_path.exists():
        print(f"[crop] manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    S.ensure_new_gen_dirs()
    single_task = args.num_shards == 1 and args.limit is None and args.pair_ids_file is None
    done_path = S.crop_done_path(args.build_id)
    if single_task and done_path.exists() and not args.overwrite:
        print(f"[crop] build {args.build_id} already done (skip; --overwrite to redo)")
        return 0

    pair_ids = None
    if args.pair_ids_file:
        pair_ids = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
    items = shard(driver_clips(manifest_path, pair_ids), args.shard_index, args.num_shards)
    if args.limit is not None:
        items = items[: args.limit]
    print(
        f"[crop] node={socket.gethostname()} build={args.build_id} shard={args.shard_index}/{args.num_shards} "
        f"clips={len(items)} concurrency={args.concurrency} "
        f"target={sorted({n for _, (_, n) in items})}frames (per manifest row; audio = frames x {S.SAMPLES_PER_FRAME})"
    )

    t0 = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(crop_one, k, c, args.overwrite, n): k for k, (c, n) in items}
        for fut in as_completed(futures):
            row = fut.result()
            results.append(row)
            if not row["ok"]:
                print(f"[crop] FAIL {row['crop_key']}: {row['error']}", file=sys.stderr)

    n_ok = sum(1 for r in results if r["ok"])
    n_fail = len(results) - n_ok
    log_path = S.crop_log_path(args.build_id)
    with log_path.open("a" if args.num_shards > 1 else "w") as f:
        for row in sorted(results, key=lambda r: r["crop_key"]):
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"[crop] {n_ok} ok, {n_fail} failed in {time.time() - t0:.1f}s -> {log_path}")
    if n_fail == 0 and single_task:
        done_path.touch()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
