"""List the VoxCeleb2 clips of one shard of people, with ffprobe.

A shard is a run of whole people (person folders sorted by id), never of single files.
Every clip gets one JSON line: its streams, its length and a drop_reason (null, "corrupt",
"too_short" or "no_audio"). Dropped clips stay in the list with their reason. Lines are
written in the order the ffprobe calls finish, so the order changes from run to run.
A shard whose .done file exists is skipped unless --overwrite is given.

    python -m mozaic.data_generation.inputs.list_clips --shard-index 0 --num-shards 60
    python -m mozaic.data_generation.inputs.list_clips --shard-index 0 --num-shards 60 --overwrite

Reads  <voxceleb2>/dev/mp4/<id>/<video>/<clip>.mp4
Writes voxceleb2/_logs/inventory_shard<i>_of_<n>.jsonl (+ .done)
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S


def find_identities() -> list[Path]:
    """Person folders (id*), sorted: the unit a shard is cut from."""
    if not S.VOXCELEB2_MP4_ROOT.exists():
        raise FileNotFoundError(f"VoxCeleb2 mp4 root not found: {S.VOXCELEB2_MP4_ROOT}")
    return sorted(p for p in S.VOXCELEB2_MP4_ROOT.iterdir() if p.is_dir() and p.name.startswith("id"))


def shard(items: list[Path], shard_index: int, num_shards: int) -> list[Path]:
    """A contiguous part: every person is in exactly one shard."""
    n = len(items)
    per = (n + num_shards - 1) // num_shards
    start = shard_index * per
    end = min(start + per, n)
    return items[start:end]


def _parse_frame_rate(rate: str | None) -> float | None:
    if not rate:
        return None
    try:
        num, _, den = rate.partition("/")
        den_f = float(den) if den else 1.0
        return float(num) / den_f if den_f else None
    except (ValueError, ZeroDivisionError):
        return None


def probe_clip(identity_id: str, path: Path) -> dict[str, Any]:
    """ffprobe one clip -> its clip-list row, with drop_reason."""
    video_id = path.parent.name
    utterance_id = path.stem
    clip_id = f"{identity_id}/{video_id}/{utterance_id}"
    rel_path = str(path.relative_to(S.VOXCELEB2_MP4_ROOT))

    row: dict[str, Any] = {
        "identity_id": identity_id,
        "video_id": video_id,
        "utterance_id": utterance_id,
        "clip_id": clip_id,
        "rel_path": rel_path,
        "duration_s": None,
        "fps": None,
        "width": None,
        "height": None,
        "video_codec": None,
        "has_audio": False,
        "audio_codec": None,
        "audio_sample_rate_hz": None,
        "corrupt": True,
        "drop_reason": S.DROP_REASON_CORRUPT,
    }

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=S.TIMEOUT_FFPROBE, check=False)
    except subprocess.TimeoutExpired:
        return row
    if result.returncode != 0:
        return row

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return row

    duration = data.get("format", {}).get("duration")
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

    if video_stream is None or duration is None:
        return row  # no video stream or no length: stays corrupt

    row["corrupt"] = False
    row["duration_s"] = float(duration)
    row["fps"] = _parse_frame_rate(video_stream.get("r_frame_rate"))
    row["width"] = video_stream.get("width")
    row["height"] = video_stream.get("height")
    row["video_codec"] = video_stream.get("codec_name")

    if audio_stream is not None:
        row["has_audio"] = True
        row["audio_codec"] = audio_stream.get("codec_name")
        sr = audio_stream.get("sample_rate")
        row["audio_sample_rate_hz"] = int(sr) if sr is not None else None

    # A clip that is too short is marked too_short, even when it also has no audio.
    if row["duration_s"] < S.MIN_CLIP_DURATION_S:
        row["drop_reason"] = S.DROP_REASON_TOO_SHORT
    elif not row["has_audio"]:
        row["drop_reason"] = S.DROP_REASON_NO_AUDIO
    else:
        row["drop_reason"] = None

    return row


def run_shard(shard_index: int, num_shards: int, concurrency: int, overwrite: bool) -> int:
    done_path = S.inventory_shard_done_path(shard_index, num_shards)
    log_path = S.inventory_shard_log_path(shard_index, num_shards)
    if done_path.exists() and not overwrite:
        print(f"[inv] shard {shard_index}/{num_shards} already done -> {log_path} (skip; --overwrite to redo)")
        return 0

    identities = find_identities()
    my_identities = shard(identities, shard_index, num_shards)
    clip_paths: list[tuple[str, Path]] = []
    for id_dir in my_identities:
        clip_paths.extend((id_dir.name, p) for p in sorted(id_dir.glob("*/*.mp4")))

    print(
        f"[inv] node={socket.gethostname()} shard={shard_index}/{num_shards} "
        f"identities={len(my_identities)} clips={len(clip_paths)} concurrency={concurrency}"
    )

    t0 = time.time()
    n_ok = n_corrupt = n_too_short = n_no_audio = 0
    tmp_path = log_path.with_suffix(".jsonl.tmp")
    S.log_dir().mkdir(parents=True, exist_ok=True)

    with open(tmp_path, "w") as f, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(probe_clip, identity_id, p): (identity_id, p) for identity_id, p in clip_paths}
        for fut in as_completed(futures):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            if row["corrupt"]:
                n_corrupt += 1
            elif row["drop_reason"] == S.DROP_REASON_TOO_SHORT:
                n_too_short += 1
            elif row["drop_reason"] == S.DROP_REASON_NO_AUDIO:
                n_no_audio += 1
            else:
                n_ok += 1

    tmp_path.replace(log_path)  # a half-written log never looks like a finished shard
    done_path.write_text(f"clips={len(clip_paths)} elapsed_s={time.time() - t0:.1f}\n")

    elapsed = time.time() - t0
    rate = len(clip_paths) / elapsed if elapsed > 0 else 0.0
    print(
        f"[inv] shard {shard_index}/{num_shards} done in {elapsed:.1f}s ({rate:.1f} clips/s)  "
        f"ok={n_ok} corrupt={n_corrupt} too_short={n_too_short} no_audio={n_no_audio}  -> {log_path}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="List the clips of one shard of VoxCeleb2 people with ffprobe.")
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=S.INVENTORY_CONCURRENCY)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    return run_shard(args.shard_index, args.num_shards, args.concurrency, args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
