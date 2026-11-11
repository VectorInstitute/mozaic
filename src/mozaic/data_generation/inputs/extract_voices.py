"""Copy the sound of usable VoxCeleb2 clips out to wav files (16 kHz mono, 16-bit PCM).

The clips come from the clip list (drop_reason null, so they have sound), never from a new
walk of the mp4 folders. VoxCeleb2 sound is already 16 kHz mono, so this only takes it out
of the mp4 once, for the later steps to read. Each wav is written to a .tmp file first and
then renamed, so a stopped run never leaves a half wav that looks finished; a wav that
exists is kept unless --overwrite.

Two ways to run:
  * a list of clips (what a build needs, see make_audio_extract_list): --clip-ids or
    --clip-ids-file; exits 1 if a clip failed;
  * one shard of people, cut like list_clips (people sorted by id): --shard-index and
    --num-shards; writes a log line per clip and a .done file, and a done shard is skipped.

    python -m mozaic.data_generation.inputs.extract_voices --clip-ids-file voice_ref_clips.txt
    python -m mozaic.data_generation.inputs.extract_voices --clip-ids id00012/abc/00001 id00012/abc/00002
    python -m mozaic.data_generation.inputs.extract_voices --shard-index 0 --num-shards 60

Reads  voxceleb2/_reports/inventory.parquet, <voxceleb2>/dev/mp4/<id>/<video>/<clip>.mp4
Writes voxceleb2/audio/<id>/<video>/<clip>.wav, voxceleb2/_logs/audio_shard<i>_of_<n>.jsonl (+ .done)
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

import pandas as pd

from mozaic.data_generation import settings as S


def load_usable_rows() -> pd.DataFrame:
    """Usable clips of the clip list (drop_reason null, so they have sound)."""
    df = pd.read_parquet(S.inventory_parquet_path())
    return df[df["drop_reason"].isna()].sort_values(["identity_id", "video_id", "utterance_id"])


def shard_by_identity(df: pd.DataFrame, shard_index: int, num_shards: int) -> pd.DataFrame:
    """A contiguous part of the sorted people, cut like list_clips.shard()."""
    identities = sorted(df["identity_id"].unique())
    n = len(identities)
    per = (n + num_shards - 1) // num_shards
    start = shard_index * per
    end = min(start + per, n)
    my_identities = set(identities[start:end])
    return df[df["identity_id"].isin(my_identities)]


def extract_one(rel_path: str, clip_id: str, overwrite: bool) -> dict[str, Any]:
    """ffmpeg one clip's sound to its wav; ffprobe the result."""
    src = S.VOXCELEB2_MP4_ROOT / rel_path
    dst = S.audio_wav_path(rel_path)
    row: dict[str, Any] = {
        "clip_id": clip_id,
        "rel_path": rel_path,
        "wav_rel_path": str(dst.relative_to(S.AUDIO_ROOT)),
        "ok": False,
        "duration_s": None,
        "sample_rate_hz": None,
        "channels": None,
        "error": None,
    }

    if dst.exists() and not overwrite:
        row["ok"] = True
        probe = _probe_wav(dst)
        row.update(probe)
        return row

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".wav.tmp")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        str(S.AUDIO_CHANNELS),
        "-ar",
        str(S.AUDIO_SAMPLE_RATE_HZ),
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",  # the .wav.tmp name does not tell ffmpeg the format
        str(tmp),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=S.TIMEOUT_FFPROBE, check=False)
    except subprocess.TimeoutExpired:
        row["error"] = "ffmpeg timeout"
        tmp.unlink(missing_ok=True)
        return row

    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        row["error"] = f"ffmpeg rc={result.returncode}: {result.stderr.decode(errors='replace')[:300]}"
        tmp.unlink(missing_ok=True)
        return row

    tmp.replace(dst)  # a stopped run never leaves a half file that looks finished
    row["ok"] = True
    row.update(_probe_wav(dst))
    return row


def _probe_wav(path: Path) -> dict[str, Any]:
    """ffprobe the wav: its length, sample rate and channels (None when unreadable)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=S.TIMEOUT_FFPROBE, check=False)
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        return {
            "duration_s": float(data["format"]["duration"]),
            "sample_rate_hz": int(stream["sample_rate"]),
            "channels": int(stream["channels"]),
        }
    except Exception:  # noqa: BLE001  any failed probe: the fields stay None
        return {"duration_s": None, "sample_rate_hz": None, "channels": None}


def run_shard(shard_index: int, num_shards: int, concurrency: int, overwrite: bool) -> int:
    done_path = S.audio_shard_done_path(shard_index, num_shards)
    log_path = S.audio_shard_log_path(shard_index, num_shards)
    if done_path.exists() and not overwrite:
        print(f"[audio] shard {shard_index}/{num_shards} already done -> {log_path} (skip; --overwrite to redo)")
        return 0

    rows = shard_by_identity(load_usable_rows(), shard_index, num_shards)
    clips = list(zip(rows["rel_path"], rows["clip_id"], strict=True))
    print(
        f"[audio] node={socket.gethostname()} shard={shard_index}/{num_shards} "
        f"clips={len(clips)} concurrency={concurrency}"
    )

    t0 = time.time()
    n_ok = n_fail = 0
    tmp_log = log_path.with_suffix(".jsonl.tmp")
    S.log_dir().mkdir(parents=True, exist_ok=True)
    S.AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

    with open(tmp_log, "w") as f, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(extract_one, rel_path, clip_id, overwrite): clip_id for rel_path, clip_id in clips}
        for fut in as_completed(futures):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            n_ok += int(row["ok"])
            n_fail += int(not row["ok"])

    tmp_log.replace(log_path)  # a half-written log never looks like a finished shard
    done_path.write_text(f"clips={len(clips)} elapsed_s={time.time() - t0:.1f}\n")

    elapsed = time.time() - t0
    rate = len(clips) / elapsed if elapsed > 0 else 0.0
    print(
        f"[audio] shard {shard_index}/{num_shards} done in {elapsed:.1f}s ({rate:.1f} clips/s)  ok={n_ok} fail={n_fail}  -> {log_path}"
    )
    return 0


def run_clip_ids(clip_ids: list[str], overwrite: bool, concurrency: int = S.AUDIO_EXTRACT_CONCURRENCY) -> int:
    """Extract the given clips (no log file, no .done), in parallel threads."""
    df = load_usable_rows()
    rows = df[df["clip_id"].isin(clip_ids)]
    missing = set(clip_ids) - set(rows["clip_id"])
    if missing:
        print(
            f"[audio] WARNING: {len(missing)} clip_id(s) not found in usable inventory: {sorted(missing)[:5]}",
            file=sys.stderr,
        )

    S.AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    pairs = list(zip(rows["rel_path"], rows["clip_id"], strict=True))
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(extract_one, rel_path, clip_id, overwrite): clip_id for rel_path, clip_id in pairs}
        for i, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            mark = "ok" if row["ok"] else f"FAIL ({row['error']})"
            print(
                f"[audio] {i}/{len(pairs)} {row['clip_id']} -> {row['wav_rel_path']}  {mark}  dur={row['duration_s']} sr={row['sample_rate_hz']}"
            )
            n_ok += int(row["ok"])
            n_fail += int(not row["ok"])
    print(f"[audio] {n_ok} ok, {n_fail} failed")
    return 1 if n_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Copy the sound of usable VoxCeleb2 clips out to wav files.")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--clip-ids", nargs="*", default=None, help="extract just these clip_ids")
    ap.add_argument(
        "--clip-ids-file",
        type=Path,
        default=None,
        help="extract the clip_ids in this file, one per line (for long lists)",
    )
    ap.add_argument("--concurrency", type=int, default=S.AUDIO_EXTRACT_CONCURRENCY)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.clip_ids_file:
        args.clip_ids = [line.strip() for line in args.clip_ids_file.read_text().splitlines() if line.strip()]

    if args.clip_ids:
        return run_clip_ids(args.clip_ids, args.overwrite, args.concurrency)
    if args.shard_index is None or args.num_shards is None:
        ap.error("either --clip-ids, or both --shard-index and --num-shards, are required")
    return run_shard(args.shard_index, args.num_shards, args.concurrency, args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
