"""Lip-sync check: SyncNet confidence of every clip in a list (make_lip_sync_list writes it).

Each shipped clip is scored on its own, so the four classes answer four questions:

    RVRA   real video + real voice        the ceiling for this clip
    RVFA   real video + converted voice   did the voice conversion break the sync
    FVRA   fake video + real voice        did the video generator keep the lip motion
    FVFA   fake video + converted voice   both together

SyncNet confidence has no absolute scale (it moves with pose, light and how much the speaker moves), so
keep_or_drop reads each fake clip against its own pair's RVRA. The face is detected, tracked and cropped by
SyncNet's own pipeline (S3FD), the crop is put back together with the same clip's own sound, and SyncNet
scores the offset and the confidence, as its authors' code does.

Runs in the SyncNet environment (requirements_syncnet.txt, CPU) with SyncNet's code (joonson/syncnet_python,
--upstream or $SYNCNET_UPSTREAM, with its two weight files) and ffmpeg on PATH; it imports nothing from this repo:

    python src/mozaic/data_generation/gates/lip_sync_check.py --pairs <gates>/sync_pairs.jsonl \\
        --out <gates>/sync_report.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

MIN_TRACK_FRAMES = 50  # about 2 s at 25 fps; SyncNet's default of 100 would drop every 77-frame clip


def run(cmd: list[str], **kw: Any) -> None:
    """Run a command, quiet on success."""
    subprocess.run(cmd, check=True, capture_output=True, **kw)  # noqa: S603


def crop_face(upstream: Path, clip: Path, work_dir: Path, reference: str) -> Path:
    """Detect, track and crop the face with SyncNet's pipeline; return the video-only crop."""
    run([sys.executable, str(upstream / "run_pipeline.py"), "--videofile", str(clip), "--reference", reference,
         "--data_dir", str(work_dir), "--min_track", str(MIN_TRACK_FRAMES), "--overwrite"], cwd=upstream)  # fmt: skip
    tracks = sorted((work_dir / "pycrop" / reference).glob("*.avi"))
    if not tracks:
        raise RuntimeError("no face track found")
    track = max(tracks, key=lambda p: p.stat().st_size)
    video_only = work_dir / "video_only.avi"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(track), "-an", "-c:v", "copy", str(video_only)])
    return video_only


def remux_with_own_audio(video_only: Path, clip: Path, out_path: Path) -> None:
    """Put the cropped frames back together with the shipped clip's own sound."""
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_only), "-i", str(clip), "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)])  # fmt: skip


def score_clip(syncnet: Any, opt: Any, upstream: Path, clip: Path, work_root: Path, job_id: str) -> dict[str, Any]:
    """Crop one clip's face, put its own sound back, and score the sync."""
    work_dir = work_root / job_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    try:
        video_only = crop_face(upstream, clip, work_dir, reference="crop")
        muxed = work_dir / "scored.mp4"
        remux_with_own_audio(video_only, clip, muxed)
        opt.tmp_dir = str(work_dir / "eval")
        opt.reference = "eval"
        offset, conf, _ = syncnet.evaluate(opt, videofile=str(muxed))
        return {"offset_frames": float(offset), "confidence": float(conf)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Score lip sync of every clip in a list with SyncNet.")
    p.add_argument("--pairs", type=Path, required=True, help="JSONL: {job_id, clip, class_name, pair_id}")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--upstream", type=Path, default=None, help="SyncNet's code (default $SYNCNET_UPSTREAM)")
    p.add_argument("--work-dir", type=Path, default=None, help="default: <out folder>/sync_work")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--vshift", type=int, default=15)
    return p.parse_args()


def main() -> int:
    """Load SyncNet once and score every clip of the list."""
    args = parse_args()
    upstream = args.upstream or (Path(os.environ["SYNCNET_UPSTREAM"]) if os.environ.get("SYNCNET_UPSTREAM") else None)
    if upstream is None:
        raise SystemExit("neither --upstream nor $SYNCNET_UPSTREAM is set")
    upstream = upstream.resolve()
    work_root = args.work_dir or (args.out.parent / "sync_work")
    work_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(upstream))
    from SyncNetInstance import SyncNetInstance  # noqa: PLC0415

    syncnet = SyncNetInstance()
    syncnet.loadParameters(str(upstream / "data" / "syncnet_v2.model"))
    opt = argparse.Namespace(batch_size=args.batch_size, vshift=args.vshift, tmp_dir="", reference="")

    rows = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    print(f"[lipsync] {len(rows)} clips", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for i, row in enumerate(rows, start=1):
            try:
                result = score_clip(syncnet, opt, upstream, Path(row["clip"]), work_root, row["job_id"])
            except Exception as exc:  # noqa: BLE001 (one bad clip must not stop the list)
                result = {"error": str(exc)[:200]}
            result.update({k: row.get(k) for k in ("job_id", "class_name", "pair_id")})
            result["scored_clip"] = row["clip"]
            msg = result.get("error") or f"conf={result['confidence']:.2f} offset={result['offset_frames']:+.0f}"
            print(f"[lipsync] {i}/{len(rows)} {row['class_name']} {row['job_id']}: {msg}", flush=True)
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
