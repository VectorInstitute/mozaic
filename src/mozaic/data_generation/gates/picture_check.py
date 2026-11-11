"""Picture check measurements: flicker, face melting and face softness of every shipped clip.

Measured on every main-arm clip of a build (or of the pairs in a list) and written per
clip, so the rule (keep_or_drop) can compare each generated clip with the REAL clip of
the same pair: camera motion, focus and sensor grain then cancel out.

Per clip, on grayscale frames scaled to 224 x 224:
    jit_whole      brightness pulsing: mean |frame-to-frame change| of each frame's mean
                   brightness, after removing a 5-frame moving average
    wobble_face    face melting: mean per-pixel |frame-to-frame change| of the central
                   face region, at 96 px after a Gaussian blur (sigma 2), so grain does not count
    sharp_face_lf  face sharpness: Laplacian variance of the central 112 x 112 face region
                   after a Gaussian blur (sigma 1.5): detail without sensor grain

    python -m mozaic.data_generation.gates.picture_check --build-id <build> --final-dir <final> \\
        [--pair-ids-file <pairs.txt>] --out <gates>/picture_report.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mozaic.data_generation import settings as S

SIZE = 224
SMALL = 96
FACE = 56  # the central 112x112 of 224: where the face is in a VoxCeleb2 crop


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Per-clip picture-quality measurements for the selection rule.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--final-dir", type=Path, required=True, help="the shipped clips to measure")
    p.add_argument("--pair-ids-file", type=Path, default=None, help="restrict to these pairs (one batch)")
    p.add_argument("--out", type=Path, required=True, help="picture_report.jsonl, one row per shipped main-arm clip")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    return p.parse_args()


def frames(path: Path, size: int = SIZE) -> np.ndarray | None:
    """Grayscale frames at size x size via ffmpeg, or None for an unreadable clip."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf", f"scale={size}:{size}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
        check=False,
    )
    n = len(proc.stdout) // (size * size)
    if n < 10:
        return None
    return np.frombuffer(proc.stdout[: n * size * size], dtype=np.uint8).reshape(n, size, size)


def measure(fr: np.ndarray) -> dict[str, float]:
    """Compute the three measurements for one clip's frames."""
    f32 = fr.astype(np.float32)
    mean = f32.mean(axis=(1, 2))
    k = 5
    hp = (mean - np.convolve(mean, np.ones(k) / k, mode="same"))[k:-k]
    jit = float(np.mean(np.abs(np.diff(hp))))
    small = np.stack([cv2.resize(f, (SMALL, SMALL), interpolation=cv2.INTER_AREA) for f in f32])
    blurred = np.stack([cv2.GaussianBlur(f, (0, 0), 2.0) for f in small])
    q = SMALL // 4
    wobble = float(np.abs(np.diff(blurred[:, q:-q, q:-q], axis=0)).mean())
    face = f32[:, FACE:-FACE, FACE:-FACE]
    sharp = float(np.mean([cv2.Laplacian(cv2.GaussianBlur(f, (0, 0), 1.5), cv2.CV_32F).var() for f in face]))
    return {"jit_whole": jit, "wobble_face": wobble, "sharp_face_lf": sharp, "n_frames": int(len(fr))}


def one(job: tuple[str, dict[str, Any], str]) -> dict[str, Any] | None:
    """Measure one shipped clip; None when it is not on disk."""
    build_id, row, final_dir = job
    path = S.find_final_sample(row["class_name"], row["sample_id"], Path(final_dir))
    if path is None:
        return None
    fr = frames(path)
    if fr is None:
        return None
    return {
        "build_id": build_id,
        "pair_id": row["pair_id"],
        "class_name": row["class_name"],
        "sample_id": row["sample_id"],
        **measure(fr),
    }


def main() -> int:
    """Measure every shipped main-arm clip and write the report."""
    args = parse_args()
    rows = [json.loads(x) for x in S.new_gen_manifest_path(args.build_id).read_text().splitlines() if x.strip()]
    if args.pair_ids_file:
        keep = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
        rows = [r for r in rows if r["pair_id"] in keep]
    rows = [r for r in rows if r["arm"] == "main"]
    final_dir = S.resolve_final_dir(args.final_dir)
    jobs = [(args.build_id, r, str(final_dir)) for r in rows]
    print(f"[picture] {args.build_id}: {len(jobs)} main-arm rows in {final_dir}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool, args.out.open("w") as fh:
        for rec in pool.map(one, jobs, chunksize=8):
            if rec:
                fh.write(json.dumps(rec) + "\n")
                n += 1
    print(f"[picture] {n} clips measured -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
