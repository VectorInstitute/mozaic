"""Build the lip-sync check's input (sync_pairs.jsonl) of one batch of a build: its first 20 pairs.

Lip sync is a spot check. SyncNet's face detector (S3FD) runs over every frame on the CPU, so scoring a whole
batch would take hours: the check scores the batch's first 20 pairs by pair id (10% of a 200-pair batch), every
shipped clip of each, so a fake clip is always read against its own pair's real clip (RVRA, the clip's ceiling:
SyncNet confidence has no absolute scale). keep_or_drop drops a sampled pair whose clips fail the limits; the
other pairs of the batch are not measured for lip sync.

Every path points at the build's final folder: what is scored is what ships.

    python -m mozaic.data_generation.gates.make_lip_sync_list --build-id <build> --final-dir <final> \\
        --pair-ids-file <pairs.txt> --out-dir <gates>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mozaic.data_generation.gates.make_voice_check_list import load_rows, shipped
from mozaic.data_generation import settings as S

SAMPLE_PAIRS = 20  # pairs per batch scored with SyncNet


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Build the lip-sync check's clip list for one batch of a build.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <new_generated>/_reports/<build-id>_gates")
    p.add_argument("--sync-pairs", type=int, default=SAMPLE_PAIRS, help="pairs to score (all their clips each)")
    p.add_argument("--final-dir", type=Path, default=None, help="default: <new_generated>/final")
    p.add_argument("--pair-ids-file", type=Path, default=None, help="restrict to these pair_ids (one batch of a build)")
    return p.parse_args()


def build_sync_pairs(rows: list[dict[str, Any]], n_pairs: int, root: Path | None = None) -> list[dict[str, Any]]:
    """One row per shipped clip of the first n_pairs pair ids (sorted), in pair and class order."""
    wanted = set(sorted({r["pair_id"] for r in rows})[:n_pairs])
    out: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: (x["pair_id"], x["class_name"])):
        if r["pair_id"] not in wanted:
            continue
        clip = shipped(r["class_name"], r["sample_id"], root)
        if not clip.exists():
            continue
        out.append({
            "job_id": r["sample_id"],
            "clip": str(clip),
            "class_name": r["class_name"],
            "pair_id": r["pair_id"],
            "audio_variant": r["audio_variant"],
            "video_driver": r["video_driver"],
        })  # fmt: skip
    return out


def main() -> int:
    """Write the clip list and say what it holds."""
    args = parse_args()
    rows = load_rows(args.build_id)
    if args.pair_ids_file:
        keep = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
        rows = [r for r in rows if r["pair_id"] in keep]
        if not rows:
            print("[gates] REFUSING: no manifest rows match --pair-ids-file", file=sys.stderr)
            return 1
    out_dir = args.out_dir or (S.new_gen_report_dir() / f"{args.build_id}_gates")
    out_dir.mkdir(parents=True, exist_ok=True)
    root = S.resolve_final_dir(args.final_dir)
    sync_pairs = build_sync_pairs(rows, args.sync_pairs, root)
    if not sync_pairs:
        print(f"[gates] REFUSING: no shipped clip of the sampled pairs under {root}", file=sys.stderr)
        return 1
    sync_path = out_dir / "sync_pairs.jsonl"
    with sync_path.open("w") as f:
        for row in sync_pairs:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    by_class: dict[str, int] = {}
    for row in sync_pairs:
        by_class[row["class_name"]] = by_class.get(row["class_name"], 0) + 1
    print(f"[gates] lip-sync list -> {sync_path}: {len(sync_pairs)} clips over {args.sync_pairs} pairs: {by_class}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
