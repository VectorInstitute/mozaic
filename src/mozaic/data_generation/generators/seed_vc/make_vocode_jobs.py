"""Vocoder job list of a build: one job per real driver sound that gets a vocoded copy.

Each job sends a crop's sound (crops/<crop_key>.wav) through BigVGAN, the vocoder Seed-VC ends in
(vocode.py), to vocoded/<crop_key>.wav, where the mux reads it. The copy is still real speech and stays
labelled real, so "this sound went through a neural vocoder" no longer means "fake": without it every
converted voice is vocoded and no real one is.

One job per distinct crop_key, not per row: several rows of the pair list can share one vocoded wav, as
they share one render or one conversion.

    python -m mozaic.data_generation.generators.seed_vc.make_vocode_jobs --build-id <build>
    python src/mozaic/data_generation/generators/seed_vc/vocode.py --jobs <manifests>/<build>/vocode_jobs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Build the vocoded-real arm's job list from a 4-class manifest.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--out", type=Path, default=None, help="jobs JSONL (default: beside the manifest)")
    p.add_argument("--pair-ids-file", type=Path, default=None, help="newline-separated pair_ids to restrict to")
    p.add_argument(
        "--allow-missing", action="store_true", help="emit jobs whose crop audio is absent (NOT for real runs)"
    )
    return p.parse_args()


def vocode_rows(build_id: str, pair_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Return the rows needing a vocoder pass, only *pair_ids*' when given."""
    rows: list[dict[str, Any]] = []
    with S.new_gen_manifest_path(build_id).open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if pair_ids is not None and row.get("pair_id") not in pair_ids:
                continue
            if row.get("needs_vocode"):
                rows.append(row)
    return rows


def build_jobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse rows to one job per distinct crop_key, in a fixed order."""
    seen: set[str] = set()
    jobs: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: str(r["crop_key"])):
        crop_key = str(row["crop_key"])
        if crop_key in seen:
            continue
        seen.add(crop_key)
        jobs.append(
            {
                "job_id": crop_key,
                "crop_key": crop_key,
                "split": row.get("split"),
                "source_wav": str(S.crop_audio_path(crop_key)),
                "out_wav": str(S.new_gen_vocoded_dir() / f"{crop_key}.wav"),
            }
        )
    return jobs


def main() -> int:
    """Write the vocode job list, refusing to run against an unvalidated manifest."""
    args = parse_args()

    report = S.new_gen_validate_report_path(args.build_id)
    try:
        verdict = json.loads(report.read_text()).get("verdict")
    except (OSError, json.JSONDecodeError):
        verdict = None
    if verdict != "PASS":
        print(f"[vocode-jobs] refusing: {args.build_id} has no PASSing validate_report", file=sys.stderr)
        return 1

    pair_ids = None
    if args.pair_ids_file is not None:
        pair_ids = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
    rows = vocode_rows(args.build_id, pair_ids)
    if not rows:
        print(f"[vocode-jobs] {args.build_id} has no needs_vocode rows — nothing to do")
        return 0

    jobs = build_jobs(rows)

    missing = [j for j in jobs if not Path(j["source_wav"]).exists()]
    if missing and not args.allow_missing:
        print(
            f"[vocode-jobs] refusing: {len(missing)}/{len(jobs)} jobs have no crop audio "
            f"(first: {missing[0]['source_wav']}) — run the crop stage first, or pass --allow-missing",
            file=sys.stderr,
        )
        return 1

    out = args.out or (S.new_gen_manifest_dir(args.build_id) / "vocode_jobs.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    S.new_gen_vocoded_dir().mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for job in jobs:
            handle.write(json.dumps(job) + "\n")

    print(f"[vocode-jobs] {len(jobs)} jobs ({len(rows)} rows) -> {out}")
    if missing:
        print(f"[vocode-jobs] WARNING: {len(missing)} jobs have no crop audio (--allow-missing)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
