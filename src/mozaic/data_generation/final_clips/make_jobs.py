"""Write the mux's job list: which video and which audio make each final clip of a build.

The manifest row decides (video_status and audio_variant), never a file name:

    class   video                      audio
    RVRA    crops/<crop_key>.mp4       crops/<crop_key>.wav
    RVFA    crops/<crop_key>.mp4       conversions/<conversion_key>.wav
    FVRA    renders/<render_key>.mp4   crops/<crop_key>.wav
    FVFA    renders/<render_key>.mp4   conversions/<conversion_key>.wav

The control arms follow the same two rules: the vocoded arm takes the crop video with
vocoded/<crop_key>.wav (labelled real), the self-drive arm its render with the crop's own audio.
A pair that lost a render or a voice upstream ships in no class; its rows go to
<jobs>_skipped.jsonl. Every input is checked to exist before the mux runs.

    python -m mozaic.data_generation.final_clips.make_jobs --build-id <build_id>
    python -m mozaic.data_generation.final_clips.make_jobs --build-id <build_id> --out-root <final_dir> --out jobs.jsonl
    python -m mozaic.data_generation.final_clips.mux --jobs jobs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S

# A file written on another machine can take a while to show up on a shared file system.
INPUT_VISIBILITY_RETRIES = 6
INPUT_VISIBILITY_WAIT_S = 20


def video_source(row: dict[str, Any]) -> Path:
    """The video this row ships: its render if generated, else the crop."""
    if row["video_status"] == "generated":
        return S.render_path(row["render_key"])
    return S.crop_video_path(row["crop_key"])


def audio_source(row: dict[str, Any]) -> Path:
    """The audio this row ships, by audio_variant (not class_name)."""
    variant = row["audio_variant"]
    if variant == S.AUDIO_VARIANT_CONVERTED:
        return S.new_gen_conversion_dir() / f"{row['conversion_key']}.wav"
    if variant == S.AUDIO_VARIANT_VOCODED:
        return S.new_gen_vocoded_dir() / f"{row['crop_key']}.wav"
    return S.crop_audio_path(row["crop_key"])


def build_job(row: dict[str, Any], out_root: Path | None = None) -> dict[str, Any]:
    """One mux job for one final clip."""
    return {
        "sample_id": row["sample_id"],
        "class_name": row["class_name"],
        "label": row["label"],
        "split": row["split"],
        "test_subset": row.get("test_subset", S.TEST_SUBSET_BOTH),
        "arm": row["arm"],
        "pair_id": row["pair_id"],
        "video": str(video_source(row)),
        "audio": str(audio_source(row)),
        "out": str(S.final_sample_path(row["class_name"], row["sample_id"], out_root)),
        "target_frames": S.row_crop_frames(row),  # the length the mux checks on both streams
        "generator_version": row.get("generator_version", S.GENERATOR_VERSION),  # written into the clip record
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the mux's job list from a 4-class manifest.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--out", type=Path, default=None, help="jobs JSONL (default: beside the manifest)")
    p.add_argument("--pair-ids-file", type=Path, default=None, help="newline-separated pair_ids to restrict to")
    p.add_argument("--classes", nargs="*", default=None, help="restrict to these class names")
    p.add_argument("--arms", nargs="*", default=None, help="restrict to these arms (e.g. main)")
    p.add_argument("--allow-missing", action="store_true", help="write jobs even when inputs are missing (not for real runs)")
    p.add_argument("--out-root", type=Path, default=None,
                   help="write the final clips under this folder instead of new_generated/final/ "
                   "(sample ids carry no build id, so each build needs its own folder)")
    return p.parse_args()


def passed_validation(build_id: str) -> bool:
    """True when the build's validate_report.json says PASS."""
    try:
        verdict = json.loads(S.new_gen_validate_report_path(build_id).read_text()).get("verdict")
    except (OSError, json.JSONDecodeError):
        verdict = None
    return verdict == "PASS"


def read_rows(
    build_id: str, pair_ids: set[str] | None, classes: list[str] | None, arms: list[str] | None
) -> list[dict[str, Any]]:
    """The manifest rows of the chosen pairs, classes and arms."""
    rows: list[dict[str, Any]] = []
    with S.new_gen_manifest_path(build_id).open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if pair_ids is not None and row["pair_id"] not in pair_ids:
                continue
            if classes and row["class_name"] not in classes:
                continue
            if arms and row["arm"] not in arms:
                continue
            rows.append(row)
    return rows


def lost_input_record(row: dict[str, Any]) -> Path | None:
    """The .refused.json or .failed.json of this row's render or voice, if there is one."""
    if row.get("render_key"):
        for cand in (S.render_refusal_path(row["render_key"]), S.render_failure_path(row["render_key"])):
            if cand.exists():
                return cand
    if row.get("conversion_key"):
        cand = S.conversion_failure_path(row["conversion_key"])
        if cand.exists():
            return cand
    return None


def drop_lost_pairs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into (kept, skipped): a pair that lost a render or a voice ships in no class.

    A pair left with only its real-video classes would let the driver's identity give the label away.
    """
    dropped_pairs = S.refused_pair_ids(rows)
    skipped: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for r in rows:
        sidecar = lost_input_record(r)
        if sidecar is not None or r.get("pair_id") in dropped_pairs:
            reason = (
                "pair_incomplete"
                if sidecar is None
                else ("refused" if sidecar.name.endswith(".refused.json") else "failed")
            )
            skipped.append({"sample_id": r["sample_id"], "pair_id": r["pair_id"], "class_name": r["class_name"],
                            "reason": reason, "sidecar": str(sidecar) if sidecar else ""})  # fmt: skip
        else:
            kept.append(r)
    return kept, skipped


def missing_inputs(jobs: list[dict[str, Any]]) -> list[str]:
    return [
        f"{j['sample_id']}: missing {kind} -> {j[kind]}"
        for j in jobs
        for kind in ("video", "audio")
        if not Path(j[kind]).exists()
    ]


def wait_for_inputs(jobs: list[dict[str, Any]]) -> list[str]:
    """The inputs still missing after a bounded wait."""
    missing = missing_inputs(jobs)
    for _ in range(INPUT_VISIBILITY_RETRIES):
        if not missing:
            break
        print(
            f"[terminal-jobs] {len(missing)} input(s) not visible yet; waiting {INPUT_VISIBILITY_WAIT_S}s", flush=True
        )
        time.sleep(INPUT_VISIBILITY_WAIT_S)
        missing = missing_inputs(jobs)
    return missing


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    """Resolve every final clip to its two input streams and write the job list."""
    args = parse_args()
    if not passed_validation(args.build_id):
        print(f"[terminal-jobs] refusing: {args.build_id} has no PASSing validate_report", file=sys.stderr)
        return 1

    pair_ids = None
    if args.pair_ids_file:
        pair_ids = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
    rows = read_rows(args.build_id, pair_ids, args.classes, args.arms)
    if not rows:
        print("[terminal-jobs] no rows selected", file=sys.stderr)
        return 1

    rows, skipped = drop_lost_pairs(rows)
    if skipped:
        pairs_skipped = sorted({s["pair_id"] for s in skipped})
        print(
            f"[terminal-jobs] {len(skipped)} sample(s) of {len(pairs_skipped)} refused/failed pair(s) skipped: {pairs_skipped[:5]}"
        )

    jobs = [build_job(r, args.out_root) for r in rows]
    missing = wait_for_inputs(jobs)
    if missing:
        print(f"[terminal-jobs] {len(missing)} missing input(s):", file=sys.stderr)
        for m in missing[:15]:
            print(f"  - {m}", file=sys.stderr)
        if not args.allow_missing:
            print(
                "[terminal-jobs] renders come from the video generators, conversions from Seed-VC, "
                "vocoded audio from the vocoder step.",
                file=sys.stderr,
            )
            return 1

    out_path = args.out or (S.new_gen_manifest_dir(args.build_id) / "terminal_jobs.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, jobs)
    write_jsonl(out_path.with_name(out_path.stem + "_skipped.jsonl"), skipped)

    by_class: dict[str, int] = {}
    for j in jobs:
        by_class[j["class_name"]] = by_class.get(j["class_name"], 0) + 1
    print(f"[terminal-jobs] wrote {len(jobs)} jobs -> {out_path}")
    print(f"[terminal-jobs] by class: {dict(sorted(by_class.items()))}")
    print(f"[terminal-jobs] container: {S.TERMINAL_CONTAINER_EXT}, audio: {S.TERMINAL_AUDIO_CODEC}")
    if args.out_root:
        print(f"[terminal-jobs] out-root: {args.out_root} (existing final/ is left untouched)")
    print(f"[terminal-jobs] run: python -m mozaic.data_generation.final_clips.mux --jobs {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
