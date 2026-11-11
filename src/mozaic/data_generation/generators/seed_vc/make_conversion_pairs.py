"""Seed-VC pairs of a build: one line per converted voice, from the build's pair list (manifest.jsonl).

RVFA and FVFA of a pair use the same converted voice, so the lines are grouped by conversion_key, never
by sample_id: one conversion serves up to two shipped clips, as one render serves FVRA and FVFA.

Two different sound files go in:
- source: the CROPPED driver sound (crops/<crop_key>.wav, exactly crop_frames x 640 samples), so the
  voice lines up with the video crop. Converting the whole clip and cutting after would cut after the
  generator instead of before it.
- reference: ONE whole clip of the other person (voxceleb2/audio/), the first of the row's
  voice_ref_clip_ids. Seed-VC takes one short reference; the other clips stay listed on the line.

target_ref_duration_s is the length of that one reference, what Seed-VC hears. target_ref_total_s is the
length of the whole listed pool, which nothing here reads.

    python -m mozaic.data_generation.generators.seed_vc.make_conversion_pairs --build-id <build>
    python -m mozaic.data_generation.generators.seed_vc.make_jobs \
        --pairs <manifests>/<build>/conversion_pairs.jsonl --out-dir <manifests>/<build>/seedvc_jobs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S

CONVERSION_CLASSES = {"RVFA", "FVFA"}


def conversion_seed(conversion_key: str) -> int:
    """The seed of one conversion: different for every voice, the same on every rerun.

    Seed-VC draws its starting noise with an unseeded torch.randn (modules/flow_matching.py), so without it
    a rerun gives a different voice. Made the same way as the render seeds.
    """
    digest = hashlib.blake2b(f"{S.SPLIT_SEED}:{conversion_key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def load_conversion_rows(build_id: str, pair_ids: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The pair-list rows that need a converted voice, grouped by conversion_key (only pair_ids' when given)."""
    manifest = S.new_gen_manifest_path(build_id)
    if not manifest.exists():
        msg = f"no manifest for build {build_id!r} at {manifest}"
        raise SystemExit(msg)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with manifest.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if pair_ids is not None and row.get("pair_id") not in pair_ids:
                continue
            if row.get("needs_conversion"):
                groups[row["conversion_key"]].append(row)
    return groups


def validate_groups(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Check that every conversion_key names exactly ONE conversion.

    If two rows sharing a key disagree on what to convert, one of them ships a voice made for the other, and
    nothing later would notice.
    """
    problems: list[str] = []
    for key, rows in sorted(groups.items()):
        for field in ("crop_key", "voice_identity_id", "driver_identity_id", "split", "same_speaker_arm"):
            values = {json.dumps(r.get(field), sort_keys=True) for r in rows}
            if len(values) > 1:
                problems.append(f"{key}: rows disagree on {field}: {sorted(values)}")
        head = rows[0]
        classes = {r["class_name"] for r in rows}
        if not classes <= CONVERSION_CLASSES:
            problems.append(f"{key}: consumed by {sorted(classes)}, expected a subset of {sorted(CONVERSION_CLASSES)}")
        refs = head.get("voice_ref_clip_ids") or []
        if not refs:
            problems.append(f"{key}: no voice_ref_clip_ids")
        # The voice must come from the voice identity and, outside the same-speaker arm, must not be the
        # driver's own: otherwise the row is not the mismatch its class name says.
        for ref in refs:
            if ref.split("/")[0] != head["voice_identity_id"]:
                problems.append(f"{key}: reference {ref} is not from voice identity {head['voice_identity_id']}")
                break
        if not head["same_speaker_arm"] and head["voice_identity_id"] == head["driver_identity_id"]:
            problems.append(f"{key}: cross-identity conversion but voice == driver ({head['voice_identity_id']})")
        if head["same_speaker_arm"] and head["voice_identity_id"] != head["driver_identity_id"]:
            problems.append(f"{key}: same-speaker arm but voice != driver")
    return problems


def ref_durations() -> dict[str, float]:
    """clip_id -> duration_s from the clip list, to record how long each reference is."""
    import pandas as pd  # only needed when building pairs

    inv = pd.read_parquet(S.inventory_parquet_path(), columns=["clip_id", "duration_s"])
    return dict(zip(inv["clip_id"], inv["duration_s"], strict=True))


def build_pair(conversion_key: str, rows: list[dict[str, Any]], durations: dict[str, float]) -> dict[str, Any]:
    """One Seed-VC pair, with the sample_ids that will use its voice."""
    head = rows[0]
    refs = head["voice_ref_clip_ids"]
    return {
        "job_id": conversion_key,
        "conversion_key": conversion_key,
        # Explicit wav paths: without them make_jobs resolves both sides from the whole-clip tree, which
        # would give Seed-VC a whole clip as the source instead of the crop the video is cut to.
        "source_wav": str(S.crop_audio_path(head["crop_key"])),
        "target_ref_wav": str(S.audio_wav_path(f"{refs[0]}.mp4")),
        "source_clip_id": head["driver_clip_id"],
        "crop_key": head["crop_key"],
        "target_identity_id": head["voice_identity_id"],
        "target_ref_clip_id": refs[0],
        "target_ref_clip_ids": refs,
        # What Seed-VC hears: the one reference clip, not the pool that target_ref_total_s describes.
        "target_ref_duration_s": round(float(durations.get(refs[0], 0.0)), 3),
        "target_ref_total_s": head.get("voice_ref_total_s"),
        "split": head["split"],
        "same_speaker_arm": head["same_speaker_arm"],
        "base_seed": conversion_seed(conversion_key),
        "length_adjust": S.SEEDVC_LENGTH_ADJUST,
        "diffusion_steps": S.SEEDVC_DIFFUSION_STEPS,
        "inference_cfg_rate": S.SEEDVC_INFERENCE_CFG_RATE,
        "consumer_sample_ids": sorted(r["sample_id"] for r in rows),
        "consumer_classes": sorted({r["class_name"] for r in rows}),
        "generator_version": S.GENERATOR_VERSION,
    }


def check_inputs_exist(pairs: list[dict[str, Any]]) -> list[str]:
    """Every wav a pair names, checked before any GPU time is spent."""
    missing: list[str] = []
    for pair in pairs:
        for key in ("source_wav", "target_ref_wav"):
            if not Path(pair[key]).exists():
                missing.append(f"{pair['job_id']}: missing {key} -> {pair[key]}")
    return missing


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Build a Seed-VC pairs file from a 4-class manifest.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--out", type=Path, default=None, help="pairs.jsonl path (default: beside the manifest)")
    p.add_argument("--allow-missing", action="store_true", help="warn instead of failing on unresolvable wavs")
    p.add_argument("--pair-ids-file", type=Path, default=None, help="newline-separated pair_ids to restrict to")
    p.add_argument(
        "--emit-missing-refs",
        type=Path,
        default=None,
        help="write the clip_ids whose reference wav is missing, for inputs.extract_voices --clip-ids-file",
    )
    return p.parse_args()


def main() -> int:
    """Group the manifest's conversion rows and write pairs.jsonl."""
    args = parse_args()

    report = S.new_gen_validate_report_path(args.build_id)
    try:
        verdict = json.loads(report.read_text()).get("verdict")
    except (OSError, json.JSONDecodeError):
        verdict = None
    if verdict != "PASS":
        print(f"[pairs] refusing: {args.build_id} has no PASSing validate_report ({report})", file=sys.stderr)
        return 1

    pair_ids = None
    if args.pair_ids_file:
        pair_ids = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
    groups = load_conversion_rows(args.build_id, pair_ids)
    if not groups:
        print(f"[pairs] no rows need conversion in {args.build_id}", file=sys.stderr)
        return 1

    problems = validate_groups(groups)
    if problems:
        print(f"[pairs] REFUSING — {len(problems)} grouping problem(s):", file=sys.stderr)
        for p in problems[:20]:
            print(f"  - {p}", file=sys.stderr)
        return 1

    durations = ref_durations()
    pairs = [build_pair(k, rows, durations) for k, rows in sorted(groups.items())]

    # Only the first reference of each pair is ever given to Seed-VC, so only those need extracting.
    if args.emit_missing_refs:
        needed = sorted({p["target_ref_clip_id"] for p in pairs if not Path(p["target_ref_wav"]).exists()})
        args.emit_missing_refs.parent.mkdir(parents=True, exist_ok=True)
        args.emit_missing_refs.write_text("\n".join(needed) + ("\n" if needed else ""))
        print(f"[pairs] {len(needed)} reference clip(s) need extraction -> {args.emit_missing_refs}")
        print(
            "[pairs] extract with: python -m mozaic.data_generation.inputs.extract_voices "
            f"--clip-ids-file {args.emit_missing_refs}"
        )

    missing = check_inputs_exist(pairs)
    if missing:
        head = "\n".join(f"  - {m}" for m in missing[:20])
        if not args.allow_missing:
            print(f"[pairs] REFUSING — {len(missing)} unresolvable wav(s):\n{head}", file=sys.stderr)
            print(
                "[pairs] crops come from crop/crop_clips.py; references from inputs/extract_voices.py.",
                file=sys.stderr,
            )
            return 1
        print(f"[pairs] WARNING — {len(missing)} unresolvable wav(s):\n{head}", file=sys.stderr)

    out_path = args.out or (S.new_gen_manifest_dir(args.build_id) / "conversion_pairs.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")

    consumers = sum(len(p["consumer_sample_ids"]) for p in pairs)
    same_speaker = sum(1 for p in pairs if p["same_speaker_arm"])
    print(f"[pairs] wrote {len(pairs)} conversion pairs -> {out_path}")
    print(f"[pairs] they serve {consumers} shipped samples ({same_speaker} same-speaker arm)")
    print(f"[pairs] settings: steps={S.SEEDVC_DIFFUSION_STEPS} cfg={S.SEEDVC_INFERENCE_CFG_RATE}")
    print(
        "[pairs] next: python -m mozaic.data_generation.generators.seed_vc.make_jobs "
        f"--pairs {out_path} --out-dir {S.new_gen_manifest_dir(args.build_id) / 'seedvc_jobs'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
