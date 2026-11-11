"""List the voice clips a build's voice conversions need as wav files.

Voice audio is extracted on demand, not for all of VoxCeleb2: only the voice_ref_clip_id of
every row that needs a conversion (the clip Seed-VC opens). One clip_id per line, sorted,
the ids of inventory.parquet. Clips already extracted are listed too (the worker skips
them), unless --missing-only.

    python -m mozaic.data_generation.inputs.make_audio_extract_list --build-id bulk_v1_wan77
    python -m mozaic.data_generation.inputs.extract_voices \
        --clip-ids-file <data>/new_generated/manifests/bulk_v1_wan77/voice_ref_clips.txt

Reads  new_generated/manifests/<build-id>/manifest.jsonl, voxceleb2/audio/ (to count what is there)
Writes new_generated/manifests/<build-id>/voice_ref_clips.txt (or --out)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mozaic.data_generation import settings as S


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="List the voice clips a build needs extracted.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--out", type=Path, default=None, help="default: beside the manifest")
    p.add_argument("--missing-only", action="store_true", help="leave out clips whose wav already exists")
    p.add_argument(
        "--pair-ids-file",
        type=Path,
        default=None,
        help="only these pair_ids (one per line), for a run on a batch of pairs (default: the whole build)",
    )
    return p.parse_args()


def voice_refs(build_id: str, pair_ids: set[str] | None = None) -> list[str]:
    """The distinct voice clips this build converts with, sorted."""
    refs: set[str] = set()
    with S.new_gen_manifest_path(build_id).open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("needs_conversion"):
                continue
            if pair_ids is not None and row.get("pair_id") not in pair_ids:
                continue
            # Seed-VC opens voice_ref_clip_id; voice_ref_clip_ids is the longer list on record.
            ref = row.get("voice_ref_clip_id")
            if ref:
                refs.add(str(ref))
    return sorted(refs)


def main() -> int:
    """Write the clip-id list and say how much of it is already extracted."""
    args = parse_args()

    manifest = S.new_gen_manifest_path(args.build_id)
    if not manifest.exists():
        print(f"[audio-list] no manifest for {args.build_id} at {manifest}", file=sys.stderr)
        return 1

    pair_ids = None
    if args.pair_ids_file:
        pair_ids = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
    refs = voice_refs(args.build_id, pair_ids)
    if not refs:
        print(f"[audio-list] {args.build_id} has no conversion rows — nothing to extract")
        return 0

    present = [r for r in refs if S.audio_wav_path(f"{r}.mp4").exists()]
    missing = [r for r in refs if r not in set(present)]
    wanted = missing if args.missing_only else refs

    out = args.out or (S.new_gen_manifest_dir(args.build_id) / "voice_ref_clips.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(wanted) + ("\n" if wanted else ""))

    print(
        f"[audio-list] {args.build_id}: {len(refs)} voice refs, "
        f"{len(present)} extracted, {len(missing)} missing -> {out} ({len(wanted)} listed)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
