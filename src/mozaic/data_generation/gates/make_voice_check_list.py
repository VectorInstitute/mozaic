"""Build the voice-gate input (sv_pairs.jsonl) of a four-class build, from the clips that shipped.

Every path points at the build's final folder, not at the intermediates: the shipped clip
went through the mux, the loudness step and the audio encode, any of which can move a
speaker embedding. Scoring conversions/*.wav would grade a file nobody receives.

voice_check scores cosine(converted, target) and cosine(converted, source). A cosine has
no absolute scale (0.45 is excellent for one speaker pair and poor for another), so every
conversion is written twice:

    <key>              the converted clip (RVFA, fake voice)
    <key>__ctrl_real   a real clip of the same pair (RVRA), against the same two recordings:
                       the pair's last RVRA row in the manifest, i.e. its vocoded twin when
                       the pair has one; keep_or_drop reads its source_cosine as "vocoded real
                       voice vs the driver's natural recording"

The target recording is read from each conversion's own record (target_ref_clip_id),
not re-derived from the manifest: the gate scores against the recording Seed-VC was given.
RVFA and FVFA share one conversion, so rows are deduplicated by conversion_key.

    python -m mozaic.data_generation.gates.make_voice_check_list --build-id <build> --final-dir <final> \\
        --pair-ids-file <pairs.txt> --out-dir <gates>
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
    p = argparse.ArgumentParser(description="Build the voice-gate pair file from a four-class build.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <new_generated>/_reports/<build-id>_gates")
    p.add_argument("--limit", type=int, default=None, help="first N conversions (to try the gate itself)")
    p.add_argument("--final-dir", type=Path, default=None, help="default: <new_generated>/final")
    p.add_argument("--pair-ids-file", type=Path, default=None, help="restrict to these pair_ids (one batch of a build)")
    return p.parse_args()


def load_rows(build_id: str) -> list[dict[str, Any]]:
    """Every manifest row for the build."""
    return [json.loads(line) for line in S.new_gen_manifest_path(build_id).read_text().splitlines() if line.strip()]


def shipped(class_name: str, sample_id: str, root: Path | None = None) -> Path:
    """Path of a shipped clip, in the container the mux writes (S.final_sample_path)."""
    return S.final_sample_path(class_name, sample_id, root)


def build_sv_pairs(rows: list[dict[str, Any]], limit: int | None, root: Path | None = None) -> list[dict[str, Any]]:
    """One converted clip and one real control per conversion."""
    by_pair_class: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        by_pair_class[(r["pair_id"], r["class_name"])] = r

    seen: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: x["sample_id"]):
        key = r.get("conversion_key")
        if not key or r["class_name"] != "RVFA" or key in seen:
            continue
        seen.add(key)

        sidecar = S.new_gen_conversion_dir() / f"{key}.json"
        if not sidecar.exists():
            print(f"[gates] no conversion sidecar for {key} — skipping", file=sys.stderr)
            continue
        prov = json.loads(sidecar.read_text())
        target_ref = S.VOXCELEB2_MP4_ROOT / f"{prov['target_ref_clip_id']}.mp4"
        source_wav = S.crop_audio_path(r["crop_key"])
        converted = shipped("RVFA", r["sample_id"], root)
        control = by_pair_class.get((r["pair_id"], "RVRA"))

        missing = [str(p) for p in (target_ref, source_wav, converted) if not p.exists()]
        if missing:
            print(f"[gates] {key}: missing {missing} — skipping", file=sys.stderr)
            continue

        common = {
            "target_ref_wav": str(target_ref),
            "source_wav": str(source_wav),
            "target_identity_id": prov["target_identity_id"],
            "source_identity_id": r["driver_identity_id"],
            "same_speaker_arm": bool(r.get("same_speaker_arm")),
        }
        pairs.append({"job_id": key, "arm": "converted", "converted_wav": str(converted), **common})

        # The control: a real clip of the same pair, scored against the same two
        # recordings, so target_cosine can be read as a distance travelled.
        if control is not None:
            ctrl_path = shipped("RVRA", control["sample_id"], root)
            if ctrl_path.exists():
                pairs.append(
                    {"job_id": f"{key}__ctrl_real", "arm": "real_control", "converted_wav": str(ctrl_path), **common}
                )

        if limit is not None and len(seen) >= limit:
            break
    return pairs


def main() -> int:
    """Write the voice-gate pair file and print what was built."""
    args = parse_args()
    rows = load_rows(args.build_id)
    if args.pair_ids_file:
        keep = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
        rows = [r for r in rows if r["pair_id"] in keep]
        print(f"[gates] restricted to {len(keep)} pair_ids from {args.pair_ids_file.name}: {len(rows)} rows")
        if not rows:
            print("[gates] REFUSING: no manifest rows match --pair-ids-file", file=sys.stderr)
            return 1
    out_dir = args.out_dir or (S.new_gen_report_dir() / f"{args.build_id}_gates")
    out_dir.mkdir(parents=True, exist_ok=True)

    root = S.resolve_final_dir(args.final_dir)
    print(f"[gates] scoring {root}")
    sv_pairs = build_sv_pairs(rows, args.limit, root)
    sv_path = out_dir / "sv_pairs.jsonl"
    with sv_path.open("w") as f:
        for row in sv_pairs:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    n_conv = sum(1 for p in sv_pairs if p["arm"] == "converted")
    n_ctrl = sum(1 for p in sv_pairs if p["arm"] == "real_control")
    n_same = sum(1 for p in sv_pairs if p["arm"] == "converted" and p["same_speaker_arm"])
    print(f"[gates] SV pairs -> {sv_path}")
    print(f"[gates]   {n_conv} converted, {n_ctrl} real controls, of which {n_same} are same-speaker-arm")

    # build_sv_pairs skips a clip that is not on disk, so a build pointed at the wrong
    # folder or container writes an empty file; the voice gate cannot run on nothing.
    if not sv_pairs:
        print(
            f"\n[gates] REFUSING: {len(sv_pairs)} SV pairs resolved under {root}. Check --final-dir and that the "
            f"build shipped {S.TERMINAL_CONTAINER_EXT} files.",
            file=sys.stderr,
        )
        return 1

    provenance = out_dir / "gate_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "build_id": args.build_id,
                "final_dir": str(root),
                "container_ext": S.TERMINAL_CONTAINER_EXT,
                "n_sv_pairs": len(sv_pairs),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[gates] provenance -> {provenance}")

    print("\n[gates] run the voice gate in the speechbrain environment:")
    print("  <voice env>/bin/python -m mozaic.data_generation.gates.voice_check \\")
    print(f"      --pairs {sv_path} --out {out_dir / 'sv_report.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
