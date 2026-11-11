"""Check a build's pair list before any GPU time is spent: it must have zero violations.

Split checks (a leak here would spoil every number measured on the dataset):
  F1  no person is in two splits
  F2  every person a row borrows from is in the row's own split (split_map.json)
  F3  each pair_id is in one split
  F4  every row made from one driver clip is in that clip's split
  F5  gender_match is true on every row (same-gender pairs only)
  F6  no person has an unknown gender

Construction checks (the pair list says what the generators will make):
  C1  sample_id is unique
  C2  every pair has exactly the four main classes
  C3  coherence and label are right for each class (RVRA/FVFA coherent, RVFA/FVRA not)
  C4  no main row borrows from its own person
  C5  FVRA+FVFA share one render_key and RVFA+FVFA one conversion_key
  C6  every row has the same crop length and duration_s (length cannot tell the label)
  C7  the face folder is never in the voice clips
  C8  the crop fits inside every driver clip (needs the clip list)
  C9  the main rows are balanced over the classes, overall and in every split
  C10 every (split, gender) cell is whole rings (no driver clip skipped)

    python -m mozaic.data_generation.inputs.check_pairs --build-id bulk_v1_wan77

Reads  new_generated/manifests/<build-id>/manifest.jsonl, voxceleb2/split_map.json,
       voxceleb2/_reports/inventory.parquet (C8)
Writes new_generated/manifests/<build-id>/validate_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from mozaic.data_generation import settings as S

MAX_EXAMPLES = 5  # examples kept per violation; the count is what matters
FLOAT_TOL = 1e-9  # float slack for duration comparisons


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Check a build's pair list.")
    p.add_argument("--build-id", required=True)
    p.add_argument(
        "--manifest", type=Path, default=None, help="default: new_generated/manifests/<build-id>/manifest.jsonl"
    )
    p.add_argument("--out", type=Path, default=None, help="default: beside the manifest")
    p.add_argument("--skip-duration-check", action="store_true", help="skip C8 (does not read the clip list)")
    return p.parse_args()


class Violations:
    """Failures as (check, message), so the report lists every one."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, check: str, message: str, examples: list[str] | None = None, count: int | None = None) -> None:
        """Record one failure with a few of its rows."""
        self.items.append(
            {
                "check": check,
                "message": message,
                "count": count if count is not None else (len(examples) if examples else 1),
                "examples": (examples or [])[:MAX_EXAMPLES],
            }
        )

    def __len__(self) -> int:
        """Number of violations recorded."""
        return len(self.items)


def check_firewall(rows: list[dict[str, Any]], split_map: dict[str, str], v: Violations) -> None:
    """F1-F6: the split checks."""
    # F1: no person in two splits, over the people the pair list really uses.
    identities_by_split: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        for key in ("driver_identity_id", "face_identity_id", "voice_identity_id", "pair_source_identity_id"):
            identities_by_split[r["split"]].add(r[key])
    splits = sorted(identities_by_split)
    for a_idx, a in enumerate(splits):
        for b in splits[a_idx + 1 :]:
            overlap = sorted(identities_by_split[a] & identities_by_split[b])
            if overlap:
                v.add("F1", f"identities appear in both {a} and {b}", overlap, len(overlap))

    # F2: every borrowed person is in the row's own split, per the split map.
    wrong_split: list[str] = []
    for r in rows:
        for key in ("driver_identity_id", "face_identity_id", "voice_identity_id", "pair_source_identity_id"):
            if split_map.get(r[key]) != r["split"]:
                wrong_split.append(
                    f"{r['sample_id']}: {key}={r[key]} is in {split_map.get(r[key])}, row says {r['split']}"
                )
    if wrong_split:
        v.add("F2", "borrowed identity outside the row's split", wrong_split, len(wrong_split))

    # F3: a pair_id is in one split.
    pair_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        pair_splits[r["pair_id"]].add(r["split"])
    straddling = sorted(p for p, s in pair_splits.items() if len(s) > 1)
    if straddling:
        v.add("F3", "pair_id spans more than one split", straddling, len(straddling))

    # F4: every row of a driver clip is in that clip's split.
    clip_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        clip_splits[r["driver_clip_id"]].add(r["split"])
    leaked = sorted(c for c, s in clip_splits.items() if len(s) > 1)
    if leaked:
        v.add("F4", "driver clip's derivatives span more than one split", leaked, len(leaked))

    # F5: same-gender pairs everywhere.
    mismatched = [
        f"{r['sample_id']}: {r['gender_driver']}->{r['gender_source']}" for r in rows if not r["gender_match"]
    ]
    if mismatched:
        v.add("F5", "cross-gender pairing (design requires same-gender only)", mismatched, len(mismatched))

    # F6: every person's gender is known.
    unknown = sorted({r[k] for r in rows for k in ("gender_driver", "gender_source") if r[k] not in S.RING_GENDERS})
    if unknown:
        v.add("F6", "unknown/invalid gender value present", unknown, len(unknown))


def ring_geometry_gaps(rows: list[dict[str, Any]]) -> list[str]:
    """Every (split, gender) cell that is not whole rings of full people."""
    per_ring = S.FOLDERS_PER_CELEB * S.CLIPS_PER_FOLDER
    pairs_per_ring: dict[tuple[str, str], set[str]] = defaultdict(set)
    identities_per_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        pairs_per_ring[(r["split"], r["ring_gender"])].add(r["pair_id"])
        identities_per_cell[(r["split"], r["ring_gender"])].add(r["driver_identity_id"])
    short_rings = [
        f"{split}/{g}: {len(pairs)} pairs, expected {len(identities_per_cell[(split, g)]) * per_ring} "
        f"({len(identities_per_cell[(split, g)])} identities x {per_ring})"
        for (split, g), pairs in sorted(pairs_per_ring.items())
        if len(pairs) != len(identities_per_cell[(split, g)]) * per_ring
    ]
    # A person missing from a cell altogether would not show above: a cell is whole rings,
    # so its people are a multiple of RING_SIZE.
    short_rings += [
        f"{split}/{g}: {len(ids)} identities, not a whole number of rings of {S.RING_SIZE}"
        for (split, g), ids in sorted(identities_per_cell.items())
        if len(ids) % S.RING_SIZE
    ]
    return short_rings


def check_construction(rows: list[dict[str, Any]], v: Violations) -> None:
    """C1-C7, C9, C10: the pair list describes what it claims to."""
    # C1: unique sample_id.
    dupes = sorted(s for s, n in Counter(r["sample_id"] for r in rows).items() if n > 1)
    if dupes:
        v.add("C1", "duplicate sample_id", dupes, len(dupes))

    main_rows = [r for r in rows if r["arm"] == "main"]

    # C2: every pair has exactly the four main classes.
    by_pair: dict[str, list[str]] = defaultdict(list)
    for r in main_rows:
        by_pair[r["pair_id"]].append(r["class_name"])
    bad_pairs = [f"{p}: {sorted(c)}" for p, c in by_pair.items() if sorted(c) != sorted(S.CLASS_NAMES)]
    if bad_pairs:
        v.add("C2", "pair does not have exactly the 4 main classes", bad_pairs, len(bad_pairs))

    # C3: coherence, and the label that goes with the class.
    expected_coherent = {"RVRA": True, "RVFA": False, "FVRA": False, "FVFA": True}
    expected_label = {"RVRA": S.LABEL_REAL, "RVFA": S.LABEL_FAKE, "FVRA": S.LABEL_FAKE, "FVFA": S.LABEL_FAKE}
    bad_coh: list[str] = []
    bad_label: list[str] = []
    for r in main_rows:
        if r["identity_coherent"] != expected_coherent[r["class_name"]]:
            bad_coh.append(f"{r['sample_id']}: coherent={r['identity_coherent']}")
        if (r["face_identity_id"] == r["voice_identity_id"]) != r["identity_coherent"]:
            bad_coh.append(f"{r['sample_id']}: identity_coherent contradicts face/voice ids")
        if r["label"] != expected_label[r["class_name"]]:
            bad_label.append(f"{r['sample_id']}: label={r['label']}")
    if bad_coh:
        v.add("C3", "coherence flag wrong for class", bad_coh, len(bad_coh))
    if bad_label:
        v.add("C3", "label wrong for class", bad_label, len(bad_label))

    # C4: main rows borrow from ANOTHER person.
    self_borrow = [
        f"{r['sample_id']}: driver={r['driver_identity_id']}"
        for r in main_rows
        if r["pair_source_identity_id"] == r["driver_identity_id"]
    ]
    if self_borrow:
        v.add("C4", "main row borrows from its own identity", self_borrow, len(self_borrow))

    # C5: the render and the voice conversion are really shared.
    render_by_pair: dict[str, set[str]] = defaultdict(set)
    conv_by_pair: dict[str, set[str]] = defaultdict(set)
    for r in main_rows:
        if r["needs_render"]:
            render_by_pair[r["pair_id"]].add(r["render_key"])
        if r["needs_conversion"]:
            conv_by_pair[r["pair_id"]].add(r["conversion_key"])
    bad_share = [f"{p}: {sorted(k)}" for p, k in render_by_pair.items() if len(k) != 1]
    bad_share += [f"{p}: {sorted(k)}" for p, k in conv_by_pair.items() if len(k) != 1]
    if bad_share:
        v.add("C5", "FVRA/FVFA (or RVFA/FVFA) do not share one artefact", bad_share, len(bad_share))
    missing_key = [
        r["sample_id"]
        for r in rows
        if (r["needs_render"] and not r["render_key"]) or (r["needs_conversion"] and not r["conversion_key"])
    ]
    if missing_key:
        v.add("C5", "row needs an artefact but names no key", missing_key, len(missing_key))

    # C6: one length for every row, a whole number of frames. Compared with a tolerance,
    # never by multiplying back: 4.48 * 25 is 112.00000000000001 in binary floats.
    durations = sorted({r["duration_s"] for r in rows})
    frames_set = sorted({S.row_crop_frames(r) for r in rows})
    if len(frames_set) != 1:
        v.add("C6", f"crop_frames is not constant: {frames_set}", [str(f) for f in frames_set], len(frames_set))
    if len(durations) != 1:
        v.add("C6", f"duration_s is not constant: {durations}", [str(d) for d in durations], len(durations))
    elif abs(durations[0] - frames_set[0] / S.TARGET_FPS) > FLOAT_TOL:
        v.add("C6", f"duration_s {durations[0]} != crop_frames/TARGET_FPS ({frames_set[0]}/{S.TARGET_FPS})")
    # A non-default length shows in crop_key and the default does not, so two lengths of one
    # clip never share a file name.
    bad_keys = [
        r["crop_key"]
        for r in rows
        if r["crop_key"].endswith(f"__f{S.row_crop_frames(r)}") != (S.row_crop_frames(r) != S.CROP_FRAMES)
    ]
    if bad_keys:
        v.add("C6", "crop_key does not carry the non-default crop length", sorted(set(bad_keys))[:10], len(bad_keys))
    if not isinstance(S.CROP_FRAMES, int):
        v.add("C6", f"CROP_FRAMES must be an int, got {S.CROP_FRAMES!r}")

    # C10: whole rings, counted from the people in each cell, so a skipped folder shows.
    short_rings = ring_geometry_gaps(rows)
    if short_rings:
        v.add("C10", "ring is missing driver clips (see the builder warnings)", short_rings, len(short_rings))

    # C7: the face folder is never in the voice clips.
    face_in_voice: list[str] = []
    for r in rows:
        if not r["needs_conversion"] or r["arm"] != "main":
            continue
        face_folder = r["face_ref_folder"]
        if any(f"/{face_folder}/" in f"/{clip}" for clip in r["voice_ref_clip_ids"]):
            face_in_voice.append(f"{r['sample_id']}: face folder {face_folder} leaked into voice refs")
    if face_in_voice:
        v.add("C7", "face-donor folder present in the voice-reference pool", face_in_voice, len(face_in_voice))

    # C9: the main rows are balanced over the classes, overall AND in every split.
    counts = Counter(r["class_name"] for r in main_rows)
    if len(set(counts.values())) > 1:
        v.add("C9", f"main-row class balance is uneven: {dict(counts)}", [f"{k}={n}" for k, n in counts.items()])
    per_split: dict[str, Counter[str]] = defaultdict(Counter)
    for r in main_rows:
        per_split[r["split"]][r["class_name"]] += 1
    for split in sorted(per_split):
        c = per_split[split]
        if sorted(c) != sorted(S.CLASS_NAMES) or len(set(c.values())) > 1:
            v.add("C9", f"main-row class balance uneven within split {split}: {dict(c)}")


def check_crop_fits(rows: list[dict[str, Any]], v: Violations) -> None:
    """C8: the crop fits inside every driver clip."""
    inv = pd.read_parquet(S.inventory_parquet_path(), columns=["clip_id", "duration_s"])
    duration_by_clip = dict(zip(inv["clip_id"], inv["duration_s"], strict=True))
    needed = S.CROP_START_S + S.crop_duration_s(S.row_crop_frames(rows[0]))
    too_short: list[str] = []
    missing: list[str] = []
    for clip_id in sorted({r["driver_clip_id"] for r in rows}):
        d = duration_by_clip.get(clip_id)
        if d is None:
            missing.append(clip_id)
        elif float(d) < needed:
            too_short.append(f"{clip_id}: {float(d):.2f}s < {needed}s")
    if too_short:
        v.add("C8", "driver clip shorter than the fixed crop window", too_short, len(too_short))
    if missing:
        v.add("C8", "driver clip absent from the inventory", missing, len(missing))


def main() -> int:
    """Run every check, write the report, exit 1 on any violation."""
    args = parse_args()
    manifest_path = args.manifest or S.new_gen_manifest_path(args.build_id)
    if not manifest_path.exists():
        print(f"[validate] manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    split_map: dict[str, str] = json.loads(S.split_map_path().read_text())["groups"]

    v = Violations()
    check_firewall(rows, split_map, v)
    check_construction(rows, v)
    if not args.skip_duration_check:
        check_crop_fits(rows, v)

    report = {
        "build_id": args.build_id,
        "manifest": str(manifest_path),
        "n_samples": len(rows),
        "n_violations": len(v),
        "verdict": "PASS" if len(v) == 0 else "FAIL",
        "violations": v.items,
        "checks_run": ["F1", "F2", "F3", "F4", "F5", "F6", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9", "C10"]
        + ([] if args.skip_duration_check else ["C8"]),
    }
    out = args.out or S.new_gen_validate_report_path(args.build_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"[validate] {report['verdict']}: {len(v)} violation(s) over {len(rows)} samples -> {out}")
    for item in v.items:
        print(f"[validate]   {item['check']}: {item['message']} (n={item['count']})", file=sys.stderr)
        for ex in item["examples"]:
            print(f"[validate]     e.g. {ex}", file=sys.stderr)
    return 0 if len(v) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
