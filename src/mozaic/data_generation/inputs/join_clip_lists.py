"""Join the inventory shard logs into one clip list, with each person's gender.

Run it after every list_clips shard has finished; --strict stops if a shard log is
missing. Rows keep the shard order (shard 0 first) and, inside a shard, the line order of
its log. Gender comes from the VoxCeleb2 meta file (its values carry stray spaces).

    python -m mozaic.data_generation.inputs.join_clip_lists --num-shards 60 --strict
    python -m mozaic.data_generation.inputs.join_clip_lists     # join whatever shard logs exist

Reads  voxceleb2/_logs/inventory_shard<i>_of_<n>.jsonl, <voxceleb2>/meta/vox2_meta.csv
Writes voxceleb2/_reports/inventory.parquet
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from mozaic.data_generation import settings as S

SHARD_RE = re.compile(r"inventory_shard(\d+)_of_(\d+)\.jsonl$")


def find_shard_logs() -> list[tuple[int, int, Path]]:
    """[(shard_index, num_shards, path), ...] for every shard log on disk, sorted."""
    out: list[tuple[int, int, Path]] = []
    for p in S.log_dir().glob("inventory_shard*_of_*.jsonl"):
        m = SHARD_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(out)


def load_gender_map() -> dict[str, str]:
    """identity_id -> 'm' or 'f', from vox2_meta.csv."""
    gender: dict[str, str] = {}
    with open(S.VOXCELEB2_META_CSV, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        id_idx = header.index("VoxCeleb2 ID")
        gender_idx = header.index("Gender")
        for row in reader:
            if not row:
                continue
            identity_id = row[id_idx].strip()
            g = row[gender_idx].strip()
            gender[identity_id] = g
    return gender


def main() -> int:
    ap = argparse.ArgumentParser(description="Join the inventory shard logs into inventory.parquet.")
    ap.add_argument("--num-shards", type=int, default=None, help="expected number of shards, to find missing ones")
    ap.add_argument("--strict", action="store_true", help="exit 1 if an expected shard log is missing")
    args = ap.parse_args()

    shard_logs = find_shard_logs()
    if not shard_logs:
        print(f"[build_inv] no shard logs found under {S.log_dir()}", file=sys.stderr)
        return 1

    declared_num_shards = {n for _, n, _ in shard_logs}
    if len(declared_num_shards) != 1:
        print(f"[build_inv] WARNING: mixed num_shards values across logs: {declared_num_shards}", file=sys.stderr)
    expected_n = args.num_shards or max(declared_num_shards)

    present_idx = {i for i, _, _ in shard_logs}
    missing = sorted(set(range(expected_n)) - present_idx)
    if missing:
        msg = f"[build_inv] missing {len(missing)}/{expected_n} shard logs: {missing[:20]}{'...' if len(missing) > 20 else ''}"
        if args.strict:
            print(msg, file=sys.stderr)
            return 1
        print(msg + " (continuing without --strict)", file=sys.stderr)

    rows = []
    for _, _, path in shard_logs:
        with open(path) as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    rows.append(json.loads(stripped))

    df = pd.DataFrame(rows)
    gender_map = load_gender_map()
    df["gender"] = df["identity_id"].map(gender_map)

    n_no_gender = int(df["gender"].isna().sum())
    if n_no_gender:
        print(
            f"[build_inv] WARNING: {n_no_gender} clips have an identity_id with no gender in vox2_meta.csv",
            file=sys.stderr,
        )

    S.report_dir().mkdir(parents=True, exist_ok=True)
    out_path = S.inventory_parquet_path()
    df.to_parquet(out_path, index=False)

    reason_counts = Counter(df["drop_reason"].fillna("ok"))
    n_identities = df["identity_id"].nunique()
    print(f"[build_inv] merged {len(shard_logs)} shard logs -> {len(df)} clip rows, {n_identities} identities")
    print(f"[build_inv] drop_reason breakdown: {dict(reason_counts)}")
    print(f"[build_inv] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
