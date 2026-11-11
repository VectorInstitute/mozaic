"""MOZAIC-Small: a smaller MOZAIC that mirrors the full one, for training and quick evaluation.

Whole pairs (every clip of a pair: real, fake and vocoded), each Small split drawn from
the same full split, so a person is still in one split only. Pairs are drawn in
proportion to the full split per (length, generators that kept the pair, gender); every
person is the source (A) of at least one pair, and of at most ceil(pairs / people).
Seeded, so the same every build.

Small is a choice of rows. Its files sit next to the full ones, with _small added, and it
shares videos/ with the full dataset (nothing is copied):

    python -m mozaic.common.subsets     # metadata_small.jsonl, pairs_small.jsonl,
                                        # summary_small.json, metadata_table_small.parquet

Every row is the full release's row, unchanged. summary_small.json is summary.json counted
on Small, plus a subset block recording how Small was drawn. Then build Small's question
files, run and score models with --subset small (common/paths.py): qas_small/,
results_small/, training_small/.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import logging
import math
import os
import random
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from mozaic.common.config import REPO_ROOT
from mozaic.common.io import FILE_MODE, write_json, write_jsonl
from mozaic.common.metadata import SPLITS, load_metadata
from mozaic.common.paths import (
    data_root,
    metadata_path,
    suffix,
    summary_path,
    use_subset,
)

Row = dict[str, Any]
log = logging.getLogger(__name__)

NAME = "small"  # the subset these files are for: metadata_small.jsonl, qas_small/, ...
SEED = 7
TARGET_CLIPS = {"train": 10_000, "val": 2_000, "test": 5_000}  # about this many clips per split
REPOSITORY = "https://github.com/VectorInstitute/mozaic"


# --------------------------------------------------------------------------- #
# the draw
# --------------------------------------------------------------------------- #
def pair_table(clips: Iterable[Row]) -> dict[str, Row]:
    """One row per pair: split, length, generators that kept it, gender, people, its clips."""
    pairs: dict[str, Row] = {}
    for c in clips:
        fields = {"pair_id": c["pair_id"], "split": c["split"], "length": c["length"], "kept_by": sorted(c["kept_by"]),
                  "gender": c["gender"], "source_id": c["source_id"], "target_id": c["target_id"]}
        p = pairs.setdefault(c["pair_id"], {**fields, "clips": []})
        if any(p[k] != v for k, v in fields.items()):
            raise ValueError(f"{c['file_name']}: its pair fields differ from the pair's other clips")
        p["clips"].append(c["file_name"])
    return pairs


def stratum(pair: Row) -> tuple[str, str, str]:
    """What the mix is kept on: length, generators that kept the pair, gender."""
    return pair["length"], "+".join(pair["kept_by"]), pair["gender"]


def quotas(pairs: list[Row], n: int) -> dict[tuple, int]:
    """Pairs per stratum, in proportion to the full split (largest remainders, then key)."""
    sizes = collections.Counter(stratum(p) for p in pairs)
    exact = {s: n * k / len(pairs) for s, k in sizes.items()}
    out = {s: math.floor(x) for s, x in exact.items()}
    for s in sorted(exact, key=lambda s: (out[s] - exact[s], s))[: n - sum(out.values())]:
        out[s] += 1
    return out


def draw(pairs: list[Row], n: int, seed: str) -> list[Row]:
    """n pairs from one split: the quotas per stratum, every source person at least once.

    First each person gets one pair (people with the fewest pairs first), then each
    stratum is filled up to its quota from the people with the fewest pairs so far, never
    more than ceil(n / people) per person. The stratum with the fewest people left to
    choose from (per pair still needed) is filled first, so small strata are not left
    without anyone who has room.
    """
    rng = random.Random(seed)
    left = quotas(pairs, n)
    by_person: dict[str, list[Row]] = collections.defaultdict(list)
    for p in sorted(pairs, key=lambda p: p["pair_id"]):
        by_person[p["source_id"]].append(p)
    people = sorted(by_person)
    if n < len(people):
        raise ValueError(f"{n} pairs cannot cover {len(people)} people")
    cap = math.ceil(n / len(people))
    chosen: dict[str, Row] = {}
    count: collections.Counter[str] = collections.Counter()

    def take(p: Row) -> None:
        chosen[p["pair_id"]] = p
        left[stratum(p)] -= 1
        count[p["source_id"]] += 1

    rng.shuffle(people)
    for person in sorted(people, key=lambda q: len(by_person[q])):
        options = [p for p in by_person[person] if left[stratum(p)] > 0]
        if options:
            take(rng.choices(options, weights=[left[stratum(p)] for p in options])[0])

    pools = collections.defaultdict(list)
    for p in sorted(pairs, key=lambda p: p["pair_id"]):
        if p["pair_id"] not in chosen:
            pools[stratum(p)].append(p)
    for pool in pools.values():
        rng.shuffle(pool)
    while any(k > 0 for k in left.values()):
        room = {s: [p for p in pools[s] if p["pair_id"] not in chosen and count[p["source_id"]] < cap]
                for s in left if left[s] > 0}
        s = min(room, key=lambda s: (len({p["source_id"] for p in room[s]}) / left[s], s))
        if not room[s]:
            raise ValueError(f"stratum {s}: {left[s]} more pairs needed, no person has room left")
        take(min(room[s], key=lambda p: count[p["source_id"]]))
    return sorted(chosen.values(), key=lambda p: p["pair_id"])


def pairs_per_split(full: list[Row]) -> int:
    """How many pairs a Small split gets: about TARGET_CLIPS clips."""
    split = full[0]["split"]
    return round(TARGET_CLIPS[split] * len(full) / sum(len(p["clips"]) for p in full))


def counts(pairs: list[Row], clips: dict[str, Row]) -> dict[str, Any]:
    """Pairs, clips, people and the mix of a set of pairs."""
    rows = [clips[name] for p in pairs for name in p["clips"]]
    sources = collections.Counter(p["source_id"] for p in pairs)

    def kind(c: Row) -> str:
        return " ".join((c["label"], c["video_generator"], c["audio_generator"], c["length"], c["arm"]))

    return {
        "pairs": len(pairs),
        "clips": len(rows),
        "people_as_source": len(sources),
        "people_any_role": len(set(sources) | {p["target_id"] for p in pairs}),
        "max_pairs_per_source": max(sources.values(), default=0),
        "by_label": dict(sorted(collections.Counter(c["label"] for c in rows).items())),
        "by_kind": dict(sorted(collections.Counter(kind(c) for c in rows).items())),
        "by_gender": dict(sorted(collections.Counter(p["gender"] for p in pairs).items())),
        "by_length_generators": dict(sorted(collections.Counter(f"{s[0]} {s[1]}" for s in map(stratum, pairs)).items())),
    }


def check(picked: dict[str, list[Row]], table: dict[str, Row], clips: dict[str, Row]) -> list[str]:
    """What is wrong with the draw: splits, sizes, people, quotas, kinds."""
    problems = []
    seen: set[str] = set()
    for split in SPLITS:
        full = [p for p in table.values() if p["split"] == split]
        part = picked[split]
        ids = [p["pair_id"] for p in part]
        n = pairs_per_split(full)
        if len(set(ids)) != len(ids) or seen & set(ids):
            problems.append(f"{split}: a pair is drawn twice")
        seen |= set(ids)
        problems += [f"{split}: {p['pair_id']} is in {p['split']}" for p in part if p["split"] != split]
        if len(ids) != n:
            problems.append(f"{split}: {len(ids)} pairs, expected {n}")
        people = {p["source_id"] for p in full}
        if {p["source_id"] for p in part} != people:
            problems.append(f"{split}: not every person is the source of a pair")
        if max(collections.Counter(p["source_id"] for p in part).values()) > math.ceil(n / len(people)):
            problems.append(f"{split}: a person is the source of too many pairs")
        if collections.Counter(stratum(p) for p in part) != {s: k for s, k in quotas(full, n).items() if k}:
            problems.append(f"{split}: the mix differs from the full split")
        missing = set(counts(full, clips)["by_kind"]) - set(counts(part, clips)["by_kind"])
        if missing:
            problems.append(f"{split}: kinds missing: {sorted(missing)}")
    return problems


def draw_small(clips: dict[str, Row]) -> tuple[dict[str, list[Row]], dict[str, Any]]:
    """Small's pairs per split, and per split its counts next to the full split's."""
    table = pair_table(clips.values())
    picked, per_split = {}, {}
    for split in SPLITS:
        full = sorted((p for p in table.values() if p["split"] == split), key=lambda p: p["pair_id"])
        picked[split] = draw(full, pairs_per_split(full), f"{SEED}:{split}")
        per_split[split] = {"counts": counts(picked[split], clips), "full_counts": counts(full, clips)}
        c = per_split[split]["counts"]
        log.info("%s: %d pairs, %d clips, %d people as source", split, c["pairs"], c["clips"], c["people_as_source"])
    problems = check(picked, table, clips)
    if problems:
        raise ValueError(f"MOZAIC-Small is not right, nothing written: {problems[:5]}")
    return picked, per_split


# --------------------------------------------------------------------------- #
# the files, next to the full ones
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lines_where(path: Path, keep: Callable[[Row], bool]) -> list[str]:
    """The lines of a jsonl file whose row passes keep, unchanged and in the file's order."""
    return [x for x in path.read_text().splitlines() if x.strip() and keep(json.loads(x))]


def write_lines(path: Path, lines: list[str]) -> None:
    """Write jsonl lines through io.py, then check each came out exactly as given."""
    write_jsonl(path, (json.loads(x) for x in lines))
    if path.read_text().splitlines() != lines:
        raise ValueError(f"{path}: a row differs from the full release")


def write_parquet(names: set[str], path: Path) -> int:
    """metadata_table.parquet's rows of these clips, same schema. Needs pyarrow."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(data_root() / "metadata_table.parquet")
    table = table.filter(pc.is_in(table["file_name"], value_set=pa.array(sorted(names))))
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.chmod(tmp, FILE_MODE)
    os.replace(tmp, path)
    return int(table.num_rows)


def recount(rows: list[Row], pair_rows: list[Row], like: dict[str, Any]) -> dict[str, Any]:
    """summary.json's counts for these clips and pairs, with the keys in the order of like."""
    folders: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for r in rows:
        folders[r["file_name"].rsplit("/", 1)[0]][r["split"]] += 1
    labels = collections.Counter(r["label"] for r in rows)
    splits = collections.Counter(r["split"] for r in rows)
    lengths = collections.Counter(p["length"] for p in pair_rows)
    return {
        "clips": len(rows),
        "pairs": {k: lengths[k] for k in like["pairs"]},
        "by_label": {k: labels[k] for k in like["by_label"]},
        "by_split": {k: splits[k] for k in like["by_split"]},
        "by_folder": {f: {k: folders[f][k] for k in v if folders[f][k]}
                      for f, v in like["by_folder"].items() if f in folders},
    }


def git_state() -> dict[str, Any]:
    """The repo commit that wrote Small, and whether its code had uncommitted changes."""

    def git(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True)
        return done.stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain", "--", "src"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def small_summary(full: dict[str, Any], rows: list[Row], pair_rows: list[Row],
                  per_split: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """summary.json counted on Small, with a subset block after schema_version: how Small was drawn."""
    people = {p[k] for p in pair_rows for k in ("source_id", "target_id")}
    splits = json.loads((data_root() / "splits.json").read_text())
    people_per_split = collections.Counter(splits[p] for p in people)
    block = {
        "name": NAME,
        "of": {"name": full["name"], "version": full["version"],
               "summary_sha256": sha256_file(summary_path(full=True)),
               "metadata_sha256": sha256_file(metadata_path(full=True))},
        "seed": SEED,
        "drawn_on": dt.date.today().isoformat(),
        "method": ("whole pairs; each Small split drawn from the same full split; pairs in proportion to the full "
                   "split per (length, generators that kept the pair, gender); every person the source (A) of at "
                   f"least one pair and of at most ceil(pairs / people); about this many clips per split: {TARGET_CLIPS}"),
        "code": {"repository": REPOSITORY, "module": "mozaic.common.subsets", **git_state()},
        "splits": per_split,
    }
    new = {"name": f"{full['name']}-Small", "source": {**full["source"], "people": len(people)},
           "counts": recount(rows, pair_rows, full["counts"])}  # fmt: skip
    if "people_per_split" in full:
        new["people_per_split"] = {k: people_per_split[k] for k in full["people_per_split"]}
    out: dict[str, Any] = {}
    for key, value in full.items():
        out[key] = new.get(key, value)
        if key == "schema_version":
            out["subset"] = block
    out.setdefault("subset", block)
    return out


def write_small() -> list[Path]:
    """Draw Small from the full metadata and write its four files next to the full ones."""
    use_subset(NAME)  # so metadata_path() and summary_path() name Small's files
    root = data_root()
    clips = load_metadata(metadata_path(full=True))
    picked, per_split = draw_small(clips)
    pair_ids = {p["pair_id"] for part in picked.values() for p in part}
    names = {name for part in picked.values() for p in part for name in p["clips"]}

    meta = lines_where(metadata_path(full=True), lambda r: r["file_name"] in names)
    pair_lines = lines_where(root / "pairs.jsonl", lambda r: r["pair_id"] in pair_ids)
    if (len(meta), len(pair_lines)) != (len(names), len(pair_ids)):
        raise ValueError("the full release does not hold Small's clips and pairs exactly once each")
    paths = [metadata_path(), root / f"pairs{suffix()}.jsonl", summary_path(), root / f"metadata_table{suffix()}.parquet"]
    write_lines(paths[0], meta)
    write_lines(paths[1], pair_lines)
    full_summary = json.loads(summary_path(full=True).read_text())
    rows, pair_rows = [json.loads(x) for x in meta], [json.loads(x) for x in pair_lines]
    write_json(paths[2], small_summary(full_summary, rows, pair_rows, per_split))
    log.info("%s: %s rows", paths[3].name, f"{write_parquet(names, paths[3]):,}")
    log.info("MOZAIC-Small: %s clips, %s pairs -> %s", f"{len(rows):,}", f"{len(pair_rows):,}",
             ", ".join(p.name for p in paths))
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    argparse.ArgumentParser(description="Draw MOZAIC-Small and write its files next to the full ones.").parse_args()
    write_small()
