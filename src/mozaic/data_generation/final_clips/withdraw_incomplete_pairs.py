"""Withdraw a pair that shipped only some of its classes.

The mux writes the classes of a pair one by one and drops a clip that misses its target, so one
drop can leave the pair's other clips on disk. A pair that ships only its real classes, or only
its generated ones, lets the driver's identity give the label away, so the whole pair is moved
aside before anything measures the folder.

Nothing is deleted: the pair's clips and records move to <final_dir>/_withdrawn/<pair_id>/ beside
a reason.json that names the classes that never arrived. Run it after the mux has finished (a
pair still being written looks incomplete). Without --apply it only reports.

    python -m mozaic.data_generation.final_clips.withdraw_incomplete_pairs --build-id <build_id> --final-dir <final_dir>
    python -m mozaic.data_generation.final_clips.withdraw_incomplete_pairs --build-id <build_id> --final-dir <final_dir> --apply
    python -m mozaic.data_generation.final_clips.withdraw_incomplete_pairs --build-id <build_id> --final-dir <final_dir> \\
        --withdraw-pairs pairs.txt --reason selection --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Withdraw pairs that shipped only some of their classes.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--final-dir", type=Path, required=True, help="the folder the mux wrote")
    p.add_argument("--pair-ids-file", type=Path, default=None,
                   help="only these pair_ids; a run on part of a build passes its own list, "
                   "so it never judges pairs that are still being muxed")
    p.add_argument("--apply", action="store_true", help="move the clips; without it, report only")
    p.add_argument("--withdraw-pairs", type=Path, default=None,
                   help="withdraw these pair_ids whole (every shipped clip of every arm), e.g. the pairs "
                   "the selection rule rejected; a reason file is written beside them")
    p.add_argument("--reason", default="selection", help="recorded with --withdraw-pairs (default: selection)")
    return p.parse_args()


def expected_by_pair(build_id: str, pair_ids: set[str] | None) -> dict[str, list[dict[str, str]]]:
    """pair_id -> the (sample_id, class_name) rows that pair is expected to ship."""
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    with S.new_gen_manifest_path(build_id).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if pair_ids is not None and row["pair_id"] not in pair_ids:
                continue
            out[row["pair_id"]].append({"sample_id": row["sample_id"], "class_name": row["class_name"]})
    return out


def classify(expected: dict[str, list[dict[str, str]]], root: Path) -> tuple[dict[str, dict[str, Any]], int, int]:
    """Count complete and absent pairs; return the incomplete ones with what is present and missing."""
    incomplete: dict[str, dict[str, Any]] = {}
    complete = absent = 0
    for pair_id, rows in sorted(expected.items()):
        present: list[dict[str, str]] = []
        missing: list[dict[str, str]] = []
        for row in rows:
            path = S.final_sample_path(row["class_name"], row["sample_id"], root)
            (present if path.exists() else missing).append(row)
        if not present:
            absent += 1
        elif not missing:
            complete += 1
        else:
            incomplete[pair_id] = {"present": present, "missing": missing}
    return incomplete, complete, absent


def withdraw(pair_id: str, detail: dict[str, Any], root: Path, *, apply: bool) -> list[str]:
    """Move one pair's shipped clips (and their records) aside and record why. Returns what moved."""
    dest = root / "_withdrawn" / pair_id
    moved: list[str] = []
    for row in detail["present"]:
        src = S.final_sample_path(row["class_name"], row["sample_id"], root)
        for path in (src, src.with_suffix(".json")):
            if not path.exists():
                continue
            moved.append(path.name)
            if apply:
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest / path.name))
    if apply:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "reason.json").write_text(
            json.dumps(
                {
                    "pair_id": pair_id,
                    "withdrawn_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "reason": "incomplete_group",
                    "detail": (
                        "the terminal chain refused at least one class of this pair, so the classes that did "
                        "ship would let driver identity predict the label"
                    ),
                    "missing": [r["sample_id"] for r in detail["missing"]],
                    "withdrawn": moved,
                },
                indent=2,
            )
            + "\n"
        )
    return moved


def withdraw_listed(args: argparse.Namespace, root: Path) -> int:
    """Withdraw every shipped clip of the listed pairs, whatever their arm."""
    listed = {x.strip() for x in args.withdraw_pairs.read_text().splitlines() if x.strip()}
    rows = [json.loads(x) for x in S.new_gen_manifest_path(args.build_id).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["pair_id"] in listed]
    by_pair: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    n_files = n_pairs = 0
    for pair_id, prs in sorted(by_pair.items()):
        present = [r for r in prs if S.find_final_sample(r["class_name"], r["sample_id"], root) is not None]
        if not present:
            continue
        n_pairs += 1
        dest = root / "_withdrawn" / pair_id
        for r in present:
            src = S.find_final_sample(r["class_name"], r["sample_id"], root)
            assert src is not None
            n_files += 1
            if args.apply:
                dest.mkdir(parents=True, exist_ok=True)
                for side in (src, src.with_suffix(".json")):
                    if side.exists():
                        side.rename(dest / side.name)
        if args.apply:
            (dest / "WITHDRAWN.txt").write_text(f"{args.reason}\n{len(present)} clip(s)\n")
    verb = "moved" if args.apply else "would move"
    print(
        f"[withdraw] {args.build_id} in {root.name}: {n_pairs} listed pair(s) on disk, {verb} {n_files} file(s) -> _withdrawn/ ({args.reason})"
    )
    if not args.apply:
        print("[withdraw] dry run — pass --apply to move them")
    return 0


def tails(rows: list[dict[str, str]]) -> str:
    """The class part of each sample_id, for the report."""
    return ", ".join(sorted(r["sample_id"].split("__x__", 1)[-1].split("__", 1)[-1] for r in rows))


def main() -> int:
    """Report, and with --apply withdraw, every half-shipped pair of one build."""
    args = parse_args()
    pair_ids = None
    if args.pair_ids_file:
        pair_ids = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
    root = S.resolve_final_dir(args.final_dir)
    if args.withdraw_pairs:
        return withdraw_listed(args, root)
    expected = expected_by_pair(args.build_id, pair_ids)
    if not expected:
        print(f"[withdraw] no manifest rows selected for {args.build_id}", file=sys.stderr)
        return 2

    incomplete, complete, absent = classify(expected, root)
    print(
        f"[withdraw] {args.build_id} in {root.name}: {complete} complete, "
        f"{absent} not shipped yet, {len(incomplete)} INCOMPLETE"
    )
    for pair_id, detail in incomplete.items():
        missing = tails(detail["missing"])
        kept = tails(detail["present"])
        print(f"[withdraw]   {pair_id}: missing {missing}; on disk {kept}")
        moved = withdraw(pair_id, detail, root, apply=args.apply)
        verb = "moved" if args.apply else "would move"
        print(f"[withdraw]     {verb} {len(moved)} file(s) -> {root.name}/_withdrawn/{pair_id}/")
    if incomplete and not args.apply:
        print("[withdraw] dry run — pass --apply to move them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
