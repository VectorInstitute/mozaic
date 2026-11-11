"""Split the people of the clip list into train, val and test, once, before any pair is drawn.

Every person with at least one usable clip (drop_reason null) goes to exactly one split, so
no person is ever in two splits. The splits are filled by usable-clip count, 70/10/20 for
each gender (run.yaml split), with seed 42: people are taken largest first (ties in a seeded
shuffle of the order they first appear in the clip list), and each goes to the split that
most needs clips of its gender. People with no usable clip are listed as excluded.

    python -m mozaic.data_generation.inputs.make_identity_split

Reads  voxceleb2/_reports/inventory.parquet
Writes voxceleb2/split_map.json (people sorted by id)
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict

import pandas as pd

from mozaic.data_generation import settings as S

_SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}  # on a tie: train, then val


def load_identity_table() -> tuple[dict[str, Counter], list[str]]:
    """(identity_id -> Counter({gender: usable clips}), people with no usable clip)."""
    df = pd.read_parquet(S.inventory_parquet_path())
    usable = df[df["drop_reason"].isna()]

    groups: dict[str, Counter] = defaultdict(Counter)
    for identity_id, gender in zip(usable["identity_id"], usable["gender"], strict=True):
        key = gender if isinstance(gender, str) and gender else "unknown"
        groups[identity_id][key] += 1

    all_identities = set(df["identity_id"].unique())
    excluded = sorted(all_identities - set(groups.keys()))
    return dict(groups), excluded


def greedy_assign(groups: dict[str, Counter], quotas: dict[str, dict[str, float]], seed: int) -> dict[str, str]:
    """Give each person a split, filling the per-gender clip quotas evenly.

    People are shuffled (seeded), then sorted by clip count, largest first. Each goes to the
    split with the most room for its clips; ties go to the split with more room in total,
    then train, val, test.
    """
    remaining = {s: {g: float(q) for g, q in quota.items()} for s, quota in quotas.items()}
    rng = random.Random(seed)
    order = list(groups.items())
    rng.shuffle(order)
    order.sort(key=lambda gc: sum(gc[1].values()), reverse=True)

    assignment: dict[str, str] = {}
    for identity_id, counts in order:
        best_split = next(iter(quotas))
        best_key: tuple | None = None
        for s in quotas:
            rem = remaining[s]
            score = sum(min(c, max(0.0, rem.get(g, 0.0))) for g, c in counts.items())
            tot_rem = sum(max(0.0, v) for v in rem.values())
            key = (score, tot_rem, -_SPLIT_ORDER.get(s, 99))
            if best_key is None or key > best_key:
                best_key, best_split = key, s
        assignment[identity_id] = best_split
        for g, c in counts.items():
            remaining[best_split][g] = remaining[best_split].get(g, 0.0) - c
    return assignment


def _assert_disjoint(groups_map: dict[str, str]) -> None:
    by_split: dict[str, set] = defaultdict(set)
    for identity_id, split in groups_map.items():
        by_split[split].add(identity_id)
    splits = list(by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = by_split[splits[i]] & by_split[splits[j]]
            if overlap:
                raise AssertionError(f"identities shared between {splits[i]}/{splits[j]}: {list(overlap)[:5]}")


def summarize(groups_map: dict[str, str], groups: dict[str, Counter]) -> dict:
    per_split: dict[str, Counter] = defaultdict(Counter)
    n_identities: Counter = Counter()
    for identity_id, split in groups_map.items():
        per_split[split].update(groups[identity_id])
        n_identities[split] += 1
    return {
        "identities_per_split": dict(n_identities),
        "clips_per_split": {s: sum(c.values()) for s, c in per_split.items()},
        "gender_per_split": {s: dict(c) for s, c in per_split.items()},
    }


def main() -> int:
    groups, excluded = load_identity_table()
    if excluded:
        print(
            f"[split] {len(excluded)} identities excluded (zero usable clips): {excluded[:10]}{'...' if len(excluded) > 10 else ''}"
        )

    totals: Counter = Counter()
    for counts in groups.values():
        totals.update(counts)
    tr, va, te = S.SPLIT_RATIO
    quotas = {
        "train": {g: tr * n for g, n in totals.items()},
        "val": {g: va * n for g, n in totals.items()},
        "test": {g: te * n for g, n in totals.items()},
    }

    assignment = greedy_assign(groups, quotas, seed=S.SPLIT_SEED)
    _assert_disjoint(assignment)
    summary = summarize(assignment, groups)

    out = {
        "dataset": "voxceleb2",
        "seed": S.SPLIT_SEED,
        "ratio": list(S.SPLIT_RATIO),
        "group_key": "identity_id",
        "stratify_key": S.STRATIFY_KEY,
        "excluded_identities_no_usable_clips": excluded,
        "summary": summary,
        "groups": dict(sorted(assignment.items())),  # sorted: the same bytes on every rerun
    }

    S.split_map_path().parent.mkdir(parents=True, exist_ok=True)
    with open(S.split_map_path(), "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")

    cps = summary["clips_per_split"]
    ips = summary["identities_per_split"]
    total_clips = sum(cps.values())
    print(f"Split map (seed={S.SPLIT_SEED}, ratio={S.SPLIT_RATIO}, stratify={S.STRATIFY_KEY}):")
    for s in ("train", "val", "test"):
        gp = summary["gender_per_split"].get(s, {})
        n_m, n_f = gp.get("m", 0), gp.get("f", 0)
        gender_pct = f"m={n_m / max(1, n_m + n_f):.1%} f={n_f / max(1, n_m + n_f):.1%}" if (n_m + n_f) else "n/a"
        print(
            f"  {s:6} identities={ips.get(s, 0):<6} clips={cps.get(s, 0):<8} "
            f"({cps.get(s, 0) / total_clips:.1%})  gender: {gender_pct}"
        )
    print(f"  -> {S.split_map_path()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
