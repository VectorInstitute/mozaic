"""Balanced samples, shared by every task."""

from __future__ import annotations

import collections
import random
from collections.abc import Callable, Hashable
from typing import Any

Row = dict[str, Any]


def shuffled_options(options: list[str], seed: int, clip_id: str) -> list[str]:
    """Options in a per-clip order, the same every time, so no position gives the answer."""
    out = list(options)
    random.Random(f"{seed}:{clip_id}").shuffle(out)
    return out


def balanced_sample(
    rows: list[Row],
    size: int,
    seed: int,
    answer: Callable[[Row], Hashable],
    kind: Callable[[Row], Hashable],
) -> list[Row]:
    """Equal per answer, spread evenly over each answer's kinds, one clip per pair.

    Rows are ordered so that any first part of the list is still balanced, in case a
    run stops early.
    """
    answers = sorted({answer(r) for r in rows})
    if not answers or size % len(answers):
        raise ValueError(f"size must be a multiple of {len(answers)} (one share per answer)")

    by_kind: dict[Hashable, list[Row]] = collections.defaultdict(list)
    for r in rows:
        by_kind[kind(r)].append(r)

    counts: dict[Hashable, int] = {}
    for a in answers:
        kinds = sorted(k for k in by_kind if answer(by_kind[k][0]) == a)
        share, extra = divmod(size // len(answers), len(kinds))
        for i, k in enumerate(kinds):
            counts[k] = share + (i < extra)

    rng = random.Random(seed)
    used_pairs: set[str] = set()
    placed = []
    for k, n in counts.items():
        candidates = [r for r in by_kind[k] if r["meta"]["pair_id"] not in used_pairs]
        if len(candidates) < n:
            raise ValueError(f"not enough clips of kind {k}: need {n}, have {len(candidates)}")
        picks = rng.sample(candidates, n)
        used_pairs.update(r["meta"]["pair_id"] for r in picks)
        placed += [((i + 0.5) / n, rng.random(), r) for i, r in enumerate(picks)]
    return [r for _, _, r in sorted(placed, key=lambda p: p[:2])]


def counted_sample(
    rows: list[Row],
    counts: dict[Hashable, int],
    seed: int,
    kind: Callable[[Row], Hashable],
) -> list[Row]:
    """Exactly counts[k] rows of each kind k, one clip per pair (t3).

    Kinds are drawn in the order of counts. Rows are then ordered so that any first part
    of the list holds about its share of every kind.
    """
    rng = random.Random(seed)
    used_pairs: set[str] = set()
    picked: dict[Hashable, list[Row]] = {}
    for k, n in counts.items():
        candidates = [r for r in rows if kind(r) == k and r["meta"]["pair_id"] not in used_pairs]
        if len(candidates) < n:
            raise ValueError(f"not enough clips of kind {k}: need {n}, have {len(candidates)}")
        picked[k] = rng.sample(candidates, n)
        used_pairs.update(r["meta"]["pair_id"] for r in picked[k])
    placed = [((i + 0.5) / counts[k], rng.random(), r) for k, picks in picked.items() for i, r in enumerate(picks)]
    return [r for _, _, r in sorted(placed, key=lambda p: p[:2])]