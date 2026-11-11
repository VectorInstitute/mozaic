"""Word-overlap scores of a predicted text against one reference text: ROUGE-L, BLEU-4 and CIDEr-D.

Plain Python, as in the COCO caption evaluation (pycocoevalcap), with one reference per item:

    rouge_l   per item: F-score of the longest common subsequence (beta 1.2), then the mean
    bleu_4    corpus: clipped 1- to 4-gram precision over all items, geometric mean, brevity penalty
    cider_d   per item: tf-idf weighted 1- to 4-gram cosine (idf from the references), clipped, with a
              length penalty (sigma 6), times 10; then the mean

They reward the reference's words, not its meaning: an explanation in other words scores low. Read them
next to the judge and the embedding cosine.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
MAX_N = 4
ROUGE_BETA = 1.2
CIDER_SIGMA = 6.0


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(words: list[str], n: int) -> Counter:
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    common = lcs(p, r)
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return (1 + ROUGE_BETA**2) * precision * recall / (recall + ROUGE_BETA**2 * precision)


def bleu_4(preds: Sequence[str], refs: Sequence[str]) -> float:
    matched, total = [0] * MAX_N, [0] * MAX_N
    pred_len = ref_len = 0
    for pred, ref in zip(preds, refs, strict=True):
        p, r = tokens(pred), tokens(ref)
        pred_len, ref_len = pred_len + len(p), ref_len + len(r)
        for n in range(1, MAX_N + 1):
            pn, rn = ngrams(p, n), ngrams(r, n)
            matched[n - 1] += sum(min(c, rn[g]) for g, c in pn.items())
            total[n - 1] += max(len(p) - n + 1, 0)
    if not pred_len or not all(matched):
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matched, total, strict=True)) / MAX_N
    brevity = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    return brevity * math.exp(log_precision)


def cider_d(preds: Sequence[str], refs: Sequence[str]) -> list[float]:
    """One score per item; idf comes from the references of all items."""
    ref_tokens = [tokens(r) for r in refs]
    pred_tokens = [tokens(p) for p in preds]
    log_docs = math.log(float(len(refs)))
    df = [Counter(g for r in ref_tokens for g in ngrams(r, n)) for n in range(1, MAX_N + 1)]

    def vector(words: list[str]) -> list[dict[tuple, float]]:
        return [{g: c * (log_docs - math.log(max(1.0, df[n - 1][g]))) for g, c in ngrams(words, n).items()}
                for n in range(1, MAX_N + 1)]

    scores = []
    for p, r in zip(pred_tokens, ref_tokens, strict=True):
        vp, vr = vector(p), vector(r)
        total = 0.0
        for n in range(MAX_N):
            norm_p = math.sqrt(sum(v * v for v in vp[n].values()))
            norm_r = math.sqrt(sum(v * v for v in vr[n].values()))
            if norm_p and norm_r:
                dot = sum(min(v, vr[n][g]) * vr[n][g] for g, v in vp[n].items() if g in vr[n])
                total += dot / (norm_p * norm_r) * math.exp(-((len(p) - len(r)) ** 2) / (2 * CIDER_SIGMA**2))
        scores.append(total / MAX_N * 10.0)
    return scores


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
