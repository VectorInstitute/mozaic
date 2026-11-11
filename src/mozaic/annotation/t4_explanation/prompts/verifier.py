"""Verifier prompt, and how its per-sentence verdicts are read. It checks the annotator's sentences against the
claims (PROMPT for fake parts, REAL_PROMPT for real parts), and the rewrite against the checked sentences."""

from __future__ import annotations

import re
from typing import Any

Row = dict[str, Any]

PROMPT = """Below are claims about a video clip (C1, C2, ...), and sentences written from them (S1, S2, ...) by someone who also watched the clip. Each sentence lists the claims it is based on.

The writer may describe what is on screen neutrally (who, where, what they do), and may say where or when a listed defect shows. A sentence fails if it states or implies any defect, flaw or oddity that no listed claim gives, if it says anything looks or sounds natural, real, fine or convincing, or if it adds a cause or a judgement. It also fails if it has claim numbers in its text (such as [3] or (2)), a timestamp, a time in seconds, or a frame number. Rewording, combining and linking the listed claims is fine.

Claims:
{claims}

Sentences:
{sentences}

Reply in JSON only, one entry per sentence: {{"S1": {{"follows": true or false, "added": "<what it adds, or empty>"}}, ...}}"""


REAL_PROMPT = """Below are claims about a video clip (C1, C2, ...), and sentences written from them (S1, S2, ...) by someone who also watched the clip. Each sentence lists the claims it is based on. Every part the sentences are about is a natural recording.

The writer may describe what is on screen or heard neutrally (who, where, what they do or say), and says the part is a natural recording, with nothing AI-generated. A sentence fails if it states or implies any defect, flaw, oddity or sign of AI generation or editing, if it describes the quality of the picture or the sound (how a face, its expressions or the light look, how the voice sounds: its tone, pace, clarity or smoothness), if it says how natural, real or convincing anything looks or sounds beyond that claim, or if it adds a cause or a judgement. It also fails if it has claim numbers in its text (such as [3] or (2)), a timestamp, a time in seconds, or a frame number. Rewording, combining and linking the listed claims is fine.

Claims:
{claims}

Sentences:
{sentences}

Reply in JSON only, one entry per sentence: {{"S1": {{"follows": true or false, "added": "<what it adds, or empty>"}}, ...}}"""


REWRITE_PROMPT = """Below are checked sentences about a video clip (C1, C2, ...), and a rewrite of them (S1, S2, ...) by someone who did not see the clip. Each sentence of the rewrite lists the sentences it is based on.

A sentence of the rewrite fails if it states anything the listed sentences do not: a detail of the scene they do not give (posture, place, clothing, an action, what is said), a defect, flaw or oddity, a cause, a judgement, or a flaw said more or less strongly. It also fails if it has claim numbers in its text (such as [3] or (2)), a timestamp, a time in seconds, or a frame number. Rewording, combining, reordering, linking points (with and, so, as) and leaving out repeats is fine.

Sentences:
{claims}

Rewrite:
{sentences}

Reply in JSON only, one entry per sentence: {{"S1": {{"follows": true or false, "added": "<what it adds, or empty>"}}, ...}}"""


def build_prompt(claims: list[Row], reply: Row, parts: list[str], template: str = PROMPT) -> tuple[str, list[str]]:
    """The verifier prompt for one annotator reply, and its sentences in the order it labels them (S1, S2, ...)."""
    texts, lines = [], []
    for part in parts:
        for s in reply[part]:
            texts.append(s["text"])
            lines.append(f"S{len(texts)}. {s['text']} (claims: {', '.join(f'C{c}' for c in s['claims'])})")
    listed = "\n".join(f"C{c['id']}. {c['text']}" for c in claims if c["part"] in parts)
    return template.format(claims=listed, sentences="\n".join(lines)), texts


def build_rewrite_prompt(reply: Row, rewrite: Row, parts: list[str]) -> tuple[str, list[str]]:
    """The verifier prompt for a rewrite (REWRITE_PROMPT): the checked sentences are its claims, each sentence of the
    rewrite cites all of its part's. The rewriter never saw the clip, so no new scene detail is allowed either."""
    claims, cited, n = [], {}, 0
    for part in parts:
        cited[part] = []
        for s in reply[part]:
            n += 1
            claims.append({"id": n, "part": part, "text": s["text"].strip()})
            cited[part].append(n)
    sentences = {p: [{"text": s, "claims": cited[p]} for s in re.split(r"(?<=[.!?])\s+", rewrite[p].strip()) if s]
                 for p in parts}
    return build_prompt(claims, sentences, parts, REWRITE_PROMPT)


def problems(verdicts: Any, texts: list[str]) -> list[str]:
    """The sentences the verifier failed, as problems; [] if every sentence follows from its claims."""
    want = [f"S{i}" for i in range(1, len(texts) + 1)]
    if not isinstance(verdicts, dict) or sorted(verdicts) != sorted(want) \
            or not all(isinstance(verdicts[k], dict) for k in want):
        return ["verifier: not one verdict per sentence"]
    return [f"verifier: {k} adds {verdicts[k].get('added')!r}: {texts[i]!r}"
            for i, k in enumerate(want) if not verdicts[k].get("follows")]
