"""A clip's numbered claims, from metadata: all its explanation may state. Two kinds of call, each with its own
claims: the fake parts (verdict, fact, signs, scene) and the real parts (verdict, scene)."""

from __future__ import annotations

from typing import Any

from mozaic.annotation.t4_explanation.facts.metadata_facts import PARTS as SECTIONS  # the order of the explanation

Row = dict[str, Any]
KINDS = ("fake", "real")  # one call per clip per kind with such a part; a saved row without "kind" is fake
REQUIRED = ("verdict", "fact", "sign")  # the kinds every reply must cite
# The fixed text real parts had before they were explained (in the settings of the folders annotated then)
REAL = {
    "video": "The video looks like a natural recording, with nothing AI-generated.",
    "audio": "The audio sounds like a natural recording, with nothing AI-generated.",
}
REAL_VERDICT = {p: f"the {p} is a natural recording, with nothing AI-generated" for p in SECTIONS}
REAL_SENTENCE = {p: v[0].upper() + v[1:] + "." for p, v in REAL_VERDICT.items()}  # ends every real part, as written
WHOLE_CLIP = ", from the start of the clip to the end"
SCENE = {"video": "what is on screen: who is shown, where, and what they do, described neutrally",
         "audio": "what is heard: who speaks and how, described neutrally"}
# A real part's scene: its content only (what is said, not how it sounds); there is nothing to find in it
REAL_SCENE = {"video": SCENE["video"], "audio": "what is heard: who speaks and what they say, described neutrally"}


def parts_of(clip_facts: Row, kind: str = "fake") -> list[str]:
    """The clip's parts of one kind, in the order of the explanation."""
    return [p for p in SECTIONS if clip_facts["parts"][p]["fake"] == (kind == "fake")]


def claims_of(clip_facts: Row, kind: str = "fake") -> list[Row]:
    """The numbered claims of a clip's parts of one kind (metadata_facts.facts): per fake part its verdict, fact,
    signs, scene; per real part its verdict, scene."""
    out: list[Row] = []

    def add(part: str, kind: str, text: str, **extra: Any) -> None:
        out.append({"id": len(out) + 1, "part": part, "kind": kind, "text": text, **extra})

    for name in parts_of(clip_facts, kind):
        part = clip_facts["parts"][name]
        if kind == "real":
            add(name, "verdict", REAL_VERDICT[name])
            add(name, "scene", REAL_SCENE[name])
            continue
        add(name, "verdict", f"the {name} is AI-generated")
        add(name, "fact", part["changed"] + WHOLE_CLIP)
        for s in part.get("signs") or []:
            add(name, "sign", s["meaning"], artifact=s["artifact"])
        add(name, "scene", SCENE[name])
    return out


def listing(claims: list[Row], part: str) -> str:
    """One part's claims as the annotator reads them: '3. the face ...'."""
    return "\n".join(f"{c['id']}. {c['text']}" for c in claims if c["part"] == part)
