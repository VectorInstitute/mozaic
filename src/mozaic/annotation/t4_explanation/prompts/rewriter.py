"""Rewriter prompt: SYSTEM says how to write, the message holds the checked sentences."""

from __future__ import annotations

from typing import Any

Row = dict[str, Any]

SYSTEM = """Rewrite the sentences of each part of a clip (the video, the audio) as one paragraph, the way a person would write it for someone watching the clip with them.

Say what the sentences say and nothing more: the same people, actions and facts, the words of each flaw exactly as they are, and "AI-generated" and "AI-altered" wherever they are used. Only the wording and the order change.

Write plainly and naturally, stating facts without contrast or drama: say each point once, let sentences of different lengths flow into each other, and use no filler, stock phrases, dashes, lists, bracketed numbers or times (no timestamps or seconds)."""

LABEL = {"video": "VIDEO", "audio": "AUDIO"}


# Added to a real part's message: its sentences name no posture ("positioned in front of"), and a rewrite that adds one
# ("stands", "sits") fails the verifier
REAL_NOTE = ("Keep every detail exactly as the sentences give it: where they say a person is positioned somewhere, do "
             "not say they stand or sit, and add no posture, direction or action they do not state.")


def build_prompt(reply: Row, parts: list[str], kind: str = "fake") -> str:
    """The rewriter message for one clip: each part's checked sentences (a real part: its scene sentences, and
    REAL_NOTE)."""
    blocks = "\n\n".join(f"{LABEL[p]}:\n" + "\n".join(s["text"].strip() for s in reply[p]) for p in parts)
    shape = "{" + ", ".join(f'"{p}": "<paragraph>"' for p in parts) + "}"
    note = f"\n\n{REAL_NOTE}" if kind == "real" else ""
    return f"{blocks}{note}\n\nReply in JSON only: {shape}"
