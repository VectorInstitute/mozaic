"""T2 prompt and reply parsing. Video only. MLLMs only."""

from __future__ import annotations

import json
from typing import Any

MODALITIES = ("video",)

DESCRIPTIONS = {
    "real": "a camera recording; nothing was generated",
    "face swap": "only the inner face was replaced; hair, head shape, body, background and all movement are from the original recording",
    "fully generated": "the whole frame was generated, including the person and the background",
    "full-body swap": "the whole frame was redrawn from one photo of another person, copying the movement of the original recording",
}

PROMPT = """The images are {n} frames sampled evenly from one video clip, in time order. There is no sound.

{question}

What each option means:
{descriptions}

Options: {options}

Reply in JSON only: {{"answer": "<one of the options, copied exactly>"}}"""


def build_prompt(record: dict[str, Any], n_frames: int) -> str:
    """Descriptions and options follow the clip's shuffled order."""
    descriptions = "\n".join(f'- "{o}": {DESCRIPTIONS[o]}' for o in record["options"])
    options = ", ".join(f'"{o}"' for o in record["options"])
    return PROMPT.format(n=n_frames, question=record["question"], descriptions=descriptions, options=options)


def parse(raw: str, options: list[str]) -> str | None:
    """The chosen option, or None if the reply is not exactly one of them."""
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        answer = json.loads(text)["answer"]
    except (ValueError, KeyError, TypeError):
        answer = text
    answer = str(answer).strip().lower()
    return answer if answer in options else None

