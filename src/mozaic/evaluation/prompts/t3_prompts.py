"""T3 prompt and reply parsing: two clips, face or voice, four options. MLLMs only.

The question text comes from the question file. What each option means is listed in the
prompt, in the same order as the question's (shuffled) options.
"""

from __future__ import annotations

import json
from typing import Any

MODALITIES = ("video", "audio")  # per question: face -> frames only, voice -> sound only

LABELS = ("Clip 1:", "Clip 2:")  # sent before each clip's frames or sound

DESCRIPTIONS = {
    "face": {
        "real, same person": "the face in clip 1 is a real recording of the person in clip 2",
        "real, different person": "the face in clip 1 is a real recording of someone else",
        "fake of this person": (
            "the face in clip 1 was swapped, changed or made by a computer to look like the person in clip 2"
        ),
        "fake of a different person": (
            "the face in clip 1 was swapped, changed or made by a computer, and looks like someone else"
        ),
    },
    "voice": {
        "real, same person": "the voice in clip 1 is a real recording of the person in clip 2",
        "real, different person": "the voice in clip 1 is a real recording of someone else",
        "fake of this person": (
            "the voice in clip 1 was converted, changed or made by a computer to sound like the person in clip 2"
        ),
        "fake of a different person": (
            "the voice in clip 1 was converted, changed or made by a computer, and sounds like someone else"
        ),
    },
}

PROMPTS = {
    "face": """Above are {n} frames sampled evenly from each clip, in time order: first clip 1, then clip 2. There is no sound.

{question}

What each option means:
{descriptions}

Options: {options}

Reply in JSON only: {{"answer": "<one of the options, copied exactly>"}}""",
    "voice": """Above are the soundtracks of clip 1 and clip 2, in that order. There are no images.

{question}

What each option means:
{descriptions}

Options: {options}

Reply in JSON only: {{"answer": "<one of the options, copied exactly>"}}""",
}

# Everything fixed that the model reads; its hash goes into inference_settings.json.
PROMPT = "\n".join([*LABELS, *PROMPTS.values(), json.dumps(DESCRIPTIONS)])


def clip_inputs(record: dict[str, Any]) -> list[tuple[str, bool, bool]]:
    """(label, frames?, sound?) for clip 1 and clip 2: frames for face, sound for voice."""
    face = record["part"] == "face"
    return [(label, face, not face) for label in LABELS]


def build_prompt(record: dict[str, Any], n_frames: int) -> str:
    """The text after both clips; n_frames is per clip (0 for voice questions).

    Descriptions and options follow the question's shuffled order.
    """
    meaning = DESCRIPTIONS[record["part"]]
    descriptions = "\n".join(f'- "{o}": {meaning[o]}' for o in record["options"])
    options = ", ".join(f'"{o}"' for o in record["options"])
    return PROMPTS[record["part"]].format(n=n_frames, question=record["question"], descriptions=descriptions,
                                          options=options)


def parse(raw: str, options: list[str]) -> str | None:
    """The chosen option, or None if the reply is not exactly one of them."""
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        answer = json.loads(text)["answer"]
    except (ValueError, KeyError, TypeError):
        answer = text
    answer = str(answer).strip().lower()
    return answer if answer in options else None
