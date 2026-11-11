"""T1 prompt and reply parsing. MLLMs only."""

from __future__ import annotations

import json
from typing import Any

MODALITIES = ("video", "audio")

PROMPT = """The images are {n} frames sampled evenly from one video clip, in time order. The audio is that clip's soundtrack.

{question}

What each option means:
{descriptions}

Options: {options}

Reply in JSON only: {{"answer": "<one of the options, copied exactly>"}}"""

DESCRIPTIONS = {
    "the video": "the video is AI-generated or edited; the audio is real",
    "the audio": "the audio is AI-generated or edited; the video is real",
    "both": "both the video and the audio are AI-generated or edited",
    "neither": "both the video and the audio are real",
}

# The prompt's first line by what the model gets of the clip (the task's modalities in a model config: video only,
# audio only; t4 too). PROMPT has the first, so its hash, and every run made with it, stay as they were.
FIRST_LINES = {
    ("video", "audio"): "The images are {n} frames sampled evenly from one video clip, in time order. The audio is "
                        "that clip's soundtrack.",
    ("video",): "The images are {n} frames sampled evenly from one video clip, in time order. The clip's soundtrack is "
                "not provided.",
    ("audio",): "The audio is one video clip's soundtrack. The clip's frames are not provided.",
}


def template(modalities: tuple[str, ...] = MODALITIES) -> str:
    """PROMPT with its first line for these modalities."""
    return PROMPT.replace(FIRST_LINES[MODALITIES], FIRST_LINES[modalities])


def build_prompt(record: dict[str, Any], n_frames: int, modalities: tuple[str, ...] = MODALITIES) -> str:
    """Descriptions and options follow the clip's shuffled order."""
    descriptions = "\n".join(f'- "{o}": {DESCRIPTIONS[o]}' for o in record["options"])
    options = ", ".join(f'"{o}"' for o in record["options"])
    return template(modalities).format(n=n_frames, question=record["question"], descriptions=descriptions, options=options)


def parse(raw: str, options: list[str]) -> str | None:
    """The chosen option, or None if the reply is not exactly one of them."""
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        answer = json.loads(text)["answer"]
    except (ValueError, KeyError, TypeError):
        answer = text
    answer = str(answer).strip().lower()
    return answer if answer in options else None

