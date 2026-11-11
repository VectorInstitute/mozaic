"""T4 prompt and reply parsing. Video and audio. MLLMs only.

The evaluated model gets only the clip: no facts and no reference clip (the facts are for
the T4 annotator). It explains the video and the audio each on its own, as the reference
does (gt_target.sections): whether the part is AI-generated, and if so what was generated.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mozaic.annotation.t4_explanation.facts.metadata_facts import REGIONS
from mozaic.evaluation.prompts.t1_prompts import FIRST_LINES  # T1's first lines

MODALITIES = ("video", "audio")

# In the references' words (the T4 annotation: metadata_facts, its rewriter's "AI-generated" and "AI-altered", and
# the FaceFusion fix's inner-face sentence), never how a part was made
DESCRIPTIONS = {
    "none": "nothing is AI-generated",
    "inner face": "only the inner face (eyes, nose, mouth, skin) is AI-generated; the hair, head, body and background "
                  "are untouched",
    "whole frame": "the whole frame, person and background, is AI-generated",
    "voice": "the voice is AI-altered",
}
if set(DESCRIPTIONS) != set(REGIONS):
    raise ValueError(f"T4 region descriptions {sorted(DESCRIPTIONS)} differ from the regions {sorted(REGIONS)}")

PROMPT = """The images are {n} frames sampled evenly from one video clip, in time order. The audio is that clip's soundtrack.

Some clips are real recordings; normal processing such as compression does not count as AI generation. In others, AI generated or altered the video, the audio, or both.

Explain the video and the audio of this clip, each on its own, in 3 to 5 full sentences per part:
- video: say clearly whether the video is AI-generated: if so, what was generated; if not, that it is a natural recording, with nothing AI-generated.
- audio: say clearly whether the audio is AI-generated: if so, what was generated or altered; if not, that it is a natural recording, with nothing AI-generated.
Describe what is on screen (who is shown, where, what they do) and what is heard (who speaks and what about), and name any visible sign of manipulation.

What can be generated or altered:
{descriptions}

Reply in JSON only, each value a paragraph: {{"video": "<3 to 5 sentences>", "audio": "<3 to 5 sentences>"}}"""


def template(modalities: tuple[str, ...] = MODALITIES) -> str:
    """PROMPT with its first line for these modalities (both parts are still asked for)."""
    return PROMPT.replace(FIRST_LINES[MODALITIES], FIRST_LINES[modalities])


def build_prompt(record: dict[str, Any], n_frames: int, modalities: tuple[str, ...] = MODALITIES) -> str:
    """Descriptions follow the order of the question's options; "none" is not listed (a real part is said in words)."""
    descriptions = "\n".join(f"- {o}: {DESCRIPTIONS[o]}" for o in record["options"] if o != "none")
    return template(modalities).format(n=n_frames, descriptions=descriptions)


def parse(raw: str, options: list[str]) -> dict[str, str] | None:
    """{"video", "audio"}, each a non-empty explanation, or None if the reply does not give both. Read from the JSON
    objects in the reply, wherever they are: inside a ```json fence, around other text, followed by stray
    characters, or one object per part ({"video": ...}, {"audio": ...}], as Uni-MoE writes); the first value of
    each part wins; keys in any case; line breaks inside strings allowed; a reply that ends without its last "}"
    is read with it. A reply that is not JSON as written is read again after _repair (a comma before "}", a quote
    inside a value). A part that is missing is never made up: the reply stays unreadable. options is unused: the
    signature every task's parse() has."""
    fixed = _repair(raw)
    # as written first; then right before its closing brace; then repaired
    for text in (raw, raw.rstrip() + "}", fixed, fixed.rstrip() + "}"):
        parts = _parts(text)
        if all(isinstance(parts.get(m), str) and parts[m].strip() for m in MODALITIES):
            return {m: parts[m].strip() for m in MODALITIES}
    return None


_RUN_IN_KEY = re.compile(r'(?<!")(?:\\n|\s)+(video|audio)"\s*:\s*"', re.IGNORECASE)  # no quote before the key


def _repair(text: str) -> str:
    """text with the two slips models make in this JSON undone: a comma before "}" or "]" (OmniVinci: "...",\n})
    is dropped, and a quote inside a value (MiniCPM-o: said "next week" and) is escaped. A quote closes the
    value only when a ",", "}", "]" or ":" comes next, or the text ends; the words stay as written. A part's key
    run into the value before it (MiniCPM-o: ...lighting.\\n\\naudio": "...) starts that part again."""
    text = _RUN_IN_KEY.sub(r'", "\1": "', text)
    out, in_string, escaped, i = [], False, False, 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                after = text[i + 1:].lstrip()
                if after and after[0] not in ",}]:":
                    out.append('\\"')  # a quote inside the value
                    i += 1
                    continue
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "," and text[i + 1:].lstrip()[:1] in ("}", "]"):
            i += 1  # a comma before "}" or "]"
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parts(text: str) -> dict[str, Any]:
    """The video and audio values of the JSON objects in text, in order; an object read is skipped as a whole."""
    decoder, parts, i = json.JSONDecoder(strict=False), {}, 0  # strict=False: raw line breaks inside a string
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).strip().lower()
                if key in MODALITIES and isinstance(v, str) and v.strip():
                    parts.setdefault(key, v)
        i = end
    return parts


def answered(raw: str) -> bool:
    """Whether a reply still being written already holds its answer for good: it ends right after a "}" that closes
    an object at the top level, and the objects before it give both parts. parse() keeps the first value of each
    part, so nothing the model writes next can change what it reads: a model that writes its object again and again
    (Ola and Uni-MoE, greedy, up to max_new_tokens) can stop there. The "}" of an object inside another is not
    enough: the outer object, read as a whole, may give other values or none."""
    tail = raw.rstrip().rstrip(",]").rstrip()  # Uni-MoE ends an object with "}," or "}]", in one token
    if not tail.endswith("}") or _depth(raw) != 0:
        return False
    parts = _parts(raw)
    return all(isinstance(parts.get(m), str) and parts[m].strip() for m in MODALITIES)


def _depth(text: str) -> int:
    """How many objects are open at the end of text: its "{" less its "}", outside strings."""
    depth, in_string, escaped = 0, False, False
    for ch in text:
        if in_string:
            escaped, in_string = (False, True) if escaped else (ch == "\\", ch != '"')
        elif ch == '"':
            in_string = True
        elif ch in "{}":
            depth += 1 if ch == "{" else -1
    return depth
