"""Annotator prompt: the clip's claims, the person's gender and the reply format. PROMPT for the fake parts,
REAL_PROMPT for the real parts."""

from __future__ import annotations

from typing import Any

from mozaic.annotation.t4_explanation.facts.claims import REQUIRED, listing

Row = dict[str, Any]

PROMPT = """You are writing the reference explanation for one clip of a deepfake benchmark. The clip is shown above. For each part, write one flowing paragraph, the way a good narrator would explain it to someone watching with you: set the scene (who is there, where, what they do or say), then explain what is AI-generated and for how long, and how the listed signs show as the clip plays. Use full, connected sentences that build on each other, not a list of short statements, and follow the clip as it unfolds (what happens first, then next).

The numbered claims below are the facts about this clip. Cite the claims each sentence uses, and use every one ("what is on screen" and "what is heard" are there to cite whenever you describe the scene). You may describe what you see and hear freely and neutrally, and say where or when a listed sign shows. But every defect, flaw or oddity you mention must be one the claims list: never add another, never say anything looks natural, real or convincing, never name how it was made (a method, a tool, "deepfake"), never compare it with real footage, and don't talk about judging or concluding.

The person in the clip is a {person}. Cite claims only in "claims", never in the text, and never give a time, a timestamp or a frame number: the times printed on the frames are only there to help you follow the clip.

{blocks}

Reply in JSON only: {reply}"""

REAL_PROMPT = """You are writing the reference explanation for one clip of a deepfake benchmark. The clip is shown above. Every part listed below is a natural recording: nothing in it is AI-generated, and there is nothing to find or prove. For each part, write one flowing paragraph that describes the clip, the way a good narrator would describe it to someone watching with you: for VIDEO, who is there, where, and what they do; for AUDIO, who speaks and what they say. Follow the clip as it unfolds (what happens first, then next), in full, connected sentences that build on each other, not a list of short statements. End the paragraph by saying that the part is a natural recording, with nothing AI-generated, in those words.

Describe what is there, never its quality: not the face's expressions, detail, sharpness or how it moves, not how the light falls, not how the voice sounds (its tone, pace, clarity or smoothness). Never mention a defect, flaw or oddity, never call anything natural, real or convincing except in the last sentence, never name how a clip is made or processed (a method, a tool, a vocoder, compression), never compare it with other footage, and don't talk about judging or concluding (no "overall", no "concludes").

The numbered claims below are the facts about this clip. Cite the claims each sentence uses, and use every one ("what is on screen" and "what is heard" are there to cite whenever you describe the scene). Each paragraph is about its own part only: the picture for VIDEO, the sound for AUDIO.

The person in the clip is a {person}. Cite claims only in "claims", never in the text, and never give a time, a timestamp or a frame number: the times printed on the frames are only there to help you follow the clip.

{blocks}

Reply in JSON only: {reply}"""

RETRY = "\n\nYour last reply failed these checks; write it again:\n- {problems}"

LABEL = {"video": "VIDEO (the picture)", "audio": "AUDIO (the sound)"}
PERSON = {"m": "man", "f": "woman"}  # metadata gender, one per clip: a pair's two people share it
# A real part has no fact claim: one more sentence asked, as many as a fake part with no sign
EXTRA = {"fake": 0, "real": 1}


def lengths(claims: list[Row], parts: list[str], kind: str = "fake") -> dict[str, tuple[int, int]]:
    """Sentences per part: the must-cite claims plus 2 to 5 (a real part: 3 to 6), room for the scene and for where
    the signs show."""
    need = {p: sum(c["part"] == p and c["kind"] in REQUIRED for c in claims) + EXTRA[kind] for p in parts}
    return {p: (need[p] + 2, need[p] + 5) for p in parts}


def build_prompt(claims: list[Row], parts: list[str], gender: str, problems: list[str] | None = None,
                 kind: str = "fake") -> str:
    """The annotator prompt for one clip and one kind of part: the person's gender, per part its claims and the
    sentences asked; on a retry, the problems of the last round."""
    span = lengths(claims, parts, kind)
    blocks = "\n\n".join(f"{LABEL[p]}, {span[p][0]} to {span[p][1]} sentences:\n{listing(claims, p)}" for p in parts)
    shape = "{" + ", ".join(f'"{p}": [{{"text": "<sentence>", "claims": [<numbers>]}}, ...]' for p in parts) + "}"
    prompt = (PROMPT if kind == "fake" else REAL_PROMPT).format(person=PERSON[gender], blocks=blocks, reply=shape)
    return prompt + RETRY.format(problems="\n- ".join(problems)) if problems else prompt
