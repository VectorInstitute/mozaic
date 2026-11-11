"""The T4 judge's prompt and how its reply is read. Text only: the reference explanation
(gt_target.sections) and the predicted one (pred_target.sections), never the clip."""

from __future__ import annotations

import json
import re
from typing import Any

PARTS = ("video", "audio")
MAX_SCORE = 10

SYSTEM = """You are an intelligent and fair evaluator of explanations of audio-video clips that may be AI-generated or altered. For one clip you get a reference explanation and a predicted explanation, each with a video part and an audio part. Score each predicted part against the same reference part.

Evaluation instructions:
- Judge meaning, not wording: accept paraphrases, synonyms and differences in style or tone.
- Ignore timing words: every manipulation covers the whole clip.
- For each part, weigh in this order of importance:
  1. Fake or real: the part called AI-generated/altered, or a real recording, as in the reference. A hedged or
     self-contradictory answer ("may be AI-generated", "real, but parts look generated") counts as wrong.
  2. What was manipulated: for a fake video, the same region as the reference: only the inner face (eyes, nose, mouth, skin) with the rest untouched, or the whole frame (person and background).
     For a fake audio, that the voice was generated or altered.
  3. Reasonableness: nothing called AI-generated that the reference says is real (a real part, or the background or body when only the inner face was generated). Visible signs the reference names (e.g. the face looks blurry, warps, barely moves) are credited when matched; signs the reference does not name are neither rewarded nor penalized.
  4. Scene: a description of who is shown or heard, where, and what about that agrees with the reference is fine; one that contradicts it (a woman instead of a man, another setting or topic) costs a little; a missing description costs nothing.
- When uncertain between two scores, give the lower one.

Scale for each part (0-10):
  10   fake or real as in the reference; for a fake part, what was manipulated right and the reference's
       signs (if any) named; nothing hallucinated. A real part called real with nothing hallucinated is 10.
  8-9  fake or real, and what was manipulated, right; a minor gap (a sign missed, a small scene contradiction)
  5-7  fake or real right, but what was manipulated is wrong or unstated, or more is called manipulated
       than the reference says
  1-4  a fake part called real, or a real part called fake
  0    no usable explanation

Return a JSON object only:
{"video": {"score": <integer 0-10>, "justification": "<concise explanation (1-2 sentences)>"},
 "audio": {"score": <integer 0-10>, "justification": "<concise explanation (1-2 sentences)>"}}"""

USER = """Please evaluate the following explanation of an audio-video clip.
Reference
  video: {gt_video}
  audio: {gt_audio}
Prediction
  video: {pred_video}
  audio: {pred_audio}
Please return your evaluation in the specified JSON format, with a score and a justification for each part."""

UNUSABLE = {p: {"score": 0, "justification": "no usable explanation: the reply could not be read"} for p in PARTS}


def build_user(reference: dict[str, str], prediction: dict[str, str]) -> str:
    """reference: gt_target.sections; prediction: pred_target.sections."""
    return USER.format(gt_video=reference["video"], gt_audio=reference["audio"],
                       pred_video=prediction["video"], pred_audio=prediction["audio"])


def parse(raw: str) -> dict[str, dict[str, Any]] | None:
    """{"video": {"score", "justification"}, "audio": {...}} with integer scores 0-10, or None. Reads the whole
    reply, else its last ```json fence, else from its first { to its last }."""
    text = raw.strip()
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    braces = text[text.find("{"):text.rfind("}") + 1] if "{" in text else ""
    for candidate in (text, *fences[-1:], braces):
        try:
            reply = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(reply, dict):
            continue
        out = {}
        for part in PARTS:
            item = reply.get(part)
            if not isinstance(item, dict):
                break
            score, why = item.get("score"), item.get("justification")
            if isinstance(score, bool) or not isinstance(score, int | float) or score != int(score):
                break
            if not 0 <= score <= MAX_SCORE or not isinstance(why, str):
                break
            out[part] = {"score": int(score), "justification": why.strip()}
        else:
            return out
    return None
