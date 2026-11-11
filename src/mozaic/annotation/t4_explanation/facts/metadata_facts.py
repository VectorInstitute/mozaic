"""A clip's facts from metadata: what is fake, where, and the signs from the quality-gate scores."""

from __future__ import annotations

from typing import Any

Row = dict[str, Any]

REGIONS = ("none", "inner face", "whole frame", "voice")
PARTS = ("video", "audio")

# video_generator -> (region, what changed, what is untouched). Only what one could see or
# hear in the clip itself: never how it was made (from a photo, copied movement, whose voice),
# since the evaluated model cannot know it. The method is kept apart, in "method".
VIDEO = {
    "facefusion": ("inner face", "the inner face (eyes, nose, mouth, skin) is AI-generated",
                   "the hair, head shape, body and background are untouched"),
    "animate": ("whole frame", "the whole frame, person and background, is AI-generated", None),
    "s2v": ("whole frame", "the whole frame, person and background, is AI-generated", None),
}
AUDIO = ("voice", "the voice is AI-altered", None)

# (score, "<" or ">", cutoff, artifact category, meaning, strong cutoff, strong meaning). A score past its
# cutoff gives the sign; past its strong cutoff too, the sign is worded as strong. The meaning says what one
# sees in the clip alone: the score compares the clip with its real recording, which the evaluated model never
# sees. The scores come from the gate; the cutoffs are our own first guess, and the golden review decides
# whether they match what people see. sharp < 0.7 keeps the clearly softer faces (0.85 was FaceFusion's
# median, half of it). Every clip already passed the gate's own, looser limits. The strong cutoffs split the
# test clips with that sign near their quartiles (sharp: 0.57 / 0.63 / 0.67; melt > 1.25: 1.32 / 1.44 / 1.56;
# melt < 0.8: 0.62 / 0.71 / 0.76).
SIGNS = (
    ("sharp", "<", 0.7, "unnatural blur", "the face looks a little soft and blurry",
     0.6, "the face looks clearly soft and blurry, without fine detail"),
    ("melt", ">", 1.25, "shape or structure distortion", "the face wobbles or warps as it moves",
     1.5, "the face strongly wobbles and warps as it moves"),
    ("melt", "<", 0.8, "unnatural human movement", "the face barely moves or changes expression",
     0.65, "the face hardly moves or changes expression at all"),
)

# Scores that give no sign for a generator, as in the gate: S2V invents its own head movement,
# so its face changes differ from the real clip's by design, not by warping.
OFF = {"s2v": ("melt",)}


def past(value: float, side: str, cutoff: float) -> bool:
    """Whether a score is past a cutoff on its side ("<" or ">")."""
    return value < cutoff if side == "<" else value > cutoff


def signs(clip: Row) -> list[Row]:
    """The signs of a fake video: one per score past its cutoff (worded by strength), except the scores off for
    its generator."""
    off = OFF.get(clip.get("video_generator"), ())
    found = []
    for score, side, cutoff, artifact, meaning, strong, strong_meaning in SIGNS:
        value = clip.get(score)
        if score not in off and value is not None and past(value, side, cutoff):
            found.append({"artifact": artifact, "meaning": strong_meaning if past(value, side, strong) else meaning,
                          "score": score, "value": round(value, 3)})
    return found


def facts(clip: Row) -> Row:
    """The facts of one clip: the fake parts, their regions, and per part its method and sentences."""
    video_fake, audio_fake = not clip["face_real"], not clip["voice_real"]
    regions: list[str] = []
    parts: dict[str, Row] = {}
    if video_fake:
        region, changed, original = VIDEO[clip["video_generator"]]
        regions.append(region)
        parts["video"] = {"fake": True, "region": region, "method": clip["video_generator"],
                          "changed": changed, "original": original, "signs": signs(clip)}
    else:
        parts["video"] = {"fake": False, "region": "none", "method": "real", "changed": None,
                          "original": "a camera recording"}
    if audio_fake:
        region, changed, original = AUDIO
        regions.append(region)
        parts["audio"] = {"fake": True, "region": region, "method": "seed_vc", "changed": changed, "original": original}
    else:
        processed = clip.get("arm") == "vocoded"
        parts["audio"] = {"fake": False, "region": "none", "method": "real", "changed": None,
                          "original": "the speaker's real voice" + (", passed through a vocoder only" if processed else "")}
    return {
        "fake_parts": [p for p in PARTS if parts[p]["fake"]],
        "regions": regions or ["none"],
        "parts": parts,
    }

