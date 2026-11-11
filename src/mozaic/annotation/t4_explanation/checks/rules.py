"""Code checks, no calls: check_reply on the annotator's reply, check_rewrite on the rewriter's paragraphs, for
either kind of part (fake, real)."""

from __future__ import annotations

import re
from typing import Any

from mozaic.annotation.t4_explanation.facts.claims import REAL_SENTENCE, REQUIRED

Row = dict[str, Any]

# How it was made: the explanation describes what is seen or heard.
METHOD_WORDS = ("facefusion", "face fusion", "face swap", "face-swap", "faceswap", "s2v", "wan2", "animate",
                "seed-vc", "seed_vc", "seedvc", "voice conversion", "voice-conversion", "deepfake", "bigvgan",
                "vocoder", "score", "metadata", "photo", "photograph", "still image")
# The annotator talking about its input instead of the clip.
INPUT_LEAKS = re.compile(r"\bdescribed as\b|\b(?:is|are|was|were|been) (?:given|described(?! as\b)|noted|listed|mentioned|"
                         r"provided|specified|reported|indicated|available)\b|\bto describe\b|\bthe signs?\b|\bthe facts?\b"
                         r"|\bclaims?\b|\bnumbered\b|\bthe list\b", re.IGNORECASE)
# Talking about deciding instead of describing the clip ("my conclusion", "judged", "taken as a whole").
DECIDING = re.compile(r"\bjudg\w*|\bmy conclusion\b|\bin conclusion\b|\bconclude[sd]?\b|\bconcluding\b|\bverdict\b"
                      r"|\btaken (?:as a whole|together)\b|\ball things considered\b|\boverall\b|\bsettle\b"
                      r"|\bthe first thing\b", re.IGNORECASE)
# Frame numbers depend on how many frames were sampled: moments are named by meaning.
_ORDINALS = ("second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|"
             "fifteenth|sixteenth")
FRAME_NUMBERS = re.compile(rf"\bframes?\s*(?:#\s*)?\d+|\b\d+\s*(?:st|nd|rd|th)?\s+frames?\b"
                           rf"|\b(?:{_ORDINALS})(?:\s+(?:and|to|through|-)\s+\w+)?\s+frames?\b", re.IGNORECASE)
# A claim number left in the text: the citations belong in "claims" ("... speaks [3].", "(2, 3)").
CLAIM_NUMBERS = re.compile(r"[\[(]\s*\d+(?:\s*,\s*\d+)*\s*[\])]")
# A timestamp or a time in seconds (the frames are shown with their times): moments are named by meaning.
TIMES = re.compile(r"\b\d{1,2}:\d{2}(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*(?:seconds?|secs?)\b", re.IGNORECASE)
# Another clip: the reader sees this clip only, so a fake part is never compared with a real one.
OTHER_CLIP = re.compile(r"\b(?:real|original|genuine|source|unaltered|natural) (?:recording|video|clip|footage|face|"
                        r"person|voice|speaker)s?\b|\bthe original\b|\bthan (?:it|they|a face|a voice) (?:would|does|do)\b",
                        re.IGNORECASE)
# A fake part said to look or sound fine: a part with no signs was simply not measured as flawed.
FAKE_FINE = re.compile(r"\bnothing\b[^.;]*\b(?:wrong|off|odd|unusual|unnatural|stands? out|gives? it away)\b"
                       r"|\bno (?:visible|audible|obvious|specific|particular|single|clear)\b[^.;]*\b(?:flaw|sign|artifact|"
                       r"problem|glitch|issue|error)s?\b|\b(?:looks?|sounds?|seems?|appears?) (?:natural|normal|fine|"
                       r"real|clean|convincing|believable)\b|\bflawless\b|\bseamless\b|\bnothing (?:further|more|else)\b"
                       r"|\bpoint(?:s|ed)? to\b", re.IGNORECASE)
RULES = (("talks about its input", INPUT_LEAKS), ("talks about deciding", DECIDING), ("names a frame", FRAME_NUMBERS),
         ("has a claim number", CLAIM_NUMBERS), ("gives a time", TIMES), ("compares with another clip", OTHER_CLIP),
         ("says a fake looks fine", FAKE_FINE))
DASHES = re.compile(r"[\u2014\u2013]")  # em and en dashes: the rewriter writes without them
VERDICTS = ("AI-generated", "AI-altered")  # two facts (the part is generated, the voice altered): a rewrite keeps each
# A real part's verdict, in these words ("nothing in the sound is AI-generated" too): said by its reply and kept by
# its rewrite. The only place a real part may say natural or AI-generated, so the rules below read a real part
# without them.
REAL_WORDS = {"natural recording": re.compile(r"\bnatural recording\b", re.IGNORECASE),
              "nothing AI-generated": re.compile(r"\bnothing\b(?:\s+\w+){0,4}?\s+(?:is\s+|was\s+)?AI-generated\b",
                                                 re.IGNORECASE)}
_REAL_VERDICT = re.compile("|".join(rx.pattern for rx in REAL_WORDS.values()), re.IGNORECASE)
# A real part called made or changed by AI, outside its verdict.
FAKE_WORDS = re.compile(r"(?-i:\bAI\b)|\bsynthetic\w*|\bgenerat\w*|\bmanipulat\w*|\baltered\b|\balteration\w*|\bedited\b"
                        r"|\bdeepfakes?\b", re.IGNORECASE)  # "AI" in capitals only: French "j'ai" is not
# What a person says, quoted: in a real part read as content, never as a defect or a claim about the clip
QUOTED = re.compile(r'"[^"]*"|\u201c[^\u201d]*\u201d')


def rules_of(text: str, kind: str) -> list[tuple[str, str]]:
    """The rules a text breaks, as (rule, matched words). A real part is read without its verdict's words."""
    rules = RULES
    if kind == "real":  # "as the scene concludes" is an event: the clip ends, nothing is decided
        deciding = re.compile(DECIDING.pattern.replace(r"|\bconclude[sd]?\b|\bconcluding\b", ""), re.IGNORECASE)
        rules = tuple((name, deciding if rx is DECIDING else rx) for name, rx in RULES)
        text, rules = QUOTED.sub(" ", _REAL_VERDICT.sub(" ", text)), (*rules, ("calls a real part fake", FAKE_WORDS))
    return [(name, m.group(0)) for name, rx in rules if (m := rx.search(text))]

# Words that name a defect, by family. Synonyms within a family are free; a family no cited claim has is a new defect.
DEFECTS = {
    "blur": r"blur\w*|soft\w*|fuzz\w*|hazy|smear\w*|sharp\w*|crisp\w*|detail\w*|textur\w*|definition",
    "warp": r"warp\w*|wobbl\w*|distort\w*|melt\w*|rippl\w*|deform\w*|shape",
    "movement": r"(?:stays?|stayed|staying|sits?|sitting|stands?|standing|holds?|held|remains?|keeps?) still|stillness"
                r"|stiff\w*|rigid\w*|frozen|fixed|motionless|barely moves?|hardly moves?|expressions?",
    "flicker": r"flicker\w*|jitter\w*|shimmer\w*|flash\w*",
    "colour": r"colou?rs?|tones?|tint\w*|light\w*|shadow\w*",
    "edges": r"edges?|borders?|outlines?|boundar\w*|seams?|blend\w*",
    "sync": r"sync\w*|out of time",
    "sound": r"metallic|robotic|buzz\w*|echo\w*|pitch\w*|monoton\w*|tinny|muffled|hiss\w*|crackl\w*",
}
_DEFECT = {k: re.compile(rf"\b(?:{v})\b", re.IGNORECASE) for k, v in DEFECTS.items()}


def defects(text: str, part: str | None = None) -> set[str]:
    """The defect families a text names. Colour is a picture defect: in an audio part it is scene ("a red shirt",
    "a neutral tone")."""
    found = {k for k, rx in _DEFECT.items() if rx.search(text)}
    return found - {"colour"} if part == "audio" else found


def real_defects(text: str, part: str) -> set[str]:
    """The defect families a real part names. None in its audio: its words are what is said ("in the best shape",
    "the pitchers"), and the verifier judges a defect by meaning. In its video, quoted words are left out."""
    return set() if part == "audio" else defects(QUOTED.sub(" ", text), part)


def sign_words(meaning: str) -> list[str]:
    """The words a sign is said in: "the face looks clearly soft and blurry, without fine detail" ->
    ["looks clearly soft and blurry", "without fine detail"]."""
    return [w.strip() for w in meaning.lower().removeprefix("the face ").split(",")]


# In a real part these are scene ("framed photographs on a shelf"): nothing in it was made from a photo
REAL_SCENE_WORDS = ("photo", "photograph", "still image", "score")


def method_word(text: str, kind: str = "fake") -> str | None:
    """The first method word in a text (whole words, any case), or None."""
    words = [w for w in METHOD_WORDS if kind == "fake" or w not in REAL_SCENE_WORDS]
    return next((w for w in words if re.search(rf"\b{re.escape(w)}s?\b", text.lower())), None)


def check_reply(reply: Any, claims: list[Row], parts: list[str], lengths: dict[str, tuple[int, int]],
                kind: str = "fake") -> list[str]:
    """Every problem of a reply of one kind of part (claims.KINDS); [] if it passes."""
    if not isinstance(reply, dict):
        return ["not a JSON object"]
    problems = []
    by_id = {c["id"]: c for c in claims}
    for part in parts:
        sentences = reply.get(part)
        if not isinstance(sentences, list) or not sentences:
            problems.append(f"{part}: no sentences")
            continue
        lo, hi = lengths[part]
        if not lo <= len(sentences) <= hi:
            problems.append(f"{part}: {len(sentences)} sentences, asked {lo} to {hi}")
        cited_all: set[int] = set()
        for i, s in enumerate(sentences, 1):
            text = (s.get("text") or "").strip() if isinstance(s, dict) else ""
            cited = (s.get("claims") or []) if isinstance(s, dict) else []
            mine = [by_id[c] for c in cited if c in by_id and by_id[c]["part"] == part]
            if not text or not mine or len(mine) != len(cited):
                problems.append(f"{part} {i}: cites no claim of its part, or an unknown one ({cited})")
                continue
            cited_all |= {c["id"] for c in mine}
            new = real_defects(text, part) if kind == "real" else \
                defects(text, part) - set().union(*(defects(c["text"]) for c in mine))
            if all(c["kind"] == "scene" for c in mine):
                new -= {"colour"}
            if new:
                problems.append(f"{part} {i}: says {', '.join(sorted(new))}, which no cited claim does: {text!r}")
            if (word := method_word(text, kind)):
                problems.append(f"{part} {i}: names a method ({word}): {text!r}")
            for name, words in rules_of(text, kind):
                problems.append(f"{part} {i}: {name} ({words}): {text!r}")
        if kind == "real":
            said = " ".join((s.get("text") or "") for s in sentences if isinstance(s, dict))
            problems += [f"{part}: does not say {w!r}" for w, rx in REAL_WORDS.items() if not rx.search(said)]
        missing = [c["id"] for c in claims if c["part"] == part and c["kind"] in REQUIRED and c["id"] not in cited_all]
        if missing:
            problems.append(f"{part}: must-cite claims not cited: {missing}")
    return problems


def check_rewrite(rewrite: Any, reply: Row, claims: list[Row], parts: list[str], kind: str = "fake") -> list[str]:
    """Every problem of the rewriter's paragraphs against the checked sentences they rewrite; [] if they pass."""
    if not isinstance(rewrite, dict):
        return ["rewriter: not a JSON object"]
    problems = []
    for part in parts:
        text = rewrite.get(part)
        if not isinstance(text, str) or not text.strip():
            problems.append(f"rewriter: {part}: no paragraph")
            continue
        before = " ".join(s["text"] for s in reply[part])
        if (new := (real_defects(text, part) - real_defects(before, part)) if kind == "real"
                else defects(text, part) - defects(before)):
            problems.append(f"rewriter: {part}: says {', '.join(sorted(new))}, which the checked sentences do not")
        for c in claims:
            if c["part"] == part and c["kind"] == "sign":
                if not defects(c["text"]) <= defects(text):
                    problems.append(f"rewriter: {part}: drops a sign ({c['text']})")
                for words in sign_words(c["text"]):  # the flaw's own words, as the checked sentences said them
                    if words in before.lower() and words not in text.lower():
                        problems.append(f"rewriter: {part}: changes the words of a sign ({words})")
        kept = REAL_WORDS if kind == "real" else {v: re.compile(re.escape(v), re.IGNORECASE) for v in VERDICTS}
        for phrase, rx in kept.items():
            if rx.search(before) and not rx.search(text):
                problems.append(f"rewriter: {part}: no longer says {phrase}")
        if kind == "real" and not text.strip().endswith(REAL_SENTENCE[part]):  # the verdict, as written, ends it
            problems.append(f"rewriter: {part}: does not end with its verdict ({REAL_SENTENCE[part]})")
        if DASHES.search(text):
            problems.append(f"rewriter: {part}: has a dash")
        if (word := method_word(text, kind)):
            problems.append(f"rewriter: {part}: names a method ({word})")
        for name, words in rules_of(text, kind):
            problems.append(f"rewriter: {part}: {name} ({words})")
    return problems
