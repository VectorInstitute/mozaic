"""Tests for T4: facts and signs, selection, claims (fake and real parts), the checks, the annotator + verifier +
rewriter loop and the final rows. No videos, no API: every model is a fake."""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any

import pytest
from made_up import clip, pair_clips

from mozaic.annotation.t4_explanation import annotate as t4a
from mozaic.annotation.t4_explanation import build_questions as t4
from mozaic.annotation.t4_explanation.checks import rules as t4c
from mozaic.annotation.t4_explanation.prompts import annotator as t4p
from mozaic.annotation.t4_explanation.prompts import verifier as t4v
from mozaic.annotation.t4_explanation.facts.artifacts import CATEGORIES
from mozaic.annotation.t4_explanation.facts.claims import REAL, claims_of, parts_of
from mozaic.annotation.t4_explanation.models.client import Generation
from mozaic.annotation.t4_explanation.facts.metadata_facts import REGIONS, facts, signs
from mozaic.common import paths
from mozaic.common.io import read_jsonl
from mozaic.common.paths import qas_dir

Row = dict[str, Any]


@pytest.fixture
def release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Row]:
    """A made-up MOZAIC-Small: 40 test pairs, 10 per length and gender, each with all three generators,
    and 2 train pairs. The golden set is 32 clips: 4 per cell, 1 per length and gender."""
    root = tmp_path / "mozaic"
    monkeypatch.setenv("MOZAIC_ROOT", str(root))
    monkeypatch.setattr(paths, "_subset", "small")
    monkeypatch.setitem(t4.CONFIG["golden"], "size", 32)
    rows = []
    for i in range(40):
        length = "77f" if i < 20 else "153f"
        rows += [{**c, "gender": "fm"[i % 2]}
                 for c in pair_clips(f"id{i:05d}_aaa_00001", f"id{90000 + i}", "test", length, ["facefusion", "animate", "s2v"])]
    for i in range(2):
        rows += [{**c, "gender": "m"} for c in pair_clips(f"id{50 + i:05d}_bbb_00001", f"id{80000 + i}", "train", "77f", ["s2v"])]
    root.mkdir(parents=True)
    (root / "metadata_small.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


BLOCK = re.compile(r"(VIDEO|AUDIO) \(the \w+\), (\d+) to \d+ sentences:\n((?:\d+\. .*(?:\n|$))+)")


def good_reply(prompt: str, extra: str = "") -> Row:
    """A reply that passes the checks, from the claims in an annotator prompt: one sentence per claim, then
    scene sentences up to the fewest asked. extra is added to the first scene sentence."""
    reply = {}
    for part, lo, block in BLOCK.findall(prompt):
        claims = [(int(n), text) for n, text in re.findall(r"(\d+)\. (.*)", block)]
        scene = claims[-1][0]
        said = [{"text": text[0].upper() + text[1:] + ".", "claims": [n]} for n, text in claims[:-1]]
        while len(said) < int(lo):
            said.append({"text": f"A person talks to the camera in a room{extra if len(said) == len(claims) - 1 else ''}.",
                         "claims": [scene]})
        reply[part.lower()] = said
    return reply


def of_real(prompt: str) -> bool:
    """A call about real parts: its prompt or its checked sentences say "natural recording"; a fake part's never do."""
    return "natural recording" in prompt


class FakeAnnotator:
    """Gemini, faked. Replies in order: "GOOD" passes the checks ("HALL": the same, in a hall, not a room),
    "FLICKER" adds a defect no claim gives, an Exception is raised, any other string is returned as it is.
    A call about real parts takes its reply from `real` instead (GOOD when empty), and is kept in real_calls."""

    def __init__(self, replies: list[str | Exception], real: list[str | Exception] | None = None) -> None:
        self.replies, self.real = list(replies), list(real or [])
        self.calls: list[tuple[str, str]] = []
        self.real_calls: list[tuple[str, str]] = []

    def generate(self, clip_file: str, prompt: str) -> Generation:
        real = of_real(prompt)
        (self.real_calls if real else self.calls).append((clip_file, prompt))
        reply = (self.real.pop(0) if self.real else "GOOD") if real else self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if reply in ("GOOD", "HALL", "FLICKER"):
            text = json.dumps(good_reply(prompt, " while the face flickers" if reply == "FLICKER" else ""))
            reply = text.replace(" a room", " a hall") if reply == "HALL" else text
        return Generation(raw=reply, usage={"total_tokens": 900}, finish_reason="stop")


class FakeVerifier:
    """Claude, faked: every sentence follows, unless the next reply is "NO" (the first sentence adds something).
    The checks of a rewrite (REWRITE_PROMPT) take their replies from `rewrites` instead; a check of real parts from
    `real` (kept in real_prompts)."""

    def __init__(self, replies: list[str] | None = None, rewrites: list[str] | None = None,
                 real: list[str] | None = None) -> None:
        self.replies, self.rewrites, self.real = list(replies or []), list(rewrites or []), list(real or [])
        self.prompts: list[str] = []
        self.rewrite_prompts: list[str] = []
        self.real_prompts: list[str] = []

    def generate(self, prompt: str) -> Generation:
        of_rewrite = "did not see the clip" in prompt
        if of_real(prompt):
            self.real_prompts.append(prompt)
            replies = self.real
        else:
            (self.rewrite_prompts if of_rewrite else self.prompts).append(prompt)
            replies = self.rewrites if of_rewrite else self.replies
        n = len(re.findall(r"^S\d+\. ", prompt, re.MULTILINE))
        no = bool(replies) and replies.pop(0) == "NO"
        verdicts = {f"S{i}": {"follows": not (no and i == 1), "added": "a timing" if no and i == 1 else ""}
                    for i in range(1, n + 1)}
        return Generation(raw=json.dumps(verdicts), finish_reason="stop")


class FakeRewriter:
    """Claude rewriting, faked: each part's sentences joined, as one paragraph. A next reply "DASH" puts an em
    dash in it, "NUMBER" a claim number, "DROP" leaves out every sentence after the first two, "HALL" says a hall,
    not a room. A rewrite of real parts (its scene sentences only) takes its reply from `real` (kept in real_prompts)."""

    def __init__(self, replies: list[str] | None = None, real: list[str] | None = None) -> None:
        self.replies, self.real = list(replies or []), list(real or [])
        self.prompts: list[str] = []
        self.real_prompts: list[str] = []

    def generate(self, prompt: str) -> Generation:
        real = "AI-" not in prompt  # a fake part's sentences say AI-generated or AI-altered; a real part's verdict is not sent
        (self.real_prompts if real else self.prompts).append(prompt)
        replies = self.real if real else self.replies
        how = replies.pop(0) if replies else "OK"
        out = {}
        for part, block in re.findall(r"(VIDEO|AUDIO):\n(.*?)(?:\n\n|$)", prompt, re.DOTALL):
            lines = block.splitlines()[:2] if how == "DROP" else block.splitlines()
            text = " ".join(lines).replace(" a room", " a hall") if how == "HALL" else " ".join(lines)
            out[part.lower()] = text + {"DASH": " \u2014 seen here.", "NUMBER": " He talks [3]."}.get(how, "")
        return Generation(raw=json.dumps(out), usage={"total_tokens": 300}, finish_reason="stop")


def pair_clip(label: str, generator: str, length: str = "77f") -> Row:
    """Pair 0's clip of this label and generator (vocoded view)."""
    return next(c for c in t4.view().values() if c["label"] == label and c["video_generator"] == generator
                and c["pair_id"].startswith("id00000") and c["length"] == length)


# --------------------------------------------------------------------------- #
# Facts, selection and calls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "generator", "parts", "regions"),
    [("RVRA", "s2v", [], ["none"]), ("RVFA", "s2v", ["audio"], ["voice"]),
     ("FVRA", "facefusion", ["video"], ["inner face"]), ("FVRA", "animate", ["video"], ["whole frame"]),
     ("FVRA", "s2v", ["video"], ["whole frame"]), ("FVFA", "animate", ["video", "audio"], ["whole frame", "voice"])],
)
def test_facts_follow_the_label_and_the_generator(label: str, generator: str, parts: list, regions: list) -> None:
    f = facts(clip(label, generator, "main", "test", "id00001_aaa_00001", "id00002", "77f"))
    assert (f["fake_parts"], f["regions"]) == (parts, regions)
    assert set(f["regions"]) <= set(REGIONS)
    assert f["parts"]["video"]["fake"] == (label[0] == "F") and f["parts"]["audio"]["fake"] == (label[2] == "F")
    assert f["parts"]["video"]["method"] == (generator if label[0] == "F" else "real")


@pytest.mark.parametrize(
    ("generator", "scores", "artifacts"),
    [("animate", {"sharp": 0.6, "melt": 1.0, "flicker": 0.2}, ["unnatural blur"]),
     ("animate", {"sharp": 1.0, "melt": 1.5, "flicker": 0.9}, ["shape or structure distortion"]),  # flicker: no sign
     ("facefusion", {"sharp": 1.0, "melt": 0.7, "flicker": 0.2}, ["unnatural human movement"]),
     ("s2v", {"sharp": 0.6, "melt": 1.9, "flicker": 0.9}, ["unnatural blur"]),  # melt is off for S2V, as in the gate
     ("animate", {"sharp": 0.7, "melt": 1.25, "flicker": 0.4}, []),  # at the cutoff: no sign
     ("animate", {}, [])],                                             # no scores (a real clip): no sign
)
def test_signs_come_from_the_scores_past_their_cutoffs(generator: str, scores: Row, artifacts: list) -> None:
    c = {**clip("FVRA", generator, "vocoded", "test", "id00001_aaa_00001", "id00002", "77f"), **scores}
    assert [s["artifact"] for s in signs(c)] == artifacts
    assert facts(c)["parts"]["video"]["signs"] == signs(c)
    assert all(a in CATEGORIES["video"] for a in artifacts)


def test_vocoded_view_keeps_one_arm_per_label(release: list[Row]) -> None:
    kept = collections.Counter((c["label"], c["arm"]) for c in release if t4.in_view(c))
    assert set(kept) == {("RVRA", "vocoded"), ("RVFA", "main"), ("FVRA", "vocoded"), ("FVFA", "main")}


def test_select_writes_every_split_and_a_golden_set_drawn_from_test(release: list[Row]) -> None:
    paths = t4.select()
    assert [p.name for p in paths] == ["train.jsonl", "val.jsonl", "test.jsonl", "golden.jsonl"]
    assert all(p.parent == t4.build_dir() / "selection" for p in paths)
    sets = {p.stem: read_jsonl(p) for p in paths}
    in_view = [c for c in release if t4.in_view(c)]
    assert len(sets["test"]) == sum(c["split"] == "test" for c in in_view) and len(sets["train"]) == 8 and not sets["val"]
    golden = sets["golden"]
    assert {r["id"] for r in golden} <= {r["id"] for r in sets["test"]}
    assert collections.Counter(t4.cell(r) for r in golden) == {
        cell: 4 for cell in ("RVRA_real", "RVFA_real", "FVRA_facefusion", "FVRA_animate", "FVRA_s2v",
                             "FVFA_facefusion", "FVFA_animate", "FVFA_s2v")}
    gender = {c["file_name"]: c["gender"] for c in release}
    assert collections.Counter(gender[r["clips"][0]] for r in golden) == {"f": 16, "m": 16}
    assert collections.Counter(r["meta"]["frames"] for r in golden) == {77: 16, 153: 16}
    assert len({r["meta"]["pair_id"] for r in golden}) == 32
    r = golden[0]
    assert r["gt_answer"] is None and r["gt_target"]["evidence"] is None and r["options"] == list(REGIONS)
    assert list(r) == ["id", "task", "split", "clips", "question", "options", "gt_answer", "gt_label", "gt_target", "meta"]
    assert all(r["meta"]["arm"] == t4.CONFIG["arms"][r["gt_label"]] for r in golden) and "gender" not in r["meta"]


def test_select_again_is_the_same_and_never_swaps_the_clips(release: list[Row], monkeypatch: pytest.MonkeyPatch) -> None:
    first = [read_jsonl(p) for p in t4.select()]
    assert [read_jsonl(p) for p in t4.select()] == first  # same metadata and config: same clips
    monkeypatch.setitem(t4.CONFIG["golden"], "seed", 1)
    with pytest.raises(SystemExit, match="golden"):
        t4.select()
    assert read_jsonl(t4.selection_path("golden")) == first[-1]


# --------------------------------------------------------------------------- #
# Claims, prompt and checks
# --------------------------------------------------------------------------- #
def test_claims_give_each_fake_part_its_reasoning_in_order(release: list[Row]) -> None:
    c = claims_of(facts({**pair_clip("FVFA", "animate"), "sharp": 0.55}))
    assert [(x["id"], x["part"], x["kind"]) for x in c] == [
        (1, "video", "verdict"), (2, "video", "fact"), (3, "video", "sign"), (4, "video", "scene"),
        (5, "audio", "verdict"), (6, "audio", "fact"), (7, "audio", "scene")]
    assert c[1]["text"] == "the whole frame, person and background, is AI-generated, from the start of the clip to the end"
    assert c[2]["text"] == "the face looks clearly soft and blurry, without fine detail"  # 0.55 < 0.6: strong
    mild = claims_of(facts({**pair_clip("FVRA", "facefusion"), "sharp": 0.65}))
    assert mild[2]["text"] == "the face looks a little soft and blurry" and {x["part"] for x in mild} == {"video"}
    assert claims_of(facts(pair_clip("RVRA", "real"))) == []  # nothing fake: no claims, no call


@pytest.mark.parametrize(
    ("label", "generator", "real_parts"),
    [("RVRA", "real", ["video", "audio"]), ("RVFA", "real", ["video"]), ("FVRA", "animate", ["audio"]),
     ("FVFA", "animate", [])],
)
def test_real_claims_give_each_real_part_a_verdict_and_a_scene(release: list[Row], label: str, generator: str,
                                                               real_parts: list[str]) -> None:
    f = facts(pair_clip(label, generator))
    c = claims_of(f, "real")
    assert parts_of(f, "real") == real_parts and parts_of(f) == f["fake_parts"]
    assert [(x["part"], x["kind"]) for x in c] == [(p, k) for p in real_parts for k in ("verdict", "scene")]
    assert [x["id"] for x in c] == list(range(1, len(c) + 1))
    assert all(x["text"] == f"the {x['part']} is a natural recording, with nothing AI-generated"
               for x in c if x["kind"] == "verdict")
    assert claims_of(f) == claims_of(f, "fake")  # the fake claims are as before: saved replies stay valid


def test_real_prompt_asks_as_many_sentences_as_a_fake_part_with_no_sign(release: list[Row]) -> None:
    f = facts(pair_clip("RVRA", "real"))
    c = claims_of(f, "real")
    prompt = t4p.build_prompt(c, ["video", "audio"], "m", kind="real")
    assert prompt.startswith(t4p.REAL_PROMPT[:80]) and "natural recording, with nothing AI-generated" in prompt
    assert "VIDEO (the picture), 4 to 7 sentences:\n1. the video is a natural recording" in prompt
    assert "AUDIO (the sound), 4 to 7 sentences:\n3. the audio is a natural recording" in prompt
    no_sign = claims_of(facts({**pair_clip("FVRA", "animate"), "sharp": 1.0, "melt": 1.0}))
    assert t4p.lengths(no_sign, ["video"])["video"] == t4p.lengths(c, ["video"], "real")["video"] == (4, 7)
    question, _ = t4v.build_prompt(c, good_reply(prompt), ["video", "audio"], t4v.REAL_PROMPT)
    assert "is a natural recording" in question and "C1. the video is a natural recording" in question


def test_prompt_lists_the_claims_and_the_sentences_asked(release: list[Row]) -> None:
    c = claims_of(facts({**pair_clip("FVRA", "facefusion"), "sharp": 0.65}))
    prompt = t4p.build_prompt(c, ["video"], "f")
    assert "VIDEO (the picture), 5 to 8 sentences:\n1. the video is AI-generated\n" in prompt  # 3 to cite, + 2 to 5
    assert "3. the face looks a little soft and blurry\n4. what is on screen" in prompt and "AUDIO" not in prompt
    assert '{"video": [{"text": "<sentence>", "claims": [<numbers>]}, ...]}' in prompt
    assert "The person in the clip is a woman." in prompt and "a man." in t4p.build_prompt(c, ["video"], "m")
    question, texts = t4v.build_prompt(c, good_reply(prompt), ["video"])
    assert "C3. the face looks a little soft and blurry" in question and "S1. The video is AI-generated. (claims: C1)" in question
    assert len(texts) == 5 and "Clip" not in question  # the verifier gets text only


FAKE_VIDEO = claims_of(facts({**clip("FVRA", "facefusion", "vocoded", "test", "id00001_aaa_00001", "id00002", "77f"),
                              "sharp": 0.65}))
GOOD = good_reply(t4p.build_prompt(FAKE_VIDEO, ["video"], "m"))


def with_scene(text: str, cite: int = 4) -> Row:
    """GOOD with its last scene sentence replaced."""
    return {"video": [*GOOD["video"][:-1], {"text": text, "claims": [cite]}]}


@pytest.mark.parametrize(
    ("reply", "problem"),
    [(GOOD, None),
     (with_scene("A man in a blue shirt sits under warm light."), None),         # colour and lighting: scene only
     (with_scene("His eyes look at the camera as he speaks."), None),             # places are free
     (with_scene("Later the soft face shows most at the mouth.", 3), None),      # where a listed sign shows
     (with_scene("The face flickers as he turns."), "video 5: says flicker"),
     (with_scene("The lighting on the face shifts.", 3), "video 5: says colour"),  # colour is a defect outside the scene
     (with_scene("Overall, the face is soft.", 3), "video 5: talks about deciding (Overall)"),
     (with_scene("This face swap shows at the jaw.", 3), "video 5: names a method (face swap)"),
     (with_scene("Unlike the original video, the face is soft.", 3), "video 5: compares with another clip"),
     (with_scene("The voice sounds natural.", 3), "video 5: says a fake looks fine (sounds natural)"),
     (with_scene("As the claims say, he talks."), "video 5: talks about its input (claims)"),
     (with_scene("Around frame 4 he talks."), "video 5: names a frame (frame 4)"),
     (with_scene("He talks [4]."), "video 5: has a claim number ([4])"),
     (with_scene("He talks (2, 4)."), "video 5: has a claim number ((2, 4))"),
     (with_scene("A man in his 30s talks (in a room)."), None),                   # an age and a bracket are fine
     (with_scene("He turns at 00:04.2."), "video 5: gives a time (00:04.2)"),
     (with_scene("He turns after 4 seconds."), "video 5: gives a time (4 seconds)"),
     (with_scene("He turns near the end."), None),                                # a moment by meaning is fine
     (with_scene("He is still talking as the clip ends."), None),                  # "still" alone is not stillness
     (with_scene("His face stays still as he talks.", 3), "video 5: says movement"),
     (with_scene("He talks.", 99), "video 5: cites no claim of its part"),
     ({"video": GOOD["video"][:2] + GOOD["video"][3:]}, "video: 4 sentences, asked 5 to 8"),
     ({"video": [s for s in GOOD["video"] if s["claims"] != [3]] + [GOOD["video"][-1]]}, "video: must-cite claims not cited: [3]"),
     ({}, "video: no sentences"),
     ("not json", "not a JSON object")],
)
def test_reply_check(reply: Any, problem: str | None) -> None:
    got = t4c.check_reply(reply, FAKE_VIDEO, ["video"], t4p.lengths(FAKE_VIDEO, ["video"]))
    assert (got == []) if problem is None else any(p.startswith(problem) for p in got), got


REAL_VIDEO = claims_of(facts(clip("RVFA", "real", "main", "test", "id00001_aaa_00001", "id00002", "77f")), "real")
REAL_GOOD = good_reply(t4p.build_prompt(REAL_VIDEO, ["video"], "m", kind="real"))


def real_with(text: str, cite: int = 2) -> Row:
    """REAL_GOOD with its last scene sentence replaced."""
    return {"video": [*REAL_GOOD["video"][:-1], {"text": text, "claims": [cite]}]}


@pytest.mark.parametrize(
    ("reply", "problem"),
    [(REAL_GOOD, None),
     (real_with("A man in a blue shirt sits under warm light."), None),           # colour and lighting: scene only
     (real_with("He speaks in a bright room, and nothing AI-generated is shown.", 1), None),  # the verdict's words
     (real_with("His face looks blurry as he turns."), "video 4: says blur"),      # a real part has no sign
     (real_with("His face is AI-generated."), "video 4: calls a real part fake (AI)"),
     (real_with("The picture was generated by a model."), "video 4: calls a real part fake (generated)"),
     (real_with("It is genuine footage of a man."), "video 4: compares with another clip"),
     (real_with("It looks natural."), "video 4: says a fake looks fine"),          # only the verdict says natural
     (real_with("The vocoder keeps his voice."), "video 4: names a method (vocoder)"),
     (real_with("His facial expressions shift as he talks."), "video 4: says movement"),  # quality, not content
     (real_with("Framed photographs hang behind him."), None),                      # scene in a real part
     (real_with("He keeps talking as the scene concludes."), None),                 # the clip ends: an event
     (real_with("Overall, he talks."), "video 4: talks about deciding (Overall)"),
     (real_with('He says "we generated a profit".'), None),                       # quoted: what he says
     (real_with("His face looks AI-made."), "video 4: calls a real part fake (AI)"),
     ({"video": [{"text": "A man talks in a room.", "claims": [2]}] * 3 + [
         {"text": "The video is real.", "claims": [1]}]}, "video: does not say 'natural recording'"),
     ({"video": REAL_GOOD["video"][1:] + [REAL_GOOD["video"][1]]}, "video: must-cite claims not cited: [1]")],
)
def test_real_reply_check(reply: Row, problem: str | None) -> None:
    got = t4c.check_reply(reply, REAL_VIDEO, ["video"], t4p.lengths(REAL_VIDEO, ["video"], "real"), "real")
    assert (got == []) if problem is None else any(p.startswith(problem) for p in got), got


def test_a_real_audio_part_says_what_is_said() -> None:
    audio = claims_of(facts(clip("FVRA", "animate", "vocoded", "test", "id00001_aaa_00001", "id00002", "77f")), "real")
    good = good_reply(t4p.build_prompt(audio, ["audio"], "f", kind="real"))
    said = lambda text: {"audio": [*good["audio"][:-1], {"text": text, "claims": [2]}]}
    check = lambda reply: t4c.check_reply(reply, audio, ["audio"], t4p.lengths(audio, ["audio"], "real"), "real")
    for words in ("She says she is in the best shape.", "He talks about the pitchers.", "He adds that they're sharpening.",
                  "Elle dit, donc j'ai fait ces deux spectacles."):
        assert check(said(words)) == [], words  # the speaker's words are content, not a defect or "AI"
    assert check(said("Her voice is AI-generated.")) != []
    from mozaic.annotation.t4_explanation.prompts import rewriter as t4w
    assert t4w.REAL_NOTE in t4w.build_prompt(good, ["audio"], "real") and t4w.REAL_NOTE not in t4w.build_prompt(good, ["audio"])


def test_real_rewrite_check_keeps_the_verdict_words() -> None:
    sentences = [x["text"] for x in REAL_GOOD["video"]]
    joined = " ".join(sentences[1:] + sentences[:1])  # the scene, then the verdict
    check = lambda text: t4c.check_rewrite({"video": text}, REAL_GOOD, REAL_VIDEO, ["video"], "real")
    assert check(joined) == []
    assert any(p.startswith("rewriter: video: does not end with its verdict")
               for p in check(joined.replace("with nothing AI-generated", "and nothing in it is AI-generated")))
    assert any(p.startswith("rewriter: video: no longer says nothing AI-generated")
               for p in check(joined.replace(", with nothing AI-generated", "")))
    assert any(p.startswith("rewriter: video: calls a real part fake") for p in check(sentences[1] + " His face is synthetic. " + sentences[0]))
    assert t4c.check_reply(GOOD, FAKE_VIDEO, ["video"], t4p.lengths(FAKE_VIDEO, ["video"])) == []  # fake: as before


def test_a_real_parts_verdict_is_not_given_to_the_rewriter_and_ends_its_paragraph() -> None:
    from mozaic.annotation.t4_explanation.facts.claims import REAL_SENTENCE

    class Rewords(FakeRewriter):  # rewrites as asked, and would reword a verdict it were given
        def generate(self, prompt: str):
            gen = super().generate(prompt)
            return Generation(raw=gen.raw.replace("natural recording", "recording that is natural"),
                              usage=gen.usage, finish_reason="stop")

    rewriter = Rewords()
    out = t4a.rewrite(REAL_GOOD, REAL_VIDEO, ["video"], rewriter, FakeVerifier(), "real")
    assert rewriter.real_prompts and "natural recording" not in rewriter.real_prompts[0]  # scene sentences only
    assert out["text"]["video"].endswith(REAL_SENTENCE["video"]) and out["problems"] == []
    assert REAL_SENTENCE["video"] == "The video is a natural recording, with nothing AI-generated."

    part = {"video": [*REAL_GOOD["video"][1:], {"text": "The part is a natural recording, with nothing AI-generated.",
                                                "claims": [1]}]}  # the annotator's own wording of the verdict
    verifier = FakeVerifier()
    t4a.rewrite(part, REAL_VIDEO, ["video"], FakeRewriter(), verifier, "real")
    checked = verifier.real_prompts[-1].split("Rewrite:")[0]
    assert REAL_SENTENCE["video"] in checked and "The part is" not in checked  # checked against the verdict as written


def test_a_reply_is_read_even_with_reasoning_before_its_json() -> None:
    from mozaic.annotation.t4_explanation.models.client import parse_json

    assert parse_json('{"S1": 1}') == {"S1": 1} and parse_json('```json\n{"S1": 1}\n```') == {"S1": 1}
    assert parse_json('S1 follows from C3.\n\n```json\n{"S1": {"follows": true}}\n```') == {"S1": {"follows": True}}
    assert parse_json('Here it is: {"S1": 2} done') == {"S1": 2} and parse_json("no json") is None


def test_verifier_reply_is_one_verdict_per_sentence() -> None:
    texts = ["A.", "B."]
    assert t4v.problems({"S1": {"follows": True}, "S2": {"follows": True}}, texts) == []
    assert t4v.problems({"S1": {"follows": True}, "S2": {"follows": False, "added": "a timing"}}, texts) == [
        "verifier: S2 adds 'a timing': 'B.'"]
    assert t4v.problems({"S1": {"follows": True}}, texts) == ["verifier: not one verdict per sentence"]
    assert t4v.problems(None, texts) == ["verifier: not one verdict per sentence"]


REPLY = {"video": GOOD["video"]}
JOINED = " ".join(x["text"] for x in GOOD["video"])


@pytest.mark.parametrize(
    ("text", "problem"),
    [(JOINED, None),
     (JOINED.replace(". ", ", and ", 1), None),                                     # rewording is free
     (JOINED + " The face also flickers.", "rewriter: video: says flicker"),
     (JOINED.replace("The face looks a little soft and blurry", "He speaks"), "rewriter: video: drops a sign"),
     (JOINED.replace("AI-generated", "made"), "rewriter: video: no longer says AI-generated"),
     (JOINED + " He talks \u2014 calmly.", "rewriter: video: has a dash"),
     (JOINED + " Overall, it is fake.", "rewriter: video: talks about deciding"),
     (JOINED + " He talks [3].", "rewriter: video: has a claim number"),
     (JOINED + " He is still talking at the end.", None),
     (JOINED.replace("a little soft", "clearly soft"), "rewriter: video: changes the words of a sign (looks a little soft and blurry)"),
     ("", "rewriter: video: no paragraph")],
)
def test_rewrite_check(text: str, problem: str | None) -> None:
    got = t4c.check_rewrite({"video": text}, REPLY, FAKE_VIDEO, ["video"])
    assert (got == []) if problem is None else any(p.startswith(problem) for p in got), got


def test_colour_words_in_an_audio_part_are_scene() -> None:
    assert t4c.defects("a man in a red shirt speaks in a neutral tone", "audio") == set()
    assert t4c.defects("the lighting on the face shifts", "video") == {"colour"}


def test_a_rewrite_is_verified_against_the_checked_sentences_as_unseen_clip() -> None:
    question, texts = t4v.build_rewrite_prompt(REPLY, {"video": JOINED}, ["video"])
    assert "did not see the clip" in question and "posture" in question
    assert f"C1. {GOOD['video'][0]['text']}" in question and len(texts) == len(GOOD["video"])


def test_rewriter_gets_the_checked_sentences_only() -> None:
    from mozaic.annotation.t4_explanation.prompts import rewriter as t4w

    prompt = t4w.build_prompt(REPLY, ["video"])
    assert prompt.startswith("VIDEO:\n" + GOOD["video"][0]["text"]) and '{"video": "<paragraph>"}' in prompt
    assert "claims" not in prompt.lower() and "AUDIO" not in prompt
    assert t4c.check_rewrite("not json", REPLY, FAKE_VIDEO, ["video"]) == ["rewriter: not a JSON object"]


# --------------------------------------------------------------------------- #
# Annotator + verifier + rewriter loop
# --------------------------------------------------------------------------- #
def saved_of(kind: str = "fake") -> dict[str, Row]:
    """The saved replies of one kind by clip, the last row of each."""
    return {r["clip"]: r for r in read_jsonl(t4.build_dir() / "replies.jsonl") if t4.kind_of(r) == kind}


def fake_clips(n: int) -> list[str]:
    """The first n golden clips with a fake part, in annotation order."""
    return [r["clips"][0] for r in t4.annotation_order(golden_only=True) if r["gt_target"]["fake_parts"]][:n]


def test_annotate_makes_one_call_per_kind_of_part_saves_each_and_resumes(release: list[Row]) -> None:
    t4.select()
    rows = t4.annotation_order(golden_only=True)[:6]
    fake = [r["clips"][0] for r in rows if r["gt_target"]["fake_parts"]]
    real = [r["clips"][0] for r in rows if r["gt_label"] != "FVFA"]  # a clip with a real part
    annotator = FakeAnnotator(["GOOD"] * len(fake))
    verifier = FakeVerifier()
    path = t4a.annotate(golden_only=True, limit=6, annotator=annotator, verifier=verifier, rewriter=FakeRewriter())
    saved = read_jsonl(path)
    assert path == t4.build_dir() / "replies.jsonl" and len(saved) == len(fake) + len(real)
    assert [r["clip"] for r in saved if r["kind"] == "fake"] == fake  # no fake call for RVRA
    assert [r["clip"] for r in saved if r["kind"] == "real"] == real  # no real call for FVFA
    assert [f for f, _ in annotator.calls] == fake and len(verifier.prompts) == len(fake)
    assert [f for f, _ in annotator.real_calls] == real and len(verifier.real_prompts) == 2 * len(real)  # + rewrites
    assert all(r["ok"] and len(r["rounds"]) == 1 and r["rounds"][0]["verifier"] for r in saved)
    annotator = FakeAnnotator([])
    t4a.annotate(golden_only=True, limit=6, annotator=annotator, verifier=FakeVerifier(), rewriter=FakeRewriter())  # all saved: no call
    assert annotator.calls == [] and annotator.real_calls == []


def test_a_failed_round_is_asked_again_with_its_problems_and_a_clip_failing_every_round_is_saved_failed(
        release: list[Row]) -> None:
    t4.select()
    first, second, third = fake_clips(3)
    # clip 1: the checks fail round 1 (a defect no claim gives), round 2 passes; the verifier is not asked in round 1
    # clip 2: round 1 passes the checks, the verifier fails it; round 2 passes
    # clip 3: fails both rounds: saved as failed
    annotator = FakeAnnotator(["FLICKER", "GOOD", "GOOD", "GOOD", "FLICKER", "FLICKER"])
    verifier = FakeVerifier(["OK", "NO", "OK"])
    limit = [r["clips"][0] for r in t4.annotation_order(True)].index(third) + 1
    t4a.annotate(golden_only=True, limit=limit, annotator=annotator, verifier=verifier, rewriter=FakeRewriter())
    saved = saved_of()
    assert [len(saved[f]["rounds"]) for f in (first, second, third)] == [2, 2, 2]
    assert [saved[f]["ok"] for f in (first, second, third)] == [True, True, False]
    assert "says flicker" in saved[first]["rounds"][0]["problems"][0] and "verifier" not in saved[first]["rounds"][0]
    assert saved[second]["rounds"][0]["problems"][0].startswith("verifier: S1 adds 'a timing'")
    assert "Your last reply failed these checks" in annotator.calls[1][1] and "says flicker" in annotator.calls[1][1]
    assert len(verifier.prompts) == 3  # rounds that failed the checks never reach the verifier

    annotator = FakeAnnotator(["GOOD"])
    t4a.annotate(golden_only=True, limit=limit, retry_failed=True, annotator=annotator, verifier=FakeVerifier(), rewriter=FakeRewriter())
    assert [f for f, _ in annotator.calls] == [third] and annotator.real_calls == []  # only the failed one, round 1
    saved = saved_of()
    assert saved[third]["ok"] and len(saved[third]["rounds"]) == 1


def test_annotate_does_not_save_a_failed_call_and_stops_after_failures_in_a_row(
        release: list[Row], caplog: pytest.LogCaptureFixture) -> None:
    t4.select()
    down = [RuntimeError("down")] * 99
    annotator = FakeAnnotator([RuntimeError("endpoint down"), "GOOD", *down], real=down)  # every real call fails too
    path = t4a.annotate(golden_only=True, annotator=annotator, verifier=FakeVerifier(), rewriter=FakeRewriter())
    assert [r["clip"] for r in read_jsonl(path)] == [annotator.calls[1][0]]  # the one call that worked
    assert "STOPPED after 3 failures in a row" in caplog.text


def test_annotate_refuses_other_settings_including_the_cutoffs(release: list[Row], monkeypatch: pytest.MonkeyPatch) -> None:
    from mozaic.annotation.t4_explanation.facts import metadata_facts

    t4.select()
    t4a.annotate(golden_only=True, limit=1, annotator=FakeAnnotator(["GOOD"]), verifier=FakeVerifier(), rewriter=FakeRewriter())
    for section in ("annotator", "verifier", "rewriter"):
        version = t4.CONFIG[section]["model_version"]
        monkeypatch.setitem(t4.CONFIG[section], "model_version", "another-model")
        with pytest.raises(SystemExit, match=section):
            t4a.annotate(golden_only=True, limit=8, annotator=FakeAnnotator(["GOOD"] * 8), verifier=FakeVerifier(), rewriter=FakeRewriter())
        monkeypatch.setitem(t4.CONFIG[section], "model_version", version)
    monkeypatch.setattr(metadata_facts, "SIGNS", metadata_facts.SIGNS[:1])
    with pytest.raises(SystemExit, match="facts_sha256"):
        t4a.annotate(golden_only=True, limit=8, annotator=FakeAnnotator(["GOOD"] * 8), verifier=FakeVerifier(), rewriter=FakeRewriter())


def test_settings_a_folder_lacks_are_added_once_and_a_changed_one_is_refused(release: list[Row]) -> None:
    t4.select()
    path = t4.build_dir() / "settings.json"
    t4a.annotate(golden_only=True, limit=1, annotator=FakeAnnotator(["GOOD"]), verifier=FakeVerifier(), rewriter=FakeRewriter())
    full = json.loads(path.read_text())
    old = {k: v for k, v in full.items() if not k.startswith("real_")}  # a folder annotated before the real parts
    path.write_text(json.dumps(old))
    t4a.annotate(golden_only=True, limit=1, annotator=FakeAnnotator([]), verifier=FakeVerifier(), rewriter=FakeRewriter())
    assert json.loads(path.read_text()) == full  # added once, the rest as it was
    path.write_text(json.dumps({**full, "real_claims_sha256": "another"}))
    with pytest.raises(SystemExit, match="real_claims_sha256"):
        t4a.annotate(golden_only=True, limit=1, annotator=FakeAnnotator([]), verifier=FakeVerifier(), rewriter=FakeRewriter())


def test_rows_saved_before_real_parts_read_as_fake_and_a_clip_waits_for_its_real_reply(release: list[Row]) -> None:
    t4.select()
    order = [r["clips"][0] for r in t4.annotation_order(True)]
    clip_file = next(f for f in order if "/FVRA/" in f)  # a fake video and a real audio: two calls
    limit = order.index(clip_file) + 1
    run = lambda annotator: t4a.annotate(golden_only=True, limit=limit, annotator=annotator, verifier=FakeVerifier(),
                                         rewriter=FakeRewriter())
    path = run(FakeAnnotator(["GOOD"] * limit))
    fake_only = [{k: v for k, v in r.items() if k != "kind"} for r in read_jsonl(path) if r["kind"] == "fake"]
    path.write_text("".join(json.dumps(r) + "\n" for r in fake_only))  # as saved before real parts were explained
    assert all(t4.kind_of(r) == "fake" for r in read_jsonl(path))
    golden = lambda: {r["clips"][0] for r in read_jsonl(t4.build_final()[-1])}
    assert clip_file not in golden()  # its real part has no reply yet: left out, never the old fixed text
    annotator = FakeAnnotator([])
    run(annotator)
    assert annotator.calls == [] and clip_file in {f for f, _ in annotator.real_calls}  # only the real calls left
    assert clip_file in golden()


def test_out_runs_every_step_in_another_folder(release: list[Row], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t4, "_out", None)  # put back after the test
    out = tmp_path / "trial"
    t4.use_out(str(out))
    assert t4.build_dir() == out / "build" and t4.final_path("golden") == out / "t4_golden.jsonl"
    t4.select()
    path = t4a.annotate(golden_only=True, limit=8, annotator=FakeAnnotator(["GOOD"] * 8), verifier=FakeVerifier(), rewriter=FakeRewriter())
    assert path == out / "build" / "replies.jsonl" and read_jsonl(path)
    assert not qas_dir("t4").exists()  # the real T4 folder is not touched




def test_annotate_goes_golden_first_and_each_clip_once(release: list[Row]) -> None:
    t4.select()
    golden = {r["clips"][0] for r in read_jsonl(t4.selection_path("golden"))}
    order = [r["clips"][0] for r in t4.annotation_order(golden_only=False)]
    assert set(order[:len(golden)]) == golden and len(order) == len(set(order))
    t4.selection_path("golden").unlink()
    with pytest.raises(SystemExit, match="run --select first"):
        t4.annotation_order(golden_only=True)


# --------------------------------------------------------------------------- #
# Final rows and stats
# --------------------------------------------------------------------------- #
def test_build_final_writes_one_row_per_clip_with_the_t1_t3_keys(release: list[Row]) -> None:
    t4.select()
    golden = t4.annotation_order(golden_only=True)
    n_fake = sum(bool(r["gt_target"]["fake_parts"]) for r in golden)
    t4a.annotate(golden_only=True, annotator=FakeAnnotator(["FLICKER", "FLICKER"] + ["GOOD"] * n_fake),
                 verifier=FakeVerifier(), rewriter=FakeRewriter())  # the first fake clip fails both rounds: left out
    paths = t4.build_final()
    assert [p.name for p in paths] == ["t4_train.jsonl", "t4_val.jsonl", "t4_test.jsonl", "t4_golden.jsonl"]
    assert all(p.parent == qas_dir("t4") for p in paths) and t4.build_dir() == qas_dir("t4") / "build"
    rows = read_jsonl(t4.final_path("golden"))
    assert len(rows) == len(golden) - 1
    test = {r["id"]: r for r in read_jsonl(t4.final_path("test"))}
    assert all(test[r["id"]] == r for r in rows)  # explained once: the same row in golden and in test
    real = saved_of("real")
    for r in rows:
        assert list(r) == ["id", "task", "split", "clips", "question", "options", "gt_answer", "gt_label", "gt_target", "meta"]
        target = r["gt_target"]
        assert set(target) == {"fake_parts", "regions", "parts", "sections", "evidence"}
        assert r["gt_answer"] == " ".join(target["sections"].values()) and list(target["sections"]) == ["video", "audio"]
        for part in ("video", "audio"):
            if part not in target["fake_parts"]:  # a real part: its own reply's rewrite, never the old fixed text
                assert target["sections"][part] == real[r["clips"][0]]["rewrite"]["text"][part] != REAL[part]
                assert "natural recording" in target["sections"][part]
        assert all(i["modality"] in target["fake_parts"] for i in target["evidence"])
        assert target["evidence"] == [{"modality": "video", "artifact": s["artifact"], "meaning": s["meaning"]}
                                      for s in target["parts"]["video"].get("signs") or []]  # signs, not the reply
        assert r["meta"]["annotator"] == t4.CONFIG["annotator"]["model_version"]
        assert r["meta"]["verifier"] == t4.CONFIG["verifier"]["model_version"] and r["meta"]["human_checked"] is True
    assert any(not r["gt_target"]["fake_parts"] for r in rows)  # RVRA: in the files, from its real call
    assert all(r["meta"]["human_checked"] is (r["id"] in {x["id"] for x in rows}) for r in test.values())  # golden only


def test_build_rechecks_the_rewrite_with_the_current_rules_and_a_retry_asks_only_the_rewriter(
        release: list[Row], monkeypatch: pytest.MonkeyPatch) -> None:
    t4.select()
    t4a.annotate(golden_only=True, limit=8, annotator=FakeAnnotator(["GOOD"] * 8), verifier=FakeVerifier(), rewriter=FakeRewriter())
    before = len(read_jsonl(t4.build_final()[-1]))
    monkeypatch.setattr(t4c, "RULES", (*t4c.RULES, ("names a room", re.compile(r"\broom\b"))))  # a stricter rule
    assert len(read_jsonl(t4.build_final()[-1])) < before  # no new call to apply it: the rewrites fail it now
    annotator, rewriter = FakeAnnotator([]), FakeRewriter(["HALL"] * 8, real=["HALL"] * 8)
    t4a.annotate(golden_only=True, limit=8, retry_failed=True, annotator=annotator, verifier=FakeVerifier(), rewriter=rewriter)
    assert annotator.calls == annotator.real_calls == [] and rewriter.prompts and rewriter.real_prompts  # only rewriters
    assert len(read_jsonl(t4.build_final()[-1])) == before

def test_a_failed_rewrite_is_left_out_and_asked_again_alone(release: list[Row]) -> None:
    t4.select()
    rewriter = FakeRewriter(["OK", "DASH", "NUMBER", "OK"])
    verifier = FakeVerifier(rewrites=["OK", "NO"])  # the 2nd rewrite it sees (the 4th clip) adds something
    t4a.annotate(golden_only=True, limit=8, annotator=FakeAnnotator(["GOOD"] * 8), verifier=verifier, rewriter=rewriter)
    saved = list(saved_of().values())
    assert len(rewriter.prompts) == len(saved) and all(r["ok"] for r in saved)
    assert [bool(r["rewrite"]["problems"]) for r in saved[:4]] == [False, True, True, True]
    assert len(verifier.rewrite_prompts) == len(saved) - 2  # a rewrite failing the code checks never reaches it
    assert saved[3]["rewrite"]["problems"][0].startswith("rewriter: verifier: S1 adds")
    rows = {r["clips"][0]: r for r in read_jsonl(t4.build_final()[-1])}
    assert saved[0]["clip"] in rows and not {r["clip"] for r in saved[1:4]} & rows.keys()  # failed rewrites: left out
    first = saved[0]
    part = first["claims"][0]["part"]
    assert rows[first["clip"]]["gt_target"]["sections"][part] == first["rewrite"]["text"][part]
    assert all(r["meta"]["rewriter"] == t4.CONFIG["rewriter"]["model_version"] for r in rows.values())

    annotator, rewriter = FakeAnnotator([]), FakeRewriter()
    t4a.annotate(golden_only=True, limit=8, retry_failed=True, annotator=annotator, verifier=FakeVerifier(), rewriter=rewriter)
    assert annotator.calls == [] and len(rewriter.prompts) == 3  # only the rewriter is asked again
    again = saved_of()
    assert all(not again[r["clip"]]["rewrite"]["problems"] and len(again[r["clip"]]["rewrites_failed"]) == 1
               for r in saved[1:4])
    assert {r["clip"] for r in saved} <= {r["clips"][0] for r in read_jsonl(t4.build_final()[-1])}
    s = json.loads((t4.build_dir() / "stats.json").read_text())["rewrites"]
    assert s["passed"] == len(saved) + len(saved_of("real")) and s["failed"] == 0 and s["fixed_by_a_retry"] == 3
    assert s["attempts_failed_by"] == {"checks": 2, "verifier": 1, "cut off": 0}


def test_after_rewrite_tries_failed_rewrites_the_annotator_is_asked_again(release: list[Row]) -> None:
    t4.select()
    clip = fake_clips(1)[0]
    limit = [r["clips"][0] for r in t4.annotation_order(True)].index(clip) + 1
    t4a.annotate(golden_only=True, limit=limit, annotator=FakeAnnotator(["GOOD"]), verifier=FakeVerifier(),
                 rewriter=FakeRewriter(["DASH"]))                                   # rewrite 1 fails
    for _ in range(t4.CONFIG["rewrite_tries"] - 1):                                 # rewrites 2 and 3 fail
        annotator = FakeAnnotator([])
        t4a.annotate(golden_only=True, limit=limit, retry_failed=True, annotator=annotator, verifier=FakeVerifier(),
                     rewriter=FakeRewriter(["DASH"]))
        assert annotator.calls == []                                                # only the rewriter
    annotator = FakeAnnotator(["GOOD"])
    t4a.annotate(golden_only=True, limit=limit, retry_failed=True, annotator=annotator, verifier=FakeVerifier(),
                 rewriter=FakeRewriter())
    assert [f for f, _ in annotator.calls] == [clip]                                 # stuck: new sentences
    saved = saved_of()[clip]
    assert saved["ok"] and not saved["rewrite"]["problems"] and saved["rewrites_failed"] == []
    dropped = read_jsonl(t4.build_dir() / "replies_dropped.jsonl")
    assert len(dropped[-1]["rewrites_failed"]) == t4.CONFIG["rewrite_tries"] - 1     # the stuck row, kept for the record


def test_stats_count_what_was_dropped_and_what_a_retry_fixed(release: list[Row]) -> None:
    t4.select()
    first, second, third = fake_clips(3)
    limit = [r["clips"][0] for r in t4.annotation_order(True)].index(third) + 1
    t4a.annotate(golden_only=True, limit=limit, annotator=FakeAnnotator(["FLICKER", "GOOD", "GOOD", "GOOD", "FLICKER", "FLICKER"]),
                 verifier=FakeVerifier(["OK", "NO", "OK"]), rewriter=FakeRewriter())
    s = json.loads((t4.build_dir() / "stats.json").read_text())
    n = len(saved_of("real"))  # real calls, each passing in round 1
    assert s["by_kind"] == {"fake": {"explained": 3, "passed": 2, "failed": 1},
                            "real": {"explained": n, "passed": n, "failed": 0}}
    assert s["clips"] == {"explained": 3 + n, "passed": 2 + n, "failed": 1}
    assert s["passed_in_round"] == {"1": n, "2": 2}
    assert s["rounds_failed_by"] == {"checks": 3, "verifier": 1, "cut off": 0}
    assert s["problems"]["says flicker"] == 3 and s["problems"]["verifier: adds"] == 1
    assert s["calls"] == {"annotator": 6 + n, "verifier": 3 + n, "rewriter": 2 + n, "rewrite verifier": 2 + n}
    assert s["rewrites"]["passed"] == 2 + n and s["rewrites"]["failed"] == 0


def test_categories_are_one_closed_list_per_part() -> None:
    assert list(CATEGORIES) == ["video", "audio"] and (len(CATEGORIES["video"]), len(CATEGORIES["audio"])) == (8, 3)
    names = [c for cats in CATEGORIES.values() for c in cats]
    assert len(names) == len(set(names)) and all(c == c.strip().lower() for c in names)


# --------------------------------------------------------------------------- #
# Inference and scores (the evaluated model sees only the clip)
# --------------------------------------------------------------------------- #
def test_t4_prompt_shows_no_facts_and_parse_checks_the_reply() -> None:
    from mozaic.evaluation.prompts import t4_prompts as p

    row = t4.make_row(clip("FVFA", "facefusion", "main", "test", "id00001_aaa_00001", "id00002", "77f"))
    prompt = p.build_prompt(row, 32)
    assert "32 frames" in prompt and "FAKE" not in prompt and "Reference" not in prompt and "- inner face:" in prompt
    assert "whole person" not in prompt and "- none:" not in prompt and '{"video": "<3 to 5 sentences>", "audio": "<3 to 5 sentences>"}' in prompt
    assert prompt.index("- inner face:") < prompt.index("- voice:")  # options' order
    ok = p.parse('```json\n{"video": " The video is AI-generated. ", "audio": "Natural."}\n```', row["options"])
    assert ok == {"video": "The video is AI-generated.", "audio": "Natural."}
    assert p.parse('{"video": "x"}', row["options"]) is None  # audio missing
    assert p.parse('{"video": "x", "audio": " "}', row["options"]) is None  # audio empty
    assert p.parse('{"video": ["x"], "audio": "y"}', row["options"]) is None
    assert p.parse("not json", row["options"]) is None


def test_t4_parse_reads_the_object_wherever_it_is_and_never_makes_a_part_up() -> None:
    from mozaic.evaluation.prompts import t4_prompts as p

    opts, want = list(REGIONS), {"video": "V.", "audio": "A."}
    assert p.parse('{"video": "V.", "audio": "A."}]', opts) == want  # a stray "]" after it (Uni-MoE)
    assert p.parse('Here it is: {"video": "V.", "audio": "A."} Hope it helps.', opts) == want
    assert p.parse('Sure.\n```json\n{"video": "V.", "audio": "A."}\n```\nDone.', opts) == want
    assert p.parse('{"Video": "V.", "AUDIO": "A."}', opts) == want  # keys in any case
    assert p.parse('{"video": "V.\nmore", "audio": "A."}', opts) == {"video": "V.\nmore", "audio": "A."}
    assert p.parse('{"note": {"x": 1}, "video": "V.", "audio": "A."}', opts) == want  # nested object before
    assert p.parse('{"video": "V."}, {"audio": "A."}]', opts) == want  # one object per part (Uni-MoE)
    assert p.parse('{"video": "V.", "audio": "A."', opts) == want  # ends right before its closing brace
    assert p.parse('{"video": "V.", "audio": "A."} {"video": "other"}', opts) == want  # the first value wins
    assert p.parse('{"video": "V. The audio is natural."}', opts) is None  # no audio part: not made up
    assert p.parse('{"video": "{\'start_time\': \'00:00\'}, "audio": "{\'a\': 1}]', opts) is None  # broken
    assert p.parse('{"video": "V.", "audio": "A', opts) is None  # cut off inside a string
    assert p.parse('```json\n{\n  "video": "V.",\n  "audio": "A.",\n}\n```', opts) == want  # comma before "}"
    assert p.parse('{"video": "V.", "audio": "A.",\n  }', opts) == want
    said = '{"video": "V.", "audio": "He says "next week" and "next month." Natural."}'
    assert p.parse(said, opts) == {"video": "V.", "audio": 'He says "next week" and "next month." Natural.'}
    assert p.parse('{"video": "A "Celebrity Apprentice" wall.", "audio": "A."}', opts) == {
        "video": 'A "Celebrity Apprentice" wall.', "audio": "A."}
    assert p.parse('{"video": "V.", "audio": "He said \\"hi\\"."}', opts) == {"video": "V.", "audio": 'He said "hi".'}
    assert p.parse('{"video": "V.", "inner face: only the face is AI-generated"}', opts) is None  # no audio key
    assert p.parse('{"video": "V.\\n\\naudio": "A."}', opts) == want  # the audio key run into the video value


def test_t4_a_reply_is_answered_once_a_top_level_object_closes_with_both_parts() -> None:
    from mozaic.evaluation.prompts import t4_prompts as p

    one = '{"video": "V.", "audio": "A."}'
    assert p.answered(one) and p.answered(one + "\n") and p.answered("```json\n" + one)
    assert p.answered('{"video": "V."}, {"audio": "A."}')  # one object per part (Uni-MoE)
    assert p.answered(one + ",") and p.answered(one + "]")  # Uni-MoE's "}," and "}]" come as one token
    assert not p.answered('{"video": "V."}')  # audio still to come
    assert not p.answered(one + "\n{")  # past the "}": already stopped there
    assert not p.answered('{"video": "V. }", "audio": "A.')  # a "}" inside a string
    # the inner object's "}" (Uni-MoE): the outer one gives the answer, and it is still open
    nested = '{"video": "V.", "audio": "A.", "generated": {"video": "No", "audio": "No"}'
    assert not p.answered(nested) and p.answered(nested + "}")
    assert p.parse(nested + "}", []) == {"video": "V.", "audio": "A."}
    # the reply Ola writes again and again: answered after its first object, and parse reads the same there
    runaway = (one + "\n") * 40 + '{\n  "video": "The video'
    first = runaway.index("}") + 1
    assert p.answered(runaway[:first]) and p.parse(runaway[:first], []) == p.parse(runaway, [])
    assert not any(p.answered(runaway[:i]) for i in range(first))


def test_retry_failed_reads_saved_replies_again_before_asking(tmp_path: Path) -> None:
    from mozaic.evaluation.predictions import to_prediction
    from mozaic.evaluation.run_inference import drop_failed, reread_failed

    qs = [{"id": f"q{i}", "options": list(REGIONS)} for i in range(4)]
    rows = [{"id": "q0", **to_prediction("t4", {"video": "V.", "audio": "A."}), "raw": "{}", "finish_reason": "stop"},
            {"id": "q1", **to_prediction("t4", None), "raw": '{"video": "V.", "audio": "A."}]', "finish_reason": "stop"},
            {"id": "q2", **to_prediction("t4", None), "raw": '{"video": "V. only"}', "finish_reason": "stop"},
            {"id": "q3", **to_prediction("t4", None), "raw": '{"video": "V.", "audio": "A."}]', "finish_reason": "length"}]
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert reread_failed(path, "t4", qs) == 1  # q1 read now; q2 has no audio; q3 was cut off: asked again
    saved = {r["id"]: r for r in read_jsonl(path)}
    assert saved["q1"]["pred_target"] == {"sections": {"video": "V.", "audio": "A."}} and saved["q1"]["reread"]
    assert saved["q1"]["raw"] == rows[1]["raw"] and saved["q0"] == rows[0]
    assert drop_failed(path) == 2 and [r["id"] for r in read_jsonl(path)] == ["q0", "q1"]
    assert reread_failed(path, "t4", qs) == 0  # nothing left to read


def test_t4_predictions_mirror_the_sections_and_scores_count_parsed() -> None:
    from mozaic.evaluation.metrics import t4_metrics
    from mozaic.evaluation.predictions import to_prediction

    rows = [t4.make_row(clip(label, gen, "main", "test", f"id0000{i}_aaa_00001", "id00009", "77f"))
            for i, (label, gen) in enumerate([("FVFA", "facefusion"), ("RVRA", "real")])]
    for r in rows:  # as in the released files: every row has its reference explanation
        r["gt_target"]["sections"] = {"video": "a reference", "audio": "b reference"}
    answers = [{"video": "a", "audio": "b"}, None]
    preds = [{"id": r["id"], **to_prediction("t4", a)} for r, a in zip(rows, answers, strict=True)]
    assert preds[0]["pred_answer"] == "a b" and preds[0]["pred_target"] == {"sections": {"video": "a", "audio": "b"}}
    assert preds[1]["pred_answer"] is None and preds[1]["pred_target"] is None  # mirrors gt_answer and gt_target
    s = t4_metrics.score({r["id"]: r for r in rows}, preds)
    assert (s["answered"], s["unreadable"], s["parse_rate"]["rate"]) == (2, 1, 0.5)
    assert s["judge"] == "not run yet" and s["text"]["video"]["n"] == 2


def test_t4_training_targets_are_the_sections_evaluation_reads() -> None:
    """The SFT target is the reference per part, as the prompt asks and the parser reads; the epoch's selection
    key is in the T4 scores."""
    dataset = pytest.importorskip("mozaic.training.mllms.dataset")  # needs the training env (torch)
    from mozaic.evaluation.metrics import t4_metrics
    from mozaic.evaluation.predictions import to_prediction
    from mozaic.evaluation.prompts import t4_prompts
    from mozaic.training.common.loop import get
    from mozaic.training.run_train import load_train_config

    rows = [t4.make_row(clip(label, gen, "main", "train", f"id0000{i}_aaa_00001", "id00009", "77f"))
            for i, (label, gen) in enumerate([("FVFA", "facefusion"), ("RVRA", "real")])]
    for r in rows:
        r["gt_target"]["sections"] = {"video": "The video ... reference.", "audio": "The audio ... reference."}
    for r in rows:
        target = json.loads(dataset.reply("t4", r))
        assert target == r["gt_target"]["sections"]
        assert t4_prompts.parse(dataset.reply("t4", r), r["options"]) == target
    rows[0]["gt_target"]["sections"]["audio"] = " "
    with pytest.raises(ValueError, match="evaluation would not read"):
        dataset.reply("t4", rows[0])  # an empty part: never trained on

    key = load_train_config("qwen2_5_omni/t4_lora")["train"]["select_on"]
    preds = [{"id": r["id"], **to_prediction("t4", {"video": "The video", "audio": "The audio"})} for r in rows]
    assert 0 < get(t4_metrics.score({r["id"]: r for r in rows}, preds), key) < 1


def test_t4_refuses_the_full_dataset(release: list[Row], monkeypatch: pytest.MonkeyPatch) -> None:
    from mozaic.evaluation.run_inference import check_t4

    monkeypatch.setattr(paths, "_subset", None)
    with pytest.raises(SystemExit, match="--subset small"):
        t4.select()
    with pytest.raises(SystemExit, match="--subset small"):
        check_t4({"model_version": "gemini-3.8-flash", "tasks": {"t4": "t4_test.jsonl"}}, "t4")


def test_t4_refuses_to_test_the_annotator_the_golden_set_and_classifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    from mozaic.evaluation.predictions import from_probs
    from mozaic.evaluation.run_inference import check_t4, run_classifier

    monkeypatch.setattr(paths, "_subset", "small")

    with pytest.raises(ValueError, match="MLLMs only"):
        from_probs("t4", {"RVRA": 1.0})
    with pytest.raises(SystemExit, match="mllm config"):
        run_classifier(None, {}, "t4", None, False, False)

    other = {"model_version": "gemini-3.8-flash", "tasks": {"t4": "t4_test.jsonl"}}
    check_t4(other, "t4")
    annotator = {"model_version": t4.CONFIG["annotator"]["model_version"], "tasks": {"t4": "t4_test.jsonl"}}
    check_t4(annotator, "t1")  # only t4 is guarded
    with pytest.raises(SystemExit, match="wrote the T4 reference"):
        check_t4(annotator, "t4")
    with pytest.raises(SystemExit, match="human review only"):
        check_t4({**other, "tasks": {"t4": "t4_golden.jsonl"}}, "t4")
