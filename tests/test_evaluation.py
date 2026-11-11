"""Tests for run_inference, run_eval and the models: Gemini through a fake client (no API calls),
Qwen3-Omni's inputs without the model (no GPU)."""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from made_up import SMALL_COUNTS, t1_rows, task3_test_rows
from PIL import Image

from mozaic.annotation.sampling import counted_sample
from mozaic.annotation.t3_identity_check import build_questions as t3
from mozaic.common.frame_sampler import image_to_b64, pcm_to_wav
from mozaic.common.io import read_jsonl, write_jsonl
from mozaic.common.paths import qas_dir
from mozaic.common.questions import select, task_entry
from mozaic.evaluation import run_eval, run_inference
from mozaic.evaluation.base import BaseMLLM, ClipInput, Generation
from mozaic.evaluation.predictions import from_probs, to_prediction

WORDS = '{"clip 1": "hello there", "clip 2": "good morning"}'


class FakeClient:
    """Stand-in for the OpenAI-compatible endpoint: replies in order, raising the exceptions.

    prompt_tokens gives each reply's prompt token count; 100 when not given.
    """

    def __init__(self, replies: Sequence[str | Exception], prompt_tokens: Sequence[int] = ()) -> None:
        self.replies = list(replies)
        self.prompt_tokens = list(prompt_tokens)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        """Record what would be sent, then reply like the OpenAI client."""
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        tokens = self.prompt_tokens.pop(0) if self.prompt_tokens else 100
        usage = SimpleNamespace(model_dump=lambda exclude_none: {"prompt_tokens": tokens, "total_tokens": tokens + 5})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")],
                               usage=usage)


Run = Callable[..., tuple[Any, Path]]


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Run:
    """Small question files in a temporary dataset root, and run(module, fake, task) to run on them.

    Frames and sound are made up (no video is read); the fake client replaces the OpenAI client.
    """
    pytest.importorskip("openai")
    gemini = importlib.import_module("mozaic.evaluation.models.api.gemini.model")
    monkeypatch.setenv("MOZAIC_ROOT", str(tmp_path))
    monkeypatch.setattr(run_inference, "sample_frames", lambda path, n=None: [Image.new("RGB", (8, 8))] * n)
    monkeypatch.setattr(run_inference, "load_audio", lambda path, rate=None: b"\x00\x00" * 160)
    monkeypatch.setattr(gemini.time, "sleep", lambda seconds: None)
    (tmp_path / "api.env").write_text("GEMINI_API_KEY=test\n")
    write_jsonl(qas_dir("t1") / "t1_4.jsonl", t1_rows()[:4])  # gt: neither, neither, the audio, the video
    write_jsonl(qas_dir("t3") / "t3_8.jsonl", counted_sample(task3_test_rows(10), SMALL_COUNTS, 1, t3.kind))
    base = {"model": "gemini", "family": "mllm", "model_version": "gemini-3.8-flash",  # the chat API, JPEG frames
            "base_url": "https://fake.vectorinstitute.ai/v1", "api_key_env": "GEMINI_API_KEY", "env_file": None,
            "experiment_name": "chat_api_test", "frames": 32, "audio_sample_rate": 16000,
            "generation": {"temperature": 1.0, "max_tokens": 8200}, "retry": {"max_attempts": 3},
            "stop_after_failures": 3, "tasks": {"t1": "t1_4.jsonl", "t3": "t3_8.jsonl"}}

    def go(module: Any, fake: FakeClient, task: str, **changes: Any) -> tuple[Any, Path]:
        config = {**base, **changes}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(gemini, "OpenAI", lambda api_key, base_url: fake)
        monkeypatch.setattr(sys, "argv", ["main", "--config", str(path), "--task", task,
                                          "--env-file", str(tmp_path / "api.env")])
        code = module.main()
        return code, tmp_path / "results" / config["model"] / config["experiment_name"] / task

    return go


def test_t1_sends_frames_then_sound_then_text_and_never_the_answer(run: Run) -> None:
    fake = FakeClient(['{"answer": "neither"}'] * 4)
    _, out = run(run_inference, fake, "t1")
    sent = fake.calls[0]
    assert set(sent) == {"model", "messages", "temperature", "max_tokens"}  # the fixture's generation
    assert (sent["temperature"], sent["max_tokens"]) == (1.0, 8200)
    parts = sent["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["image_url"] * 32 + ["file", "text"]  # unchanged for one-clip tasks
    assert parts[32]["file"]["file_data"].startswith("data:audio/wav;base64,")  # the sound, as a file
    q = read_jsonl(qas_dir("t1") / "t1_4.jsonl")[0]
    assert not [w for w in ("gt_", "RVRA", q["id"], q["clips"][0]) if w in json.dumps(sent)]  # no answer leaks
    saved = read_jsonl(out / "predictions.jsonl")
    assert saved[0]["preprocessing"] == {"frames": 32, "audio_sample_rate": 16000}
    with pytest.raises(SystemExit, match="other settings"):
        run(run_inference, FakeClient([]), "t1", frames=16)


def test_t1_asks_failed_calls_again_and_stops_after_three_failures_in_a_row(run: Run) -> None:
    down = RuntimeError("503 unavailable")
    replies = ['{"answer": "neither"}', down, down, down, "I am not sure", '```json\n{"answer": "the video"}\n```']
    _, out = run(run_inference, FakeClient(replies), "t1")
    saved = read_jsonl(out / "predictions.jsonl")
    assert [s["pred_answer"] for s in saved] == ["neither", None, "the video"]  # the 2nd question's call failed
    again = FakeClient(['{"answer": "both"}'])
    run(run_inference, again, "t1")
    assert len(again.calls) == 1  # only the 2nd question was asked again
    stopped = FakeClient([RuntimeError("429 daily limit reached")] * 12)
    _, out = run(run_inference, stopped, "t1", experiment_name="other")
    assert len(stopped.calls) == 9  # 3 questions x 3 tries, then it stopped; the 4th was never asked
    assert not (out / "predictions.jsonl").exists()


def test_t3_sends_both_clips_as_frames_or_sound_and_never_the_answer(run: Run) -> None:
    key = read_jsonl(qas_dir("t3") / "t3_8.jsonl")
    fake = FakeClient(['{"answer": "real, same person"}'] * len(key))
    _, out = run(run_inference, fake, "t3")
    for q, sent in zip(key, fake.calls, strict=True):
        parts = sent["messages"][0]["content"]
        clip = ["image_url"] * 32 if q["part"] == "face" else ["file"]  # face: frames only; voice: sound only
        assert [p["type"] for p in parts] == ["text", *clip, "text", *clip, "text"]
        assert (parts[0]["text"], parts[len(clip) + 1]["text"]) == ("Clip 1:", "Clip 2:")
        if q["part"] == "voice":
            assert parts[1]["file"]["file_data"].startswith("data:audio/wav;base64,")
        else:
            assert "32 frames" in parts[-1]["text"]
        text = parts[-1]["text"]
        assert q["question"] in text and "Options: " + ", ".join(f'"{o}"' for o in q["options"]) in text
        # what each option means, in the question's (shuffled) option order
        assert [text.index(f'- "{o}": ') for o in q["options"]] == sorted(text.index(f'- "{o}": ') for o in q["options"])
        sent_text = json.dumps(sent)
        assert not [w for w in ("gt_", "same_person", "RVRA", q["id"], *q["clips"]) if w in sent_text]
    saved = read_jsonl(out / "predictions.jsonl")
    assert {s["preprocessing"]["clips"] for s in saved} == {2}


def test_t1_video_only_or_audio_only_sends_that_modality_and_says_the_other_is_not_given(run: Run) -> None:
    for kept, sent_types, missing in (("video", ["image_url"] * 32, "soundtrack"), ("audio", ["file"], "frames")):
        fake = FakeClient(['{"answer": "neither"}'] * 4)
        _, out = run(run_inference, fake, "t1", experiment_name=f"{kept}only",
                     tasks={"t1": {"file": "t1_4.jsonl", "modalities": [kept]}})
        parts = fake.calls[0]["messages"][0]["content"]
        assert [p["type"] for p in parts] == [*sent_types, "text"]
        assert f"The clip's {missing} {'is' if kept == 'video' else 'are'} not provided." in parts[-1]["text"]
        assert json.loads((out / "inference_settings.json").read_text())["modalities"] == [kept]
    with pytest.raises(SystemExit, match="only t1, t4 take"):  # t3 sends each question's own modality
        run(run_inference, FakeClient([]), "t3", tasks={"t3": {"file": "t3_8.jsonl", "modalities": ["audio"]}})


def test_t1_and_t4_prompts_keep_their_hash_and_the_modality_ablation_runs_theirs() -> None:
    t1 = importlib.import_module("mozaic.evaluation.prompts.t1_prompts")
    t4 = importlib.import_module("mozaic.evaluation.prompts.t4_prompts")
    assert t1.template() == t1.PROMPT and t4.template() == t4.PROMPT  # every run made so far
    hashes = [hashlib.sha256(t1.template((m,)).encode()).hexdigest()[:10] for m in ("video", "audio")]
    assert hashes == ["1b4187c431", "b2f0116ce1"]  # the runs in ablations/modality
    record = {"options": ["none", "inner face", "voice"]}
    assert "The clip's frames are not provided." in t4.build_prompt(record, 0, ("audio",))
    assert '"video": "<3 to 5 sentences>", "audio"' in t4.build_prompt(record, 0, ("audio",))  # both parts still asked


def test_t3_scores_the_answers_by_part_and_kind(run: Run) -> None:
    key = read_jsonl(qas_dir("t3") / "t3_8.jsonl")  # 8 questions
    down = RuntimeError("503 unavailable")
    fake = '```json\n{"answer": "fake of this person"}\n```'
    replies = ['{"answer": "real, same person"}', down, down, down, "Real.", *[fake] * 5]
    _, out = run(run_inference, FakeClient(replies), "t3")  # the 2nd question is left
    run(run_inference, FakeClient(['{"answer": "fake of this person"}']), "t3")
    run_eval.eval_task("gemini", out.parent.name, "t3")  # the config's experiment_name
    scores = json.loads((out / "scores.json").read_text())
    right = ((key[0]["gt_answer"] == "real, same person")
             + sum(q["gt_answer"] == "fake of this person" for q in key[1:2] + key[3:]))
    assert (scores["answered"], scores["missing"], scores["unreadable"]) == (8, 0, 1)  # "Real." is not an option
    assert scores["accuracy"] == round(right / 8, 4)
    assert scores["pred_answers"] == {"real, same person": 1, "None": 1, "fake of this person": 6}
    assert set(scores["per_part"]) == {"face", "voice"}
    # a real voice is asked as its vocoded copy (SMALL_COUNTS)
    assert set(scores["per_clip1"]) == {"face real", "face facefusion", "voice vocoded", "voice converted"}
    assert set(scores["per_class"]) == {"real_same_person", "real_different_person", "fake_this_person",
                                        "fake_different_person"}
    fakes = sum(q["gt_target"]["fake"] for q in key)
    assert (scores["fake"]["positives"], scores["same_person"]["positives"]) == (fakes, 4)  # the two halves
    assert scores["prompt_tokens"] == {"face 77": [100], "voice 77": [100]}
    first = read_jsonl(out / "predictions.jsonl")[0]
    assert (first["pred_target"], first["pred_target_name"]) == ({"fake": 0, "same_person": 1}, "real_same_person")


def test_scoring_the_full_test_split_also_scores_small_from_the_same_answers(
        run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = t1_rows()[:4]  # gt: neither, neither, the audio, the video
    small_file = tmp_path / "qas_small" / qas_dir("t1").name / "t1_test.jsonl"
    write_jsonl(qas_dir("t1") / "t1_test.jsonl", rows)
    write_jsonl(small_file, rows[2:])  # Small: copies of two full questions
    replies = [f'{{"answer": "{a}"}}' for a in ("neither", "both", "the audio", "neither")]
    _, out = run(run_inference, FakeClient(replies), "t1", tasks={"t1": "t1_test.jsonl"})
    monkeypatch.setattr(sys, "argv", ["main", "--config", str(tmp_path / "config.yaml"), "--task", "t1"])
    run_eval.main()
    small = tmp_path / "results_small" / out.relative_to(tmp_path / "results")
    full = read_jsonl(out / "predictions.jsonl")
    copied = lambda rows: [{**r, "copied_from": str(out)} for r in rows]
    assert read_jsonl(small / "predictions.jsonl") == copied(full[2:])  # the full answers, marked as copies
    assert json.loads((small / "inference_settings.json").read_text()) == {
        **json.loads((out / "inference_settings.json").read_text()), "question_file": str(small_file),
        "derived_from": str(out)}
    scores = json.loads((small / "scores.json").read_text())
    assert (scores["question_file"], scores["questions"], scores["answered"]) == (str(small_file), 2, 2)
    assert scores["accuracy"] == 0.5  # the audio: right; neither: wrong
    run_eval.main()  # scored again: the same answers, the same scores
    assert json.loads((small / "scores.json").read_text()) == scores
    changed = [*full[:3], {**full[3], "raw": "asked again"}]
    write_jsonl(out / "predictions.jsonl", changed)  # the full run changed (a --retry-failed there)
    run_eval.main()
    assert read_jsonl(small / "predictions.jsonl") == copied(changed[2:])  # a copy follows the full answer


def test_overwriting_a_full_run_drops_its_small_copy_but_never_a_small_runs_own_answer(
        run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = t1_rows()[:4]
    write_jsonl(qas_dir("t1") / "t1_test.jsonl", rows)
    write_jsonl(tmp_path / "qas_small" / qas_dir("t1").name / "t1_test.jsonl", rows[2:])
    _, out = run(run_inference, FakeClient(['{"answer": "neither"}'] * 4), "t1", tasks={"t1": "t1_test.jsonl"})
    small = tmp_path / "results_small" / out.relative_to(tmp_path / "results")
    config = ["main", "--config", str(tmp_path / "config.yaml"), "--task", "t1"]
    monkeypatch.setattr(sys, "argv", config)
    run_eval.main()
    assert (small / "scores.json").exists()

    def rerun(answer: str) -> None:  # --overwrite with other settings, as a retrained model's other best epoch
        gemini = importlib.import_module("mozaic.evaluation.models.api.gemini.model")
        monkeypatch.setattr(gemini, "OpenAI", lambda api_key, base_url: FakeClient([f'{{"answer": "{answer}"}}'] * 4))
        cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
        cfg["generation"]["temperature"] /= 2
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
        monkeypatch.setattr(sys, "argv", [*config, "--overwrite", "--env-file", str(tmp_path / "api.env")])
        run_inference.main()
        monkeypatch.setattr(sys, "argv", config)
        run_eval.main()

    rerun("the audio")
    new = read_jsonl(out / "predictions.jsonl")
    assert read_jsonl(small / "predictions.jsonl") == [{**r, "copied_from": str(out)} for r in new[2:]]  # not the old
    assert json.loads((small / "scores.json").read_text())["accuracy"] == 0.5  # the new answers, scored
    copies = read_jsonl(small / "predictions.jsonl")
    own = {k: v for k, v in copies[0].items() if k != "copied_from"}  # as if a --subset small run answered it
    write_jsonl(small / "predictions.jsonl", [own, *copies[1:]])
    rerun("the video")
    assert read_jsonl(small / "predictions.jsonl")[0] == own  # a Small result with its own answer is kept


def test_small_waits_for_every_answer_and_keeps_the_answers_of_a_subset_small_run(
        run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mozaic.common.paths._subset", None)  # back to the full dataset after the --subset small run
    rows = t1_rows()[:4]
    write_jsonl(qas_dir("t1") / "t1_test.jsonl", rows)
    write_jsonl(tmp_path / "qas_small" / qas_dir("t1").name / "t1_test.jsonl", rows[:2])
    down = RuntimeError("503 unavailable")
    replies = ['{"answer": "neither"}', down, down, down, '{"answer": "the audio"}', '{"answer": "the video"}']
    _, out = run(run_inference, FakeClient(replies), "t1", tasks={"t1": "t1_test.jsonl"})  # the 2nd is left
    small = tmp_path / "results_small" / out.relative_to(tmp_path / "results")
    config = ["main", "--config", str(tmp_path / "config.yaml"), "--task", "t1"]
    monkeypatch.setattr(sys, "argv", config)
    run_eval.main()
    assert (out / "scores.json").exists() and not (small / "scores.json").exists()  # a Small question has no answer
    assert [r["id"] for r in read_jsonl(small / "predictions.jsonl")] == [rows[0]["id"]]  # the one there is, copied
    gemini = importlib.import_module("mozaic.evaluation.models.api.gemini.model")
    fake = FakeClient(['{"answer": "both"}'] * 2)
    monkeypatch.setattr(gemini, "OpenAI", lambda api_key, base_url: fake)
    monkeypatch.setattr(sys, "argv", [*config, "--subset", "small", "--env-file", str(tmp_path / "api.env")])
    run_inference.main()  # a --subset small run: asks only the question the full run has no answer to
    assert len(fake.calls) == 1
    own = read_jsonl(small / "predictions.jsonl")[1]
    assert own["id"] == rows[1]["id"] and own["pred_answer"] == "both" and "copied_from" not in own
    run(run_inference, FakeClient(['{"answer": "neither"}']), "t1", tasks={"t1": "t1_test.jsonl"})  # full run done
    monkeypatch.setattr(sys, "argv", config)
    run_eval.main()
    assert read_jsonl(small / "predictions.jsonl")[1] == own  # the Small run's own answer is kept
    assert json.loads((small / "scores.json").read_text())["answered"] == 2  # every Small question answered: scored


def test_a_subset_small_run_after_a_full_run_copies_its_answers_and_asks_nothing(
        run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mozaic.common.paths._subset", None)
    rows = t1_rows()[:4]
    write_jsonl(qas_dir("t1") / "t1_test.jsonl", rows)
    write_jsonl(tmp_path / "qas_small" / qas_dir("t1").name / "t1_test.jsonl", rows[1:3])
    replies = [f'{{"answer": "{a}"}}' for a in ("neither", "both", "the audio", "neither")]
    _, out = run(run_inference, FakeClient(replies), "t1", tasks={"t1": "t1_test.jsonl"})  # the full run, done
    small = tmp_path / "results_small" / out.relative_to(tmp_path / "results")
    gemini = importlib.import_module("mozaic.evaluation.models.api.gemini.model")
    fake = FakeClient([])
    monkeypatch.setattr(gemini, "OpenAI", lambda api_key, base_url: fake)
    config = ["main", "--config", str(tmp_path / "config.yaml"), "--task", "t1"]
    monkeypatch.setattr(sys, "argv", [*config, "--subset", "small", "--env-file", str(tmp_path / "api.env")])
    run_inference.main()  # by mistake: every answer is in the full run
    assert fake.calls == []  # nothing asked, nothing paid
    full = {r["id"]: r for r in read_jsonl(out / "predictions.jsonl")}
    assert read_jsonl(small / "predictions.jsonl") == [{**full[q["id"]], "copied_from": str(out)} for q in rows[1:3]]
    assert json.loads((small / "inference_settings.json").read_text())["derived_from"] == str(out)
    run_inference.main()  # again: nothing to copy, nothing to ask
    assert fake.calls == [] and len(read_jsonl(small / "predictions.jsonl")) == 2
    monkeypatch.setattr(sys, "argv", [*config, "--subset", "small"])
    run_eval.main()  # scored as any --subset small run
    assert json.loads((small / "scores.json").read_text())["answered"] == 2


def test_sound_check_passes_only_when_both_sounds_arrive(run: Run) -> None:
    sound_check = importlib.import_module("mozaic.evaluation.models.api.gemini.sound_check")
    voice = [r for r in task3_test_rows(2) if r["part"] == "voice"][:2]
    voice[1] = {**voice[1], "meta": {**voice[1]["meta"], "frames": 153}}  # one question of each length
    write_jsonl(qas_dir("t3") / "check.jsonl", voice)
    tasks = {"t3": "check.jsonl"}
    good = FakeClient([WORDS, WORDS], prompt_tokens=[300, 452])
    code, out = run(sound_check, good, "t3", tasks=tasks)
    assert code == 0
    assert [[p["type"] for p in c["messages"][0]["content"]] for c in good.calls] == [
        ["text", "file", "text", "file", "text"]] * 2
    record = json.loads(next(out.glob("sound_check_*.json")).read_text())
    assert (record["verdict"], [t["prompt_tokens"] for t in record["tests"]]) == ("PASS: the sound arrived", [300, 452])
    deaf = FakeClient(['{"clip 1": "NO AUDIO", "clip 2": "hi"}', WORDS], prompt_tokens=[300, 376])  # a sound missing
    code, out = run(sound_check, deaf, "t3", tasks=tasks, experiment_name="deaf")
    assert code == 1
    problems = json.loads(next(out.glob("sound_check_*.json")).read_text())["problems"]
    assert any("NO AUDIO" in p for p in problems) and any("152" in p for p in problems)
    assert not (out / "predictions.jsonl").exists()  # the sound check asks no question


def test_interactions_api_sends_each_clip_as_one_video_and_the_chat_api_stays_as_before(
        run: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    gemini = importlib.import_module("mozaic.evaluation.models.api.gemini.model")
    calls = []

    def fake_post(url: str, body: dict[str, Any], key: str) -> dict[str, Any]:
        calls.append((url, body, key))
        text = {"type": "text", "text": '{"answer": "real, same person"}'}
        return {"status": "completed", "steps": [{"type": "model_output", "content": [text]}], "model": "gemini-x-0101",
                "usage": {"total_input_tokens": 300, "total_output_tokens": 11, "total_thought_tokens": 20}}

    monkeypatch.setattr(gemini, "post_json", fake_post)
    monkeypatch.setattr(gemini, "clip_video", lambda path, with_sound: (b"frames+sound" if with_sound else b"frames", 3.08))
    monkeypatch.setattr(gemini, "load_audio", lambda path, rate=None: b"\x00\x00" * 160)
    generation = {"temperature": 0.0, "max_output_tokens": 8200, "thinking_level": "low"}
    interactions = {"api": "interactions", "base_url": "https://gateway/v1beta", "generation": generation, "video_resolution": "low"}

    def video(data: bytes) -> dict[str, Any]:  # 32 frames over the clip's 3.08 s, at the config's resolution
        return {"type": "video", "data": base64.b64encode(data).decode(), "mime_type": "video/mp4",
                "processing": {"type": "static", "fps": 32 / 3.08}, "resolution": "low"}

    _, out = run(run_inference, FakeClient([]), "t3", experiment_name="interactions", **interactions)
    for q, (url, body, key) in zip(read_jsonl(qas_dir("t3") / "t3_8.jsonl"), calls, strict=True):
        assert (url, key, body["generation_config"], body["store"]) == ("https://gateway/v1beta/interactions", "test", generation, False)
        items = body["input"]
        assert [items[0], items[2], items[4]["type"], len(items)] == [{"type": "text", "text": "Clip 1:"},
                                                                     {"type": "text", "text": "Clip 2:"}, "text", 5]
        if q["part"] == "face":  # each clip as a video, without its sound
            assert items[1] == items[3] == video(b"frames")
        else:  # each sound alone
            assert [(i["type"], i["mime_type"]) for i in (items[1], items[3])] == [("audio", "audio/wav")] * 2
    saved = read_jsonl(out / "predictions.jsonl")
    assert (saved[0]["usage"]["prompt_tokens"], saved[0]["usage"]["completion_tokens"]) == (300, 31)  # thinking is output
    assert saved[0]["usage"]["served_model"] == "gemini-x-0101"  # the model that answered, as Google names it
    assert (saved[0]["finish_reason"], saved[0]["pred_answer"]) == ("stop", "real, same person")
    settings = json.loads((out / "inference_settings.json").read_text())
    assert (settings["api"], settings["video_resolution"]) == ("interactions", "low")
    calls.clear()
    run(run_inference, FakeClient([]), "t1", experiment_name="interactions_t1", **interactions)
    assert calls[0][1]["input"][0] == video(b"frames+sound")  # t1: the clip with its sound
    jpeg = FakeClient(['{"answer": "neither"}'] * 4)
    _, out = run(run_inference, jpeg, "t1")  # the fixture: the chat API with JPEG frames, as before
    assert jpeg.calls[0]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "api" not in json.loads((out / "inference_settings.json").read_text())


def test_gemini_test_configs_ask_geminis_questions_greedy_as_whole_clips_one_folder_each() -> None:
    for task in ("t1", "t2", "t3"):
        flash, pro = (run_inference.load_model_config(f"{m}/{task}/zeroshot") for m in ("gemini_3_8_flash", "gemini_3_1_pro"))
        assert flash["tasks"] == pro["tasks"]  # the same file and filters
    configs = [run_inference.load_model_config(f"{m}/t1/zeroshot") for m in ("gemini_3_8_flash", "gemini_3_1_pro")]
    for config in configs:
        assert all(config[k] == configs[0][k] for k in ("model", "frames", "audio_sample_rate", "retry"))
        assert (config["api"], config["video_resolution"]) == ("interactions", "low")
        assert config["generation"] == {"temperature": 0.0, "max_output_tokens": 2048, "thinking_level": "low"}
    assert [c["model_version"] for c in configs] == ["gemini-3.8-flash", "gemini-3.1-pro-preview"]
    assert configs[0]["experiment_name"] != configs[1]["experiment_name"]  # the same model key: a folder each


def test_a_model_with_only_generate_still_takes_one_clip_questions() -> None:
    class OneClip(BaseMLLM):
        def generate(self, frames: list, audio_wav: bytes | None, prompt: str) -> Generation:
            return Generation(raw=f"{len(frames)} {audio_wav} {prompt}")

    model = OneClip({})
    assert model.generate_clips([ClipInput(frames=[1, 2])], "p").raw == "2 None p"
    with pytest.raises(NotImplementedError):
        model.generate_clips([ClipInput("Clip 1:"), ClipInput("Clip 2:")], "p")


@pytest.mark.parametrize("name", ["qwen3_omni_vllm", "qwen2_5_omni_vllm"])  # <name>/t<N>/zeroshot
def test_vllm_configs_ask_geminis_questions_with_32_frames_at_224(name: str) -> None:
    vllm_qwen = importlib.import_module("mozaic.evaluation.models.mllms.qwen_omni_vllm.model")
    for task in ("t1", "t2", "t3"):
        config = run_inference.load_model_config(f"{name}/{task}/zeroshot")
        gemini = run_inference.load_model_config(f"gemini_3_8_flash/{task}/zeroshot")
        assert isinstance(run_inference.load_model(config), vllm_qwen.Model)  # nothing loaded yet
        assert {k: config[k] for k in ("frames", "audio_sample_rate")} == {k: gemini[k] for k in ("frames", "audio_sample_rate")}
        assert config["frame_size"] == 224  # the clips' own size, not enlarged
        run_inference.use_split(config, None)  # the test split by default
        assert config["tasks"] == gemini["tasks"]  # the same file and filters


def test_split_names_the_question_files_and_the_results_folder() -> None:
    for task in ("t1", "t2", "t3"):
        config = run_inference.load_model_config(f"qwen3_omni_vllm/{task}/zeroshot")
        run_inference.use_split(config, "val")
        assert config["experiment_name"] == "zeroshot_32f_224_val"  # one results folder for the three tasks
        assert task_entry(config, task)[0] == f"{task}_val.jsonl"
    gemini = run_inference.load_model_config("gemini_3_8_flash/t1/zeroshot")
    with pytest.raises(SystemExit, match="split"):  # Gemini's configs name their files
        run_inference.use_split(gemini, "val")
    run_inference.use_split(gemini, None)
    assert gemini == run_inference.load_model_config("gemini_3_8_flash/t1/zeroshot")  # unchanged without --split


def test_vllm_qwen_keeps_224_and_puts_the_t1_sound_inside_the_video() -> None:
    vllm_qwen = importlib.import_module("mozaic.evaluation.models.mllms.qwen_omni_vllm.model")
    t1 = ClipInput(path="/data/clip.mov", send_frames=True, send_audio=True)
    msgs, sound_in_video = vllm_qwen.message([t1], "prompt", 224)
    assert msgs[0]["content"] == [
        {"type": "video", "video": "/data/clip.mov", "resized_height": 224, "resized_width": 224},
        {"type": "text", "text": "prompt"}] and sound_in_video
    rate = 32 / 77 * 25  # qwen_omni_utils: 32 frames of a 77-frame clip at 25 fps
    sent = vllm_qwen.request("chat", ["video"], ["sound"], [rate], True, 224, 2)["mm_processor_kwargs"]
    assert sent == {"use_audio_in_video": True, "fps": rate, "second_per_grid_ts": [2 / rate],  # the rate, twice
                    "size": {"shortest_edge": 224 * 224, "longest_edge": 224 * 224}}

    voice = [ClipInput("Clip 1:", path="/data/a.mov", send_audio=True), ClipInput("Clip 2:", path="/data/b.mov", send_audio=True)]
    msgs, sound_in_video = vllm_qwen.message(voice, "prompt", 224)
    assert [c["type"] for c in msgs[0]["content"]] == ["text", "audio", "text", "audio", "text"] and not sound_in_video
    assert vllm_qwen.request("chat", [], ["a", "b"], [], False, 224, 2) == {
        "prompt": "chat", "multi_modal_data": {"audio": ["a", "b"]}, "mm_processor_kwargs": {"use_audio_in_video": False}}


def test_nemotron_config_asks_the_qwen_vllm_questions_greedy_without_reasoning() -> None:
    nemotron = importlib.import_module("mozaic.evaluation.models.mllms.nemotron_omni_vllm.model")
    config = run_inference.load_model_config("nemotron_omni_vllm/t1/zeroshot")
    qwen = run_inference.load_model_config("qwen3_omni_vllm/t1/zeroshot")
    assert isinstance(run_inference.load_model(config), nemotron.Model)  # nothing loaded yet
    same = ("frames", "frame_size", "audio_sample_rate", "experiment_name", "generation", "tasks")
    assert {k: config[k] for k in same} == {k: qwen[k] for k in same}  # same questions and input, greedy, 256 tokens
    assert config["chat_template_kwargs"] == {"enable_thinking": False}  # reasoning off


def test_nemotron_keeps_224_and_each_label_next_to_its_clip() -> None:
    nemotron = importlib.import_module("mozaic.evaluation.models.mllms.nemotron_omni_vllm.model")
    assert nemotron.engine_args(32, 224, 16) == {  # (224 / 16)^2 patches: frames not enlarged to 512
        "hf_overrides": {"vision_config": {"video_target_num_patches": 196}},
        "media_io_kwargs": {"video": {"num_frames": 32}}}

    t1 = ClipInput(path="/data/clip.mov", send_frames=True, send_audio=True)
    assert nemotron.message([t1], "prompt") == ("<video>\nprompt", True)  # vLLM puts the sound after the frames
    assert nemotron.request("chat", ["video"], ["sound"], True) == {
        "prompt": "chat", "multi_modal_data": {"video": ["video"], "audio": ["sound"]},
        "mm_processor_kwargs": {"use_audio_in_video": True}}

    voice = [ClipInput("Clip 1:", path="/data/a.mov", send_audio=True), ClipInput("Clip 2:", path="/data/b.mov", send_audio=True)]
    assert nemotron.message(voice, "prompt") == ("Clip 1:\n<so_embedding>\nClip 2:\n<so_embedding>\nprompt", False)
    face = [ClipInput("Clip 1:", path="/data/a.mov", send_frames=True), ClipInput("Clip 2:", path="/data/b.mov", send_frames=True)]
    assert nemotron.message(face, "prompt") == ("Clip 1:\n<video>\nClip 2:\n<video>\nprompt", False)
    assert nemotron.request("chat", [], ["a", "b"], False) == {
        "prompt": "chat", "multi_modal_data": {"audio": ["a", "b"]}, "mm_processor_kwargs": {"use_audio_in_video": False}}


def test_gemma_config_asks_the_qwen_vllm_questions_greedy() -> None:
    gemma = importlib.import_module("mozaic.evaluation.models.mllms.gemma4_vllm.model")
    config = run_inference.load_model_config("gemma4_vllm/t1/zeroshot")
    qwen = run_inference.load_model_config("qwen3_omni_vllm/t1/zeroshot")
    assert isinstance(run_inference.load_model(config), gemma.Model)  # nothing loaded yet
    same = ("frames", "audio_sample_rate", "experiment_name", "generation", "tasks")
    assert {k: config[k] for k in same} == {k: qwen[k] for k in same}  # same questions and input, greedy, 256 tokens
    assert config["vllm"]["attention_backend"] == "TRITON_ATTN"  # FlashInfer would need nvcc


def test_gemma_sends_each_clip_after_its_label_and_t1_sound_after_its_video() -> None:
    gemma = importlib.import_module("mozaic.evaluation.models.mllms.gemma4_vllm.model")
    t1 = ClipInput(path="/data/clip.mov", send_frames=True, send_audio=True)
    assert gemma.message([t1], "prompt")[0]["content"] == [
        {"type": "video"}, {"type": "audio"}, {"type": "text", "text": "prompt"}]  # the sound as its own part
    assert gemma.request("chat", ["video"], ["sound"]) == {
        "prompt": "chat", "multi_modal_data": {"video": ["video"], "audio": ["sound"]}}

    voice = [ClipInput("Clip 1:", path="/data/a.mov", send_audio=True), ClipInput("Clip 2:", path="/data/b.mov", send_audio=True)]
    assert [c["type"] for c in gemma.message(voice, "prompt")[0]["content"]] == ["text", "audio", "text", "audio", "text"]
    face = [ClipInput("Clip 1:", path="/data/a.mov", send_frames=True), ClipInput("Clip 2:", path="/data/b.mov", send_frames=True)]
    assert [c["type"] for c in gemma.message(face, "prompt")[0]["content"]] == ["text", "video", "text", "video", "text"]
    assert gemma.request("chat", [], ["a", "b"]) == {"prompt": "chat", "multi_modal_data": {"audio": ["a", "b"]}}


def test_minicpm_o_config_asks_the_qwen_vllm_questions_greedy_without_thinking() -> None:
    minicpm = importlib.import_module("mozaic.evaluation.models.mllms.minicpm_o_vllm.model")
    config = run_inference.load_model_config("minicpm_o_vllm/t1/zeroshot")
    qwen = run_inference.load_model_config("qwen3_omni_vllm/t1/zeroshot")
    assert isinstance(run_inference.load_model(config), minicpm.Model)  # nothing loaded yet
    same = ("frames", "audio_sample_rate", "experiment_name", "generation", "tasks")
    assert {k: config[k] for k in same} == {k: qwen[k] for k in same}  # same questions and input, greedy, 256 tokens
    assert config["chat_template_kwargs"] == {"enable_thinking": False}  # thinking off
    assert config["answer_choices"] is True  # the reply held to the options


def test_minicpm_o_answer_choices_are_the_prompts_json_with_each_option() -> None:
    minicpm = importlib.import_module("mozaic.evaluation.models.mllms.minicpm_o_vllm.model")
    t1 = importlib.import_module("mozaic.evaluation.prompts.t1_prompts")
    t3 = importlib.import_module("mozaic.evaluation.prompts.t3_prompts")
    record = {"question": "Which is fake?", "options": ["both", "neither", "the audio", "the video"]}
    choices = minicpm.answer_choices(t1.build_prompt(record, 32))
    assert choices == ['{"answer": "both"}', '{"answer": "neither"}', '{"answer": "the audio"}', '{"answer": "the video"}']
    assert [t1.parse(c, record["options"]) for c in choices] == record["options"]  # parse() reads every one

    options = ["fake of this person", "real, same person", "fake of a different person", "real, different person"]
    choices = minicpm.answer_choices(t3.build_prompt({"part": "voice", "question": "Who?", "options": options}, 0))
    assert [t3.parse(c, options) for c in choices] == options  # commas inside an option stay in it


def test_minicpm_o_writes_each_tag_after_its_label_and_counts_tokens_between_marks() -> None:
    minicpm = importlib.import_module("mozaic.evaluation.models.mllms.minicpm_o_vllm.model")
    t1 = ClipInput(path="/data/clip.mov", send_frames=True, send_audio=True)
    assert minicpm.message([t1], "prompt") == "(<video>./</video>)\n(<audio>./</audio>)\nprompt"  # the sound after its video
    assert minicpm.request("chat", ["frames"], ["sound"]) == {
        "prompt": "chat", "multi_modal_data": {"video": ["frames"], "audio": ["sound"]}}

    voice = [ClipInput("Clip 1:", path="/data/a.mov", send_audio=True), ClipInput("Clip 2:", path="/data/b.mov", send_audio=True)]
    assert minicpm.message(voice, "prompt") == "Clip 1:\n(<audio>./</audio>)\nClip 2:\n(<audio>./</audio>)\nprompt"
    face = [ClipInput("Clip 1:", path="/data/a.mov", send_frames=True), ClipInput("Clip 2:", path="/data/b.mov", send_frames=True)]
    assert minicpm.message(face, "prompt") == "Clip 1:\n(<video>./</video>)\nClip 2:\n(<video>./</video>)\nprompt"
    assert minicpm.request("chat", [], ["a", "b"]) == {"prompt": "chat", "multi_modal_data": {"audio": ["a", "b"]}}
    assert minicpm.count_between([7, 1, 0, 0, 2, 9, 1, 0, 2], start=1, end=2) == 3  # two pieces: 2 + 1 tokens


def test_choice_filter_lets_through_only_the_replies_still_possible_then_the_end() -> None:
    """The filter Uni-MoE-2.0-Omni, Ola and VITA-1.5 answer through."""
    from mozaic.evaluation.models.mllms.answer_choices import choice_filter

    allowed = choice_filter([[1, 2, 3], [1, 4]], end_ids=[9])  # two replies as tokens
    assert allowed(0, np.array([])) == [1]
    assert allowed(0, np.array([1])) == [2, 4]  # both still possible
    assert allowed(0, np.array([1, 2])) == [3]
    assert allowed(0, np.array([1, 4])) == [9]  # a whole reply: only the end
    assert allowed(0, np.array([7])) == [9]  # off every reply: only the end

def test_videollama2_config_asks_the_qwen_vllm_questions_greedy() -> None:
    videollama2 = importlib.import_module("mozaic.evaluation.models.mllms.videollama2_av.model")
    config = run_inference.load_model_config("videollama2_av/t2/zeroshot")
    qwen = run_inference.load_model_config("qwen3_omni_vllm/t2/zeroshot")
    assert isinstance(run_inference.load_model(config), videollama2.Model)  # nothing loaded yet
    same = ("frames", "audio_sample_rate", "experiment_name", "tasks")
    assert {k: config[k] for k in same} == {k: qwen[k] for k in same}  # same questions and input
    # greedy, 256 tokens, and no repetition penalty (the checkpoint's own file would add 1.05), as every model
    assert config["generation"] == {"do_sample": False, "max_new_tokens": 256, "repetition_penalty": 1.0}
    assert config["code"] == "external-repos/videollama2"  # the authors' code, git-ignored
    assert config["soundtrack"] == "pieces"  # t2 sends no sound: the authors' default, as run


def test_videollama2_nopad_config_is_the_standard_one_with_only_the_clips_own_sound() -> None:
    """The run reported for t1 and t3: every setting as t2's (pieces) config but the sound (and its own folder)."""
    config = run_inference.load_model_config("videollama2_av/t2/zeroshot")
    qwen = {t: run_inference.load_model_config(f"qwen3_omni_vllm/{t}/zeroshot") for t in ("t1", "t3")}
    differ = ("experiment_name", "soundtrack", "tasks")
    for task in ("t1", "t3"):  # t2 has no sound: it is the pieces run
        nopad = run_inference.load_model_config(f"videollama2_av/{task}/zeroshot")
        assert {k: v for k, v in nopad.items() if k not in differ} == {k: v for k, v in config.items() if k not in differ}
        assert nopad["soundtrack"] == "nopad" and nopad["experiment_name"] == "zeroshot_32f_224_nopad"
        assert nopad["tasks"] == qwen[task]["tasks"]

def test_videollama2_writes_each_tag_after_its_label_one_kind_per_question() -> None:
    videollama2 = importlib.import_module("mozaic.evaluation.models.mllms.videollama2_av.model")
    t1 = ClipInput(frames=["frame"] * 32, audio_wav=b"sound")
    assert videollama2.message([t1], "prompt") == ("<video>\nprompt", "<video>")  # one video with its soundtrack
    face = [ClipInput("Clip 1:", frames=["a"]), ClipInput("Clip 2:", frames=["b"])]
    assert videollama2.message(face, "prompt") == ("Clip 1:\n<video>\nClip 2:\n<video>\nprompt", "<video>")
    voice = [ClipInput("Clip 1:", audio_wav=b"a"), ClipInput("Clip 2:", audio_wav=b"b")]
    assert videollama2.message(voice, "prompt") == ("Clip 1:\n<audio>\nClip 2:\n<audio>\nprompt", "<audio>")
    with pytest.raises(ValueError, match="one kind of clip"):  # the authors' tokenizer helper takes one tag
        videollama2.message([face[0], voice[1]], "prompt")


def test_omnivinci_config_asks_the_qwen_vllm_questions_greedy() -> None:
    omnivinci = importlib.import_module("mozaic.evaluation.models.mllms.omnivinci.model")
    config = run_inference.load_model_config("omnivinci/t1/zeroshot")
    qwen = run_inference.load_model_config("qwen3_omni_vllm/t1/zeroshot")
    model = run_inference.load_model(config)
    assert isinstance(model, omnivinci.Model) and model.reads_clips  # nothing loaded yet; it reads its clips
    same = ("frames", "audio_sample_rate", "experiment_name", "tasks")
    assert {k: config[k] for k in same} == {k: qwen[k] for k in same}  # same questions and input
    # greedy, 256 tokens, and no repetition penalty (its language model's own file would add 1.05), as every model
    assert config["generation"] == {"do_sample": False, "max_new_tokens": 256, "repetition_penalty": 1.0}


def test_omnivinci_writes_its_tags_and_keeps_only_each_clips_own_sound() -> None:
    omnivinci = importlib.import_module("mozaic.evaluation.models.mllms.omnivinci.model")
    t1 = ClipInput(path="a.mp4", send_frames=True, send_audio=True)
    # the parts joined with nothing in between, as their processor joins them; the model puts the sound in the video
    assert omnivinci.message([t1], "prompt") == "<sound><vila/video>prompt"
    face = [ClipInput("Clip 1:", path="a.mp4", send_frames=True), ClipInput("Clip 2:", path="b.mp4", send_frames=True)]
    assert omnivinci.message(face, "prompt") == "Clip 1:<vila/video>Clip 2:<vila/video>prompt"
    voice = [ClipInput("Clip 1:", path="a.mp4", send_audio=True), ClipInput("Clip 2:", path="b.mp4", send_audio=True)]
    assert omnivinci.message(voice, "prompt") == "Clip 1:<sound>Clip 2:<sound>prompt"

    # their bookkeeping of a video with its soundtrack, in pieces of 30 s: a 77-frame clip at 25 fps (3.08 s) is one
    # piece, all 32 frames, the sound up to 3.08 s (308 spectrogram frames); a 40 s video would be two
    sound = {"audio_start_sec": 0.0, "audio_end_sample_sec": 3.08}
    indices = np.linspace(0, 76, 32).round().astype(int).tolist()
    assert omnivinci.segments(indices, 77, 25.0, sound, 30, 100) == ([list(range(32))], [[0, 308]])
    sound = {"audio_start_sec": 0.0, "audio_end_sample_sec": 40.0}
    indices = np.linspace(0, 999, 32).round().astype(int).tolist()
    assert omnivinci.segments(indices, 1000, 25.0, sound, 30, 100) == (
        [list(range(24)), list(range(24, 32))], [[0, 3000], [3000, 4000]])

    # t3 voice: of the 750 pieces of 30 s, only those of the clip's own sound (3.08 s: 77, 6.12 s: 153), by the
    # model's own rule for a soundtrack; a t1 soundtrack (no end) stays whole, the model cuts it itself
    model = omnivinci.Model(run_inference.load_model_config("omnivinci/t1/zeroshot"))
    model.ends = [308, 612, None]
    info = {"audio_info": [[{"new_audio_n_stft_frames": 3000}]] * 3}
    kept = model.keep_own_sound(None, (None, None, info), [np.zeros((750, 2))] * 3)
    assert [len(f) for f in kept] == [77, 153, 750]


def test_uni_moe_config_asks_the_qwen_vllm_questions_greedy() -> None:
    uni_moe = importlib.import_module("mozaic.evaluation.models.mllms.uni_moe_2_omni.model")
    config = run_inference.load_model_config("uni_moe_2_omni/t1/zeroshot")
    qwen = run_inference.load_model_config("qwen3_omni_vllm/t1/zeroshot")
    model = run_inference.load_model(config)
    assert isinstance(model, uni_moe.Model) and model.reads_clips  # nothing loaded yet; it reads its clips
    same = ("frames", "audio_sample_rate", "experiment_name", "tasks")
    assert {k: config[k] for k in same} == {k: qwen[k] for k in same}  # same questions and input
    assert config["generation"] == {"do_sample": False, "max_new_tokens": 256, "repetition_penalty": 1.0}
    assert config["code"] == "external-repos/uni_moe_2/Uni-MoE-2" and config["silence"] in ("kept", "cut")
    assert config["answer_choices"] is True  # its reply can only be one of the question's options, as MiniCPM-o's


def test_uni_moe_writes_tags_on_their_own_lines_and_counts_each_windows_own_sound() -> None:
    uni_moe = importlib.import_module("mozaic.evaluation.models.mllms.uni_moe_2_omni.model")
    t1 = ClipInput(path="a.mov", send_frames=True, send_audio=True)
    assert uni_moe.message([t1], "prompt") == "<video>\n<audio>\nprompt"  # the authors' example: video, then its sound
    face = [ClipInput("Clip 1:", path="a.mov", send_frames=True), ClipInput("Clip 2:", path="b.mov", send_frames=True)]
    assert uni_moe.message(face, "prompt") == "Clip 1:\n<video>\nClip 2:\n<video>\nprompt"
    voice = [ClipInput("Clip 1:", path="a.mov", send_audio=True), ClipInput("Clip 2:", path="b.mov", send_audio=True)]
    assert uni_moe.message(voice, "prompt") == "Clip 1:\n<audio>\nClip 2:\n<audio>\nprompt"
    # encoder steps of each 30 s window that hold the sound itself (1,500 a window): 3.08 s 154, 6.12 s 306; their
    # processor cuts exactly 30 s as one window, and keeps at most whisper_audio_time windows
    assert [uni_moe.own_steps(n, 20) for n in (49_280, 97_920, 480_000, 480_001, 960_000)] == [
        [154], [306], [1500], [1500, 1], [1500, 1500]]
    assert uni_moe.own_steps(960_000, 1) == [1500]


def test_ola_config_asks_the_qwen_vllm_questions_greedy() -> None:
    ola = importlib.import_module("mozaic.evaluation.models.mllms.ola.model")
    config = run_inference.load_model_config("ola/t1/zeroshot")
    qwen = run_inference.load_model_config("qwen3_omni_vllm/t1/zeroshot")
    model = run_inference.load_model(config)
    assert isinstance(model, ola.Model) and model.reads_clips  # nothing loaded yet; it reads its clips
    same = ("frames", "audio_sample_rate", "experiment_name", "tasks")
    assert {k: config[k] for k in same} == {k: qwen[k] for k in same}  # same questions and input
    # greedy, 256 tokens, and no repetition penalty (its generation_config.json would add 1.05), as every model
    assert config["generation"] == {"do_sample": False, "max_new_tokens": 256, "repetition_penalty": 1.0}
    assert config["code"] == "external-repos/ola" and config["silence"] == "cut"
    assert config["answer_choices"] is True  # its reply can only be one of the question's options, as MiniCPM-o's


def test_ola_writes_their_tags_and_cuts_each_sound_at_its_own_steps() -> None:
    ola = importlib.import_module("mozaic.evaluation.models.mllms.ola.model")
    t1 = ClipInput(path="a.mov", send_frames=True, send_audio=True)
    assert ola.message([t1], "prompt") == "<speech><image>\nprompt"  # the authors' example for a video with its sound
    face = [ClipInput("Clip 1:", path="a.mov", send_frames=True), ClipInput("Clip 2:", path="b.mov", send_frames=True)]
    assert ola.message(face, "prompt") == "Clip 1:\n<image>\nClip 2:\n<image>\nprompt"
    voice = [ClipInput("Clip 1:", path="a.mov", send_audio=True), ClipInput("Clip 2:", path="b.mov", send_audio=True)]
    assert ola.message(voice, "prompt") == "Clip 1:\n<speech>\nClip 2:\n<speech>\nprompt"
    # encoder steps that hold the sound itself (1,500 in 30 s): 3.08 s 154, 6.12 s 306; a step partly filled counts
    assert [ola.own_steps(n) for n in (49_280, 97_920, 49_281, 480_000)] == [154, 306, 155, 1500]
    with pytest.raises(ValueError):
        ola.own_steps(480_001)  # the cut takes one 30 s window; our clips are 3 or 6 s

    # with silence cut, only those steps go on to their projector; kept (None): the whole window
    model = ola.Model(run_inference.load_model_config("ola/t1/zeroshot"))
    model.steps, model.read = [154, None], []
    assert model.cut(None, None, np.zeros((1, 1500, 2))).shape == (1, 154, 2)
    assert model.cut(None, None, np.zeros((1, 1500, 2))).shape == (1, 1500, 2)
    assert model.read == [154, 1500] and model.steps == []


def test_vita_config_asks_the_qwen_vllm_questions_greedy() -> None:
    vita = importlib.import_module("mozaic.evaluation.models.mllms.vita_1_5.model")
    config = run_inference.load_model_config("vita_1_5/t1/zeroshot")
    qwen = run_inference.load_model_config("qwen3_omni_vllm/t1/zeroshot")
    model = run_inference.load_model(config)
    assert isinstance(model, vita.Model) and model.reads_clips  # nothing loaded yet; it reads its clips
    same = ("frames", "audio_sample_rate", "experiment_name", "tasks")
    assert {k: config[k] for k in same} == {k: qwen[k] for k in same}  # same questions and input
    # greedy, 256 tokens, and no repetition penalty (its generation_config.json would add 1.05), as every model
    assert config["generation"] == {"do_sample": False, "max_new_tokens": 256, "repetition_penalty": 1.0}
    assert config["code"] == "external-repos/vita"
    assert config["input_cap"] is None  # 32 frames are 8,192 pieces: their 6,200 cut would drop the question
    assert isinstance(config["seed"], int)  # their sound step's random noise, the same every run
    assert config["answer_choices"] is True  # its reply can only be one of the question's options, as MiniCPM-o's


def test_vita_writes_a_tag_per_frame_then_the_sound_then_the_question() -> None:
    vita = importlib.import_module("mozaic.evaluation.models.mllms.vita_1_5.model")
    t1 = ClipInput(path="a.mov", send_frames=True, send_audio=True)
    assert vita.message([t1], "prompt", 3) == "<image><image><image><audio>\nprompt"  # frames, its sound, the question
    face = [ClipInput("Clip 1:", path="a.mov", send_frames=True), ClipInput("Clip 2:", path="b.mov", send_frames=True)]
    assert vita.message(face, "prompt", 2) == "Clip 1:\n<image><image>\nClip 2:\n<image><image>\nprompt"
    voice = [ClipInput("Clip 1:", path="a.mov", send_audio=True), ClipInput("Clip 2:", path="b.mov", send_audio=True)]
    assert vita.message(voice, "prompt", 0) == "Clip 1:\n<audio>\nClip 2:\n<audio>\nprompt"
    with pytest.raises(ValueError):
        vita.tags(ClipInput(path="a.mov"), 32)  # neither frames nor sound
    # its answers start with a mark of the question it heard, which their evaluations drop (kept in the usage)
    assert vita.drop_mark('☜{"answer": "neither"}') == ('{"answer": "neither"}', "☜")
    assert vita.drop_mark('{"answer": "neither"}') == ('{"answer": "neither"}', "")


def test_run_inference_gives_batch_size_questions_and_clip_files_to_a_model_that_reads_them(
        run: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    class ReadsClips(BaseMLLM):
        reads_clips = True

        def __init__(self, config: dict[str, Any]) -> None:
            super().__init__(config)
            self.calls: list = []

        def generate_batch(self, questions: list) -> list[Generation]:
            self.calls.append(questions)
            return [Generation(raw='{"answer": "neither"}', finish_reason="stop") for _ in questions]

    model = ReadsClips({})
    monkeypatch.setattr(run_inference, "load_model", lambda config: model)
    _, out = run(run_inference, FakeClient([]), "t1", batch_size=3)
    assert [len(batch) for batch in model.calls] == [3, 1]  # the 4 questions of t1_4.jsonl
    clips, prompt = model.calls[0][0]
    assert clips[0].path.endswith(".mov") and clips[0].send_frames and clips[0].send_audio and clips[0].frames == []
    assert prompt.startswith("The images are 32 frames")  # the frames the model will read
    saved = read_jsonl(out / "predictions.jsonl")
    assert [r["pred_answer"] for r in saved] == ["neither"] * 4
    assert saved[0]["preprocessing"] == {"frames": 32, "audio_sample_rate": 16000}


def test_arms_filter_keeps_t3_face_questions_and_drops_only_raw_real_voices() -> None:
    rows = task3_test_rows(1)  # one pair: 2 pictures and 2 sounds (the vocoded view), each with clip 2 = A and B
    vocoded = next(r for r in rows if r["part"] == "voice" and r["meta"]["arm"] == "vocoded")
    raw = {**vocoded, "id": "raw", "meta": {**vocoded["meta"], "arm": "main"}}  # a raw real voice, as in older files
    assert select([*rows, raw], {"arms": ["vocoded"]}) == rows  # face questions (no sound) and converted voices kept


def test_t3_predictions_mirror_the_gt_fields() -> None:
    assert to_prediction("t3", "fake of this person") == {
        "pred_answer": "fake of this person", "pred_target": {"fake": 1, "same_person": 1},
        "pred_target_name": "fake_this_person"}
    assert to_prediction("t3", "real, different person")["pred_target"] == {"fake": 0, "same_person": 0}
    assert to_prediction("t3", None)["pred_target"] is None
    probs = {"real_same_person": 0.1, "real_different_person": 0.2, "fake_this_person": 0.3,
             "fake_different_person": 0.4}
    assert from_probs("t3", probs)["pred_answer"] == "fake of a different person"



CONFIGS = sorted(str(p.relative_to(run_inference.CONFIG_DIR).with_suffix("")) for p in run_inference.CONFIG_DIR.glob("*/t*/*.yaml"))
TRAINED = [c for c in CONFIGS if "_train" in c]
ZERO_SHOT = [c for c in CONFIGS if c.endswith("/zeroshot")]
T4_ZERO_SHOT = [c for c in ZERO_SHOT if "/t4/" in c]


def test_config_names_resolve_to_model_task_run(monkeypatch: pytest.MonkeyPatch) -> None:
    load = run_inference.load_configs
    names = lambda configs: [sorted(c["tasks"]) for c in configs]
    assert names(load("qwen3_omni_vllm", "t2")) == [["t2"]]  # qwen3_omni_vllm/t2/zeroshot
    assert load("qwen3_omni_vllm", "t2")[0] == run_inference.load_model_config("qwen3_omni_vllm/t2/zeroshot")
    assert names(load("qwen3_omni_vllm")) == [["t1", "t2", "t3"]]  # one config: the three differ only in their task
    assert names(load("avhubert/ln_trainsmall")) == [["t1"], ["t2"], ["t3"]]  # three trained runs
    assert names(load("avhubert/ln_trainsmall", "t3")) == [["t3"]]
    assert names(load("videollama2_av")) == [["t1", "t3"], ["t2"]]  # nopad, and t2's own sound setting
    assert names(load("qwen3_omni_vllm/t1/zeroshot")) == [["t1"]]  # a full name as it is
    monkeypatch.setattr(run_inference, "subset", lambda: "small")
    assert names(load("qwen3_omni_vllm")) == [["t1", "t2", "t3"], ["t4"]]  # t4 with --subset small only
    for bad, task in [("qwen3_omni_vllm", "t9"), ("nobody", None), ("qwen3_omni_vllm/t1/zeroshot", "t2")]:
        with pytest.raises(SystemExit):
            load(bad, task)


def test_every_config_is_model_task_name_and_asks_only_its_task() -> None:
    assert len(CONFIGS) == len(list(run_inference.CONFIG_DIR.rglob("*.yaml")))  # <model>/t<N>/<name>.yaml, nothing else
    for name in CONFIGS:
        folder, task, stem = name.split("/")
        assert list(run_inference.load_model_config(name)["tasks"]) == [task]
        assert stem == "zeroshot" or re.fullmatch(r"(ln|full|adapters|lora)_train(small|full)", stem)


@pytest.mark.parametrize("name", T4_ZERO_SHOT)
def test_t4_zero_shot_configs_are_their_t1_t3_config_with_a_t4_answer(name: str) -> None:
    folder = name.split("/")[0]
    config, base = run_inference.load_model_config(name), run_inference.load_model_config(f"{folder}/t1/zeroshot")
    gemini = config["model"] == "gemini"  # one folder per Gemini model (gemini_3_8_flash/), one environment (.venv_gemini)
    assert folder == config["model"] or gemini  # the folder is the environment (.venv_<model>)
    # as its T1 config: {split} files save to <experiment_name>_test (Gemini's name the test file and folder)
    assert config["tasks"] == {"t4": {"file": "t4_test.jsonl" if gemini else "t4_{split}.jsonl"}}
    assert config.get("answer_choices", False) is False  # T4 is free text
    limit = next(k for k in ("max_tokens", "max_new_tokens", "max_output_tokens") if k in base["generation"])
    assert config["generation"] == {**base["generation"], limit: 8192}
    same = set(base) - {"tasks", "generation", "answer_choices"}
    assert {k: config[k] for k in same} == {k: base[k] for k in same}  # the same input, weights and results folder


@pytest.mark.parametrize("name", TRAINED)
def test_trained_configs_test_their_training_run_on_the_test_split_with_its_filters(name: str) -> None:
    config = run_inference.load_model_config(name)
    trained = yaml.safe_load((run_inference.TRAIN_CONFIG_DIR / f"{config['trained']['config']}.yaml").read_text())
    folder, task, stem = name.split("/")
    on, mode = config["trained"]["data"], config["trained"]["config"].rsplit("_", 1)[1]  # avhubert/t1_ln: ln
    assert folder == config["model"]  # the folder is the environment (.venv_<model>)
    assert task == trained["task"] and stem == f"{mode}_train{on}" and list(config["tasks"]) == [task]
    assert config["experiment_name"].endswith(f"_train{on}")  # trained on Small or full: never the same results folder
    file, filters = task_entry(config, trained["task"])
    val = trained["questions"]["val"]
    assert (file, filters) == (val["file"].replace("_val", "_test"), val.get("filters"))
    if config["family"] == "classifier":
        assert config["experiment_name"] == f"{trained['experiment_name']}_train{on}"


@pytest.mark.parametrize("task", ["t1", "t2", "t4"])
def test_sft_configs_get_the_zero_shot_input(task: str) -> None:
    sft = run_inference.load_model_config(f"qwen2_5_omni_vllm/{task}/lora_trainsmall")
    zero = run_inference.load_model_config("qwen2_5_omni_vllm/t1/zeroshot")
    keys = ("model", "family", "model_version", "frames", "frame_size", "audio_sample_rate", "vllm")
    assert {k: sft[k] for k in keys} == {k: zero[k] for k in keys}
    assert sft["generation"] == (zero["generation"] if task != "t4" else {"temperature": 0.0, "max_tokens": 8192})


def test_a_trained_model_gets_its_runs_weights_whatever_subset_is_tested_and_the_test_split_only(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mozaic.evaluation.base import BaseClassifier

    monkeypatch.setenv("MOZAIC_ROOT", str(tmp_path))
    write_jsonl(qas_dir("t1") / "t1_test.jsonl", t1_rows()[:4])
    write_jsonl(qas_dir("t1") / "t1_val.jsonl", t1_rows()[:4])  # the full dataset's questions
    loaded: list[str] = []

    class Uniform(BaseClassifier):
        def load(self) -> None:
            loaded.append(self.config["weights"])

        def predict_proba(self, questions: list) -> list[dict[str, float]]:
            return [{"RVRA": 0.4, "RVFA": 0.2, "FVRA": 0.2, "FVFA": 0.2} for _ in questions]

    monkeypatch.setattr(run_inference, "load_model", lambda config: Uniform(config))
    config = {"model": "avhubert", "family": "classifier", "experiment_name": "ln_mean_mlp", "trained": {"config": "avhubert/t1_ln", "data": "small"},
              "batch_size": 3, "stop_after_failures": 1, "tasks": {"t1": "t1_test.jsonl"}}
    path = tmp_path / "config.yaml"

    def go(**changes: Any) -> None:
        path.write_text(yaml.safe_dump({**config, **changes}))
        monkeypatch.setattr(sys, "argv", ["main", "--config", str(path)])
        run_inference.main()

    with pytest.raises(SystemExit, match="train avhubert/t1_ln on small first"):
        go()
    with pytest.raises(SystemExit, match="trained on t1"):
        go(tasks={"t2": "t2_test.jsonl"})
    weights = tmp_path / "training_small" / "avhubert" / "ln_mean_mlp" / "t1" / "best.pt"  # trained on Small
    weights.parent.mkdir(parents=True)
    weights.touch()
    go()
    assert loaded == [str(weights)]
    out = tmp_path / "results" / "avhubert" / "ln_mean_mlp" / "t1"  # tested on the full test split
    assert [r["pred_answer"] for r in read_jsonl(out / "predictions.jsonl")] == ["neither"] * 4
    assert json.loads((out / "inference_settings.json").read_text())["weights"] == str(weights)
    with pytest.raises(SystemExit, match="test split only"):
        go(tasks={"t1": "t1_val.jsonl"}, experiment_name="ln_val")
