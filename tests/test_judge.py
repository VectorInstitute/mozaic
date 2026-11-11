"""Tests for the T4 explanation scores: the text metrics, the judge's prompt and reply, run_judge (resumable,
unreadable predictions, failed replies) and the scores run_eval writes. No API: the client is a fake."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mozaic.common import paths
from mozaic.common.io import read_jsonl, write_jsonl
from mozaic.evaluation import run_eval
from mozaic.evaluation.judge import prompt as jp
from mozaic.evaluation.judge import run_judge
from mozaic.evaluation.metrics import t4_metrics
from mozaic.evaluation.metrics import text_metrics as tm

Row = dict[str, Any]

REFS = ["The whole frame including the woman and the background is AI-generated from the start of the clip to the end",
        "The audio sounds like a natural recording with nothing AI-generated",
        "Only the inner face of the video is AI-generated the hair head body and background are untouched",
        "A man talks about his work and his voice is AI-altered from the start of the clip to the end"]
PREDS = ["The video is AI-generated the whole frame with the woman and the background was generated",
         "The audio is a real recording with nothing AI-generated",
         "The video is a real recording of a man speaking",
         "A man talks about work and the voice is AI-altered"]


# --------------------------------------------------------------------------- #
# Text metrics
# --------------------------------------------------------------------------- #
def test_text_metrics_match_the_coco_caption_evaluation() -> None:
    # pycocoevalcap 1.2 on the same lower-cased words: Bleu(4)[3], Rouge(), Cider()
    assert round(tm.bleu_4(PREDS, REFS), 4) == 0.2305
    assert round(sum(tm.rouge_l(p, r) for p, r in zip(PREDS, REFS, strict=True)) / 4, 4) == 0.5064
    assert round(sum(tm.cider_d(PREDS, REFS)) / 4, 4) == 2.1641


def test_text_metrics_edges() -> None:
    assert tm.rouge_l("the same words", "The same, words!") == pytest.approx(1.0)
    assert tm.rouge_l("", "a reference") == 0.0 and tm.bleu_4([""], ["a reference"]) == 0.0
    assert tm.bleu_4(["a b c d e"], ["a b c d e"]) == pytest.approx(1.0)
    assert tm.cider_d(["", "x"], ["a b", "y"]) == [0.0, 0.0]
    assert tm.cosine([1, 0], [1, 0]) == pytest.approx(1.0) and tm.cosine([1, 0], [0, 1]) == 0.0
    assert tm.cosine([0, 0], [1, 0]) == 0.0


# --------------------------------------------------------------------------- #
# The judge's prompt and reply
# --------------------------------------------------------------------------- #
def test_judge_prompt_fills_both_parts_and_parse_checks_the_reply() -> None:
    user = jp.build_user({"video": "GT-V", "audio": "GT-A"}, {"video": "P-V", "audio": "P-A"})
    assert "video: GT-V\n  audio: GT-A\nPrediction\n  video: P-V\n  audio: P-A" in user
    assert "0-10" in jp.SYSTEM and "{" not in jp.USER.replace("{gt_video}", "").replace("{gt_audio}", "") \
        .replace("{pred_video}", "").replace("{pred_audio}", "")
    good = {"video": {"score": 3, "justification": " wrong verdict "}, "audio": {"score": 10.0, "justification": "ok"}}
    assert jp.parse(json.dumps(good)) == {"video": {"score": 3, "justification": "wrong verdict"},
                                          "audio": {"score": 10, "justification": "ok"}}
    assert jp.parse("Let me think.\n```json\n" + json.dumps(good) + "\n```") is not None
    for bad in ({"video": good["video"]},                                          # audio missing
                {**good, "audio": {"score": 11, "justification": "x"}},            # past the scale
                {**good, "audio": {"score": 7.5, "justification": "x"}},           # not an integer
                {**good, "audio": {"score": True, "justification": "x"}},
                {**good, "audio": {"score": 7}}):                                   # no justification
        assert jp.parse(json.dumps(bad)) is None
    assert jp.parse("not json") is None


# --------------------------------------------------------------------------- #
# run_judge and run_eval
# --------------------------------------------------------------------------- #
class FakeClient:
    """Judges by the first word of each predicted part ("FAKE" scores 9, else 2); embeds a text by whether it
    says AI. replies: raw replies to give instead, in order."""

    def __init__(self, replies: list[str] | None = None, fail: set[str] = frozenset()):
        self.replies, self.fail, self.calls = list(replies or []), fail, 0

    def judge(self, user: str) -> tuple[str, Row]:
        self.calls += 1
        if any(f in user for f in self.fail):
            raise ConnectionError("endpoint down")
        if self.replies:
            return self.replies.pop(0), {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        pred = user.split("Prediction")[1]
        score = {p: 9 if f"{p}: FAKE" in pred else 2 for p in jp.PARTS}
        reply = {p: {"score": s, "justification": "x"} for p, s in score.items()}
        return json.dumps(reply), {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "AI" in t else [0.0, 1.0] for t in texts]


@pytest.fixture
def predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Row:
    """Four T4 questions and a model's predictions: two read, one unreadable, one missing."""
    (tmp_path / "mozaic").mkdir()
    monkeypatch.setenv("MOZAIC_ROOT", str(tmp_path / "mozaic"))
    monkeypatch.setattr(paths, "_subset", "small")
    config = {"model": "some_mllm", "model_version": "some/model", "experiment_name": "zs_test", "tasks": {"t4": "x"}}
    questions = [{"id": f"q{i}", "gt_label": "FVFA", "meta": {"video_method": "animate", "frames": 77},
                  "gt_target": {"sections": {"video": REFS[0], "audio": REFS[3]}}} for i in range(4)]
    qfile = tmp_path / "t4_test.jsonl"
    write_jsonl(qfile, questions)
    out = paths.results_dir("some_mllm", "zs_test", "t4")
    write_jsonl(out / "predictions.jsonl", [
        {"id": "q0", "pred_target": {"sections": {"video": "FAKE: " + PREDS[0], "audio": "FAKE: " + PREDS[3]}}},
        {"id": "q1", "pred_target": {"sections": {"video": "FAKE: " + PREDS[0], "audio": "real: " + PREDS[1]}}},
        {"id": "q2", "pred_target": None},
    ])
    (out / "inference_settings.json").write_text(json.dumps({"question_file": str(qfile), "filters": None}))
    return {"config": config, "out": out}


def test_run_judge_scores_resumes_and_never_calls_for_an_unreadable_reply(predictions: Row) -> None:
    config, out = predictions["config"], predictions["out"]
    client = FakeClient()
    run_judge.run(config, "t4", limit=1, client=client)
    run_judge.run(config, "t4", client=client)  # continues: the other two
    rows = {r["id"]: r for r in read_jsonl(out / "judge.jsonl")}
    assert set(rows) == {"q0", "q1", "q2"} and client.calls == 2  # q2 unreadable: no call
    assert rows["q0"]["scores"]["video"]["score"] == 9 and rows["q1"]["scores"]["audio"]["score"] == 2
    assert rows["q2"]["scores"] == jp.UNUSABLE and rows["q2"]["cosine"] == {"video": 0.0, "audio": 0.0}
    assert rows["q1"]["cosine"] == {"video": 1.0, "audio": 1.0}  # both say AI in the fake embedder
    run_judge.run(config, "t4", client=client)
    assert client.calls == 2  # nothing left
    assert json.loads((out / "judge_settings.json").read_text())["judge"] == run_judge.CONFIG["judge"]["model_version"]


def test_run_judge_keeps_bad_replies_as_failed_and_retry_asks_again(predictions: Row) -> None:
    config, out = predictions["config"], predictions["out"]
    run_judge.run(config, "t4", client=FakeClient(replies=["no json here", "{}"]))
    assert sorted(r["ok"] for r in read_jsonl(out / "judge.jsonl")) == [False, False, True]
    client = FakeClient()
    run_judge.run(config, "t4", retry_failed=True, client=client)
    rows = read_jsonl(out / "judge.jsonl")
    assert client.calls == 2 and len(rows) == 3 and all(r["ok"] for r in rows)


def test_run_judge_retry_judges_a_reply_that_reads_now(predictions: Row) -> None:
    config, out = predictions["config"], predictions["out"]
    run_judge.run(config, "t4", client=FakeClient())
    preds = read_jsonl(out / "predictions.jsonl")
    preds[2]["pred_target"] = {"sections": {"video": "FAKE: " + PREDS[0], "audio": "FAKE: " + PREDS[3]}}
    write_jsonl(out / "predictions.jsonl", preds)  # q2 reads now (a newer parse)
    client = FakeClient()
    run_judge.run(config, "t4", client=client)
    assert client.calls == 0  # without --retry-failed its 0 stays
    run_judge.run(config, "t4", retry_failed=True, client=client)
    rows = {r["id"]: r for r in read_jsonl(out / "judge.jsonl")}
    assert client.calls == 1 and rows["q2"]["scores"]["video"]["score"] == 9 and len(rows) == 3


def test_run_judge_does_not_save_failed_calls_and_stops(predictions: Row, monkeypatch: pytest.MonkeyPatch) -> None:
    config, out = predictions["config"], predictions["out"]
    monkeypatch.setitem(run_judge.CONFIG, "stop_after_failures", 2)
    monkeypatch.setitem(run_judge.CONFIG, "workers", 1)
    with pytest.raises(SystemExit, match="in a row failed"):
        run_judge.run(config, "t4", client=FakeClient(fail={"FAKE"}))
    saved = read_jsonl(out / "judge.jsonl") if (out / "judge.jsonl").exists() else []
    assert [r["id"] for r in saved] in ([], ["q2"])  # only the no-call row, if it was reached


def test_run_judge_refuses_other_tasks_the_judge_itself_and_other_settings(predictions: Row) -> None:
    config, out = predictions["config"], predictions["out"]
    with pytest.raises(SystemExit, match="t4 only"):
        run_judge.run(config, "t1", client=FakeClient())
    with pytest.raises(SystemExit, match="model being judged"):
        run_judge.run({**config, "model_version": run_judge.CONFIG["judge"]["model_version"]}, "t4", client=FakeClient())
    (out / "judge_settings.json").write_text(json.dumps({"judge": "another-judge"}))
    run_judge.run(config, "t4", limit=1, client=FakeClient())  # nothing judged before: the new settings win
    (out / "judge_settings.json").write_text(json.dumps({"judge": "another-judge"}))
    with pytest.raises(SystemExit, match="other settings"):
        run_judge.run(config, "t4", client=FakeClient())


def test_t4_scores_before_and_after_the_judge(predictions: Row) -> None:
    out = predictions["out"]
    run_eval.eval_task("some_mllm", "zs_test", "t4")
    s = json.loads((out / "scores.json").read_text())
    assert (s["questions"], s["answered"], s["missing"], s["unreadable"]) == (4, 3, 1, 1)
    assert s["judge"] == "not run yet" and s["cosine"] == "not run yet"
    assert s["text"]["video"]["n"] == 3 and 0 < s["text"]["video"]["rouge_l"] < 1 and s["text"]["both"]["bleu_4"] > 0

    run_judge.run(predictions["config"], "t4", client=FakeClient())
    run_eval.eval_task("some_mllm", "zs_test", "t4")
    s = json.loads((out / "scores.json").read_text())
    assert (s["judged"], s["not_judged"], s["failed"]) == (3, 0, 0)
    assert s["judge"]["video"]["mean"] == pytest.approx((9 + 9 + 0) / 3, abs=1e-4)
    assert s["judge"]["audio"]["mean"] == pytest.approx((9 + 2 + 0) / 3, abs=1e-4)
    assert s["judge"]["both"]["mean"] == pytest.approx((9 + 5.5 + 0) / 3, abs=1e-4)
    assert s["judge_per_cell"] == {"FVFA_animate": {"n": 3, "mean": s["judge"]["both"]["mean"]}}
    assert s["cosine"] == {"video": pytest.approx(2 / 3, abs=1e-4), "audio": pytest.approx(2 / 3, abs=1e-4)}


class NoEmbedder(FakeClient):
    def embed(self, texts: list[str]) -> None:
        return None


def test_judge_without_an_embedder_leaves_the_cosine_out(predictions: Row) -> None:
    out = predictions["out"]
    run_judge.run(predictions["config"], "t4", client=NoEmbedder())
    rows = {r["id"]: r for r in read_jsonl(out / "judge.jsonl")}
    assert rows["q0"]["cosine"] is None and rows["q0"]["calls"] == 1 and rows["q0"]["ok"]
    run_eval.eval_task("some_mllm", "zs_test", "t4")
    s = json.loads((out / "scores.json").read_text())
    assert s["cosine"] == "not run" and s["judge"]["video"]["mean"] == pytest.approx(6, abs=1e-4)


def test_t4_scores_count_an_unreadable_reply_as_empty_text() -> None:
    q = {"id": "a", "gt_label": "RVRA", "meta": {"video_method": "real", "frames": 77},
         "gt_target": {"sections": {"video": REFS[1], "audio": REFS[1]}}}
    s = t4_metrics.score({"a": q}, [{"id": "a", "pred_target": None}])
    assert s["text"]["video"] == {"n": 1, "rouge_l": 0.0, "bleu_4": 0.0, "cider_d": 0.0}
