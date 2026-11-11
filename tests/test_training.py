"""Training's shared parts: pooling, padding, tuning modes, class weights, and the models' inputs.

Needs torch: run in a model environment (e.g. .venv_avhubert), skipped in the core one.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mozaic.evaluation.predictions import from_probs
from mozaic.training.base import BaseClassifier, BaseProcessor
from mozaic.training.common import tuning
from mozaic.training.common.data import CLASSES, label
from mozaic.training.common.heads import make_head, masked_mean
from mozaic.training.common.loop import class_weights, get


class Toy(BaseClassifier):
    def build_backbone(self, config):
        self.backbone = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.BatchNorm1d(8))
        self.extra = nn.Linear(8, 8)
        return 8

    def encode(self, batch):
        return self.extra(self.backbone[1](self.backbone[0](batch["x"]))), batch["valid"]

    def tunable(self):
        return {"backbone": self.backbone, "new": [self.extra]}


def trainable(model):
    return {name for name, p in model.named_parameters() if p.requires_grad}


def test_masked_mean_ignores_padding():
    x = torch.tensor([[[1.0], [3.0], [100.0]]])
    assert masked_mean(x, torch.tensor([[True, True, False]])).item() == 2.0


def test_collate_pads_and_marks_valid():
    class P(BaseProcessor):
        time_dims: ClassVar[dict[str, int]] = {"audio": 1}

    items = [{"audio": torch.ones(2, 3), "length": 3, "label": 0},
             {"audio": torch.ones(2, 5), "length": 5, "label": 1}]
    batch = P({}).collate(items)
    assert batch["audio"].shape == (2, 2, 5)
    assert batch["audio"][0, :, 3:].abs().sum() == 0
    assert batch["valid"].tolist() == [[True] * 3 + [False] * 2, [True] * 5]
    assert batch["label"].tolist() == [0, 1]


def test_tuning_modes():
    model = Toy({}, 4)
    tuning.apply(model, "frozen")
    assert trainable(model) == {n for n, _ in model.named_parameters() if n.startswith(("head", "extra"))}
    tuning.apply(model, "layernorm")
    assert {"backbone.1.weight", "backbone.1.bias"} <= trainable(model)
    assert "backbone.0.weight" not in trainable(model) and "backbone.2.weight" not in trainable(model)
    tuning.train_mode(model, "layernorm")
    assert model.training and not model.backbone.training  # BatchNorm statistics stay fixed
    tuning.apply(model, "full")
    assert trainable(model) == {n for n, _ in model.named_parameters()}
    with pytest.raises(NotImplementedError):
        tuning.apply(model, "lora")


def test_trainable_state_holds_only_trained_weights():
    model = Toy({}, 4)
    tuning.apply(model, "layernorm")
    assert set(tuning.trainable_state(model, "layernorm")) == trainable(model)  # the backbone's statistics never change


def test_full_tuning_checkpoint_reloads_to_the_same_outputs():
    """BatchNorm statistics are buffers, not parameters: without them the reloaded backbone normalises wrongly."""
    torch.manual_seed(0)
    model = Toy({}, 4)
    tuning.apply(model, "full")
    tuning.train_mode(model, "full")
    for _ in range(20):  # train mode: BatchNorm moves its running statistics towards this data
        model.backbone(torch.randn(16, 4) * 3 + 5)
    state = tuning.trainable_state(model, "full")
    assert {"backbone.2.running_mean", "backbone.2.running_var", "backbone.2.num_batches_tracked"} <= set(state)

    reloaded = Toy({}, 4)  # as evaluation builds it: the original backbone, then best.pt on top
    assert not reloaded.load_state_dict(state, strict=False).unexpected_keys
    x = torch.randn(8, 4)
    model.eval(), reloaded.eval()
    assert torch.equal(model.backbone(x), reloaded.backbone(x))
    stale = {k: v for k, v in state.items() if "running_" not in k and "num_batches" not in k}
    old = Toy({}, 4)
    old.load_state_dict(stale, strict=False)
    assert not torch.allclose(model.backbone(x), old.eval().backbone(x))  # the bug: weights saved, statistics not


def test_heads():
    assert make_head({}, 8, 4)(torch.zeros(2, 8)).shape == (2, 4)  # mlp by default
    assert make_head({"type": "linear"}, 8, 4)(torch.zeros(2, 8)).shape == (2, 4)


def test_class_weights_balance_t1():
    rows = [{"gt_label": label} for label in ["FVFA"] * 2 + ["FVRA"] * 2 + ["RVFA", "RVRA"]]
    w = class_weights("t1", rows, "balanced")
    assert w == [1.5, 1.5, 0.75, 0.75]  # RVRA, RVFA, FVRA, FVFA
    assert class_weights("t1", rows, "none") is None


def test_classes_match_evaluation():
    assert CLASSES["t1"] == ["RVRA", "RVFA", "FVRA", "FVFA"]
    assert CLASSES["t2"] == ["real", "face_swap", "fully_generated", "full_body_swap"]
    probs = dict(zip(CLASSES["t1"], [0.1, 0.2, 0.3, 0.4], strict=True))
    assert from_probs("t1", probs)["pred_label"] == "FVFA"
    probs = dict(zip(CLASSES["t2"], [0.1, 0.6, 0.2, 0.1], strict=True))
    assert from_probs("t2", probs)["pred_answer"] == "face swap"
    assert label("t2", {"gt_target_name": "full_body_swap"}) == 3


def test_get_dotted_score():
    assert get({"macro": {"f1": 0.5}}, "macro.f1") == 0.5
    assert get({"macro": {"f1": None}}, "macro.f1") == float("-inf")


def test_avhubert_input():
    pytest.importorskip("python_speech_features")
    from PIL import Image

    from mozaic.training.classifiers.avhubert.preprocess import Processor

    total, indices = 77, [0, 2, 5, 76]
    frames = [Image.new("RGB", (224, 224), (200, 200, 200)) for _ in indices]
    audio = np.random.default_rng(0).normal(0, 3000, 77 * 640).astype(np.float32)
    item = Processor({"audio_sample_rate": 16000})(frames, indices, total, audio)
    assert item["video"].shape == (1, 77, 88, 88) and item["audio"].shape == (104, 77)
    placed = (item["video"][0].abs().sum((1, 2)) > 0).nonzero().flatten().tolist()
    assert placed == indices  # each frame at its own step, the rest empty
    assert item["length"] == 77
    video_only = Processor({"audio_sample_rate": 16000})(frames, indices, total, None)  # t2
    assert "audio" not in video_only and video_only["video"].shape == (1, 77, 88, 88)


def test_limit_takes_every_class(monkeypatch):
    from mozaic.training.common import data

    rows = [{"gt_label": c} for c in ["FVFA"] * 50 + ["FVRA"] * 50 + ["RVFA"] * 50 + ["RVRA"] * 50]
    monkeypatch.setattr(data, "read_jsonl", lambda path: rows)
    monkeypatch.setattr(data, "qas_dir", lambda task: Path("."))
    _, picked = data.load_questions("t1", "t1_train.jsonl", 8)
    assert [r["gt_label"] for r in picked] == ["RVRA"] * 2 + ["RVFA"] * 2 + ["FVRA"] * 2 + ["FVFA"] * 2


def test_dfd_hr_input():
    from PIL import Image

    from mozaic.training.classifiers.dfd_hr.model import tiles
    from mozaic.training.classifiers.dfd_hr.preprocess import Processor

    frames = [Image.new("RGB", (224, 224), (255, 0, 0)) for _ in range(32)]
    item = Processor({})(frames, list(range(32)), 77, None)
    assert item["frames"].shape == (32, 3, 448, 448) and item["length"] == 32
    assert item["frames"].min() >= -1 and item["frames"].max() <= 1
    assert tiles(item["frames"][:2], 2).shape == (8, 3, 224, 224)  # 2 x 2 tiles of 224 per frame


def test_per_step_head_averages_step_logits():
    model = Toy({}, 4)
    batch = {"x": torch.randn(2, 3, 4), "valid": torch.tensor([[True, True, False], [True, True, True]])}
    clip, steps = model(batch)
    assert steps is None and clip.shape == (2, 4)
    model.per_step = True
    clip, steps = model(batch)
    assert steps.shape == (2, 3, 4)
    assert torch.allclose(clip[0], steps[0, :2].mean(0), atol=1e-6)  # the padded step is left out


def test_havic_input_and_stretch():
    pytest.importorskip("torchaudio")
    pytest.importorskip("timm")
    pytest.importorskip("einops")
    from PIL import Image

    from mozaic.training.classifiers.havic.model import split_cls, stretch_time
    from mozaic.training.classifiers.havic.preprocess import Processor

    frames = [Image.new("RGB", (320, 240), (10, 20, 30)) for _ in range(32)]
    audio = np.random.default_rng(0).normal(0, 3000, 16000 * 3).astype(np.float32)
    item = Processor({"audio_sample_rate": 16000})(frames, list(range(32)), 77, audio)
    assert item["frames"].shape == (32, 3, 224, 224) and item["audio"].shape == (1024, 128)
    pos = torch.arange(8.0).repeat_interleave(196)[None, :, None].expand(1, 8 * 196, 2)  # segment k holds k
    stretched = stretch_time(pos, 8, 16)
    assert stretched.shape == (1, 16 * 196, 2)
    assert stretched[0, 0, 0] == 0 and stretched[0, -1, 0] == 7  # the same range, at half steps
    cls, patches = split_cls(torch.zeros(2, 16 * 197, 4), 16)
    assert cls.shape == (2, 16, 4) and patches.shape == (2, 16 * 196, 4)


def test_layernorm_mode_takes_rmsnorms():
    class LlamaRMSNorm(nn.Module):  # a model's own class, as transformers defines one per model
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(8))

    model = Toy({}, 4)
    model.backbone.append(LlamaRMSNorm())
    tuning.apply(model, "layernorm")
    assert {"backbone.1.weight", "backbone.3.weight"} <= trainable(model)


def test_pe_av_input():
    from PIL import Image

    from mozaic.training.classifiers.pe_av.preprocess import Processor

    frames = [Image.new("RGB", (224, 224), (255, 0, 0)) for _ in range(32)]
    audio = np.random.default_rng(0).normal(0, 3000, 77 * 1920).astype(np.float32)  # 77 frames at 25 fps, 48 kHz
    process = Processor({"audio_sample_rate": 48000})
    item = process(frames, list(range(32)), 77, audio)
    assert item["video"].shape == (32, 3, 336, 336) and item["video"].min() >= -1 and item["video"].max() <= 1
    assert item["audio"].shape == (77, 1920) and item["length"] == 77  # one audio token per clip frame
    assert process(frames, list(range(32)), 77, audio[:-100])["audio"].shape == (77, 1920)  # reflect-padded
    video_only = process(frames, list(range(32)), 77, None)  # t2
    assert "audio" not in video_only and video_only["length"] == 32
    with pytest.raises(ValueError):
        Processor({"audio_sample_rate": 16000})


def t3_question(part):
    return {"id": f"t3_{part}_A_x", "task": "task3_identity_check", "part": part,
            "clips": ["videos/1.mov", "videos/2.mov"], "question": "Same person?",
            "options": ["real, same person", "real, different person", "fake of this person",
                        "fake of a different person"],
            "gt_answer": "fake of this person", "gt_target_name": "fake_this_person", "gt_label": "FVRA"}


def test_t3_rows_send_two_videos_or_two_sounds_as_evaluation():
    from mozaic.common.paths import resolve
    from mozaic.training.mllms.dataset import example, placeholders
    from mozaic.training.mllms.run_config import with_sound

    clips = [str(resolve("videos/1.mov")), str(resolve("videos/2.mov"))]
    face, voice = example("t3", t3_question("face"), 32), example("t3", t3_question("voice"), 32)
    assert face["messages"][0]["content"].startswith("Clip 1:<video>Clip 2:<video>Above are 32 frames")
    assert (face["videos"], face["audios"]) == (clips, [])
    assert voice["messages"][0]["content"].startswith("Clip 1:<audio>Clip 2:<audio>Above are the soundtracks")
    assert (voice["videos"], voice["audios"]) == ([], clips)
    assert face["messages"][1] == {"role": "assistant", "content": '{"answer": "fake of this person"}'}
    assert not with_sound("t3") and with_sound("t1") and not with_sound("t2")  # never the sound inside a video
    assert placeholders(False) == "<video>" and placeholders(True) == "<video><audio>"  # t1, t2 and t4 as before


def test_t3_reply_evaluation_would_not_read_is_refused():
    from mozaic.training.mllms.dataset import reply

    with pytest.raises(ValueError, match="would not read"):
        reply("t3", {**t3_question("face"), "gt_answer": "not an option"})


def test_t3_limit_takes_every_class_of_both_parts(monkeypatch):
    from mozaic.training.mllms import dataset

    names = ["fake_different_person", "fake_this_person", "real_different_person", "real_same_person"]
    rows = [{"part": part, "gt_target_name": name} for part in ("face", "voice") for name in names for _ in range(9)]
    monkeypatch.setattr(dataset, "read_jsonl", lambda path: rows)
    monkeypatch.setattr(dataset, "qas_dir", lambda task: Path("."))
    _, picked = dataset.load_questions("t3", "t3_train.jsonl", 16)
    assert len(picked) == 16
    wanted = {(part, name) for part in ("face", "voice") for name in names}
    assert {(r["part"], r["gt_target_name"]) for r in picked} == wanted


def test_t3_config_keeps_a_face_question_whole():
    from mozaic.training.run_train import load_train_config

    config = load_train_config("qwen2_5_omni/t3_lora")
    assert config["task"] == "t3" and config["train"]["select_on"] == "macro.f1"
    assert config["llamafactory"]["cutoff_len"] > 2338 + 20  # a face question's prompt tokens + the reply
    t1 = load_train_config("qwen2_5_omni/t1_lora")  # the recipe as t1's, but for the workers of an A100 node
    assert {**config["train"], "workers": 8} == t1["train"] and config["lora"] == t1["lora"]


def test_t3_voice_clip_is_read_as_sound_alone(monkeypatch):
    pytest.importorskip("llamafactory")
    from mozaic.training.mllms.qwen_omni_inputs import EvalReaderPlugin

    seen = {}

    def process_mm_info(msgs, use_audio_in_video):
        seen.update(content=msgs[0]["content"], in_video=use_audio_in_video)
        return ["the sound"], None, None

    monkeypatch.setattr(EvalReaderPlugin, "process_mm_info", staticmethod(process_mm_info))
    monkeypatch.setattr(EvalReaderPlugin, "sound", False)
    monkeypatch.setattr(EvalReaderPlugin, "frame_size", 224)
    plugin = EvalReaderPlugin.__new__(EvalReaderPlugin)
    dummy = np.zeros(1600)
    assert plugin._regularize_audios(["1.mov", dummy], 16000)["audios"][0] == "the sound"
    assert plugin._regularize_audios([dummy], 16000)["audios"][0] is dummy  # the collator's dummy, as it is
    assert seen["content"][0] == {"type": "audio", "audio": "1.mov"} and seen["in_video"] is False


def test_share_tree_gives_files_written_by_other_libraries_the_group_modes(tmp_path):
    from mozaic.common.io import DIR_MODE, FILE_MODE, share_tree

    shard = tmp_path / "final" / "sub" / "model-00001-of-00002.safetensors"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"x")
    shard.chmod(0o600)  # as safetensors saves it
    share_tree(tmp_path / "final")
    assert shard.stat().st_mode & 0o7777 == FILE_MODE
    assert all(d.stat().st_mode & 0o7777 == DIR_MODE for d in (tmp_path / "final", shard.parent))


def test_a_run_is_finished_once_history_holds_every_epoch(tmp_path):
    from mozaic.training.common import store

    assert not store.finished(tmp_path, 2)  # no history yet
    (tmp_path / "history.jsonl").write_text('{"epoch": 1}\n')
    assert not store.finished(tmp_path, 2)
    (tmp_path / "history.jsonl").write_text('{"epoch": 1}\n{"epoch": 2}\n')
    assert store.finished(tmp_path, 2)
# T3 classifiers: two clips per question (common/pairs.py)


def test_t3_classes_match_evaluation():
    assert CLASSES["t3"] == ["real_same_person", "real_different_person", "fake_this_person", "fake_different_person"]
    probs = dict(zip(CLASSES["t3"], [0.1, 0.2, 0.6, 0.1], strict=True))
    assert from_probs("t3", probs)["pred_answer"] == "fake of this person"
    assert label("t3", t3_question("voice")) == 2


def test_t3_pairs_limit_takes_every_class_of_both_parts(monkeypatch):
    from mozaic.training.common import data, pairs

    rows = [{"part": part, "gt_target_name": name} for part in ("face", "voice") for name in CLASSES["t3"] for _ in range(9)]
    monkeypatch.setattr(data, "read_jsonl", lambda path: rows)
    monkeypatch.setattr(data, "qas_dir", lambda task: Path("."))
    _, picked = pairs.load_questions("t3", "t3_train.jsonl", 16)
    assert len(picked) == 16
    assert {(r["part"], r["gt_target_name"]) for r in picked} == {(p, n) for p in ("face", "voice") for n in CLASSES["t3"]}
    assert pairs.load_questions("t3", "t3_train.jsonl", None)[1] == rows


def test_t3_pairs_send_frames_for_face_and_sound_for_voice(monkeypatch):
    from mozaic.training.common import pairs

    read = []
    monkeypatch.setattr(pairs, "resolve", lambda path: path)
    monkeypatch.setattr(pairs, "read_frames", lambda clip, n: read.append(("frames", clip, n)) or ([clip], [0], 77))
    monkeypatch.setattr(pairs, "load_audio", lambda clip, rate: read.append(("audio", clip, rate)) or bytes(8))

    class P(pairs.PairProcessor, BaseProcessor):
        def __call__(self, frames, indices, total, audio):
            return {"video": frames, "audio": audio}

        def voice(self, audio):
            return {"audio": audio}

    config = {"frames": 32, "audio_sample_rate": 16000}
    face, voice = (pairs.PairDataset("t3", [t3_question(part)], P(config), config)[0] for part in ("face", "voice"))
    assert face["clips"] == [{"video": ["videos/1.mov"], "audio": None}, {"video": ["videos/2.mov"], "audio": None}]
    assert [c["audio"].shape for c in voice["clips"]] == [(4,), (4,)] and "video" not in voice["clips"][0]
    assert read == [("frames", "videos/1.mov", 32), ("frames", "videos/2.mov", 32),
                    ("audio", "videos/1.mov", 16000), ("audio", "videos/2.mov", 16000)]
    assert (face["part"], face["label"], voice["part"], voice["index"]) == ("face", 2, "voice", 0)


def test_t3_pair_batches_keep_the_parts_apart_and_the_order():
    from mozaic.training.common import pairs

    class ToyPair(pairs.PairClassifier, Toy):  # Toy's backbone for both parts, a pair head per part
        def encode_part(self, part, batch):
            return self.encode({"x": batch["x"] if part == "face" else -batch["x"], "valid": batch["valid"]})

    class P(pairs.PairProcessor, BaseProcessor):
        time_dims: ClassVar[dict[str, int]] = {"x": 0}

    def item(part, index, steps):
        clips = [{"x": torch.randn(steps, 4), "length": steps} for _ in range(2)]
        return {"part": part, "clips": clips, "label": index % 4, "index": index}

    items = [item("voice", 0, 3), item("face", 1, 2), item("voice", 2, 5)]
    batch = P({}).collate(items)
    assert batch["face.pos"].tolist() == [1] and batch["voice.pos"].tolist() == [0, 2]
    assert batch["voice.1.x"].shape == (2, 5, 4) and batch["voice.0.valid"].tolist()[0] == [True] * 3 + [False] * 2
    assert batch["label"].tolist() == [0, 1, 2] and batch["index"].tolist() == [0, 1, 2]
    model = ToyPair({}, 4).eval()
    assert model.dynamic_graph and set(model.head) == {"face", "voice"} and model.head["face"][1].in_features == 4 * 8
    logits, steps = model(batch)
    assert steps is None and logits.shape == (3, 4)
    for j, one in enumerate(items):  # each question's logits, as if it were alone in its batch
        assert torch.allclose(logits[j], model(P({}).collate([one]))[0][0], atol=1e-6)
    tuning.apply(model, "frozen")
    assert {n for n in trainable(model) if n.startswith("head")} == {n for n, _ in model.head.named_parameters(prefix="head")}


def test_t3_pair_head_is_blind_to_the_encoders_feature_size():
    from mozaic.training.common import pairs

    class Scaled(pairs.PairClassifier, Toy):  # the same features, 100 times larger (as some of PE-AV's)
        scale = 1.0

        def encode_part(self, part, batch):
            x, valid = self.encode(batch)
            return self.scale * x, valid

    class P(pairs.PairProcessor, BaseProcessor):
        time_dims: ClassVar[dict[str, int]] = {"x": 0}

    clips = [{"x": torch.randn(3, 4), "length": 3} for _ in range(2)]
    batch = P({}).collate([{"part": "face", "clips": clips, "label": 0, "index": 0}])
    model = Scaled({}, 4).eval()
    small = model(batch)[0]
    model.scale = 100.0
    assert torch.allclose(model(batch)[0], small, atol=1e-4)  # each clip's pooled features layer-normed first


def test_t3_pe_av_inputs():
    from PIL import Image

    from mozaic.training.classifiers.pe_av.pair import Processor
    from mozaic.training.classifiers.pe_av.preprocess import Processor as T2Processor

    frames = [Image.new("RGB", (224, 224), (255, 0, 0)) for _ in range(32)]
    audio = np.random.default_rng(0).normal(0, 3000, 77 * 1920).astype(np.float32)  # 77 frames at 25 fps, 48 kHz
    process = Processor({"audio_sample_rate": 48000})
    face, t2 = process.face(frames, list(range(32)), 77), T2Processor({"audio_sample_rate": 48000})(frames, list(range(32)), 77, None)
    assert face.keys() == t2.keys() == {"video", "length"} and torch.equal(face["video"], t2["video"])
    voice = process.voice(audio)  # T1's waveform steps, without the frames
    assert voice.keys() == {"audio", "length"} and voice["audio"].shape == (77, 1920) and voice["length"] == 77


def test_t3_avhubert_inputs():
    pytest.importorskip("python_speech_features")
    from PIL import Image

    from mozaic.training.classifiers.avhubert.pair import Processor
    from mozaic.training.classifiers.avhubert.preprocess import Processor as T1Processor

    total, indices = 77, [0, 2, 5, 76]
    frames = [Image.new("RGB", (224, 224), (200, 200, 200)) for _ in indices]
    audio = np.random.default_rng(0).normal(0, 3000, 77 * 640).astype(np.float32)
    process = Processor({"audio_sample_rate": 16000})
    face = process.face(frames, indices, total)
    assert face.keys() == {"video", "length"} and face["video"].shape == (1, 77, 88, 88)
    voice, t1 = process.voice(audio), T1Processor({"audio_sample_rate": 16000})(frames, indices, total, audio)
    assert voice.keys() == {"audio", "length"} and voice["length"] == 77
    assert torch.equal(voice["audio"], t1["audio"])  # every step of the audio: T1's steps at 25 fps


def test_t3_configs_follow_t1_full():
    from mozaic.evaluation.run_inference import load_model_config
    from mozaic.training.run_train import load_model, load_train_config

    for model in ("pe_av", "avhubert"):
        t3, t1 = load_train_config(f"{model}/t3_full"), load_train_config(f"{model}/t1_full")
        assert (t3["task"], t3["experiment_name"]) == ("t3", t1["experiment_name"])
        assert (t3["questions"]["train"]["file"], t3["questions"]["val"]["file"]) == ("t3_train.jsonl", "t3_val.jsonl")
        same = ("frames", "audio_sample_rate", "backbone", "head", "tuning", "log")
        assert {k: t3[k] for k in same} == {k: t1[k] for k in same}
        memory = {"micro_batch_size": None, "workers": None}  # memory and the CPUs of an A100 node only
        assert {**t3["train"], **memory} == {**t1["train"], **memory}
        test, t1_test = load_model_config(f"{model}/t3/full_trainsmall"), load_model_config(f"{model}/t1/full_trainsmall")
        assert test["trained"] == {"config": f"{model}/t3_full", "data": "small"} and list(test["tasks"]) == ["t3"]
        assert test["experiment_name"] == t1_test["experiment_name"]
        assert test["tasks"]["t3"]["filters"] == t3["questions"]["train"]["filters"] == {"arms": ["vocoded"]}
    assert load_model({"model": "pe_av", "task": "t3"}).__module__ == "mozaic.training.classifiers.pe_av.pair"
    with pytest.raises(ValueError, match="no t3 version"):
        load_model({"model": "havic", "task": "t3"})
