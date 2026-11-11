"""VideoLLaMA2.1-7B-AV on one GPU with transformers (its own environment: .venv_videollama2_av).

VideoLLaMA2 is in neither transformers nor vLLM: the authors' code (GitHub DAMO-NLP-SG/VideoLLaMA2, branch
audio_visual) is cloned into external-repos/videollama2 (git-ignored, see the README) and called from here.

How a question goes through this file:
1. run_inference gives each clip as it gives it to every model: its 32 frames (224 x 224) and its sound
   (16 kHz WAV), and in t3 its label ("Clip 1:").
2. clip_input() turns a clip into VideoLLaMA2's input with the authors' own steps: the frames through its
   picture processor (SigLIP, which enlarges them to 384 x 384), the sound through its sound steps (a
   128-band spectrogram of a 30 s window). t1: a clip's frames and sound go in together, as one video with its
   soundtrack. t2 and t3 face: silent videos. t3 voice: two sounds.
   The sound three ways (soundtrack in the config): pieces, the authors' audio-visual way for t1 (8 overlapping
   pieces of 2 s, then silence; t3's voice clips: the clip's sound once, then silence); plain, t1's sound once, then
   silence, as t3's voice clips (a check); or nopad, only the clip's own sound (3 s or 6 s),
   without silence or repeats, for t1, t4 and t3's voice clips alike (videollama2_av/t1/zeroshot.yaml,
   videollama2_av/t4/zeroshot.yaml).
3. message() writes the question: per clip its label and its tag (<video> or <audio>), then the prompt.
   VideoLLaMA2 puts each clip's features where its tag is, in order.
4. Greedy answer, one question at a time, in float16, as the authors' mm_infer runs it.
"""

from __future__ import annotations

import io
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from mozaic.evaluation.base import BaseMLLM, ClipInput, Generation

# The authors' tags (videollama2/constants.py); load() checks they still match.
TAGS = {"video": "<video>", "audio": "<audio>"}


def kind(clip: ClipInput) -> str:
    """What VideoLLaMA2 gets of a clip: a video (its frames, with its sound when it has one) or a sound."""
    if clip.frames:
        return "video"
    if clip.audio_wav is not None:
        return "audio"
    raise ValueError("a clip with neither frames nor sound")


def message(clips: list[ClipInput], prompt: str) -> tuple[str, str]:
    """The user's text and its tag: per clip its label (t3) and its tag, then the prompt. One kind of tag per
    question, as the authors' tokenizer helper takes one: t1 and t2 one video, t3 two videos or two sounds."""
    kinds = {kind(clip) for clip in clips}
    if len(kinds) != 1:
        raise ValueError(f"one kind of clip per question, got {sorted(kinds)}")
    tag = TAGS[kinds.pop()]
    lines = []
    for clip in clips:
        if clip.label is not None:  # "Clip 1:" / "Clip 2:" in t3; t1 and t2 have no label
            lines.append(clip.label)
        lines.append(tag)
    return "\n".join([*lines, prompt]), tag  # one clip: "<video>\n{prompt}", the authors' layout


def wav_samples(wav: bytes, rate: int) -> np.ndarray:
    """16-bit mono WAV bytes as float samples in [-1, 1]."""
    with wave.open(io.BytesIO(wav)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (rate, 1, 2):
            raise ValueError(f"expected {rate} Hz mono 16-bit sound, got {w.getframerate()} Hz, "
                             f"{w.getnchannels()} channels, {8 * w.getsampwidth()} bits")
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def soundtrack(samples: np.ndarray, rate: int):
    """A video's sound as the authors' process_audio_from_video makes it, from our samples instead of their
    ffmpeg call (which, failing, would quietly give silence): 8 windows of 2 s spread over the clip, joined,
    padded to 30 s, then a 128-band spectrogram."""
    import torch
    import torchaudio.compliance.kaldi as ta_kaldi
    from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
    from videollama2.mm_utils import get_clip_timepoints

    sampler = ConstantClipsPerVideoSampler(clip_duration=2, clips_per_video=8)
    windows = [samples[int(start * rate) : int(end * rate)]
               for start, end in get_clip_timepoints(sampler, len(samples) / rate)]
    wav = torch.cat([torch.from_numpy(w) for w in windows])
    if len(wav) > 30 * rate:  # theirs takes a random 30 s; 8 windows of 2 s never get there
        raise ValueError(f"{len(wav) / rate:.1f} s of sound windows, more than 30 s")
    wav = torch.nn.functional.pad(wav, (0, 30 * rate - len(wav)))
    fbank = ta_kaldi.fbank(wav.unsqueeze(0) * 2**15, num_mel_bins=128, sample_frequency=16000, frame_length=25,
                           frame_shift=10)
    return fbank.to(torch.bfloat16).unsqueeze(0)


def sound_once(wav: bytes):
    """A clip's sound once, as the authors' process_audio_file makes it (it reads a WAV file): the clip's own sound,
    padded with silence to 30 s, then the same spectrogram. t3's voice questions, and t1 with soundtrack: plain."""
    from videollama2.mm_utils import process_audio_file

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(wav)
        f.flush()
        return process_audio_file(f.name)


def nopad(samples: np.ndarray, rate: int):
    """Only the clip's own sound: process_audio_file's spectrogram of our samples (as float64, as it reads a WAV),
    without its padding to 30 s. soundtrack: nopad, for t1, t4 and t3's voice clips."""
    import torch
    import torchaudio.compliance.kaldi as ta_kaldi

    wav = torch.from_numpy(samples.astype(np.float64)).unsqueeze(0) * 2**15
    fbank = ta_kaldi.fbank(wav, num_mel_bins=128, sample_frequency=rate, frame_length=25, frame_shift=10)
    return fbank.to(torch.bfloat16).unsqueeze(0)


def code_commit(code: Path) -> str:
    """The commit of the authors' code in use, from the clone's .git."""
    head = (code / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        return (code / ".git" / head[5:]).read_text().strip()
    return head


class Model(BaseMLLM):
    def load(self) -> None:
        import torch

        self.code = Path(self.config["code"]).resolve()
        sys.path.insert(0, str(self.code))
        from transformers import AutoConfig, AutoTokenizer
        from videollama2.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_VIDEO_TOKEN
        from videollama2.mm_utils import MAX_FRAMES
        from videollama2.model import Videollama2Qwen2ForCausalLM

        if TAGS != {"video": DEFAULT_VIDEO_TOKEN, "audio": DEFAULT_AUDIO_TOKEN}:
            raise ValueError("the authors' tags changed")
        if self.config["frames"] > MAX_FRAMES:  # their video code keeps the first MAX_FRAMES frames only
            raise ValueError(f"VideoLLaMA2 takes at most {MAX_FRAMES} frames per video")
        if self.config["audio_sample_rate"] != 16000:  # their sound steps are written for 16 kHz
            raise ValueError("VideoLLaMA2 takes 16 kHz sound")
        if self.config.get("soundtrack", "pieces") not in ("pieces", "plain", "nopad"):
            raise ValueError("soundtrack is pieces (the authors' way), plain (the clip's sound once) or nopad "
                             "(only the clip's own sound)")

        # As the authors' load_pretrained_model and model_init load this checkpoint (videollama2_qwen2, float16),
        # with one change: the picture encoder comes from our copy, not by its hub id (runs set HF_HUB_OFFLINE=1).
        path = self.config["weights"]
        config = AutoConfig.from_pretrained(path)
        config.mm_vision_tower = self.config["vision_tower"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        if self.tokenizer.pad_token is None and self.tokenizer.unk_token is not None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model = Videollama2Qwen2ForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, config=config,
                                                                 torch_dtype=torch.float16, device_map="auto")
        tower = self.model.get_vision_tower()
        if not tower.is_loaded:
            tower.load_model()
        tower.to(device="cuda", dtype=torch.float16)
        self.processor = tower.image_processor
        eos = self.model.generation_config.eos_token_id  # <|im_end|> and <|endoftext|>
        self.end_ids = set(eos) if isinstance(eos, list) else {eos}

        # The sound check of every question: features per kind, counted at the two projectors that turn a
        # clip's frames and sound into the language model's input (called once per question).
        self.counts = {"video": 0, "audio": 0}
        for name, projector in (("video", self.model.get_model().mm_projector),
                                ("audio", self.model.get_model().mm_projector_a)):
            projector.register_forward_hook(lambda _m, _i, out, name=name: self.count(name, out))

    def count(self, name: str, out) -> None:
        self.counts[name] += out.shape[0] * out.shape[1]  # clips x features per clip

    def settings(self) -> dict[str, Any]:
        """What inference_settings.json saves about the run. A run resumed on another GPU is refused."""
        import torch
        import transformers

        way = self.config.get("soundtrack", "pieces")
        plain = {"soundtrack": way} if way != "pieces" else {}  # only when set
        return {"engine": f"transformers {transformers.__version__}, torch {torch.__version__}",
                "code": f"github.com/DAMO-NLP-SG/VideoLLaMA2 {code_commit(self.code)}",
                "vision_tower": self.config["vision_tower"], "dtype": "float16",
                "gpu": torch.cuda.get_device_name(0), **plain}

    def clip_input(self, clip: ClipInput):
        """A clip as VideoLLaMA2 takes it, and its kind: (tensor, "audio"), (tensor, "video"), or
        ({"video", "audio"}, "video") for frames with their sound."""
        from videollama2.mm_utils import process_video

        rate = self.config["audio_sample_rate"]
        way = self.config.get("soundtrack", "pieces")
        if kind(clip) == "audio":
            if way == "nopad":
                return nopad(wav_samples(clip.audio_wav, rate), rate), "audio"
            return sound_once(clip.audio_wav), "audio"
        video = process_video(clip.frames, self.processor, aspect_ratio=None, num_frames=len(clip.frames))
        if clip.audio_wav is None:
            return video, "video"
        if way == "nopad":
            return {"video": video, "audio": nopad(wav_samples(clip.audio_wav, rate), rate)}, "video"
        if way == "plain":
            return {"video": video, "audio": sound_once(clip.audio_wav)}, "video"
        return {"video": video, "audio": soundtrack(wav_samples(clip.audio_wav, rate), rate)}, "video"

    def generate(self, frames: list, audio_wav: bytes | None, prompt: str) -> Generation:
        return self.generate_clips([ClipInput(frames=frames, audio_wav=audio_wav)], prompt)

    def generate_clips(self, clips: list[ClipInput], prompt: str) -> Generation:
        """One question, as the authors' mm_infer asks it: the chat template, then generate() with the clips."""
        import torch
        from videollama2.mm_utils import KeywordsStoppingCriteria, tokenizer_multimodal_token

        text, tag = message(clips, prompt)
        images = []
        for clip in clips:
            data, modal = self.clip_input(clip)
            data = {k: v.half().cuda() for k, v in data.items()} if isinstance(data, dict) else data.half().cuda()
            images.append((data, modal))
        chat = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                  add_generation_prompt=True)
        input_ids = tokenizer_multimodal_token(chat, self.tokenizer, tag, return_tensors="pt").unsqueeze(0).cuda()
        stop = KeywordsStoppingCriteria([self.tokenizer.eos_token], self.tokenizer, input_ids)

        self.counts = {"video": 0, "audio": 0}
        start = time.time()
        with torch.inference_mode():
            out = self.model.generate(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id).long(),
                                      images=images, use_cache=True, stopping_criteria=[stop],
                                      pad_token_id=self.tokenizer.eos_token_id, **self.config["generation"])
        new = out[0].tolist()  # the answer only: the prompt went in as embeddings
        usage = {
            "prompt_tokens": int(input_ids.shape[1]),  # text tokens, each clip's tag counted once
            "completion_tokens": len(new),
            "video_tokens": self.counts["video"],
            "audio_tokens": self.counts["audio"],
            "seconds": round(time.time() - start, 2),
        }
        return Generation(
            raw=self.tokenizer.decode(new, skip_special_tokens=True).strip(),
            usage=usage,
            finish_reason="stop" if new and new[-1] in self.end_ids else "length",
        )

    def unload(self) -> None:
        import torch

        self.model = None
        torch.cuda.empty_cache()
