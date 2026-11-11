"""Uni-MoE-2.0-Omni (HIT-TMG) on one GPU with transformers (its own environment: .venv_uni_moe_2_omni).

Uni-MoE-2.0-Omni is in neither transformers nor vLLM: the authors' code (GitHub HITsz-TMG/Uni-MoE, folder Uni-MoE-2)
is cloned into external-repos/uni_moe_2 (git-ignored, see the README) and called from here, unchanged.

How a question goes through this file:
1. run_inference gives each clip's file and what to send of it; clip_media() reads it with the same reader as every
   model: 32 frames (224 x 224), evenly spaced, and the sound at 16 kHz.
2. clip_media() turns the clip into Uni-MoE's input with the authors' own steps: each frame enlarged to 384 x 384 and
   through its picture processor (the model makes 144 pieces of each), with the time between frames as its video
   reader gives it; the sound cut in 30 s windows, as its sound encoder takes them (the clip's sound, then silence).
3. message() writes the question as the authors' example does: per clip its label and its tags (<video>, <audio>),
   each on its own line, then the prompt; their processor writes the chat template and the pieces' places.
4. Its sound part sums each 30 s window up in 200 pieces. silence (config): kept, the authors' way (the 200 pieces
   read every step of the window), or cut (they read only the steps of the clip's own sound): summarize().
5. Greedy answer, one question at a time, in bfloat16, as the authors' examples run it. With answer_choices on (the
   config), the reply can only be {"answer": "<one of the question's options>"}, as MiniCPM-o's: Uni-MoE often writes
   the option alone ("face swap"), which the prompts' parse() can't read.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from mozaic.common.frame_sampler import load_audio, read_frames
from mozaic.evaluation.base import BaseMLLM, ClipInput, Generation
from mozaic.evaluation.models.mllms.answer_choices import choice_filter
from mozaic.evaluation.models.mllms.minicpm_o_vllm.model import answer_choices
from mozaic.evaluation.models.mllms.ola.model import when_answered

# The authors' tags and what they become (examples/cookbook_video.py).
TAGS = {"<video>": "<|vision_start|><|video_pad|><|vision_end|>", "<audio>": "<|audio_start|><|audio_pad|><|audio_end|>"}
WINDOW = 480_000  # sound samples of one 30 s window at 16 kHz, as their processor cuts the sound
STEP = 320  # sound samples per step of its Whisper encoder (1,500 steps a window): 3.08 s are 154 steps, 6.12 s 306


def tags(clip: ClipInput) -> list[str]:
    """A clip's tags: a video with its soundtrack <video> then <audio>, as the authors' example asks about a video
    with its sound (the same file as video and as sound); a silent video <video>; a sound alone <audio>."""
    out = (["<video>"] if clip.send_frames else []) + (["<audio>"] if clip.send_audio else [])
    if not out:
        raise ValueError("a clip with neither frames nor sound")
    return out


def message(clips: list[ClipInput], prompt: str) -> str:
    """The user's text: per clip its label (t3) and its tags, each on its own line, then the prompt (the authors'
    example: "<video>\\n<audio>\\n" then the question)."""
    lines = []
    for clip in clips:
        if clip.label is not None:  # "Clip 1:" / "Clip 2:" in t3; t1 and t2 have no label
            lines.append(clip.label)
        lines += tags(clip)
    return "\n".join([*lines, prompt])


def own_steps(samples: int, windows: int) -> list[int]:
    """Per 30 s window of a sound, the encoder steps that hold the sound itself: 1,500 for a full window, the rest
    for the last one. At most `windows` windows, as their processor keeps (whisper_audio_time)."""
    full, rest = divmod(samples, WINDOW)
    if full and not rest:  # their processor cuts exactly 30 s as one window
        full, rest = full - 1, WINDOW
    return ([WINDOW // STEP] * full + [math.ceil(rest / STEP)])[:windows]


def code_commit(code: Path) -> str:
    """The commit of the authors' code in use, from the clone's .git (one folder up: the repo holds Uni-MoE-2)."""
    head = (code.parent / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        return (code.parent / ".git" / head[5:]).read_text().strip()
    return head


class Model(BaseMLLM):
    reads_clips = True  # run_inference gives us each clip's file: we read its frames, their numbers and its sound

    def load(self) -> None:
        import torch

        if self.config["audio_sample_rate"] != 16000:  # its sound steps are written for 16 kHz
            raise ValueError("Uni-MoE takes 16 kHz sound")
        if self.config.get("silence", "kept") not in ("kept", "cut"):
            raise ValueError("silence is kept (the authors' way) or cut (only the clip's own sound)")
        # deepspeed (its MoE layers) asks a CUDA compiler for its version when imported on a GPU node, and ours have
        # none: cuda_home in the config, CUDA 12.1's nvcc as torch's. Nothing is compiled.
        os.environ.setdefault("CUDA_HOME", self.config["cuda_home"])
        self.code = Path(self.config["code"]).resolve()
        sys.path.insert(0, str(self.code))
        from uni_moe.model import deepspeed_moe_inference_utils  # noqa: F401  their patch: its MoE layers on one GPU
        from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
        from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
        from uni_moe.vision_utils import process_unires_video

        # As the authors' examples load it (examples/cookbook_video.py): bfloat16, FlashAttention 2, on the GPU.
        path = self.config["weights"]
        self.processor = Qwen2VLProcessor.from_pretrained(path)
        self.model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(
            path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).cuda()
        self.processor.data_args = self.model.config
        self.process_unires_video = process_unires_video
        tokenizer = self.processor.tokenizer
        self.pad_ids = {name: tokenizer.convert_tokens_to_ids(f"<|{name}_pad|>") for name in ("video", "audio")}
        eos = self.model.generation_config.eos_token_id  # <|im_end|>, where generate stops
        self.end_ids = sorted(set(eos) if isinstance(eos, list) else {eos})

        # Their summary step (200 pieces per 30 s window, read from the window's 1,500 encoder steps), called here
        # per window with the steps to read: all of them (kept) or those of the clip's own sound (cut).
        aligner = self.model.audio_aligner
        whole = aligner.forward
        self.steps: list[int] = []  # per window of the question's sounds, the steps of its own sound
        self.read: list[int] = []  # per window, the steps its 200 pieces read (the check of the cut)

        def summarize(encoder_output, **kwargs):
            if self.config.get("silence", "kept") == "kept":
                self.read += [encoder_output.shape[1]] * encoder_output.shape[0]
                return whole(encoder_output=encoder_output, **kwargs)
            pieces = []
            for window, steps in zip(encoder_output, self.steps, strict=True):
                self.read.append(steps)
                pieces.append(whole(encoder_output=window[None, :steps], **kwargs))
            return torch.cat(pieces)

        aligner.forward = summarize

    def settings(self) -> dict[str, Any]:
        """What inference_settings.json saves about the run. A run resumed on another GPU is refused."""
        import deepspeed
        import torch
        import transformers

        return {"engine": f"transformers {transformers.__version__}, torch {torch.__version__}, "
                          f"deepspeed {deepspeed.__version__}",
                "code": f"github.com/HITsz-TMG/Uni-MoE {code_commit(self.code)}", "dtype": "bfloat16",
                "silence": self.config.get("silence", "kept"), "answer_choices": self.config["answer_choices"],
                "gpu": torch.cuda.get_device_name(0)}

    def clip_media(self, clip: ClipInput) -> dict[str, Any]:
        """A clip as their processor makes it from a video file: the frames enlarged to 384 x 384 (video_pixel_upbound)
        and through process_unires_video, with the seconds between frames (their process_video_with_decord: frames
        sent / frames in the clip x fps); the sound as samples, which their processor cuts in 30 s windows."""
        import av

        out: dict[str, Any] = {}
        if clip.send_frames:
            frames, indices, total = read_frames(clip.path, self.config["frames"])
            with av.open(clip.path) as container:
                fps = float(container.streams.video[0].average_rate)
            size = self.model.config.video_pixel_upbound
            frames = np.array([np.array(frame.resize((size, size))) for frame in frames])
            pixels, grid = self.process_unires_video(frames, self.processor.image_processor,
                                                     self.model.config.image_grid_pinpoints)
            out["video"] = (pixels, grid, 1 / (len(indices) / total * fps))
        if clip.send_audio:
            samples = np.frombuffer(load_audio(clip.path, self.config["audio_sample_rate"]), dtype="<i2")
            out["audio"] = samples.astype(np.float32) / 32768.0
        return out

    def inputs(self, clips: list[ClipInput], prompt: str):
        """A question as the authors' example builds it (initial_input): the chat template, the tags turned into their
        tokens, then their processor (the sound and the text); the videos, read here, added as it would add them."""
        import torch

        videos, sounds = [], []
        for clip in clips:
            got = self.clip_media(clip)
            if "video" in got:
                videos.append(got["video"])
            if "audio" in got:
                sounds.append(got["audio"])
        text = self.processor.apply_chat_template([{"role": "user", "content": [{"type": "text",
                                                   "text": message(clips, prompt)}]}],
                                                  tokenize=False, add_generation_prompt=True)
        for tag, tokens in TAGS.items():
            text = text.replace(tag, tokens)
        pool = self.model.config.vision_spatial_pool_stride ** 2  # pieces per picture tile: 12 x 12
        for _, grid, _ in videos:  # each video's places, as their processor writes them when it reads the video
            text = text.replace("<|video_pad|>", "<|placeholder|>" * int(grid.prod() * pool), 1)
        text = text.replace("<|placeholder|>", "<|video_pad|>")
        inputs = self.processor(text=text, audios=sounds or None, padding=True, return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        if videos:
            inputs["pixel_values_videos"] = torch.cat([pixels for pixels, _, _ in videos])
            inputs["video_grid_thw"] = torch.cat([grid for _, grid, _ in videos])
            inputs["second_per_grid_ts"] = torch.tensor([seconds for _, _, seconds in videos])
        windows = self.model.config.whisper_audio_time
        self.steps = [steps for sound in sounds for steps in own_steps(len(sound), windows)]
        return inputs

    def generate_clips(self, clips: list[ClipInput], prompt: str) -> Generation:
        """One question, as the authors' example asks it (infer_single): their inputs, then their generate()."""
        import torch

        inputs = self.inputs(clips, prompt).to(device="cuda")
        for name in ("pixel_values_videos", "audio_features"):
            if name in inputs:
                inputs[name] = inputs[name].to(dtype=torch.bfloat16)
        ids = inputs["input_ids"]
        tokenizer = self.processor.tokenizer
        extra = {}
        if self.config["answer_choices"]:  # generate passes the prompt too: the filter sees the reply only
            replies = [tokenizer.encode(choice, add_special_tokens=False) for choice in answer_choices(prompt)]
            allowed, length = choice_filter(replies, self.end_ids), ids.shape[-1]
            extra["prefix_allowed_tokens_fn"] = lambda batch_id, so_far: allowed(batch_id, so_far[length:])
        stop = None
        if self.config.get("stop_when_answered"):  # T4: once the reply holds both parts (the config)
            stop = when_answered(tokenizer)
            extra["stopping_criteria"] = [stop]
        self.read = []
        start = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, use_cache=True, pad_token_id=tokenizer.eos_token_id,
                                      **self.config["generation"], **extra)
        new = out[0, ids.shape[-1]:].tolist()
        usage = {
            "prompt_tokens": int(ids.shape[-1]),  # all the language model got: text and the pieces' places
            "completion_tokens": len(new),
            "video_tokens": int((ids == self.pad_ids["video"]).sum()),  # the model checks it made as many pieces
            "audio_tokens": int((ids == self.pad_ids["audio"]).sum()),  # 200 per 30 s window
            "audio_steps": sum(self.read),  # encoder steps the sound's 200 pieces read: 1,500 a window, or the cut
            "seconds": round(time.time() - start, 2),
        }
        answered = stop is not None and stop.stopped
        if answered:  # its answer was whole: the reply ends there, as at its end word
            usage["stopped_when_answered"] = True
        return Generation(
            raw=tokenizer.decode(new, skip_special_tokens=True).strip(),
            usage=usage,
            finish_reason="stop" if answered or (new and new[-1] in self.end_ids) else "length",
        )

    def unload(self) -> None:
        import torch

        self.model = None
        torch.cuda.empty_cache()
