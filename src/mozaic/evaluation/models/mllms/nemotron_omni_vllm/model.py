"""Nemotron 3 Nano Omni, run with vLLM (its own environment: .venv_nemotron_omni_vllm).

How a question goes through this file:
1. run_inference gives the model a batch of questions (batch_size in the config). For each clip it gives
   the clip's file and what to send: its frames, its sound, or both.
2. read() opens each clip with vLLM's own readers: 32 frames spread evenly over the clip (kept at
   224 x 224) and the sound (16 kHz).
3. message() writes the question: per clip its label ("Clip 1:" in t3), a tag where the clip goes, then
   the prompt. t1: one video with its sound. t2 and t3 face: videos without sound. t3 voice: two sounds.
4. vLLM answers the whole batch at once. Greedy answers, reasoning off.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from mozaic.evaluation.base import BaseMLLM, ClipInput, Generation

# Tags that mark where a clip goes in the text: vLLM swaps each tag for the clip's frames or sound.
# A video's frames end up as <image> tokens (we count them to check what the model got).
VIDEO_TAG, SOUND_TAG, FRAME_TOKEN = "<video>", "<so_embedding>", "<image>"


def message(clips: list[ClipInput], prompt: str) -> tuple[str, bool]:
    """The text of the question: per clip its label, then its tag, then the prompt.

        t1:        "<video>\\n" + prompt                     (the sound goes with the video)
        t3 voice:  "Clip 1:\\n<so_embedding>\\nClip 2:\\n<so_embedding>\\n" + prompt

    We write the tags into the text ourselves: if the clips were sent as separate parts, Nemotron's chat
    template would move them all to the start of the message, away from their labels.
    Also returns sound_in_video: True when a clip sends its frames and its sound together (t1).
    """
    text = ""
    for clip in clips:
        if clip.label is not None:  # "Clip 1:" / "Clip 2:" in t3; t1 and t2 have no label
            text += f"{clip.label}\n"
        if clip.send_frames:  # a video (in t1 its sound comes with it)
            text += f"{VIDEO_TAG}\n"
        elif clip.send_audio:  # a sound only (t3 voice questions)
            text += f"{SOUND_TAG}\n"
    sound_in_video = any(c.send_frames and c.send_audio for c in clips)
    return text + prompt, sound_in_video


def request(text: str, videos: list, audios: list, sound_in_video: bool) -> dict[str, Any]:
    """One question as vLLM wants it: the text, the clips, and whether the sound belongs to the video.

    A video is (frames, metadata). The metadata (the clip's 25 fps and which frames were taken) lets vLLM
    write each frame's time next to it ("Frame 1 sampled at 0.00 seconds ..."). A sound is (samples, rate).
    With use_audio_in_video, vLLM puts the clip's sound right after its 32 frames, as one piece.
    """
    media = {"video": videos, "audio": audios}
    return {"prompt": text, "multi_modal_data": {k: v for k, v in media.items() if v},  # only what this question has
            "mm_processor_kwargs": {"use_audio_in_video": sound_in_video}}


def engine_args(frames: int, frame_size: int, patch_size: int) -> dict[str, Any]:
    """Two vLLM settings that come from the config's frames and frame_size.

    Frame size: Nemotron cuts each frame into 16 x 16 pieces (patches), and by default it enlarges a frame to
    1,024 pieces (512 x 512). Our clips are 224 x 224: enlarging adds no detail, only about 5 times more
    tokens. (224 / 16)^2 = 196 pieces keeps the frames at 224, as we did for Qwen.
    num_frames: vLLM tests its memory at start-up with a video of this many frames (otherwise far more).
    """
    return {"hf_overrides": {"vision_config": {"video_target_num_patches": (frame_size // patch_size) ** 2}},
            "media_io_kwargs": {"video": {"num_frames": frames}}}


class Model(BaseMLLM):
    reads_clips = True  # run_inference gives us each clip's file; we read the frames and sound ourselves

    def load(self) -> None:
        # vLLM 0.28 builds a FlashInfer sampling program when it starts, which needs CUDA's compiler (nvcc),
        # and our GPU nodes have none. This uses vLLM's PyTorch sampler instead: the same greedy answers.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        import vllm  # imported here: only this model's environment has vLLM 0.28
        from vllm.multimodal.media import AudioMediaIO, ImageMediaIO, VideoMediaIO

        weights = self.config["weights"]
        model_config = json.loads((Path(weights) / "config.json").read_text())  # the model's own settings
        rate = model_config["sound_config"]["sampling_rate"]
        if self.config["audio_sample_rate"] != rate:  # safety check: our config and the model must agree
            raise ValueError(f"Nemotron hears sound at {rate} Hz, not the config's {self.config['audio_sample_rate']}")
        # vLLM's own readers, the same ones vLLM uses when it serves this model
        self.video_reader = VideoMediaIO(ImageMediaIO(), num_frames=self.config["frames"])  # 32 frames, evenly spaced
        self.audio_reader = AudioMediaIO()  # the clip's sound at its own rate; vLLM changes the rate if needed
        # Load the model on the GPUs (the config's vllm settings: 2 GPUs, memory use, ...)
        self.llm = vllm.LLM(model=weights, **engine_args(self.config["frames"], self.config["frame_size"],
                                                         model_config["patch_size"]), **self.config["vllm"])
        self.tokenizer = self.llm.get_tokenizer()  # turns text into tokens; also holds the chat template
        self.sampling = vllm.SamplingParams(**self.config["generation"])  # greedy, at most 256 tokens
        self.engine = f"vllm {vllm.__version__}"
        self.readers = ThreadPoolExecutor(self.config["readers"])  # read a batch's clips in parallel
        # The ids of a sound token and a frame token, to count them in every prompt (the sound check)
        self.token_ids = {kind: self.tokenizer.convert_tokens_to_ids(token)
                          for kind, token in (("audio", SOUND_TAG), ("video", FRAME_TOKEN))}

    def settings(self) -> dict[str, Any]:
        """What inference_settings.json saves about the run. A run resumed on another GPU or with other vLLM
        settings is refused, as either can change a few answers."""
        from vllm import envs
        from vllm.platforms import current_platform  # the GPU's name through NVML: no CUDA started here

        return {"engine": self.engine, "reader": f"{self.engine} ({envs.VLLM_VIDEO_LOADER_BACKEND} video)",
                "frame_size": self.config["frame_size"], "vllm": self.config["vllm"],
                "chat_template_kwargs": self.config["chat_template_kwargs"],
                "gpu": current_platform.get_device_name(0),
                **({"normalize_audio": True} if self.config.get("normalize_audio") else {})}

    def generate_clips(self, clips: list[ClipInput], prompt: str) -> Generation:
        """One question: a batch of one."""
        return self.generate_batch([(clips, prompt)])[0]

    def generate_batch(self, questions: list[tuple[list[ClipInput], str]]) -> list[Generation]:
        """Read the clips of all the questions in parallel, then vLLM answers them together."""
        requests = list(self.readers.map(lambda question: self.read(*question), questions))
        outs = self.llm.generate(requests, self.sampling, use_tqdm=False)
        return [self.generation(out) for out in outs]

    def read(self, clips: list[ClipInput], prompt: str) -> dict[str, Any]:
        """One question, ready for vLLM: its text in Nemotron's chat format, and its clips' frames and sounds."""
        try:
            text, sound_in_video = message(clips, prompt)
            # Nemotron's chat format around our text; chat_template_kwargs turns reasoning off
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                      add_generation_prompt=True, **self.config["chat_template_kwargs"])
            videos = [self.video_reader.load_file(Path(c.path)).media for c in clips if c.send_frames]  # (frames, metadata)
            audios = [self.audio_reader.load_file(Path(c.path)) for c in clips if c.send_audio]  # (samples, rate)
            if self.config.get("normalize_audio"):  # vLLM 0.28 gives 16-bit sound unscaled: back to [-1, 1]
                audios = [(s / 32768.0 if np.abs(s).max() > 1.0 else s, r) for s, r in audios]
            return request(text, videos, audios, sound_in_video)
        except Exception as e:
            raise RuntimeError(f"{clips[0].path}: {e}") from e  # say which clip failed

    def generation(self, out) -> Generation:
        """vLLM's output as our Generation: the reply, why it stopped, and token counts."""
        ids, reply = list(out.prompt_token_ids), out.outputs[0]
        usage = {"prompt_tokens": len(ids), "completion_tokens": len(reply.token_ids),
                 # the sound check: how many sound and frame tokens the model got (t1, 3 s clip: 39 and 784)
                 **{f"{kind}_tokens": ids.count(i) for kind, i in self.token_ids.items()}}
        return Generation(raw=reply.text, usage=usage, finish_reason=reply.finish_reason)

    def unload(self) -> None:
        self.readers.shutdown()
        self.llm = None
