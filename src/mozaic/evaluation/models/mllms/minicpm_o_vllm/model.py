"""MiniCPM-o 4.5, run with vLLM (its own environment: .venv_minicpm_o_vllm, the same versions as Nemotron's).

How a question goes through this file:
1. run_inference gives the model a batch of questions (batch_size in the config). For each clip it gives
   the clip's file and what to send: its frames, its sound, or both.
2. read() opens each clip with vLLM's own readers: 32 frames spread evenly over the clip and the sound
   (16 kHz). MiniCPM-o turns each frame into 64 tokens itself.
3. message() writes the question: per clip its label ("Clip 1:" in t3), then a tag for its video and/or its
   sound, then the prompt. t1: the video, then its sound (vLLM takes MiniCPM-o's sound as its own part).
   t2 and t3 face: videos without sound. t3 voice: two sounds.
4. vLLM answers the whole batch at once. Greedy answers; thinking off (the config's chat_template_kwargs).
   With answer_choices on (the config), the reply can only be {"answer": "<one of the question's options>"}.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from mozaic.evaluation.base import BaseMLLM, ClipInput, Generation

# Tags that mark where a clip goes in the text: vLLM swaps each tag for the clip's frames or sound.
VIDEO_TAG, SOUND_TAG = "(<video>./</video>)", "(<audio>./</audio>)"


def message(clips: list[ClipInput], prompt: str) -> str:
    """The text of the question: per clip its label, then its tag(s), then the prompt.

        t1:        "(<video>./</video>)\\n(<audio>./</audio>)\\n" + prompt
        t3 voice:  "Clip 1:\\n(<audio>./</audio>)\\nClip 2:\\n(<audio>./</audio>)\\n" + prompt

    MiniCPM-o's chat template takes plain text only, so the tags are written into the text, each after its
    clip's label.
    """
    text = ""
    for clip in clips:
        if clip.label is not None:  # "Clip 1:" / "Clip 2:" in t3; t1 and t2 have no label
            text += f"{clip.label}\n"
        if clip.send_frames:  # a video (its frames)
            text += f"{VIDEO_TAG}\n"
        if clip.send_audio:  # a sound: in t1 right after its video, in t3 voice questions on its own
            text += f"{SOUND_TAG}\n"
    return text + prompt


def request(text: str, videos: list, audios: list) -> dict[str, Any]:
    """One question as vLLM wants it: the text and the clips. A video is its frames (MiniCPM-o takes no frame
    times), a sound is (samples, rate)."""
    media = {"video": videos, "audio": audios}
    return {"prompt": text, "multi_modal_data": {k: v for k, v in media.items() if v}}  # only what this question has


def answer_choices(prompt: str) -> list[str]:
    """The only replies allowed: the prompts' JSON with each of the question's options, read from its Options line.

        'Options: "neither", "both", ...'  ->  ['{"answer": "neither"}', '{"answer": "both"}', ...]

    Without it MiniCPM-o often writes a sentence around its JSON, which the prompts' parse() can't read.
    """
    line = re.findall(r"^Options: (.*)$", prompt, flags=re.MULTILINE)[-1]  # the prompt's own Options line (the last)
    return [json.dumps({"answer": option}) for option in re.findall(r'"([^"]*)"', line)]  # t3 options have commas


def count_between(ids: list[int], start: int, end: int) -> int:
    """How many tokens sit between the start and end marks (every pair): a clip's frame or sound tokens."""
    count, inside = 0, False
    for token in ids:
        if token == start:
            inside = True
        elif token == end:
            inside = False
        elif inside:
            count += 1
    return count


class Model(BaseMLLM):
    reads_clips = True  # run_inference gives us each clip's file; we read the frames and sound ourselves

    def load(self) -> None:
        # vLLM 0.28 builds a FlashInfer sampling program when it starts, which needs CUDA's compiler (nvcc),
        # and our GPU nodes have none. This uses vLLM's PyTorch sampler instead: the same greedy answers.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        import vllm  # imported here: only this model's environment has vLLM 0.28
        from vllm.multimodal.media import AudioMediaIO, ImageMediaIO, VideoMediaIO

        if self.config["audio_sample_rate"] != 16000:  # safety check: MiniCPM-o's sound encoder (Whisper) hears 16 kHz
            raise ValueError(f"MiniCPM-o hears sound at 16000 Hz, not the config's {self.config['audio_sample_rate']}")
        # vLLM's own readers, the same ones vLLM uses when it serves a model
        self.video_reader = VideoMediaIO(ImageMediaIO(), num_frames=self.config["frames"])  # 32 frames, evenly spaced
        self.audio_reader = AudioMediaIO()  # the clip's sound at its own rate; vLLM changes the rate if needed
        self.llm = vllm.LLM(model=self.config["weights"], **self.config["vllm"])  # loads the model on the GPU
        self.tokenizer = self.llm.get_tokenizer()  # turns text into tokens; also holds the chat template
        self.sampling = vllm.SamplingParams(**self.config["generation"])  # greedy, at most 256 tokens
        self.engine = f"vllm {vllm.__version__}"
        self.readers = ThreadPoolExecutor(self.config["readers"])  # read a batch's clips in parallel
        # The marks around a frame's and a sound's tokens, to count them in every prompt (the sound check)
        self.marks = {"audio": (self.tokenizer.audio_start_id, self.tokenizer.audio_end_id),
                      "video": (self.tokenizer.im_start_id, self.tokenizer.im_end_id)}

    def settings(self) -> dict[str, Any]:
        """What inference_settings.json saves about the run. A run resumed on another GPU or with other vLLM
        settings is refused, as either can change a few answers."""
        from vllm import envs
        from vllm.platforms import current_platform  # the GPU's name through NVML: no CUDA started here

        return {"engine": self.engine, "reader": f"{self.engine} ({envs.VLLM_VIDEO_LOADER_BACKEND} video)",
                "vllm": self.config["vllm"], "chat_template_kwargs": self.config["chat_template_kwargs"],
                "answer_choices": self.config["answer_choices"], "gpu": current_platform.get_device_name(0),
                **({"normalize_audio": True} if self.config.get("normalize_audio") else {})}

    def generate_clips(self, clips: list[ClipInput], prompt: str) -> Generation:
        """One question: a batch of one."""
        return self.generate_batch([(clips, prompt)])[0]

    def generate_batch(self, questions: list[tuple[list[ClipInput], str]]) -> list[Generation]:
        """Read the clips of all the questions in parallel, then vLLM answers them together."""
        requests = list(self.readers.map(lambda question: self.read(*question), questions))
        outs = self.llm.generate(requests, [self.sampling_for(prompt) for _, prompt in questions], use_tqdm=False)
        return [self.generation(out) for out in outs]

    def sampling_for(self, prompt: str) -> Any:
        """How vLLM picks the reply's tokens: greedy, at most 256 tokens. With answer_choices on (the config), it may
        only write one of answer_choices(prompt) (vLLM's structured outputs); MiniCPM-o still picks which one."""
        if not self.config["answer_choices"]:
            return self.sampling
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        choices = StructuredOutputsParams(choice=answer_choices(prompt))
        return SamplingParams(**self.config["generation"], structured_outputs=choices)

    def read(self, clips: list[ClipInput], prompt: str) -> dict[str, Any]:
        """One question, ready for vLLM: its text in MiniCPM-o's chat format, and its clips' frames and sounds."""
        try:
            # MiniCPM-o's chat format around our text; chat_template_kwargs turns thinking off
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": message(clips, prompt)}], tokenize=False,
                                                      add_generation_prompt=True, **self.config["chat_template_kwargs"])
            videos = [self.video_reader.load_file(Path(c.path)).media[0] for c in clips if c.send_frames]  # frames
            audios = [self.audio_reader.load_file(Path(c.path)) for c in clips if c.send_audio]  # (samples, rate)
            if self.config.get("normalize_audio"):  # vLLM 0.28 gives 16-bit sound unscaled: back to [-1, 1]
                audios = [(s / 32768.0 if np.abs(s).max() > 1.0 else s, r) for s, r in audios]
            return request(text, videos, audios)
        except Exception as e:
            raise RuntimeError(f"{clips[0].path}: {e}") from e  # say which clip failed

    def generation(self, out) -> Generation:
        """vLLM's output as our Generation: the reply, why it stopped, and token counts."""
        ids, reply = list(out.prompt_token_ids), out.outputs[0]
        usage = {"prompt_tokens": len(ids), "completion_tokens": len(reply.token_ids),
                 # the sound check: how many sound and frame tokens the model got
                 **{f"{kind}_tokens": count_between(ids, *marks) for kind, marks in self.marks.items()}}
        return Generation(raw=reply.text, usage=usage, finish_reason=reply.finish_reason)

    def unload(self) -> None:
        self.readers.shutdown()
        self.llm = None
