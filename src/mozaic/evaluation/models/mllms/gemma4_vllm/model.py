"""Gemma 4 (12B), run with vLLM (its own environment: .venv_gemma4_vllm, the same versions as Nemotron's).

How a question goes through this file:
1. run_inference gives the model a batch of questions (batch_size in the config). For each clip it gives
   the clip's file and what to send: its frames, its sound, or both.
2. read() opens each clip with vLLM's own readers: 32 frames spread evenly over the clip and the sound
   (16 kHz). Gemma itself enlarges each 224 x 224 frame to 384 x 384 (its smallest size for video frames).
3. message() writes the question: per clip its label ("Clip 1:" in t3), then its video and/or its sound,
   then the prompt. t1: the video, then its sound (Gemma cannot take the sound inside the video).
   t2 and t3 face: videos without sound. t3 voice: two sounds.
4. vLLM answers the whole batch at once. Greedy answers; Gemma's thinking is off by default.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from mozaic.evaluation.base import BaseMLLM, ClipInput, Generation


def message(clips: list[ClipInput], prompt: str) -> list[dict[str, Any]]:
    """The user message, as the other models get it: per clip its label (t3), its video and/or its sound,
    then the prompt. Gemma's chat template keeps the parts in this order."""
    content: list[dict[str, Any]] = []
    for clip in clips:
        if clip.label is not None:  # "Clip 1:" / "Clip 2:" in t3; t1 and t2 have no label
            content.append({"type": "text", "text": clip.label})
        if clip.send_frames:  # a video (its frames)
            content.append({"type": "video"})
        if clip.send_audio:  # a sound: in t1 right after its video, in t3 voice questions on its own
            content.append({"type": "audio"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def request(text: str, videos: list, audios: list) -> dict[str, Any]:
    """One question as vLLM wants it: the text and the clips.

    A video is (frames, metadata). The metadata (the clip's 25 fps and which frames were taken) lets vLLM
    write each frame's time before it ("00:01 ..."). A sound is (samples, rate).
    """
    media = {"video": videos, "audio": audios}
    return {"prompt": text, "multi_modal_data": {k: v for k, v in media.items() if v}}  # only what this question has


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
        processor_config = json.loads((Path(weights) / "processor_config.json").read_text())
        rate = processor_config["feature_extractor"]["sampling_rate"]
        if self.config["audio_sample_rate"] != rate:  # safety check: our config and the model must agree
            raise ValueError(f"Gemma hears sound at {rate} Hz, not the config's {self.config['audio_sample_rate']}")
        # vLLM's own readers, the same ones vLLM uses when it serves a model
        self.video_reader = VideoMediaIO(ImageMediaIO(), num_frames=self.config["frames"])  # 32 frames, evenly spaced
        self.audio_reader = AudioMediaIO()  # the clip's sound at its own rate; vLLM changes the rate if needed
        self.llm = vllm.LLM(model=weights, **self.config["vllm"])  # loads the model on the GPU
        self.tokenizer = self.llm.get_tokenizer()  # turns text into tokens; also holds the chat template
        self.sampling = vllm.SamplingParams(**self.config["generation"])  # greedy, at most 256 tokens
        self.engine = f"vllm {vllm.__version__}"
        self.readers = ThreadPoolExecutor(self.config["readers"])  # read a batch's clips in parallel
        # The ids of a sound token and a frame token, to count them in every prompt (the sound check)
        self.token_ids = {kind: model_config[f"{kind}_token_id"] for kind in ("audio", "video")}

    def settings(self) -> dict[str, Any]:
        """What inference_settings.json saves about the run. A run resumed on another GPU or with other vLLM
        settings is refused, as either can change a few answers."""
        from vllm import envs
        from vllm.platforms import current_platform  # the GPU's name through NVML: no CUDA started here

        return {"engine": self.engine, "reader": f"{self.engine} ({envs.VLLM_VIDEO_LOADER_BACKEND} video)",
                "vllm": self.config["vllm"], "gpu": current_platform.get_device_name(0),
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
        """One question, ready for vLLM: its text in Gemma's chat format, and its clips' frames and sounds."""
        try:
            text = self.tokenizer.apply_chat_template(message(clips, prompt), tokenize=False, add_generation_prompt=True)
            videos = [self.video_reader.load_file(Path(c.path)).media for c in clips if c.send_frames]  # (frames, metadata)
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
                 # the sound check: how many sound and frame tokens the model got (t1, 3 s clip: 77 and 2,048)
                 **{f"{kind}_tokens": ids.count(i) for kind, i in self.token_ids.items()}}
        return Generation(raw=reply.text, usage=usage, finish_reason=reply.finish_reason)

    def unload(self) -> None:
        self.readers.shutdown()
        self.llm = None
