"""Qwen3-Omni or Qwen2.5-Omni with vLLM, in its own environment (scripts/evaluation/inference.sh).

run_inference hands it each clip's file (reads_clips) and batch_size questions at a time, which vLLM
answers together. Each clip is read by Qwen's helper qwen_omni_utils with its defaults, except the
config's frames and frame_size (32 frames at 224 x 224). t1 sends each clip as one video with its sound
inside (use_audio_in_video); t2 and t3's face questions send silent videos, t3's voice questions the two
sounds.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from typing import Any

from mozaic.evaluation.base import BaseMLLM, ClipInput, Generation


def message(clips: list[ClipInput], prompt: str, size: int) -> tuple[list[dict[str, Any]], bool]:
    """The user message, as the other models get it: per clip its label (t3), its video or its sound, then
    the prompt. A clip with both frames and sound (t1) is one video with its sound inside: sound_in_video."""
    content: list[dict[str, Any]] = []
    for clip in clips:
        if clip.label is not None:
            content.append({"type": "text", "text": clip.label})
        if clip.send_frames:  # resized_*: qwen_omni_utils keeps the clip's size instead of enlarging it
            content.append({"type": "video", "video": clip.path, "resized_height": size, "resized_width": size})
        elif clip.send_audio:
            content.append({"type": "audio", "audio": clip.path})
    content.append({"type": "text", "text": prompt})
    sound_in_video = any(c.send_frames and c.send_audio for c in clips)
    return [{"role": "user", "content": content}], sound_in_video


def request(text: str, videos: list, audios: list, fps: list[float], sound_in_video: bool, size: int,
            step: int) -> dict[str, Any]:
    """One vLLM request. The frames' real rate goes in twice: fps for Qwen's processor (the frames' times)
    and second_per_grid_ts for vLLM's own placing of the sound between frame pairs (without it vLLM assumes
    2 s per pair and the sound lands next to the wrong frames). size stops the processor enlarging frames."""
    kwargs: dict[str, Any] = {"use_audio_in_video": sound_in_video}
    if videos:
        kwargs |= {"fps": fps[0],  # one rate per question: both t3 clips always have the same length
                   "second_per_grid_ts": [step / f for f in fps],
                   "size": {"shortest_edge": size * size, "longest_edge": size * size}}
    media = {"video": videos, "audio": audios}
    return {"prompt": text, "multi_modal_data": {k: v for k, v in media.items() if v}, "mm_processor_kwargs": kwargs}


def use_qwen_omni_utils(config: dict[str, Any]) -> Callable:
    """qwen_omni_utils taking the config's frames from every clip; returns its process_mm_info."""
    import av

    # qwen_omni_utils always reads videos with torchvision (it would pick torchcodec or decord when one is
    # installed): the reader every result was made with. torchvision passes PyAV an argument PyAV 19 dropped.
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")
    av_open = av.open
    av.open = lambda *args, metadata_errors=None, **kwargs: av_open(*args, **kwargs)

    from qwen_omni_utils import process_mm_info
    from qwen_omni_utils.v2_5 import audio_process, vision_process

    if config["audio_sample_rate"] != audio_process.SAMPLE_RATE:
        raise ValueError(f"qwen_omni_utils reads sound at {audio_process.SAMPLE_RATE} Hz, "
                         f"not the config's {config['audio_sample_rate']}")
    # That many frames from every clip, evenly spaced; everything else is qwen_omni_utils' default
    vision_process.FPS_MIN_FRAMES = vision_process.FPS_MAX_FRAMES = config["frames"]
    return process_mm_info


class Model(BaseMLLM):
    reads_clips = True

    def load(self) -> None:
        import vllm
        from transformers import AutoProcessor

        self.process_mm_info = use_qwen_omni_utils(self.config)
        self.processor = AutoProcessor.from_pretrained(self.config["weights"])
        self.llm = vllm.LLM(model=self.config["weights"], **self.config["vllm"])
        self.sampling = vllm.SamplingParams(**self.config["generation"])
        self.engine = f"vllm {vllm.__version__}"
        self.readers = ThreadPoolExecutor(self.config["readers"])  # read a batch's clips in parallel
        tokenizer = self.processor.tokenizer
        self.token_ids = {kind: tokenizer.convert_tokens_to_ids(getattr(self.processor, f"{kind}_token"))
                          for kind in ("audio", "image", "video")}

    def settings(self) -> dict[str, Any]:
        """Saved in inference_settings.json. With the engine's settings and the GPU, a run resumed with another
        tensor_parallel_size or on another GPU is refused: either can change a few answers."""
        from vllm.platforms import current_platform  # the GPU's name through NVML: no CUDA started here

        return {"engine": self.engine, "reader": f"qwen-omni-utils {version('qwen-omni-utils')}",
                "frame_size": self.config["frame_size"], "vllm": self.config["vllm"],
                "gpu": current_platform.get_device_name(0)}

    def generate_clips(self, clips: list[ClipInput], prompt: str) -> Generation:
        return self.generate_batch([(clips, prompt)])[0]

    def generate_batch(self, questions: list[tuple[list[ClipInput], str]]) -> list[Generation]:
        """The questions' clips read in parallel, then answered together by vLLM. Greedy answers."""
        requests = list(self.readers.map(lambda question: self.read(*question), questions))
        outs = self.llm.generate(requests, self.sampling, use_tqdm=False)
        return [self.generation(out) for out in outs]

    def read(self, clips: list[ClipInput], prompt: str) -> dict[str, Any]:
        """One question's vLLM request: its clips read by qwen_omni_utils, and the prompt."""
        from transformers import ProcessorMixin

        try:
            msgs, sound_in_video = message(clips, prompt, self.config["frame_size"])
            # Qwen2.5-Omni's own apply_chat_template only adds a warning, on every call, about speech output
            text = ProcessorMixin.apply_chat_template(self.processor, msgs, add_generation_prompt=True, tokenize=False)
            audios, _, videos, video_kwargs = self.process_mm_info(msgs, use_audio_in_video=sound_in_video,
                                                                   return_video_kwargs=True)
            step = self.processor.video_processor.temporal_patch_size
            return request(text, videos or [], audios or [], video_kwargs["fps"], sound_in_video,
                           self.config["frame_size"], step)
        except Exception as e:
            raise RuntimeError(f"{clips[0].path}: {e}") from e

    def generation(self, out) -> Generation:
        ids, reply = list(out.prompt_token_ids), out.outputs[0]
        usage = {"prompt_tokens": len(ids), "completion_tokens": len(reply.token_ids),
                 # the sound check of every question: tokens of each kind
                 **{f"{kind}_tokens": ids.count(i) for kind, i in self.token_ids.items()}}
        return Generation(raw=reply.text, usage=usage, finish_reason=reply.finish_reason)

    def unload(self) -> None:
        self.readers.shutdown()
        self.llm = None
