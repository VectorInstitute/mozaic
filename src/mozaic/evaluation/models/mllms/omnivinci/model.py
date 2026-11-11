"""OmniVinci (NVIDIA) on one GPU with transformers (its own environment: .venv_omnivinci).

OmniVinci is in neither transformers nor vLLM: its code comes with its weights on the hub (nvidia/omnivinci,
loaded with trust_remote_code) and is called from here, unchanged.

How a question goes through this file:
1. run_inference gives each clip's file and what to send of it; clip_media() reads it with the same reader as every
   model: 32 frames (224 x 224), evenly spaced, and the sound at 16 kHz, plus each frame's number.
2. clip_media() turns the clip into OmniVinci's input with the authors' own steps (their media code): the frames
   through its picture processor (enlarged to 448 x 448), each with its time (frame number / fps); the sound padded
   with silence to 30 s, as its sound encoder takes it. t1: frames and sound together, as one video with its
   soundtrack; the model keeps only the sound pieces that cover the clip. t2 and t3 face: silent videos. t3 voice:
   two sounds; keep_own_sound() keeps only their pieces that cover the clip, as the model does in t1.
3. message() writes the question: per clip its label and its tags, then the prompt, joined as their processor does.
4. Inside the model every 8 neighbouring frames are averaged into one picture (as it was trained): 4 per clip.
5. Greedy answer, one question at a time, in float16, as the authors' example runs it.
"""

from __future__ import annotations

import importlib
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mozaic.common.frame_sampler import load_audio, read_frames
from mozaic.evaluation.base import BaseMLLM, ClipInput, Generation

# Their tags (media.py MEDIA_TOKENS); load() checks they still match.
TAGS = {"video": "<vila/video>", "sound": "<sound>"}

log = logging.getLogger(__name__)


def tags(clip: ClipInput) -> str:
    """A clip's tags, as their processor writes them: a video with its soundtrack <sound><vila/video> (the model
    puts the sound inside the video), a silent video <vila/video>, a sound alone <sound>."""
    if clip.send_frames:
        return (TAGS["sound"] if clip.send_audio else "") + TAGS["video"]
    if clip.send_audio:
        return TAGS["sound"]
    raise ValueError("a clip with neither frames nor sound")


def message(clips: list[ClipInput], prompt: str) -> str:
    """The user's text: per clip its label (t3) and its tags, then the prompt, joined as their processor joins a
    message's parts (media.extract_media): with nothing in between, as in the authors' example (video, then text)."""
    parts = []
    for clip in clips:
        if clip.label is not None:  # "Clip 1:" / "Clip 2:" in t3; t1 and t2 have no label
            parts.append(clip.label)
        parts.append(tags(clip))
    return "".join([*parts, prompt])


def segments(indices: list[int], total: int, fps: float, sound: dict[str, Any], seconds: int,
             per_second: int) -> tuple[list[list[int]], list[list[int]]]:
    """Their media._load_video bookkeeping for a video with its soundtrack: per piece of `seconds` (30), which of the
    sent frames and which part of the sound (in spectrogram frames, `per_second` a second) belong to it. Our clips are
    shorter than 30 s: one piece, all the frames, the sound up to the end of the clip."""
    duration = total / fps
    start, end = sound["audio_start_sec"], sound["audio_end_sample_sec"]
    frames, sounds, k, done = [], [], 0, 0
    for i in range(math.ceil(duration / seconds)):
        last = min((i + 1) * seconds * fps, total)
        piece = []
        while k < len(indices) and indices[k] < last:
            piece.append(k)
            k += 1
        frames.append(piece)
        overlap = (max(i * seconds, start), min(min(i * seconds + seconds, duration), end))
        if overlap[0] < overlap[1]:
            upto = round((overlap[1] - start) * per_second)
            sounds.append([done, upto])
            done = upto
        else:
            sounds.append([])
    return frames, sounds


def hub_revision(path: Path) -> str:
    """The hub commit of the downloaded weights and code (hf download writes it in .cache)."""
    return (path / ".cache" / "huggingface" / "download" / "config.json.metadata").read_text().split()[0]


class Model(BaseMLLM):
    reads_clips = True  # run_inference gives us each clip's file: we read its frames, their numbers and its sound

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        if self.config["audio_sample_rate"] != 16000:  # its sound steps are written for 16 kHz
            raise ValueError("OmniVinci takes 16 kHz sound")
        self.weights = Path(self.config["weights"]).resolve()

        # As the authors' example loads it (its README): float16, its code from the weights folder, then the same
        # settings on the model and the processor. 32 frames is also its default; audio_chunk_length takes the
        # whole sound (ours are 3 or 6 s).
        self.model = AutoModel.from_pretrained(str(self.weights), trust_remote_code=True,
                                               torch_dtype="torch.float16", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(str(self.weights), trust_remote_code=True)
        for config in (self.model.config, self.processor.config):
            config.load_audio_in_video = True
            config.num_video_frames = self.config["frames"]
            config.audio_chunk_length = "max_3600"
        code = type(self.processor).__module__.rsplit(".", 1)[0]  # their code, as transformers loaded it
        self.media = importlib.import_module(f"{code}.media")
        self.process_images = importlib.import_module(f"{code}.mm_utils").process_images
        self.tokenize = importlib.import_module(f"{code}.tokenizer_utils").tokenize_conversation
        if TAGS != {name: self.media.MEDIA_TOKENS[name] for name in TAGS}:
            raise ValueError("their tags changed")
        tokenizer = self.processor.tokenizer
        self.tag_ids = {tokenizer.convert_tokens_to_ids(tag) for tag in TAGS.values()}

        self.generation = self.model.default_generation_config  # as the example starts from it
        self.generation.update(**self.config["generation"])
        eos = self.generation.eos_token_id
        self.end_ids = set(eos) if isinstance(eos, list) else {eos}

        # The cut of t3's sounds, and the check of every question: picture pieces at the video encoder (after the
        # averaging), and the whole input the language model gets (text, pictures, sound) at its first step.
        self.ends: list[int | None] = []
        self.seen = {"video": 0, "prompt": 0}
        self.model.encoders["sound"].register_forward_hook(self.keep_own_sound)
        self.model.encoders["video"].register_forward_hook(
            lambda _m, _i, out: self.seen.update(video=self.seen["video"] + sum(f.shape[0] for f in out)))
        self.model.llm.register_forward_pre_hook(self.see_prompt, with_kwargs=True)
        self.dtype = str(next(self.model.llm.parameters()).dtype).removeprefix("torch.")
        if self.dtype != "float16" or torch.cuda.device_count() < 1:
            raise ValueError(f"expected float16 on a GPU, got {self.dtype}")
        vision, sound = self.model.vision_tower, self.model.sound_tower
        log.info("language model %s on %s (%s), picture encoder %s (%s), sound encoder %s (%s); answers %s",
                 self.dtype, self.model.llm.device, self.model.llm.config._attn_implementation,
                 next(vision.parameters()).dtype, vision.vision_tower.config._attn_implementation,
                 next(sound.parameters()).dtype, sound.audio_tower.config._attn_implementation,
                 self.generation.to_diff_dict())

    def keep_own_sound(self, _module, args, features: list) -> list:
        """t3 voice: keep only the sound pieces that cover the clip's own sound, by the rule the model itself
        uses for a video's soundtrack (modeling_vila _embed): pieces up to the sound's end in spectrogram frames.
        A t1 soundtrack (end None) stays whole here: the model cuts it itself."""
        sounds = [info for question in args[2]["audio_info"] for info in question]
        kept = []
        for feature, end, info in zip(features, self.ends, sounds, strict=True):
            if end is not None:
                feature = feature[: int(np.ceil(end * (feature.shape[0] / info["new_audio_n_stft_frames"])))]
            kept.append(feature)
        return kept

    def see_prompt(self, _module, _args, kwargs: dict) -> None:
        if kwargs.get("inputs_embeds") is not None:  # the first step: the whole prompt (then one token at a time)
            self.seen["prompt"] = kwargs["inputs_embeds"].shape[1]

    def settings(self) -> dict[str, Any]:
        """What inference_settings.json saves about the run. A run resumed on another GPU is refused."""
        import torch
        import transformers

        return {"engine": f"transformers {transformers.__version__}, torch {torch.__version__}",
                "code": f"huggingface.co/nvidia/omnivinci {hub_revision(self.weights)}", "dtype": self.dtype,
                "gpu": torch.cuda.get_device_name(0)}

    def clip_media(self, clip: ClipInput) -> dict[str, Any]:
        """A clip as their processor makes it from a file (media.extract_media, then VILAProcessor), from our frames
        and sound: the frames as one tensor and the video's info (frame times, and with a soundtrack its pieces),
        the sound padded to 30 s and its info."""
        config = self.processor.config
        out: dict[str, Any] = {}
        sound = None
        if clip.send_audio:
            samples = np.frombuffer(load_audio(clip.path, self.config["audio_sample_rate"]), dtype="<i2")
            samples = samples.astype(np.float32) / 32768.0
            wave, sound = self.media._load_speech(
                self.media.Sound({"sample": samples, "sample_rate": self.config["audio_sample_rate"]}), config)
            out["sound"], out["audio_info"], out["end"] = wave, sound, None
            if not clip.send_frames:  # a sound alone (t3 voice): its end, for keep_own_sound
                out["end"] = round(sound["audio_end_sample_sec"] * (config.audio_sampling_rate // config.audio_hop_length))
        if clip.send_frames:
            import av

            frames, indices, total = read_frames(clip.path, self.config["frames"])
            with av.open(clip.path) as container:
                fps = float(container.streams.video[0].average_rate)
            info = {"video_path": clip.path, "has_audio": sound is not None, "video_duration": total / fps,
                    "audio_info": sound, "video_frame_times": [i / fps for i in indices]}
            if sound is not None:
                sound["video_path"] = clip.path
            if sound is not None and config.interleaved_vis_aud_in_video:
                info["segment_vis_indices_list"], info["segment_aud_indices_list"] = segments(
                    indices, total, fps, sound, config.interleaved_video_segment_duration,
                    config.audio_sampling_rate // config.audio_hop_length)
                info["expected_frame_count"] = len(indices)
            out["video"] = self.process_images(frames, self.processor.image_processor, config).half()
            out["video_info"] = info
        return out

    def inputs(self, clips: list[ClipInput], prompt: str) -> tuple[Any, dict[str, list], list[dict[str, Any]]]:
        """A question as their processor makes it (VILAProcessor): its text as token ids, its media, and its
        videos' info."""
        media: dict[str, list] = defaultdict(list)
        videos, sounds = [], []
        self.ends = []
        for clip in clips:
            got = self.clip_media(clip)
            if "video" in got:
                media["video"].append(got["video"])
                videos.append(got["video_info"])
            if "sound" in got:
                media["sound"].append(got["sound"])
                sounds.append(got["audio_info"])
                self.ends.append(got["end"])
        if videos:
            media["video_info"] = [videos]  # one question: one list of its videos, as their processor makes it
        if sounds:
            # One list per sound. Their processor makes one list per question, but their generate reads the sounds'
            # lengths list by list per sound (modeling_vila __embed_media_tokens), so two sounds in one list (t3
            # voice) stop it. The same sounds, read in the same order.
            media["audio_info"] = [[sound] for sound in sounds]
        if len(sounds) > 1 and videos:
            raise ValueError("two sounds go only as sounds alone (t3 voice)")

        config = self.processor.config
        ids = self.tokenize([{"from": "human", "value": message(clips, prompt)}], self.processor.tokenizer,
                            mm_use_bos_eos_tokens=config.mm_use_bos_eos_tokens,
                            unified_audio_encoder=config.unified_audio_encoder, add_generation_prompt=True)
        return ids.unsqueeze(0), media, videos

    def generate_clips(self, clips: list[ClipInput], prompt: str) -> Generation:
        """One question, as the authors' example asks it: their processor's input, then their generate()."""
        import torch

        ids, media, videos = self.inputs(clips, prompt)
        ids = ids.cuda()
        tokenizer = self.processor.tokenizer
        self.seen = {"video": 0, "prompt": 0}
        start = time.time()
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids, media=media, media_config=defaultdict(dict),
                                      generation_config=self.generation)
        new = out[0].tolist()  # the answer only: the prompt went in as embeddings
        text_tokens = sum(1 for t in ids[0].tolist() if t not in self.tag_ids)
        newlines = sum(2 * len(v["segment_vis_indices_list"]) for v in videos if v["has_audio"])  # around each soundtrack
        usage = {
            "prompt_tokens": self.seen["prompt"],  # all the language model got: text, pictures and sound
            "completion_tokens": len(new),
            "video_tokens": self.seen["video"],
            "audio_tokens": self.seen["prompt"] - text_tokens - self.seen["video"] - newlines,
            "seconds": round(time.time() - start, 2),
        }
        return Generation(
            raw=tokenizer.decode(new, skip_special_tokens=True).strip(),
            usage=usage,
            finish_reason="stop" if new and new[-1] in self.end_ids else "length",
        )

    def unload(self) -> None:
        import torch

        self.model = None
        torch.cuda.empty_cache()
