"""Ola-7B (Tsinghua, Tencent Hunyuan, NTU) on one GPU with transformers (its own environment: .venv_ola).

Ola is in neither transformers nor vLLM: the authors' code (GitHub Ola-Omni/Ola) is cloned into external-repos/ola
(git-ignored, see the README) and called from here, unchanged.

How a question goes through this file:
1. run_inference gives each clip's file and what to send of it; clip_media() reads it with the same reader as every
   model: 32 frames (224 x 224), evenly spaced, and the sound at 16 kHz.
2. clip_media() turns the clip into Ola's input with the authors' own steps (their inference/infer.py): each frame
   through their picture step (process_anyres_video: enlarged to 256 x 256), the model makes 73 pieces of each; the
   sound as their load_audio makes it (sound()): padded with silence to 30 s, as their Whisper encoder takes it, as a
   spectrogram and as samples (their second sound encoder, BEATs, reads those).
3. message() writes the question as the authors' example does: per clip its label (t3) and its tags, each on its own
   line, then the prompt; a video with its sound "<speech><image>"; their tokenizer functions make the token ids.
4. The videos go to their generate() as a list, as their training passes them, so all 32 frames reach the language
   model. Their inference scripts pass a video as one block, which lets only its first frame through (see the README).
5. Its two sound encoders read the whole 30 s. silence (config): kept, the authors' way (their projector makes 150
   pieces, most of them silence), or cut (only the encoder steps of the clip's own sound go on: cut()).
6. Greedy answer, one question at a time, in bfloat16, as the authors' example runs it. With answer_choices on (the
   config), the reply can only be {"answer": "<one of the question's options>"}, as MiniCPM-o's: Ola often writes the
   option alone ("the video"), which the prompts' parse() can't read.
"""

from __future__ import annotations

import logging
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

# Their settings for inference, read when their code is imported (inference/infer.py sets them so). The video ones
# enlarge a frame to at least 288 x 288 in area, rounded down to a multiple of 64: 224 x 224 becomes 256 x 256.
SETTINGS = {"LOWRES_RESIZE": "384x32", "HIGHRES_BASE": "0x32", "VIDEO_RESIZE": "0x64", "VIDEO_MAXRES": "480",
            "VIDEO_MINRES": "288", "MAXRES": "1536", "MINRES": "0", "FORCE_NO_DOWNSAMPLE": "1",
            "LOAD_VISION_EARLY": "1", "PAD2STRIDE": "1"}
SPEECH, IMAGE = "<speech>", "<image>"  # their tags (ola/constants.py)
WINDOW = 480_000  # sound samples of one 30 s window at 16 kHz, as their Whisper encoder takes it
STEP = 320  # sound samples per step of its sound encoders (1,500 steps a window): 3.08 s are 154 steps, 6.12 s 306
PAD = 151643  # the id their scripts leave out of the attention mask (<|endoftext|>), never in a question

log = logging.getLogger(__name__)


def tags(clip: ClipInput) -> str:
    """A clip's tags, as the authors write them: a video with its soundtrack "<speech><image>" (their example), a
    silent video "<image>" (their training's videos without sound), a sound alone "<speech>" (their example)."""
    if clip.send_frames:
        return (SPEECH if clip.send_audio else "") + IMAGE
    if clip.send_audio:
        return SPEECH
    raise ValueError("a clip with neither frames nor sound")


def message(clips: list[ClipInput], prompt: str) -> str:
    """The user's text: per clip its label (t3) and its tags, each on its own line, then the prompt (the authors'
    example: "<speech><image>\\n" then the question)."""
    lines = []
    for clip in clips:
        if clip.label is not None:  # "Clip 1:" / "Clip 2:" in t3; t1 and t2 have no label
            lines.append(clip.label)
        lines.append(tags(clip))
    return "\n".join([*lines, prompt])


def own_steps(samples: int) -> int:
    """The encoder steps that hold a sound's own samples (of 1,500 in its 30 s window): 154 for 3.08 s, 306 for
    6.12 s. Their projector then makes one piece of every 10 steps and drops the steps left over: 15 and 30 pieces."""
    if samples > WINDOW:
        raise ValueError("the cut takes sounds of at most 30 s (one window); ours are 3 or 6 s")
    return math.ceil(samples / STEP)


def sound(samples: np.ndarray) -> tuple[Any, Any, Any, Any]:
    """A sound as their load_audio (inference/infer.py) makes it from a file, from our samples: 30 s windows (the
    last, or only, one padded with silence), each one's Whisper spectrogram (128 bands) and its samples, at most 25
    windows; with the spectrogram length and the number of windows."""
    import torch
    import whisper

    windows = [whisper.pad_or_trim(samples[i : i + WINDOW]) for i in range(0, max(len(samples), 1), WINDOW)]
    mels = torch.cat([whisper.log_mel_spectrogram(w, n_mels=128).permute(1, 0).unsqueeze(0) for w in windows])[:25]
    waves = torch.cat([torch.from_numpy(w).unsqueeze(0) for w in windows])[:25]
    return mels, torch.LongTensor([mels.shape[1]] * mels.shape[0]), torch.LongTensor([mels.shape[0]]), waves


def only_choices(allowed):
    """The choice filter (answer_choices.choice_filter) as a logits processor for generate(): every token it does not
    allow gets no chance. transformers 4.43's own filter (prefix_allowed_tokens_fn) cannot start from an empty reply,
    which Ola's is (its prompt goes in as embeddings); the 4.51 one MiniCPM-o uses can."""
    import torch
    from transformers import LogitsProcessor

    class OnlyChoices(LogitsProcessor):
        def __call__(self, input_ids, scores):
            mask = torch.full_like(scores, -math.inf)
            for row, done in enumerate(input_ids):  # the reply's tokens so far
                mask[row, allowed(row, done)] = 0
            return scores + mask

    return OnlyChoices()


def when_answered(tokenizer):
    """A stop for generate() once the reply holds its T4 answer for good (t4_prompts.answered), checked when a token
    with a "}" comes. Whether its ids hold the prompt too depends on the model's generate() (Ola's: never, the prompt
    goes in as embeddings; Uni-MoE's: as it is called), and the prompt's braces must not count: the reply starts
    where the ids stood at the first call, one token in. stopped says whether it fired."""
    from transformers import StoppingCriteria

    from mozaic.evaluation.prompts.t4_prompts import answered

    class WhenAnswered(StoppingCriteria):
        stopped = False
        start: int | None = None

        def __call__(self, output_ids, scores, **kwargs) -> bool:
            if self.start is None:
                self.start = output_ids.shape[-1] - 1
            if "}" in tokenizer.decode(output_ids[0, -1:]) and answered(
                    tokenizer.decode(output_ids[0, self.start:], skip_special_tokens=True)):
                self.stopped = True
            return self.stopped

    return WhenAnswered()


def code_commit(code: Path) -> str:
    """The commit of the authors' code in use, from the clone's .git."""
    head = (code / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        return (code / ".git" / head[5:]).read_text().strip()
    return head


def hub_revision(path: Path) -> str:
    """The hub commit of the downloaded weights (hf download writes it in .cache)."""
    return (path / ".cache" / "huggingface" / "download" / "config.json.metadata").read_text().split()[0]


class Model(BaseMLLM):
    reads_clips = True  # run_inference gives us each clip's file: we read its frames and its sound

    def load(self) -> None:
        import torch

        if self.config["audio_sample_rate"] != 16000:  # its sound steps are written for 16 kHz
            raise ValueError("Ola takes 16 kHz sound")
        if self.config.get("silence", "kept") not in ("kept", "cut"):
            raise ValueError("silence is kept (the authors' way) or cut (only the clip's own sound)")
        # deepspeed (imported by their picture encoder's file) asks a CUDA compiler for its version when imported on a
        # GPU node, and ours have none: cuda_home in the config, CUDA 12.1's nvcc as torch's. Nothing is compiled.
        os.environ.setdefault("CUDA_HOME", self.config["cuda_home"])
        os.environ.update(SETTINGS)
        self.code = Path(self.config["code"]).resolve()
        sys.path.insert(0, str(self.code))
        from ola.conversation import conv_templates
        from ola.datasets.preprocess import tokenizer_image_token, tokenizer_speech_image_token, tokenizer_speech_token
        from ola.mm_utils import KeywordsStoppingCriteria, process_anyres_video
        from ola.model.builder import load_pretrained_model

        # As the authors' example loads it (inference/infer.py): their loader, then bfloat16 on the GPU. Its config
        # names the two sound encoders' files from the working folder ("./large-v3.pt"): given here with their full
        # path (the files that came with the weights), as its README asks ("replace the path in config.json").
        self.weights = Path(self.config["weights"]).resolve()
        self.tokenizer, model, self.image_processor, _ = load_pretrained_model(
            str(self.weights), None, speech_encoder=str(self.weights / "large-v3.pt"),
            music_encoder=str(self.weights / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"))
        self.model = model.to("cuda").eval().bfloat16()
        self.image_processor.do_resize = False  # as their script sets it before each frame: their step resizes
        self.image_processor.do_center_crop = False
        self.process_anyres_video = process_anyres_video
        self.conv = conv_templates["qwen_1_5"]  # their example's chat template: system "You are a helpful assistant."
        self.tokenize = {"video and sound": tokenizer_speech_image_token, "video": tokenizer_image_token,
                         "sound": tokenizer_speech_token}
        self.stopping = KeywordsStoppingCriteria
        eos = self.model.generation_config.eos_token_id  # <|im_end|> and <|endoftext|>, where generate stops
        self.end_ids = set(eos) if isinstance(eos, list) else {eos}

        # The cut of the sound, and the check of every question: the pieces of each video at its projector, and the
        # whole input the language model gets (text, pictures, sound) at its first step.
        self.steps: list[int | None] = []  # per sound sent to the model, in order: encoder steps to pass on (None: all)
        self.read: list[int] = []  # per sound, the encoder steps passed on
        self.seen = {"video": 0, "prompt": 0}
        inner = self.model.get_model()
        inner.speech_encoder.register_forward_hook(self.cut)
        inner.mm_projector.register_forward_hook(self.see_video)
        inner.register_forward_pre_hook(self.see_prompt, with_kwargs=True)
        self.dtype = str(next(inner.layers.parameters()).dtype).removeprefix("torch.")
        if self.dtype != "bfloat16" or torch.cuda.device_count() < 1:
            raise ValueError(f"expected bfloat16 on a GPU, got {self.dtype}")
        log.info("language model %s (%s), picture encoder %s, sound encoders Whisper %s and BEATs (float32 in its "
                 "forward); answers %s", self.dtype, self.model.config._attn_implementation,
                 next(inner.vision_tower.parameters()).dtype, next(inner.speech_encoder.whisper_model.parameters()).dtype,
                 self.model.generation_config.to_diff_dict())

    def cut(self, _module, _args, out):
        """Their sound encoders' output for one sound, [windows, 1,500 steps, 2,048]: with silence cut, only the steps
        of the clip's own sound go on to their projector, which drops the steps left over after its groups of 10."""
        steps = self.steps.pop(0)
        if steps is not None:
            out = out[:, :steps]
        self.read.append(out.shape[0] * out.shape[1])
        return out

    def see_video(self, _module, args, out) -> None:
        if args[4] == "video":  # a video's pieces (a question without one gets their blank picture: never placed)
            self.seen["video"] += out.shape[1]

    def see_prompt(self, _module, _args, kwargs: dict) -> None:
        if kwargs.get("inputs_embeds") is not None:  # the first step: the whole prompt (then one token at a time)
            self.seen["prompt"] = kwargs["inputs_embeds"].shape[1]

    def settings(self) -> dict[str, Any]:
        """What inference_settings.json saves about the run. A run resumed on another GPU is refused."""
        import deepspeed
        import torch
        import transformers

        return {"engine": f"transformers {transformers.__version__}, torch {torch.__version__}, "
                          f"deepspeed {deepspeed.__version__}",
                "code": f"github.com/Ola-Omni/Ola {code_commit(self.code)}",
                "weights_revision": f"huggingface.co/THUdyh/Ola-7b {hub_revision(self.weights)}",
                "dtype": self.dtype, "silence": self.config.get("silence", "kept"),
                "answer_choices": self.config["answer_choices"], "gpu": torch.cuda.get_device_name(0)}

    def clip_media(self, clip: ClipInput) -> dict[str, Any]:
        """A clip as their example makes it from a video file (inference/infer.py), from our frames and sound: the
        frames through process_anyres_video, as one video [frames, 1, 3, 256, 256]; the sound as samples."""
        import torch

        out: dict[str, Any] = {}
        if clip.send_frames:
            frames, _, _ = read_frames(clip.path, self.config["frames"])
            out["video"] = torch.cat([self.process_anyres_video(f, self.image_processor).unsqueeze(0) for f in frames])
        if clip.send_audio:
            samples = np.frombuffer(load_audio(clip.path, self.config["audio_sample_rate"]), dtype="<i2")
            out["sound"] = samples.astype(np.float32) / 32768.0
        return out

    def inputs(self, clips: list[ClipInput], prompt: str) -> tuple[Any, dict[str, Any], bool]:
        """A question as their example builds it: the chat template around message(), their tokenizer function for
        its tags, its videos and its sounds as their generate() takes them (still on the CPU); whether it sends a
        sound. A question with no video gets their blank picture, one with no sound their silent sound: neither has a
        place in the text, so neither reaches the language model."""
        import torch

        videos, sounds = [], []
        for clip in clips:
            got = self.clip_media(clip)
            if "video" in got:
                videos.append(got["video"])
            if "sound" in got:
                sounds.append(got["sound"])
        kind = "video and sound" if any(c.send_frames and c.send_audio for c in clips) else (
            "video" if videos else "sound")
        if kind == "video and sound" and len(clips) > 1:
            raise ValueError("a video with its sound goes alone (t1)")

        conv = self.conv.copy()
        conv.append_message(conv.roles[0], message(clips, prompt))
        conv.append_message(conv.roles[1], None)
        ids = self.tokenize[kind](conv.get_prompt(), self.tokenizer, return_tensors="pt").unsqueeze(0)

        if videos:  # a list of videos, as their training passes them; their scripts pass one block (first frame only)
            media: dict[str, Any] = {"images": videos, "images_highres": videos, "modalities": ["video"] * len(videos)}
        else:  # their example's blank picture for a sound alone
            blank = [torch.zeros(1, 3, 224, 224)]
            media = {"images": blank, "images_highres": blank, "image_sizes": [(224, 224)], "modalities": ["text"]}
        cut = self.config.get("silence", "kept") == "cut"
        if sounds:
            parts = [sound(s) for s in sounds]
            self.steps = [own_steps(len(s)) if cut else None for s in sounds]
        else:  # their example's silent sound for a question without one
            parts = [(torch.zeros(1, 3000, 128), torch.LongTensor([3000]), torch.LongTensor([1]), torch.zeros(1, WINDOW))]
            self.steps = [None]
        media["speech"] = [mels for mels, _, _, _ in parts]
        media["speech_lengths"] = [length for _, length, _, _ in parts]
        media["speech_chunks"] = [windows for _, _, windows, _ in parts]
        media["speech_wav"] = [waves for _, _, _, waves in parts]
        return ids, media, bool(sounds)

    def generate_clips(self, clips: list[ClipInput], prompt: str) -> Generation:
        """One question, as the authors' example asks it: their input, then their generate()."""
        import torch

        ids, media, has_sound = self.inputs(clips, prompt)
        ids = ids.cuda()
        for key in ("images", "speech"):  # in bfloat16 on the GPU, as their example sends pictures and spectrograms
            media[key] = [t.to(device="cuda", dtype=torch.bfloat16) for t in media[key]]
        media["images_highres"] = media["images"]  # the same tensors, as their example passes them
        for key in ("speech_lengths", "speech_chunks", "speech_wav"):
            media[key] = [t.to("cuda") for t in media[key]]
        stops = [self.stopping([self.conv.sep], self.tokenizer, ids)]  # their stop word, <|im_end|>
        if self.config.get("stop_when_answered"):  # T4: once the reply holds both parts (the config)
            stops.append(when_answered(self.tokenizer))
        extra = {}
        if self.config["answer_choices"]:  # the prompt goes in as embeddings: the filter sees the reply only
            from transformers import LogitsProcessorList

            replies = [self.tokenizer.encode(choice, add_special_tokens=False) for choice in answer_choices(prompt)]
            extra["logits_processor"] = LogitsProcessorList([only_choices(choice_filter(replies, sorted(self.end_ids)))])
        self.read, self.seen = [], {"video": 0, "prompt": 0}
        start = time.time()
        with torch.inference_mode():
            out = self.model.generate(inputs=ids, attention_mask=ids.ne(PAD).long(), use_cache=True,
                                      stopping_criteria=stops, **media, **self.config["generation"], **extra)
        new = out[0].tolist()  # the answer only: the prompt went in as embeddings
        reply = new[:-1] if new and new[-1] in self.end_ids else new  # its tokenizer would keep <|endoftext|> as text
        text_tokens = int((ids >= 0).sum())  # the text; the tags are negative ids, replaced by the pieces
        usage = {
            "prompt_tokens": self.seen["prompt"],  # all the language model got: text, pictures and sound
            "completion_tokens": len(new),
            "video_tokens": self.seen["video"],  # 73 per frame and 2 more per video: 2,338 per clip
            "audio_tokens": self.seen["prompt"] - text_tokens - self.seen["video"],  # 150 pieces + 3 per sound (kept)
            "audio_steps": sum(self.read) if has_sound else 0,  # encoder steps passed on: 154 / 306 (cut), 1,500 (kept)
            "seconds": round(time.time() - start, 2),
        }
        answered = len(stops) > 1 and stops[1].stopped
        if answered:  # its answer was whole: the reply ends there, as at its end word
            usage["stopped_when_answered"] = True
        return Generation(
            raw=self.tokenizer.decode(reply, skip_special_tokens=True).strip(),
            usage=usage,
            finish_reason="stop" if answered or (new and new[-1] in self.end_ids) else "length",
        )

    def unload(self) -> None:
        import torch

        self.model = None
        torch.cuda.empty_cache()
