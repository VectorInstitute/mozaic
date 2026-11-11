"""VITA-1.5 (VITA-MLLM, Tencent's licence) on one GPU with transformers (its own environment: .venv_vita_1_5).

VITA-1.5 is in neither transformers nor vLLM: the authors' code (GitHub VITA-MLLM/VITA) is cloned into
external-repos/vita (git-ignored, see the README) and called from here, unchanged.

How a question goes through this file:
1. run_inference gives each clip's file and what to send of it; clip_media() reads it with the same reader as every
   model: 32 frames (224 x 224), evenly spaced, and the sound at 16 kHz.
2. clip_media() turns the clip into VITA's input with the authors' own steps (their video_audio_demo.py): each frame
   through its picture processor (enlarged to 448 x 448; the model makes 256 pieces of each), the sound through their
   sound step (a Kaldi filterbank of the clip's own sound, no padding; 38 pieces for 3 s, 76 for 6 s). Their sound
   step adds random noise (dither, from their training settings): seeded per question (seed in the config), so a
   rerun gives the same answer.
3. message() writes the question: per clip its label (t3) and its tags (one <image> per frame, then <audio> for its
   sound), each clip on its own line, then the prompt, as every model gets it. VITA's sound channel was made for the
   user's spoken question (their demo takes a question in text or in sound, not both, and puts the sound last); here
   it carries the clip's sound, before the question.
4. Their code cuts the input at 6,200 pieces (their training length), also when answering: 32 frames are 8,192, so
   input_cap (config) lifts it; its language model takes 32,768.
5. Greedy answer, one question at a time, in float16, as the authors' example runs it. It starts each answer with a
   mark of the question it heard (☜ text, ☞ spoken, ☟ a sound to ignore), which their evaluations drop: drop_mark(),
   the mark kept in usage. With answer_choices on (the config), the reply can only be its mark, then
   {"answer": "<one of the question's options>"}, as MiniCPM-o's: VITA sometimes writes the option alone.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

from mozaic.common.frame_sampler import load_audio, pcm_to_wav, read_frames
from mozaic.evaluation.base import BaseMLLM, ClipInput, Generation
from mozaic.evaluation.models.mllms.answer_choices import choice_filter
from mozaic.evaluation.models.mllms.minicpm_o_vllm.model import answer_choices

IMAGE, AUDIO = "<image>", "<audio>"  # their tags (vita/constants.py)
# The marks its answers start with, as it was trained (vita/util/data_utils_video_audio_neg_patch.py): ☜ the user asked
# in text, ☞ the user spoke the question (<audio> in the user's turn), ☟ the sound was noise to ignore.
MARKS = ("☜", "☞", "☟")

log = logging.getLogger(__name__)


def tags(clip: ClipInput, frames: int) -> str:
    """A clip's tags: one <image> per frame, as their example writes a video (one after the other), then <audio> for
    its sound (their example puts the sound tag after the text; here it is the clip's sound, so it goes with the
    clip, before the question)."""
    out = (IMAGE * frames if clip.send_frames else "") + (AUDIO if clip.send_audio else "")
    if not out:
        raise ValueError("a clip with neither frames nor sound")
    return out


def message(clips: list[ClipInput], prompt: str, frames: int) -> str:
    """The user's text: per clip its label (t3) and its tags, each on its own line, then the prompt (their example:
    the frames' tags, a new line, then the question)."""
    lines = []
    for clip in clips:
        if clip.label is not None:  # "Clip 1:" / "Clip 2:" in t3; t1 and t2 have no label
            lines.append(clip.label)
        lines.append(tags(clip, frames))
    return "\n".join([*lines, prompt])


def drop_mark(text: str) -> tuple[str, str]:
    """A reply without its leading mark, as their evaluations read it (videomme, VLMEvalKit: the first character when
    it is a mark), and the mark ("" if none)."""
    if text[:1] in MARKS:
        return text[1:], text[0]
    return text, ""


def only_choices(allowed):
    """The choice filter (answer_choices.choice_filter) as a logits processor for generate(): every token it does not
    allow gets no chance. transformers 4.44's own filter (prefix_allowed_tokens_fn) cannot start from an empty reply,
    which VITA's is (its prompt goes in as embeddings); the 4.51 one MiniCPM-o uses can. As ola's."""
    import math

    import torch
    from transformers import LogitsProcessor

    class OnlyChoices(LogitsProcessor):
        def __call__(self, input_ids, scores):
            mask = torch.full_like(scores, -math.inf)
            for row, done in enumerate(input_ids):  # the reply's tokens so far
                mask[row, allowed(row, done)] = 0
            return scores + mask

    return OnlyChoices()


def code_commit(code: Path) -> str:
    """The commit of the authors' code in use, from the clone's .git."""
    head = (code / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        return (code / ".git" / head[5:]).read_text().strip()
    return head


def hub_revision(path: Path) -> str:
    """The hub commit of downloaded weights (hf download writes it in .cache)."""
    metadata = sorted((path / ".cache" / "huggingface" / "download").rglob("*.metadata"))
    return metadata[0].read_text().split()[0]


class Model(BaseMLLM):
    reads_clips = True  # run_inference gives us each clip's file: we read its frames and its sound

    def load(self) -> None:
        import torch

        if self.config["audio_sample_rate"] != 16000:  # its sound step is written for 16 kHz
            raise ValueError("VITA takes 16 kHz sound")
        self.code = Path(self.config["code"]).resolve()
        sys.path.insert(0, str(self.code))
        from vita.constants import IMAGE_TOKEN_INDEX
        from vita.conversation import conv_templates
        from vita.model.builder import load_pretrained_model
        from vita.util.mm_utils import KeywordsStoppingCriteria, tokenizer_image_audio_token, tokenizer_image_token

        # As the authors' example loads it (video_audio_demo.py): their loader (float16), the picture encoder's
        # processor, the sound encoder in float16. Its config names its two encoders by hub id, read when the model is
        # built: given here as our local copies (runs set HF_HUB_OFFLINE=1); the checkpoint's own encoder weights then
        # load over them, as from the hub.
        self.weights = Path(self.config["weights"]).resolve()
        self.encoders = {name: Path(self.config[name]).resolve() for name in ("vision_encoder", "audio_encoder")}
        self.tokenizer, self.model, _, _ = load_pretrained_model(
            str(self.weights), None, "vita-1.5", "qwen2p5_instruct",
            mm_vision_tower=str(self.encoders["vision_encoder"]), mm_audio_encoder=str(self.encoders["audio_encoder"]))
        vision = self.model.get_vision_tower()
        if not vision.is_loaded:
            vision.load_model()
        self.image_processor = vision.image_processor
        audio = self.model.get_audio_encoder()
        audio.to(dtype=torch.float16)
        self.audio_processor = audio.audio_processor
        self.model.eval()
        self.model.config.tokenizer_model_max_length = self.config["input_cap"]  # None: no cut (see the config)

        self.conv = conv_templates["qwen2p5_instruct"]  # its system line changes with the input (video, or none)
        self.image_index = IMAGE_TOKEN_INDEX
        self.tokenize = {"with sound": tokenizer_image_audio_token, "without sound": tokenizer_image_token}
        self.stopping = KeywordsStoppingCriteria
        eos = self.model.generation_config.eos_token_id  # <|im_end|> and <|endoftext|>, where generate stops
        self.end_ids = set(eos) if isinstance(eos, list) else {eos}

        # The check of every question: the pieces of the frames at the picture projector, and the whole input the
        # language model gets (text, pictures, sound) at its first step.
        self.seen = {"video": 0, "prompt": 0}
        self.has_frames = False
        inner = self.model.get_model()
        inner.mm_projector.register_forward_hook(self.see_video)
        inner.register_forward_pre_hook(self.see_prompt, with_kwargs=True)
        self.dtype = str(next(inner.layers.parameters()).dtype).removeprefix("torch.")
        if self.dtype != "float16" or torch.cuda.device_count() < 1:
            raise ValueError(f"expected float16 on a GPU, got {self.dtype}")
        log.info("language model %s (%s), picture encoder %s, sound encoder %s; input cap %s; answers %s", self.dtype,
                 self.model.config._attn_implementation, next(vision.parameters()).dtype,
                 next(audio.parameters()).dtype, self.config["input_cap"], self.model.generation_config.to_diff_dict())

    def see_video(self, _module, _args, out) -> None:
        if self.has_frames:  # a question without frames gets their blank picture: never placed
            self.seen["video"] += out.shape[0] * out.shape[1]

    def see_prompt(self, _module, _args, kwargs: dict) -> None:
        if kwargs.get("inputs_embeds") is not None:  # the first step: the whole prompt (then one token at a time)
            self.seen["prompt"] = kwargs["inputs_embeds"].shape[1]

    def settings(self) -> dict[str, Any]:
        """What inference_settings.json saves about the run. A run resumed on another GPU is refused."""
        import torch
        import transformers

        return {"engine": f"transformers {transformers.__version__}, torch {torch.__version__}",
                "code": f"github.com/VITA-MLLM/VITA {code_commit(self.code)}",
                "weights_revision": f"VITA-1.5 {hub_revision(self.weights)}, InternViT-300M-448px "
                                    f"{hub_revision(self.encoders['vision_encoder'])}, VITA-1.5_AudioEnc "
                                    f"{hub_revision(self.encoders['audio_encoder'])}",
                "dtype": self.dtype, "input_cap": self.config["input_cap"], "seed": self.config["seed"],
                "answer_choices": self.config["answer_choices"], "gpu": torch.cuda.get_device_name(0)}

    def clip_media(self, clip: ClipInput) -> dict[str, Any]:
        """A clip as their example makes it from files (video_audio_demo.py), from our frames and sound: each frame
        through the picture processor (their config's aspect ratio is square: no padding), [frames, 3, 448, 448];
        the sound through their sound step (from our sound as a WAV), its filterbank and the pieces it makes."""
        import io

        import torch

        out: dict[str, Any] = {}
        if clip.send_frames:
            frames, _, _ = read_frames(clip.path, self.config["frames"])
            out["video"] = torch.stack([self.image_processor.preprocess(f, return_tensors="pt")["pixel_values"][0]
                                        for f in frames])
        if clip.send_audio:
            rate = self.config["audio_sample_rate"]
            out["sound"] = self.audio_processor.process(io.BytesIO(pcm_to_wav(load_audio(clip.path, rate), rate)))
        return out

    def inputs(self, clips: list[ClipInput], prompt: str) -> tuple[Any, Any, dict[str, Any], int]:
        """A question as their example builds it: the chat template (its system line for a video, or for no picture)
        around message(), their tokenizer function for its tags, its frames and its sounds as their generate() takes
        them (still on the CPU); and its frames. A question with no frames gets their blank picture, one with no sound
        their blank sound: neither has a place in the text, so neither reaches the language model."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        torch.manual_seed(self.config["seed"])  # their sound step's random noise: the same for every run
        videos, sounds = [], []
        for clip in clips:
            got = self.clip_media(clip)
            if "video" in got:
                videos.append(got["video"])
            if "sound" in got:
                sounds.append(got["sound"])
        frames = videos[0].shape[0] if videos else 0
        if any(v.shape[0] != frames for v in videos):
            raise ValueError("clips with different frame counts")

        conv = self.conv.copy()
        conv.append_message(conv.roles[0], message(clips, prompt, frames))
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt("video" if videos else "lang")
        ids = self.tokenize["with sound" if sounds else "without sound"](text, self.tokenizer, self.image_index,
                                                                           return_tensors="pt").unsqueeze(0)
        images = torch.cat(videos) if videos else torch.zeros((1, 3, 448, 448))  # their blank picture
        if sounds:  # several sounds (t3 voice) padded to one length, as their training batches them
            feats = pad_sequence([mat for mat, _ in sounds], batch_first=True, padding_value=0)
            lengths = torch.tensor([mat.shape[0] for mat, _ in sounds])
            pieces = torch.tensor([n for _, n in sounds])
        else:  # their blank sound
            feats, lengths, pieces = torch.zeros(1, 400, 80), torch.tensor([400]), torch.tensor([60])
        return ids, images, {"audios": feats, "lengths": lengths, "lengths_for_llm": pieces}, frames

    def generate_clips(self, clips: list[ClipInput], prompt: str) -> Generation:
        """One question, as the authors' example asks it: their input, then their generate()."""
        import torch

        ids, images, audios, frames = self.inputs(clips, prompt)
        ids = ids.cuda()
        images = images.half().cuda()  # as their example sends them: float16, lengths too
        audios = {"audios": audios["audios"].half().cuda(), "lengths": audios["lengths"].half().cuda(),
                  "lengths_for_llm": audios["lengths_for_llm"].cuda()}
        stop = self.stopping([self.conv.sep], self.tokenizer, ids)  # their stop word, <|im_start|>
        extra = {}
        if self.config["answer_choices"]:  # its mark first (or none), then the JSON; the filter sees the reply only
            from transformers import LogitsProcessorList

            replies = [self.tokenizer.encode(mark + choice, add_special_tokens=False)
                       for choice in answer_choices(prompt) for mark in ("", *MARKS)]
            extra["logits_processor"] = LogitsProcessorList([only_choices(choice_filter(replies, sorted(self.end_ids)))])
        self.has_frames, self.seen = frames > 0, {"video": 0, "prompt": 0}
        start = time.time()
        with torch.inference_mode():
            out = self.model.generate(ids, images=images, audios=audios, use_cache=True, stopping_criteria=[stop],
                                      **self.config["generation"], **extra)
        new = out[0].tolist()  # the answer only: the prompt went in as embeddings
        reply = new[:-1] if new and new[-1] in self.end_ids else new
        raw, mark = drop_mark(self.tokenizer.decode(reply, skip_special_tokens=True).strip())
        text_tokens = int((ids >= 0).sum())  # the text; the tags are negative ids, replaced by the pieces
        usage = {
            "prompt_tokens": self.seen["prompt"],  # all the language model got: text, pictures and sound
            "completion_tokens": len(new),
            "video_tokens": self.seen["video"],  # 256 per frame: 8,192 per clip
            "audio_tokens": self.seen["prompt"] - text_tokens - self.seen["video"],  # 38 per 3 s sound, 76 per 6 s
            "answer_mark": mark,  # the question it heard: ☜ text, ☞ spoken, ☟ a sound to ignore ("" none)
            "seconds": round(time.time() - start, 2),
        }
        return Generation(
            raw=raw,
            usage=usage,
            finish_reason="stop" if new and new[-1] in self.end_ids else "length",
        )

    def unload(self) -> None:
        import torch

        self.model = None
        torch.cuda.empty_cache()
