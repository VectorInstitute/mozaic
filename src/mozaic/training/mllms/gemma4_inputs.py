"""Gemma 4 (12B)'s inputs, built with its own processor as its T4 test run builds them (gemma4_vllm).

The same 32 frames with their times (OpenCV, evenly spaced), then the sound, then the prompt; thinking off. One
difference on purpose: the sound is in [-1, 1]. vLLM 0.28's reader gives this data's 16-bit sound unscaled; the trained
test run (normalize_audio) corrects it too.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

END_OF_TURN = "<turn|>\n"
SAMPLE_RATE = 16000


def read_frames(path: str, frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    """vLLM's OpenCV reader: frames evenly spaced over the clip, as RGB, and their metadata."""
    import cv2

    capture = cv2.VideoCapture(path)
    total, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
    indices = np.linspace(0, total - 1, frames, dtype=int).tolist()
    wanted, out, i = set(indices), [], 0
    while capture.grab():
        if i in wanted:
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError(f"{path}: frame {i} could not be read")
            out.append(frame[:, :, ::-1].copy())
        i += 1
    capture.release()
    if len(out) != frames:
        raise ValueError(f"{path}: {len(out)} frames read, not {frames}")
    return np.stack(out), {"total_num_frames": total, "fps": fps, "duration": total / fps, "video_backend": "opencv",
                           "frames_indices": indices}


def read_sound(path: str) -> np.ndarray:
    import librosa

    return librosa.load(path, sr=SAMPLE_RATE, mono=True)[0]


def patch_video_features() -> None:
    """transformers 5.20's forward leaves out num_frames_per_video; here every question has one video."""
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedModel

    original = Gemma4UnifiedModel.get_video_features

    def get_video_features(self, pixel_values_videos, video_position_ids=None, num_frames_per_video=None, **kwargs):
        if num_frames_per_video is None:
            num_frames_per_video = torch.tensor([pixel_values_videos.shape[0]], device=pixel_values_videos.device)
        return original(self, pixel_values_videos, video_position_ids, num_frames_per_video, **kwargs)

    Gemma4UnifiedModel.get_video_features = get_video_features


def load_template(config: dict[str, Any]):
    """(None, tokenizer, processor): no LLaMA-Factory template here."""
    from transformers import AutoProcessor

    patch_video_features()
    processor = AutoProcessor.from_pretrained(config["weights"])
    return None, processor.tokenizer, processor


def prompt_text(tokenizer, row: dict[str, Any]) -> str:
    """The test run's chat text: its video, its sound, the prompt; the row's <video><audio> marks removed."""
    prompt = row["messages"][0]["content"].replace("<video><audio>", "", 1)
    content = [{"type": "video"}, {"type": "audio"}, {"type": "text", "text": prompt}]
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)


def encode(processor, row: dict[str, Any], frames: int, reply: str = "") -> dict[str, torch.Tensor]:
    """A row's model inputs, with the reply after the prompt if given (batch of one)."""
    video, metadata = read_frames(row["videos"][0], frames)
    text = prompt_text(processor.tokenizer, row) + reply
    return dict(processor(text=[text], videos=[video], audio=[read_sound(row["audios"][0])], video_metadata=[metadata],
                          sampling_rate=SAMPLE_RATE, return_tensors="pt", add_special_tokens=False, do_sample_frames=False))


def training_example(processor, row: dict[str, Any], frames: int) -> dict[str, torch.Tensor]:
    """Model inputs and labels: only the reply is learned."""
    reply = row["messages"][1]["content"] + END_OF_TURN
    inputs = encode(processor, row, frames, reply)
    labels = inputs["input_ids"].clone()
    count = len(processor.tokenizer(reply, add_special_tokens=False)["input_ids"])
    labels[:, :-count] = -100
    return {**inputs, "labels": labels}


def answer(model, template, tokenizer, processor, example: dict[str, Any], generate_args: dict[str, Any]) -> str:
    inputs = {k: v.to(model.device) for k, v in encode(processor, example, 32).items()}
    args = {k: v for k, v in generate_args.items() if k != "use_audio_in_video"}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(**inputs, use_cache=True, **args)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def merge(weights, adapter, dest) -> None:
    from mozaic.training.mllms.minicpm_o_inputs import merge as merge_lora

    merge_lora(weights, adapter, dest)
