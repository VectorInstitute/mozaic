"""Annotator: Gemini, sees the clip (timed frames + audio) and explains its claims."""

from __future__ import annotations

from typing import Any

from mozaic.annotation.t4_explanation.models.client import Generation, Model
from mozaic.common.frame_sampler import bytes_to_b64, frame_indices, image_to_b64, load_audio, pcm_to_wav
from mozaic.common.paths import resolve

Row = dict[str, Any]


class Annotator(Model):
    """Gemini: the clip, then the prompt."""

    def generate(self, clip_file: str, prompt: str) -> Generation:
        return self.ask(clip_content(clip_file, self.config["frames"], self.config["audio_sample_rate"])
                        + [{"type": "text", "text": prompt}])


def timed_frames(clip_file: str, n: int) -> list:
    """n frames evenly spread over the clip, each with its time (mm:ss.s) printed above it, so the
    annotator can say when a sign shows."""
    import av
    from PIL import Image, ImageDraw, ImageFont

    with av.open(str(resolve(clip_file))) as container:
        fps = float(container.streams.video[0].average_rate)
        decoded = [f.to_image() for f in container.decode(video=0)]
    font = ImageFont.load_default(size=15)
    frames = []
    for i in frame_indices(len(decoded), n):
        img = decoded[i]
        canvas = Image.new("RGB", (img.width, img.height + 22), "black")
        canvas.paste(img, (0, 22))
        seconds = i / fps
        text = f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
        draw = ImageDraw.Draw(canvas)
        draw.text(((canvas.width - draw.textlength(text, font=font)) / 2, 3), text, fill="white", font=font)
        frames.append(canvas)
    return frames


def clip_content(clip_file: str, frames: int, sample_rate: int) -> list[Row]:
    """A clip as message parts: its timed frames, then its sound as a file part (some OpenAI-compatible gateways drop input_audio)."""
    parts: list[Row] = [{"type": "text", "text": "Clip:"}]
    parts += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_b64(f)}"}}
              for f in timed_frames(clip_file, frames)]
    wav = pcm_to_wav(load_audio(resolve(clip_file), sample_rate), sample_rate)
    parts.append({"type": "file", "file": {"file_data": f"data:audio/wav;base64,{bytes_to_b64(wav)}"}})
    return parts
