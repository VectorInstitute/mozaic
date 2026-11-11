"""Read frames and audio from a clip, the same way for every model.

No model uses its own fps. We pick the frames, so every model sees the same input.
"""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path

import numpy as np

from mozaic.common.config import load_config


def frame_indices(total: int, n: int) -> list[int]:
    """Uniform sampling: n indices evenly spaced from the first to the last frame."""
    if total <= n:
        return list(range(total))
    return np.linspace(0, total - 1, n).round().astype(int).tolist()


def sample_frames(video_path: str | Path, n: int | None = None) -> list:
    """Return n uniformly sampled frames as PIL images."""
    return read_frames(video_path, n)[0]


def read_frames(video_path: str | Path, n: int | None = None) -> tuple[list, list[int], int]:
    """The n frames sample_frames returns, their indices in the clip, and the clip's frame count."""
    import av

    n = n or load_config()["sampling"]["frames"]
    with av.open(str(video_path)) as container:
        frames = [f.to_image() for f in container.decode(video=0)]
    indices = frame_indices(len(frames), n)
    return [frames[i] for i in indices], indices, len(frames)


def sampled_rate(video_path: str | Path, n: int | None = None) -> float:
    """Frames per second of the n frames sample_frames picks: their real rate, for a model that takes
    them as a video. From the clip's header (frame count and rate)."""
    import av

    n = n or load_config()["sampling"]["frames"]
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total, rate = stream.frames, float(stream.average_rate)
    return rate if total <= n else (n - 1) * rate / (total - 1)


def load_audio(video_path: str | Path, sample_rate: int | None = None) -> bytes:
    """Return the clip's audio as raw 16-bit mono PCM."""
    import av

    sample_rate = sample_rate or load_config()["sampling"]["audio_sample_rate"]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    pcm = bytearray()
    with av.open(str(video_path)) as container:
        for frame in container.decode(audio=0):
            for f in resampler.resample(frame):
                pcm += bytes(f.planes[0])[: f.samples * 2]
        for f in resampler.resample(None):
            pcm += bytes(f.planes[0])[: f.samples * 2]
    return bytes(pcm)


def pcm_to_wav(pcm: bytes, sample_rate: int | None = None) -> bytes:
    """Wrap raw 16-bit mono PCM in a wav header."""
    sample_rate = sample_rate or load_config()["sampling"]["audio_sample_rate"]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def image_to_b64(img, quality: int = 90) -> str:
    """Encode a PIL image as base64 JPEG."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
