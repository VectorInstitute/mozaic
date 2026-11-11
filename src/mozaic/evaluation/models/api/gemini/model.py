"""Gemini through its OpenAI-compatible chat API, or through Google's Interactions API
(api: interactions), which takes each clip as one video."""

from __future__ import annotations

import io
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from openai import OpenAI

from mozaic.common.config import load_env
from mozaic.common.frame_sampler import bytes_to_b64, image_to_b64, load_audio, pcm_to_wav
from mozaic.evaluation.base import BaseMLLM, ClipInput, Generation


def post_json(url: str, body: dict[str, Any], key: str) -> dict[str, Any]:
    """POST a JSON body to the endpoint and return its JSON reply."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def clip_video(path: str, with_sound: bool) -> tuple[bytes, float]:
    """The clip as a video and its seconds: the file as it is, or without its sound (the video stream copied into
    an mp4, nothing re-encoded)."""
    import av

    with av.open(path) as clip:
        stream = clip.streams.video[0]
        seconds = stream.frames / float(stream.average_rate)
        if with_sound:
            return Path(path).read_bytes(), seconds
        buf = io.BytesIO()
        with av.open(buf, "w", format="mp4") as out:
            copy = out.add_stream_from_template(stream)
            for packet in clip.demux(stream):
                if packet.dts is not None:  # the last packet only flushes
                    packet.stream = copy
                    out.mux(packet)
    return buf.getvalue(), seconds


class Model(BaseMLLM):
    def load(self) -> None:
        load_env(self.config.get("env_file"))
        key = os.environ.get(self.config["api_key_env"])
        if not key:
            raise RuntimeError(f"{self.config['api_key_env']} is not set in the env file")
        self.key = key
        self.client = OpenAI(api_key=key, base_url=self.config["base_url"])
        self.reads_clips = self.config.get("api") == "interactions"  # it sends the clip files, not our frames

    def generate(self, frames: list, audio_wav: bytes | None, prompt: str) -> Generation:
        return self.generate_clips([ClipInput(frames=frames, audio_wav=audio_wav)], prompt)

    def generate_clips(self, clips: list[ClipInput], prompt: str) -> Generation:
        """One user message: per clip its label, frames and sound, then the prompt."""
        if self.config.get("api") == "interactions":
            return self.interaction(clips, prompt)
        content = []
        for clip in clips:
            if clip.label is not None:
                content.append({"type": "text", "text": clip.label})
            content += [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_b64(f)}"}}
                for f in clip.frames
            ]
            if clip.audio_wav is not None:
                # A file part: some OpenAI-compatible gateways silently drop input_audio parts.
                content.append(
                    {"type": "file", "file": {"file_data": f"data:audio/wav;base64,{bytes_to_b64(clip.audio_wav)}"}}
                )
        content.append({"type": "text", "text": prompt})

        attempts = self.config["retry"]["max_attempts"]
        for attempt in range(attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config["model_version"],
                    messages=[{"role": "user", "content": content}],
                    **self.config["generation"],
                )
                choice = resp.choices[0]
                return Generation(
                    raw=choice.message.content or "",
                    usage=resp.usage.model_dump(exclude_none=True) if resp.usage else {},
                    finish_reason=choice.finish_reason,
                )
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("max_attempts must be at least 1")

    def interaction(self, clips: list[ClipInput], prompt: str) -> Generation:
        """Google's Interactions API: per clip its label, then the clip as one video (with its sound when the task
        uses sound) at a rate that gives the config's frames, or its sound alone; then the prompt."""
        items = []
        for clip in clips:
            if clip.label is not None:
                items.append({"type": "text", "text": clip.label})
            if clip.send_frames:
                data, seconds = clip_video(clip.path, with_sound=clip.send_audio)
                video = {"type": "video", "data": bytes_to_b64(data), "mime_type": "video/mp4",
                         "processing": {"type": "static", "fps": self.config["frames"] / seconds}}
                if "video_resolution" in self.config:  # tokens per frame: low 64 (Google's default), high 256
                    video["resolution"] = self.config["video_resolution"]
                items.append(video)
            elif clip.send_audio or clip.audio_wav is not None:  # the sound alone: t3 voice, the sound check
                rate = self.config["audio_sample_rate"]
                wav = clip.audio_wav or pcm_to_wav(load_audio(clip.path, rate), rate)
                items.append({"type": "audio", "data": bytes_to_b64(wav), "mime_type": "audio/wav"})
        items.append({"type": "text", "text": prompt})
        body = {"model": self.config["model_version"], "input": items,
                "generation_config": self.config["generation"], "store": False}  # store: nothing kept at Google

        attempts = self.config["retry"]["max_attempts"]
        for attempt in range(attempts):
            try:
                reply = post_json(f"{self.config['base_url']}/interactions", body, self.key)
                text = "".join(c.get("text", "") for step in reply.get("steps", []) if step.get("type") == "model_output"
                               for c in step.get("content", []) if c.get("type") == "text")
                u = reply.get("usage", {})
                thought = u.get("total_thought_tokens", 0)
                # also under the chat API's names, which the scorer and the checks read; thinking is billed as output
                usage = {**u, "prompt_tokens": u.get("total_input_tokens"),
                         "completion_tokens": u.get("total_output_tokens", 0) + thought,
                         "completion_tokens_details": {"reasoning_tokens": thought},
                         # the model that answered, as Google names it: a preview model can change during a run
                         "served_model": reply.get("model"), "service_tier": reply.get("service_tier")}
                status = reply.get("status")
                return Generation(raw=text, usage=usage, finish_reason="stop" if status == "completed" else status)
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("max_attempts must be at least 1")

    def settings(self) -> dict[str, Any]:
        """The API and the videos' resolution, when the config sets them (older runs saved none)."""
        return {k: self.config[k] for k in ("api", "video_resolution") if k in self.config}
