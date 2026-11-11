"""The contract every evaluated model follows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Generation:
    """One model reply."""

    raw: str
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None


@dataclass
class ClipInput:
    """One clip of a question, as the model gets it."""

    label: str | None = None  # text sent before the clip, e.g. "Clip 1:" (t3); None for one-clip tasks
    frames: list = field(default_factory=list)
    audio_wav: bytes | None = None
    fps: float | None = None  # real rate of the frames, set when the config has frames_as: video
    # For a model that reads its clips itself (BaseMLLM.reads_clips): the clip's file and what to send of it
    path: str | None = None
    send_frames: bool = False
    send_audio: bool = False


class BaseMLLM:
    """A model that answers in text. Load once, answer many questions, unload.

    run_inference samples the frames and audio, builds the prompt, and parses the
    reply. A model only turns that input into text. A model with reads_clips gets each
    clip's file instead (ClipInput.path) and reads it itself.
    """

    reads_clips = False

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def load(self) -> None:
        pass

    def generate(self, frames: list, audio_wav: bytes | None, prompt: str) -> Generation:
        """Answer one question. Raise on failure, so the question is asked again later."""
        raise NotImplementedError

    def generate_clips(self, clips: list[ClipInput], prompt: str) -> Generation:
        """Answer one question about one or more clips, in order, then the prompt.

        One clip without a label goes to generate(). A model that takes tasks with two
        clips (t3) overrides this.
        """
        if len(clips) == 1 and clips[0].label is None:
            return self.generate(clips[0].frames, clips[0].audio_wav, prompt)
        raise NotImplementedError(f"{type(self).__name__} does not take questions with several clips")

    def generate_batch(self, questions: list[tuple[list[ClipInput], str]]) -> list[Generation]:
        """Answer several questions, each (clips, prompt): the config's batch_size at a time. By
        default one after the other; a model that answers them together overrides this."""
        return [self.generate_clips(clips, prompt) for clips, prompt in questions]

    def settings(self) -> dict[str, Any]:
        """Anything else the run's inference_settings.json should record, e.g. the engine's version."""
        return {}

    def unload(self) -> None:
        pass


class BaseClassifier:
    """A model that outputs class probabilities. Load once, predict in batches, unload.

    The classifier reads and preprocesses the clips itself (question["clips"]).
    run_inference turns the probabilities into a prediction.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def load(self) -> None:
        pass

    def predict_proba(self, questions: list[dict[str, Any]]) -> list[dict[str, float]]:
        """One probability per class for each question, keyed like the metrics expect.

        t1: RVRA, RVFA, FVRA, FVFA. t2: real, face_swap, fully_generated,
        full_body_swap. t3: real_same_person, real_different_person,
        fake_this_person, fake_different_person. Raise on failure, so the batch is
        asked again later.
        """
        raise NotImplementedError

    def settings(self) -> dict[str, Any]:
        """Anything else the run's inference_settings.json should record, e.g. the input's frames."""
        return {}

    def unload(self) -> None:
        pass
