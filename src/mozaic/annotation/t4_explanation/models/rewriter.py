"""Rewriter: Claude, text only, turns the checked sentences into one paragraph per part."""

from __future__ import annotations

from mozaic.annotation.t4_explanation.models.client import Generation, Model
from mozaic.annotation.t4_explanation.prompts.rewriter import SYSTEM


class Rewriter(Model):
    """Claude: the system prompt and the sentences."""

    def generate(self, prompt: str) -> Generation:
        return self.ask(prompt, system=SYSTEM)
