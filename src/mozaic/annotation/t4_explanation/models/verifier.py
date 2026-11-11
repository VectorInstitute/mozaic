"""Verifier: Claude, text only, checks each sentence against the claims."""

from __future__ import annotations

from mozaic.annotation.t4_explanation.models.client import Generation, Model


class Verifier(Model):
    """Claude: the prompt only."""

    def generate(self, prompt: str) -> Generation:
        return self.ask(prompt)
