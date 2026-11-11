"""One model behind an OpenAI-compatible endpoint, with retries. Kept apart from evaluation/."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from mozaic.common.config import load_env

Row = dict[str, Any]


@dataclass
class Generation:
    """One reply."""

    raw: str
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None


class Model:
    """One model on the endpoint: one user message (and an optional system prompt), one reply, with retries and
    backoff. Settings: one section of config.yaml (annotator, verifier or rewriter) plus retry."""

    def __init__(self, config: Row):
        self.config = config

    def load(self) -> None:
        from openai import OpenAI

        load_env(self.config.get("env_file"))
        key = os.environ.get(self.config["api_key_env"])
        if not key:
            raise RuntimeError(f"{self.config['api_key_env']} is not set in the env file")
        self.client = OpenAI(api_key=key, base_url=self.config["base_url"])

    def ask(self, content: str | list[Row], system: str | None = None) -> Generation:
        attempts = self.config["retry"]["max_attempts"]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        for attempt in range(attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config["model_version"],
                    messages=messages,
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


def parse_json(raw: str) -> Any:
    """The JSON of a reply: the whole reply, else its last ```json fence (a model may reason before it),
    else from its first { to its last }; None if none is JSON."""
    text = raw.strip()
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    braces = text[text.find("{"):text.rfind("}") + 1] if "{" in text else ""
    for candidate in (text, *(fences[-1:]), braces):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def cut_off(gen: Generation) -> bool:
    """Whether a reply was cut off (it hit max_tokens)."""
    return gen.finish_reason not in ("stop", "end_turn", None)
