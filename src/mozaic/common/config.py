"""Load the shared config.yaml and the .env file."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")
REPO_ROOT = Path(__file__).resolve().parents[3]


@cache
def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_env(path: str | Path | None = None) -> Path:
    """Load KEY=VALUE lines into the environment. Variables already set win.

    Uses the given path, else env_file from config.yaml. Returns the file used.
    """
    env = Path(path or load_config()["env_file"]).expanduser()
    if not env.is_absolute():
        env = REPO_ROOT / env
    if not env.exists():
        raise FileNotFoundError(f"env file not found: {env} (copy .env.example)")
    for line in env.read_text().splitlines():
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip("'\""))
    return env
