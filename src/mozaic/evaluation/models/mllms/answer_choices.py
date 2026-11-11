"""The token filter that keeps a model's reply to one of the options (answer_choices: true in a config).

For models whose own code generates token by token with transformers (MiniCPM-o's own mode, Uni-MoE-2.0-Omni,
Ola, VITA-1.5): generate's prefix_allowed_tokens_fn lets through only the next tokens of the replies still
possible, then the end, so the reply is always exactly one of them (e.g. {"answer": "the video"}).
"""

from __future__ import annotations


def choice_filter(replies: list[list[int]], end_ids: list[int]):
    """A token filter for generate (prefix_allowed_tokens_fn): only the next tokens of the replies still possible,
    then the end. `generated` holds the reply's tokens so far (the prompt goes in as embeddings)."""

    def allowed(batch_id: int, generated) -> list[int]:
        done = generated.tolist()
        nxt = {r[len(done)] for r in replies if len(r) > len(done) and r[: len(done)] == done}
        return sorted(nxt) or end_ids

    return allowed
