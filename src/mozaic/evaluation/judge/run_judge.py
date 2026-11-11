"""Judge saved T4 predictions: gpt-5.4-mini scores each part (0-10) against the reference, and an embedding
model (all-MiniLM-L6-v2, run here) gives the cosine of reference and prediction per part. Writes judge.jsonl
next to predictions.jsonl; run_eval then puts the judge's scores in scores.json. Resumable: the same command continues.

    python -m mozaic.evaluation.judge.run_judge --config qwen3_omni_vllm --task t4 --subset small --limit 100
    python -m mozaic.evaluation.judge.run_judge --config qwen3_omni_vllm --task t4 --subset small
    python -m mozaic.evaluation.judge.run_judge --config qwen3_omni_vllm --task t4 --subset small --retry-failed

Run it with .venv_gemini (it needs openai and sentence-transformers), in a Slurm job. A reply that is not the
judge's JSON is saved with ok: false; --retry-failed asks again. A prediction that could not be read gets 0 per part with no call;
--retry-failed judges it once it reads (inference's --retry-failed, with a newer parse).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mozaic.common.config import load_env
from mozaic.common.io import append_jsonl, read_jsonl, write_json, write_jsonl
from mozaic.common.paths import results_dir, use_subset
from mozaic.common.questions import select
from mozaic.evaluation.judge import prompt as jp
from mozaic.evaluation.metrics.text_metrics import cosine
from mozaic.evaluation.run_inference import load_configs, use_split

CONFIG = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
TASKS = ("t4",)

log = logging.getLogger(__name__)
Row = dict[str, Any]


class Client:
    """The judge on its OpenAI-compatible endpoint, with retries and backoff, and the embedder from its local weights."""

    def __init__(self, config: Row):
        from openai import OpenAI

        self.config = config
        c = config["judge"]
        load_env(c.get("env_file"))
        key = os.environ.get(c["api_key_env"])
        if not key:
            raise RuntimeError(f"{c['api_key_env']} is not set in the env file")
        self.chat_client = OpenAI(api_key=key, base_url=c["base_url"])
        self.embedder, self.embed_lock = None, threading.Lock()  # the clips are judged in threads: one encode at a time
        if config["embedder"] is not None:  # no embedder: no cosine
            from sentence_transformers import SentenceTransformer

            e = config["embedder"]
            self.embedder = SentenceTransformer(e["weights"], device=e.get("device"))

    def retry(self, call):
        attempts = self.config["retry"]["max_attempts"]
        for attempt in range(attempts):
            try:
                return call()
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("max_attempts must be at least 1")

    def judge(self, user: str) -> tuple[str, Row]:
        c = self.config["judge"]
        resp = self.retry(lambda: self.chat_client.chat.completions.create(
            model=c["model_version"],
            messages=[{"role": "system", "content": jp.SYSTEM}, {"role": "user", "content": user}],
            **c["generation"]))
        return resp.choices[0].message.content or "", resp.usage.model_dump(exclude_none=True) if resp.usage else {}

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if self.embedder is None:
            return None
        with self.embed_lock:
            return self.embedder.encode(texts, show_progress_bar=False).tolist()


def settings(config: Row) -> Row:
    """What makes two judge runs comparable; a folder judged with other settings is refused."""
    text = jp.SYSTEM + jp.USER
    return {"judge": config["judge"]["model_version"], "generation": config["judge"]["generation"],
            "embedder": (config["embedder"] or {}).get("model_version"),
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}


def judge_one(client: Client, question: Row, prediction: Row) -> Row:
    """One clip: the judge's scores and the cosine per part (None without an embedder). An unreadable
    prediction scores 0, with no call."""
    row: Row = {"id": prediction["id"]}
    sections = (prediction.get("pred_target") or {}).get("sections")
    if not sections:
        return {**row, "ok": True, "scores": jp.UNUSABLE, "cosine": {p: 0.0 for p in jp.PARTS}, "calls": 0,
                "time": datetime.now().isoformat(timespec="seconds")}
    reference = question["gt_target"]["sections"]
    raw, usage = client.judge(jp.build_user(reference, sections))
    vectors = client.embed([reference[p] for p in jp.PARTS] + [sections[p] for p in jp.PARTS])
    n = len(jp.PARTS)
    scores = jp.parse(raw)
    sims = None if vectors is None else {p: round(cosine(vectors[i], vectors[n + i]), 4) for i, p in enumerate(jp.PARTS)}
    return {**row, "ok": scores is not None, "scores": scores, "cosine": sims, "raw": raw, "usage": usage,
            "calls": 1, "time": datetime.now().isoformat(timespec="seconds")}


def run(model_config: Row, task: str, limit: int | None = None, retry_failed: bool = False,
        client: Client | None = None, config: Row = CONFIG) -> Path:
    if task not in TASKS:
        raise SystemExit(f"the judge scores explanations: {', '.join(TASKS)} only")
    if model_config.get("model_version") == config["judge"]["model_version"]:
        raise SystemExit(f"{config['judge']['model_version']} is the model being judged; pick another judge")
    out_dir = results_dir(model_config["model"], model_config["experiment_name"], task)
    inference = json.loads((out_dir / "inference_settings.json").read_text())
    questions = {q["id"]: q for q in select(read_jsonl(inference["question_file"]), inference.get("filters"))}
    predictions = read_jsonl(out_dir / "predictions.jsonl")

    judge_path, settings_path = out_dir / "judge.jsonl", out_dir / "judge_settings.json"
    now = settings(config)
    judged_before = judge_path.exists() and judge_path.stat().st_size > 0  # nothing judged yet: any settings
    if judged_before and settings_path.exists() and json.loads(settings_path.read_text()) != now:
        raise SystemExit(f"{out_dir} was judged with other settings; move judge.jsonl and judge_settings.json away")
    write_json(settings_path, now)

    done = read_jsonl(judge_path) if judge_path.exists() else []
    if retry_failed:
        # also the 0 given to an unreadable reply that reads now (inference's --retry-failed, a newer parse)
        readable = {p["id"] for p in predictions if (p.get("pred_target") or {}).get("sections")}
        kept = [r for r in done if r["ok"] and not (r.get("calls") == 0 and r["id"] in readable)]
        log.info("[%s] asking again %d failed judgements and 0s of replies that read now", task, len(done) - len(kept))
        write_jsonl(judge_path, kept)
        done = kept
    seen = {r["id"] for r in done}
    todo = [p for p in predictions if p["id"] not in seen][:limit]
    log.info("[%s] %d judged, %d to judge with %s", task, len(seen), len(todo), now["judge"])
    if not todo:
        return judge_path

    client = client or Client(config)
    failures = 0
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = {pool.submit(judge_one, client, questions[p["id"]], p): p["id"] for p in todo}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except Exception as e:  # a failed call is not saved, so the next run asks again
                failures += 1
                log.warning("[%d/%d] %s failed: %s", i, len(todo), futures[future], e)
                if failures >= config["stop_after_failures"]:
                    for f in futures:
                        f.cancel()
                    raise SystemExit(f"{failures} clips in a row failed; stopping (rerun to continue)") from e
                continue
            failures = 0
            append_jsonl(judge_path, row)
            scores = row["scores"] or {}
            log.info("[%d/%d] %s  ok=%s  video=%s audio=%s", i, len(todo), row["id"], row["ok"],
                     scores.get("video", {}).get("score"), scores.get("audio", {}).get("score"))

    rows = read_jsonl(judge_path)
    tokens = {k: sum((r.get("usage") or {}).get(k, 0) for r in rows)
              for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    log.info("[%s] %d judged (%d failed), judge tokens %s; judgements in %s", task, len(rows),
             sum(not r["ok"] for r in rows), tokens, judge_path)
    return judge_path


def main() -> None:
    p = argparse.ArgumentParser(description="Judge saved T4 predictions against the reference explanations.")
    p.add_argument("--config", required=True, help="the model config the predictions were made with: <model>/<run> or a full name")
    p.add_argument("--task", default="t4", help="t4 (the only task answered in text)")
    p.add_argument("--subset", help="run on a subset, e.g. small (T4: small only)")
    p.add_argument("--split", help="for configs whose files say {split}: test (default), val, train")
    p.add_argument("--limit", type=int, help="judge at most this many more predictions (e.g. 100 for a cost check)")
    p.add_argument("--retry-failed", action="store_true", help="ask again the judgements that could not be read")
    p.add_argument("--judge-model", help="another judge than config.yaml's (e.g. one the endpoint's key can use)")
    p.add_argument("--no-embedder", action="store_true", help="no cosine: the embedder is not loaded")
    args = p.parse_args()
    if args.judge_model:
        CONFIG["judge"]["model_version"] = args.judge_model
    if args.no_embedder:
        CONFIG["embedder"] = None
    use_subset(args.subset)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for config in load_configs(args.config, args.task):
        use_split(config, args.split)
        run(config, args.task, args.limit, args.retry_failed)


if __name__ == "__main__":
    main()
