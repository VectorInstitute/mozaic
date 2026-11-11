"""Score saved predictions. Never loads a model, so it runs anywhere.

    python -m mozaic.evaluation.run_eval --config gemini_3_8_flash --task t1
    python -m mozaic.evaluation.run_eval --config avhubert/ln_trainsmall --task t1
    python -m mozaic.evaluation.run_eval --config gemini_3_8_flash --task t1 --subset small   # results_small/
    python -m mozaic.evaluation.run_eval --config qwen3_omni_vllm --task t1 --split val      # files named t1_{split}.jsonl

Scoring a full run of t1, t2 or t3 also scores MOZAIC-Small from its answers (small_from_full).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from mozaic.common.io import iter_jsonl, read_jsonl, write_json
from mozaic.common.paths import qas_dir, results_dir, subset, use_subset
from mozaic.common.questions import select
from mozaic.evaluation.run_inference import SMALL, fill_from_full, load_configs, use_split

log = logging.getLogger(__name__)

def eval_task(model: str, experiment: str, task: str) -> None:
    out_dir = results_dir(model, experiment, task)
    settings = json.loads((out_dir / "inference_settings.json").read_text())
    filters = settings.get("filters")
    questions = {q["id"]: q for q in select(read_jsonl(settings["question_file"]), filters)}
    predictions = read_jsonl(out_dir / "predictions.jsonl")

    if task == "t1":
        from mozaic.evaluation.metrics import t1_metrics as metrics
    elif task == "t2":
        from mozaic.evaluation.metrics import t2_metrics as metrics
    elif task == "t3":
        from mozaic.evaluation.metrics import t3_metrics as metrics
    elif task == "t4":
        from mozaic.evaluation.metrics import t4_metrics as metrics
    else:
        raise ValueError(f"unknown task {task}")
    if task == "t4":  # the explanation's judge and cosine, from judge/run_judge.py when it has run
        judge = out_dir / "judge.jsonl"
        result = metrics.score(questions, predictions, read_jsonl(judge) if judge.exists() else None)
    else:
        result = metrics.score(questions, predictions)
    scores = {"model": model, "experiment": experiment, "task": task,
              "question_file": settings["question_file"], "filters": filters, **result}
    write_json(out_dir / "scores.json", scores)

    if task == "t4":
        log.info("[%s] %d/%d answered, %d unreadable, judge %s", task, scores["answered"], scores["questions"],
                 scores["unreadable"], scores.get("judge"))
    else:
        log.info("[%s] %d/%d answered, accuracy %s %s, macro F1 %s", task, scores["answered"], scores["questions"],
                 scores["accuracy"], scores["ci95"], scores["macro"]["f1"])
    log.info("[%s] scores in %s", task, out_dir / "scores.json")


def small_from_full(model: str, experiment: str, task: str) -> None:
    """MOZAIC-Small's answers and scores from a full run of t1, t2 or t3, without running the model.

    A Small question is a copy of the full question with its id (same clips, same option order), so the model's Small
    answer is its full answer: run_inference.fill_from_full copies it into results_small/ (with copied_from), keeping
    any answer of a --subset small run, and Small is scored once every Small question has one answer.
    """
    full_dir = results_dir(model, experiment, task)
    settings = json.loads((full_dir / "inference_settings.json").read_text())
    if task not in ("t1", "t2", "t3"):
        return  # t4 is on Small only
    use_subset(SMALL)
    try:
        small_file = qas_dir(task) / Path(settings["question_file"]).name
        if not small_file.exists():
            return
        small_dir = results_dir(model, experiment, task)
        made = small_dir / "inference_settings.json"
        own = json.loads(made.read_text()) if made.exists() else {**settings, "question_file": str(small_file)}
        questions = select(read_jsonl(small_file), own.get("filters"))
        fill_from_full(model, experiment, task, questions, settings=own)
        pred_path = small_dir / "predictions.jsonl"
        answered = {r["id"] for r in iter_jsonl(pred_path)} if pred_path.exists() else set()
        missing = sum(q["id"] not in answered for q in questions)
        if missing:
            log.info("[%s] Small not scored yet: %d of its %d questions have no answer", task, missing, len(questions))
            return
        eval_task(model, experiment, task)
    finally:
        use_subset(None)


def main() -> None:
    p = argparse.ArgumentParser(description="Score saved predictions.")
    p.add_argument("--config", required=True, help="<model>/<run> (zeroshot if left out), a full config name, or a path")
    p.add_argument("--task", help="score only this task (default: every task the config has)")
    p.add_argument("--subset", help="run on a subset, e.g. small (default: the full dataset)")
    p.add_argument("--split", help="the split scored, for configs whose files say {split}: test (default), val, train")
    args = p.parse_args()
    use_subset(args.subset)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for name in ("httpx", "httpx2"):  # openai >= 3 sends requests through httpx2
        logging.getLogger(name).setLevel(logging.WARNING)

    for config in load_configs(args.config, args.task):
        use_split(config, args.split)
        for task in config["tasks"]:
            eval_task(config["model"], config["experiment_name"], task)
            if subset() is None:
                small_from_full(config["model"], config["experiment_name"], task)


if __name__ == "__main__":
    main()
