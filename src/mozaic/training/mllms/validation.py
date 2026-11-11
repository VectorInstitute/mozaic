"""Validation at each epoch's end, as the classifiers: the model answers val, evaluation's code scores it.

Only evaluation's generation settings, parser, prediction rows and metrics are used: no vLLM. The
replies come from HF generate (greedy, the same input as evaluation).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from mozaic.common.io import write_json, write_jsonl
from mozaic.evaluation.predictions import to_prediction
from mozaic.evaluation.run_inference import load_model_config
from mozaic.training.common.loop import get
from mozaic.training.mllms.dataset import example
from mozaic.training.mllms.inputs import inputs_of
from mozaic.training.mllms.run_config import prompts, with_sound

log = logging.getLogger(__name__)


def load_metrics(task: str):
    if task == "t1":
        from mozaic.evaluation.metrics import t1_metrics as metrics
    elif task == "t2":
        from mozaic.evaluation.metrics import t2_metrics as metrics
    elif task == "t3":
        from mozaic.evaluation.metrics import t3_metrics as metrics
    elif task == "t4":
        from mozaic.evaluation.metrics import t4_metrics as metrics
    else:
        raise ValueError(f"no metrics for task {task}")
    return metrics


def generate_args(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluation's generation settings for this model (eval_config), as HF generate arguments."""
    gen = load_model_config(config["eval_config"])["generation"]
    if gen.get("temperature", 0.0) != 0.0:
        raise ValueError(f"{config['eval_config']} samples (temperature {gen['temperature']}); validation is greedy")
    longest = config["train"].get("val_max_new_tokens") or gen.get("max_tokens", gen.get("max_new_tokens"))  # a cap for slow models
    return {"max_new_tokens": longest, "do_sample": False, "use_audio_in_video": with_sound(config["task"])}


class Validation(TrainerCallback):
    """Writes <run>/val_predictions/epoch_<n>.jsonl, val_scores/epoch_<n>.json and, for the best epoch
    by train.select_on, val_scores.json (with the checkpoint saved right after)."""

    def __init__(self, config: dict[str, Any], out: Path, questions: list[dict[str, Any]]):
        self.config, self.out, self.questions = config, out, questions
        self.task, self.key = config["task"], config["train"]["select_on"]
        self.generate_args = generate_args(config)
        self.inputs = inputs_of(config)
        self.template, self.tokenizer, self.processor = self.inputs.load_template(config)

    @torch.no_grad()
    def answer(self, model, question: dict[str, Any]) -> str:
        row = example(self.task, question, self.config["frames"])
        if hasattr(self.inputs, "answer"):  # a model with its own generate
            return self.inputs.answer(model, self.template, self.tokenizer, self.processor, row, self.generate_args)
        prompt_ids, mm = self.inputs.model_inputs(self.template, self.tokenizer, self.processor, row)
        ids = torch.tensor([prompt_ids], device=model.device)
        mm = {k: v.to(model.device) for k, v in mm.items() if isinstance(v, torch.Tensor)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                                 **mm, **self.generate_args)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if (self.out / "val_scores" / f"epoch_{epoch}.json").exists():  # validated before a resume
            return
        rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        model.eval()
        mine = {q["id"]: self.answer(model, q) for q in self.questions[rank::world]}  # the GPUs split val
        model.train()
        parts = [mine]
        if world > 1:
            parts = [None] * world
            dist.all_gather_object(parts, mine)
        if rank == 0:
            self.score(epoch, f"checkpoint-{state.global_step}", {k: v for p in parts for k, v in p.items()})
        if world > 1:
            dist.barrier()

    def score(self, epoch: int, checkpoint: str, raws: dict[str, str]) -> None:
        parse = prompts(self.task).parse
        predictions = [{"id": q["id"], "task": q["task"], **to_prediction(self.task, parse(raws[q["id"]], q["options"])),
                        "score": None, "raw": raws[q["id"]]} for q in self.questions]
        scores = load_metrics(self.task).score({q["id"]: q for q in self.questions}, predictions)
        write_jsonl(self.out / "val_predictions" / f"epoch_{epoch}.jsonl", predictions)
        write_json(self.out / "val_scores" / f"epoch_{epoch}.json", scores)
        best_path = self.out / "val_scores.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else None
        if best is None or get(scores, self.key) > get(best, self.key):
            best = {"epoch": epoch, "checkpoint": checkpoint, **scores}
            write_json(best_path, best)
        log.info("epoch %d: val %s %.4f, %s/%d unreadable (best: epoch %d)", epoch, self.key, get(scores, self.key),
                 scores.get("unreadable"), scores["questions"], best["epoch"])
        self.log_wandb(epoch, scores)

    def log_wandb(self, epoch: int, scores: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError:  # an environment without it (the models' own)
            return
        if wandb.run is None:
            return
        wandb.define_metric("val/epoch")
        wandb.define_metric("val/*", step_metric="val/epoch")  # HF's W&B callback owns the step
        row = {"val/epoch": epoch, f"val/{self.key}": get(scores, self.key), "val/unreadable": scores.get("unreadable")}
        if "accuracy" in scores:  # t1, t2, t3; t4 has none (its scores are text metrics)
            row["val/accuracy"] = scores["accuracy"]
        row |= {f"val/{k}": v["accuracy"] for k, v in scores.items() if isinstance(v, dict) and "accuracy" in v}
        wandb.log(row)
