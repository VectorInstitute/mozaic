"""Train and validate one model on one task.

AdamW on the trainable parameters, weighted cross-entropy, bf16, one process per GPU (torchrun).
Micro-batches are accumulated to batch_size per optimizer step. Each epoch the val split is
scored with evaluation's own metrics; best.pt is the epoch with the best select_on score.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from mozaic.common.io import write_json
from mozaic.evaluation.predictions import from_probs
from mozaic.training.common import store, tuning
from mozaic.training.common.data import CLASSES, ClipDataset, label
from mozaic.training.common.pairs import PairDataset
from mozaic.training.common.tracking import Tracker, epoch_metrics

log = logging.getLogger(__name__)


def start_distributed() -> tuple[int, int, torch.device]:
    """Rank, world size and this process's GPU. One process without torchrun."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if world > 1:
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    return int(os.environ.get("RANK", "0")), world, device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(task: str, questions: list[dict[str, Any]], mode: str) -> list[float] | None:
    """balanced: n / (classes x count) per class, from the train questions. none: no weights."""
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"unknown class_weights {mode}, expected balanced or none")
    counts = np.bincount([label(task, q) for q in questions], minlength=len(CLASSES[task]))
    return (len(questions) / (len(counts) * np.maximum(counts, 1))).round(6).tolist()


def load_metrics(task: str):
    if task == "t1":
        from mozaic.evaluation.metrics import t1_metrics as metrics
    elif task == "t2":
        from mozaic.evaluation.metrics import t2_metrics as metrics
    elif task == "t3":
        from mozaic.evaluation.metrics import t3_metrics as metrics
    else:
        raise ValueError(f"unknown task {task}")
    return metrics


def get(scores: dict[str, Any], key: str) -> float:
    """A nested score by dotted key, e.g. macro.f1. None (a class never predicted) ranks last."""
    for part in key.split("."):
        scores = scores[part]
    return float("-inf") if scores is None else float(scores)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_loader(dataset, processor, sampler, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                      collate_fn=processor.collate, pin_memory=True, persistent_workers=workers > 0)


@torch.no_grad()
def validate(net: nn.Module, loader: DataLoader, device: torch.device, world: int, amp) -> tuple[dict[int, list[float]], float]:
    """Class probabilities per question position, and the unweighted val loss, over every GPU."""
    net.eval()
    probs, losses = {}, {}
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = net(batch)[0].float()
        loss = nn.functional.cross_entropy(logits, batch["label"], reduction="none")
        for i, p, l in zip(batch["index"].tolist(), logits.softmax(-1).tolist(), loss.tolist(), strict=True):
            probs[i], losses[i] = p, l
    if world > 1:  # the sampler pads with repeats; positions keep each question once
        gathered = [None] * world
        dist.all_gather_object(gathered, (probs, losses))
        probs = {i: p for part, _ in gathered for i, p in part.items()}
        losses = {i: l for _, part in gathered for i, l in part.items()}
    return probs, float(np.mean(list(losses.values())))


def score(task: str, questions: list[dict[str, Any]], probs: dict[int, list[float]], config: dict[str, Any]) -> dict[str, Any]:
    """Evaluation's scores for the val split, from the class probabilities."""
    classes = CLASSES[task]
    rows = [{"id": questions[i]["id"], "model": config["model"], "task": task,
             **from_probs(task, dict(zip(classes, p, strict=True)))} for i, p in sorted(probs.items())]
    return load_metrics(task).score({q["id"]: q for q in questions}, rows)


def train(model: nn.Module, processor, config: dict[str, Any], train_qs: list[dict[str, Any]],
          val_qs: list[dict[str, Any]], weights: list[float] | None, out: Path, tracker: Tracker,
          rank: int, world: int, device: torch.device) -> None:
    task, t = config["task"], config["train"]
    mode = config["tuning"]["mode"]
    micro = t["micro_batch_size"]
    if t["batch_size"] % (micro * world):
        raise SystemExit(f"batch_size {t['batch_size']} is not a multiple of micro_batch_size x GPUs ({micro} x {world})")
    accum = t["batch_size"] // (micro * world)
    amp = {"bf16": torch.bfloat16, "none": None}[t.get("amp", "bf16")]

    model.to(device)
    net = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False,
                                  find_unused_parameters=model.dynamic_graph) if world > 1 else model
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device) if weights else None)

    dataset = PairDataset if task == "t3" else ClipDataset  # t3: two clips per question (common/pairs.py)
    train_set = dataset(task, train_qs, processor, config)
    val_set = dataset(task, val_qs, processor, config)
    train_sampler = DistributedSampler(train_set, world, rank, shuffle=True, seed=t["seed"])
    val_sampler = DistributedSampler(val_set, world, rank, shuffle=False)
    train_loader = make_loader(train_set, processor, train_sampler, micro, t["workers"])
    val_loader = make_loader(val_set, processor, val_sampler, micro, t["workers"])

    if store.finished(out, t["epochs"]) and not (out / "last.pt").exists():  # done, last.pt removed: not again
        if rank == 0:
            log.info("already trained: %d epochs in %s", t["epochs"], out / "history.jsonl")
        tracker.finish()
        return
    start, best, history, steps = 1, float("-inf"), [], 0
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        start, best, history, steps = state["epoch"] + 1, state["best"], state["history"], state["steps"]
        if rank == 0:
            log.info("resumed after epoch %d (best %s %.4f)", state["epoch"], t["select_on"], best)

    for epoch in range(start, t["epochs"] + 1):
        began = time.time()
        train_sampler.set_epoch(epoch)
        tuning.train_mode(model, mode)
        total, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            batch = to_device(batch, device)
            sync = step % accum == 0 or step == len(train_loader)
            # t3 shares the gradients every micro-step: a micro-batch may leave a part unused, and DDP (torch 2.3.1, as
            # in .venv_avhubert) then keeps that part's gradients from earlier no_sync micro-steps on each GPU apart
            context = net.no_sync() if world > 1 and not sync and task != "t3" else contextlib.nullcontext()
            with context:
                with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                    logits, step_logits = net(batch)
                if step_logits is None:
                    loss = loss_fn(logits.float(), batch["label"])
                else:  # every valid step carries its clip's label
                    valid = batch["valid"]
                    loss = loss_fn(step_logits[valid].float(), batch["label"][:, None].expand_as(valid)[valid])
                (loss / accum).backward()
            if sync:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            total += loss.item() * len(batch["label"])
            seen += len(batch["label"])
            if rank == 0 and step % t.get("log_every", 50) == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch, step, len(train_loader), total / seen)
                tracker.log({"train/loss": loss.item(), "epoch": epoch}, steps)

        if world > 1:  # every GPU validates with rank 0's BatchNorm statistics, the ones saved
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)
        probs, val_loss = validate(net, val_loader, device, world, amp)
        if rank == 0:
            scores = score(task, val_qs, probs, config)
            value = get(scores, t["select_on"])
            row = {"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss,
                   "accuracy": scores["accuracy"], "macro_f1": scores["macro"]["f1"],
                   "balanced_accuracy": scores["balanced_accuracy"], "cohen_kappa": scores["cohen_kappa"],
                   "seconds": round(time.time() - began)}
            history.append(row)
            store.save_epoch(out, epoch, history, scores)
            tracker.log({**epoch_metrics(scores, val_loss, row["train_loss"], CLASSES[task]),
                         "epoch": epoch, "val/best": max(best, value)}, steps)
            weights_now = tuning.trainable_state(model, mode)
            if value > best:
                best = value
                store.save_checkpoint(out / "best.pt", {"model": weights_now, "epoch": epoch, "score": value,
                                                        "select_on": t["select_on"], "classes": CLASSES[task],
                                                        "config": config})
                write_json(out / "val_scores.json", {"epoch": epoch, **scores})
            store.save_checkpoint(out / "last.pt", {"model": weights_now, "optimizer": optimizer.state_dict(),
                                                    "epoch": epoch, "best": best, "history": history, "steps": steps})
            log.info("epoch %d | train loss %.4f | val loss %.4f | %s %.4f (best %.4f) | %ds", epoch,
                     row["train_loss"], val_loss, t["select_on"], value, best, row["seconds"])
        if world > 1:
            dist.barrier()
    if rank == 0:  # every epoch done: best.pt is the result; last.pt (weights and optimizer) only resumed a run
        (out / "last.pt").unlink(missing_ok=True)
    tracker.finish()
