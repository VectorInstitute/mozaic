"""LoRA SFT for models whose authors' code takes clips as lists: the protocol's recipe in a small loop, one process per
GPU, saved and validated as LLaMA-Factory's runs are. The model and inputs come from its test code.

The language model alone trains; its towers stay in eval (dropout and layer drop would change the input). Gradients are
averaged by hand: their forward takes lists of clips.
"""

from __future__ import annotations

import json
import random
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from mozaic.evaluation.run_inference import load_model, load_model_config
from mozaic.training.mllms.dataset import example, load_questions
from mozaic.training.mllms.ola_train import log, resume, save
from mozaic.training.mllms.validation import Validation


def run(config: dict[str, Any], out: Path, limit: int | None, world: int, inputs) -> None:
    from peft import LoraConfig, get_peft_model

    dist.init_process_group("nccl", timeout=timedelta(hours=12))  # one long reply in validation keeps the others waiting
    rank, t, lora = dist.get_rank(), config["train"], config["lora"]
    if t["micro_batch_size"] != 1 or t["batch_size"] % world:
        raise ValueError("micro_batch_size must be 1, and batch_size a multiple of the GPUs")
    if hasattr(inputs, "load_tested"):
        tested = inputs.load_tested(config)
    else:
        tested = load_model({**load_model_config(config["eval_config"]), "weights": config["weights"]})
        tested.load()
    inputs.TESTED = tested
    tested.model.requires_grad_(False)
    language = inputs.language_model(tested) if hasattr(inputs, "language_model") else tested.model
    model = get_peft_model(language, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                                    target_modules=inputs.LAYERS))
    if rank == 0:
        model.print_trainable_parameters()
    if hasattr(inputs, "enable_checkpointing"):
        inputs.enable_checkpointing(language)
    else:
        language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=inputs.SCALE_LOSS)
    first_epoch, step = resume(model, optimizer, out)
    _, train = load_questions(config["task"], config["questions"]["train"], limit)
    _, val = load_questions(config["task"], config["questions"]["val"], limit or t.get("val_questions"))
    validation = Validation(config, out, val)
    every = t["batch_size"] // world

    for epoch in range(first_epoch + 1, t["epochs"] + 1):
        order = list(range(len(train)))
        random.Random(t["seed"] + epoch).shuffle(order)
        mine = order[:len(order) // world * world][rank::world]
        if hasattr(inputs, "train_layers"):
            inputs.train_layers(tested)
        else:
            model.eval()
            inner = tested.model.get_model()
            inner.layers.train()
            inner.training = True  # the module's own flag, not its towers': checkpointing reads it
        language.config.use_cache = False
        for i, index in enumerate(mine, 1):
            row = example(config["task"], train[index], config["frames"])
            with torch.autocast("cuda", dtype=inputs.DTYPE):
                loss = inputs.loss(tested, row) if hasattr(inputs, "loss") else tested.model(
                    **inputs.training_inputs(tested, row), use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError(f"loss {loss.item()} on {row['videos'][0]}: no labels left, or an overflow")
            scaler.scale(loss / every).backward()
            if i % every == 0 or i == len(mine):
                scaler.unscale_(optimizer)
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and step % 10 == 1:
                    log.info("epoch %d step %d: loss %.4f, GPU peak %.1f GB", epoch, step, loss.item(),
                             torch.cuda.max_memory_allocated() / 2**30)
        language.config.use_cache = True
        validation.on_epoch_end(None, SimpleNamespace(epoch=epoch, global_step=step), None, model=model)
        if rank == 0:
            save(model, optimizer, out, epoch, step)
        dist.barrier()
    dist.destroy_process_group()
