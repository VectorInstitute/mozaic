"""Ola's LoRA SFT: the protocol's recipe in a small loop, one process per GPU, as LLaMA-Factory's runs save and validate.

The language model alone trains; its picture and sound towers stay in eval (their dropout and layer drop would change
the input). Gradients are averaged by hand: the authors' forward takes lists of clips.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from mozaic.evaluation.run_inference import load_model, load_model_config
from mozaic.training.mllms import ola_inputs
from mozaic.training.mllms.dataset import example, load_questions
from mozaic.training.mllms.validation import Validation

log = logging.getLogger(__name__)
LAYERS = r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def save(model, optimizer, out: Path, epoch: int, step: int) -> None:
    path = out / "checkpoints" / f"checkpoint-{step}"
    model.save_pretrained(path, safe_serialization=True)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    (path / "trainer_state.json").write_text(json.dumps({"epoch": epoch, "global_step": step}))


def resume(model, optimizer, out: Path) -> tuple[int, int]:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    saved = sorted((out / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not saved:
        return 0, 0
    set_peft_model_state_dict(model, load_file(saved[-1] / "adapter_model.safetensors"))
    optimizer.load_state_dict(torch.load(saved[-1] / "optimizer.pt", map_location="cuda"))
    state = json.loads((saved[-1] / "trainer_state.json").read_text())
    return state["epoch"], state["global_step"]


def run(config: dict[str, Any], out: Path, limit: int | None, world: int) -> None:
    from peft import LoraConfig, get_peft_model

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]  # its loader takes the first GPU it sees
    dist.init_process_group("nccl", timeout=timedelta(hours=12))  # one long reply in validation keeps the others waiting
    rank, t, lora = dist.get_rank(), config["train"], config["lora"]
    if t["micro_batch_size"] != 1 or t["batch_size"] % world:
        raise ValueError("micro_batch_size must be 1, and batch_size a multiple of the GPUs")
    tested_config = {**load_model_config(config["eval_config"]), "weights": config["weights"]}
    tested = load_model(tested_config)
    tested.load()
    ola_inputs.TESTED = tested
    model = get_peft_model(tested.model, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                                    target_modules=LAYERS))
    if rank == 0:
        model.print_trainable_parameters()
    tested.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    first_epoch, step = resume(model, optimizer, out)
    (_, train), (_, val) = (load_questions(config["task"], config["questions"][s], limit) for s in ("train", "val"))
    validation = Validation(config, out, val)
    every, inner = t["batch_size"] // world, tested.model.get_model()

    for epoch in range(first_epoch + 1, t["epochs"] + 1):
        order = list(range(len(train)))
        random.Random(t["seed"] + epoch).shuffle(order)
        mine = order[:len(order) // world * world][rank::world]
        model.eval()
        inner.layers.train()  # the language model alone: the towers stay in eval
        inner.training = True  # the module's own flag, not its towers': checkpointing and the loss read it
        tested.model.config.use_cache = False
        for i, index in enumerate(mine, 1):
            inputs = ola_inputs.training_inputs(tested, example(config["task"], train[index], config["frames"]))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = tested.model(**inputs, use_cache=False).loss
            (loss / every).backward()
            if i % every == 0 or i == len(mine):
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and step % 10 == 1:
                    log.info("epoch %d step %d: loss %.4f, GPU peak %.1f GB", epoch, step, loss.item(),
                             torch.cuda.max_memory_allocated() / 2**30)
        tested.model.config.use_cache = True
        validation.on_epoch_end(None, SimpleNamespace(epoch=epoch, global_step=step), None, model=model)
        if rank == 0:
            save(model, optimizer, out, epoch, step)
        dist.barrier()
    dist.destroy_process_group()
