"""Qwen3-Omni's LoRA SFT with Hugging Face's Trainer: its thinker's attention layers, the protocol's recipe, validated and
saved as the other runs are. Its towers stay in eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mozaic.training.mllms.dataset import example, load_questions
from mozaic.training.mllms.qwen3_inputs import load_template, training_example
from mozaic.training.mllms.validation import Validation

LAYERS = r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"


class Questions(torch.utils.data.Dataset):
    def __init__(self, config: dict[str, Any], rows: list[dict[str, Any]], processor):
        self.config, self.rows, self.processor = config, rows, processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return training_example(self.processor, example(self.config["task"], self.rows[i], self.config["frames"]))


def one_question(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError(f"micro_batch_size must be 1, not {len(batch)}")
    return batch[0]


def run(config: dict[str, Any], out: Path, limit: int | None, world: int) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForTextToWaveform, Trainer, TrainingArguments

    t, lora = config["train"], config["lora"]
    per_step = t["micro_batch_size"] * world
    if t["batch_size"] % per_step:
        raise ValueError(f"batch_size {t['batch_size']} is not a multiple of micro_batch_size x GPUs ({per_step})")
    _, _, processor = load_template(config)
    _, train = load_questions(config["task"], config["questions"]["train"], limit)
    _, val = load_questions(config["task"], config["questions"]["val"], limit or t.get("val_questions"))
    thinker = AutoModelForTextToWaveform.from_pretrained(config["weights"], torch_dtype=torch.bfloat16,
                                                         enable_audio_output=False).thinker
    for tower in (thinker.visual, thinker.audio_tower):
        tower.eval()
        tower.train = lambda mode=True, tower=tower: tower  # the Trainer's train() stays out of the towers
    model = get_peft_model(thinker, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                               target_modules=LAYERS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=t["epochs"], learning_rate=t["lr"],
        weight_decay=t["weight_decay"], lr_scheduler_type="constant", warmup_steps=0,  # AdamW, no schedule
        per_device_train_batch_size=t["micro_batch_size"], gradient_accumulation_steps=t["batch_size"] // per_step,
        dataloader_num_workers=t["workers"], seed=t["seed"], bf16=True, save_strategy="epoch", save_only_model=False,
        logging_steps=10, report_to="none", remove_unused_columns=False, ddp_find_unused_parameters=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, ddp_timeout=180000000)
    trainer = Trainer(model=model, args=args, train_dataset=Questions(config, train, processor), data_collator=one_question,
                      callbacks=[Validation(config, out, val)])
    trainer.train(resume_from_checkpoint=any((out / "checkpoints").glob("checkpoint-*")))
