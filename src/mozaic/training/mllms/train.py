"""Step 2: LoRA SFT with LLaMA-Factory, validated each epoch (torchrun, one process per GPU).

A checkpoint per epoch in <run>/checkpoints/; a rerun resumes from the last one.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import yaml


def config_model() -> str:
    """The config's `model:`, read before anything imports torch (run_config's loader does)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", default="")
    path = Path(p.parse_known_args()[0].config)
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "configs" / f"{path}.yaml"
    return yaml.safe_load(path.read_text())["model"] if path.is_file() else ""


# One GPU per process for the authors' loaders (native_train.py), before anything imports torch: it caches the GPU count.
MODEL = config_model()
if "LOCAL_RANK" in os.environ and MODEL in ("videollama2_av", "omnivinci", "vita_1_5"):
    os.environ["CUDA_VISIBLE_DEVICES"], os.environ["LOCAL_RANK"] = os.environ["LOCAL_RANK"], "0"
if MODEL == "nemotron":  # its layers span GPUs: asynchronous launches gave out-of-range labels in the loss
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

from mozaic.common.io import share_tree
from mozaic.training.common.tracking import ID_FILE
from mozaic.training.mllms.dataset import load_questions
from mozaic.training.mllms.inputs import inputs_of
from mozaic.training.mllms.run_config import load, parse_args, run_name, with_sound
from mozaic.training.mllms.validation import Validation


def llamafactory_args(config: dict[str, Any], out: Path, world: int) -> dict[str, Any]:
    """Our config → LLaMA-Factory's arguments: train: (the recipe), lora:, and llamafactory: as it is."""
    t, lora = config["train"], config["lora"]
    per_step = t["micro_batch_size"] * world
    if t["batch_size"] % per_step:
        raise ValueError(f"batch_size {t['batch_size']} is not a multiple of micro_batch_size x GPUs ({per_step})")
    return {
        "stage": "sft", "do_train": True,
        "model_name_or_path": config["weights"],
        "template": inputs_of(config).register(config),
        "use_audio_in_video": with_sound(config["task"]) and config["model"] != "minicpm_o",  # MiniCPM-o: the sound apart
        "dataset_dir": str(out / "data"), "dataset": f"{config['task']}_train",
        "finetuning_type": "lora", "lora_rank": lora["rank"], "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"], "lora_target": lora["target"],
        "output_dir": str(out / "checkpoints"),
        "num_train_epochs": t["epochs"], "learning_rate": t["lr"], "weight_decay": t["weight_decay"],
        "lr_scheduler_type": "constant", "warmup_ratio": 0.0,  # AdamW, no schedule
        "per_device_train_batch_size": t["micro_batch_size"],
        "gradient_accumulation_steps": t["batch_size"] // per_step,
        "dataloader_num_workers": t["workers"], "seed": t["seed"], "bf16": True,
        "save_strategy": "epoch", "save_only_model": False,  # with the optimizer, to resume
        "report_to": "wandb" if (config.get("log") or {}).get("wandb") else "none",
        "run_name": run_name(config),
        **config["llamafactory"],
    }


def use_wandb(config: dict[str, Any], out: Path) -> None:
    """HF's W&B callback reads these: the same project, group and run id scheme as the classifiers."""
    options = config.get("log") or {}
    os.environ.setdefault("WANDB_PROJECT", options.get("project", "mozaic"))
    if options.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", options["entity"])
    os.environ["WANDB_RUN_GROUP"] = f"{config['model']}/{config['task']}"
    os.environ["WANDB_DIR"] = str(out)
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = (out / ID_FILE).read_text().strip()


def main() -> None:
    args = parse_args("Step 2: LoRA SFT, validated each epoch.")
    logging.basicConfig(level=logging.INFO if int(os.environ.get("RANK", "0")) == 0 else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("qwen_omni_utils").setLevel(logging.WARNING)  # a line per clip read would bury the loss and val lines
    config, out = load(args)
    if not (out / "data" / "dataset_info.json").exists():
        raise SystemExit(f"no dataset in {out / 'data'}: run step 1 (mozaic.training.mllms.dataset) first")
    use_wandb(config, out)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if config["model"] == "gemma4":  # its own Trainer: LLaMA-Factory's transformers is too old for it
        from mozaic.training.mllms.gemma4_train import run

        run(config, out, args.limit, world)
    elif config["model"] == "qwen3_omni":  # its own Trainer: LLaMA-Factory lays out its sound in video differently
        from mozaic.training.mllms.qwen3_train import run

        run(config, out, args.limit, world)
    elif hasattr(inputs_of(config), "LAYERS"):  # authors' code that takes clips as lists: a small loop
        from mozaic.training.mllms.native_train import run

        run(config, out, args.limit, world, inputs_of(config))
    elif config["model"] == "ola":  # its own loop: the authors' model takes its clips as lists
        from mozaic.training.mllms.ola_train import run

        run(config, out, args.limit, world)
    else:
        if hasattr(inputs_of(config), "patch"):
            inputs_of(config).patch(config)
        from llamafactory.train.tuner import run_exp

        _, val = load_questions(config["task"], config["questions"]["val"], args.limit)
        run_exp(llamafactory_args(config, out, world), callbacks=[Validation(config, out, val)])
    if int(os.environ.get("RANK", "0")) == 0:
        share_tree(out / "checkpoints")


if __name__ == "__main__":
    main()
