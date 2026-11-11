"""A LoRA written into a copy of a model's released files, so its own code and other weights stay as released."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def merge(weights: str, adapter: Path, dest: Path, prefix: str = "") -> None:
    """The LoRA written into a copy of the released files, so its own code and TTS weights stay as released."""
    from safetensors.torch import load_file, save_file

    settings = json.loads((adapter / "adapter_config.json").read_text())
    scale = settings["lora_alpha"] / settings["r"]
    lora = {k: v for k, v in load_file(adapter / "adapter_model.safetensors").items() if ".lora_" in k}  # peft may add frozen embeddings
    pairs = {prefix + k[len("base_model.model."):-len(".lora_A.weight")]: k for k in lora if k.endswith(".lora_A.weight")}
    if not pairs or len(lora) != 2 * len(pairs):
        raise ValueError(f"{adapter}: {len(lora)} tensors, {len(pairs)} LoRA pairs")
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    done = set()
    for path in Path(weights).iterdir():
        if path.name.startswith("."):  # the download's own cache
            continue
        if path.suffix == ".safetensors":
            shard = load_file(path)
            for module in set(pairs) & {k[:-len(".weight")] for k in shard if k.endswith(".weight")}:
                a, b = lora[pairs[module]], lora[pairs[module].replace("lora_A", "lora_B")]
                name = f"{module}.weight"
                shard[name] = (shard[name].float() + scale * (b.float() @ a.float())).to(shard[name].dtype)
                done.add(module)
            save_file(shard, tmp / path.name, metadata={"format": "pt"})
        elif path.is_dir():
            shutil.copytree(path, tmp / path.name)
        else:
            shutil.copy(path, tmp / path.name)
    if done != set(pairs):
        raise ValueError(f"{len(set(pairs) - done)} LoRA layers not in the weights, e.g. {sorted(set(pairs) - done)[:3]}")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
