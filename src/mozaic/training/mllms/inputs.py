"""Which module builds a model's inputs in LLaMA-Factory: register, load_template, model_inputs."""

from __future__ import annotations

from typing import Any


def inputs_of(config: dict[str, Any]):
    if config["model"] == "gemma4":
        from mozaic.training.mllms import gemma4_inputs as inputs
    elif config["model"] == "nemotron":
        from mozaic.training.mllms import nemotron_inputs as inputs
    elif config["model"] == "qwen3_omni":
        from mozaic.training.mllms import qwen3_inputs as inputs
    elif config["model"] == "vita_1_5":
        from mozaic.training.mllms import vita_inputs as inputs
    elif config["model"] == "omnivinci":
        from mozaic.training.mllms import omnivinci_inputs as inputs
    elif config["model"] == "videollama2_av":
        from mozaic.training.mllms import videollama2_inputs as inputs
    elif config["model"] == "ola":
        from mozaic.training.mllms import ola_inputs as inputs
    elif config["model"] == "minicpm_o":
        from mozaic.training.mllms import minicpm_o_inputs as inputs
    else:
        from mozaic.training.mllms import qwen_omni_inputs as inputs
    return inputs
