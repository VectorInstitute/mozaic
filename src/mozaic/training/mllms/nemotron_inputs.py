"""Nemotron 3 Nano Omni's training inputs and answers, with its own processor, as its T4 test run builds them
(nemotron_omni_vllm). Its weights don't fit one A40, and its Mamba layers run without their compiled kernel: one copy over four GPUs. Its code has no gradient checkpointing
and its forward takes no sound, so this checkpoints its layers and builds the clips into the embeddings as its generate
does. One difference on purpose: the sound is in [-1, 1], as the trained test run (normalize_audio) corrects it too.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from mozaic.training.mllms.gemma4_inputs import read_frames
from mozaic.training.mllms.lora_merge import merge as merge_lora

PLACEHOLDERS, END_OF_TURN = "<video><audio>", "<|im_end|>\n"
VIDEO_TAG, SOUND_TAG = "<video>", "<so_embedding>"
LAYERS = r"backbone\.layers\.\d+\.mixer\.(in_proj|out_proj|q_proj|k_proj|v_proj|o_proj)"  # not its 128 experts per layer
DTYPE, SCALE_LOSS, PATCHES = torch.bfloat16, False, 196  # 224 x 224 frames, as the test config's video_target_num_patches
TESTED = None  # set by native_train.run


def read_sound(path: str):
    """The test run's reader (vLLM's), with its sound back in [-1, 1]."""
    from pathlib import Path

    from vllm.multimodal.media import AudioMediaIO

    samples = AudioMediaIO().load_file(Path(path))[0]
    return samples / 32768.0 if abs(samples).max() > 1.0 else samples


def load_processor(weights: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(weights, trust_remote_code=True)
    processor.image_processor.video_target_num_patches = PATCHES
    return processor


def device_map(layers: int) -> dict[str, int]:
    """Its towers whole on the first GPU (its vision code doesn't take a split), the language model's layers on the rest."""
    out = {"vision_model": 0, "mlp1": 0, "sound_encoder": 0, "sound_projection": 0, "language_model.backbone.embeddings": 0,
           "language_model.backbone.norm_f": 3, "language_model.lm_head": 3}
    out |= {f"language_model.backbone.layers.{i}": 1 + 3 * i // layers for i in range(layers)}
    return out


def use_local_kernels() -> None:
    """Its Mamba layers ask the kernel hub for mamba-ssm and causal-conv1d; here they are built locally (the training
    folder). Without them it falls back to a slow torch path whose numerics differ from its test run's. Patched where
    transformers defines the lookup, before its code imports it."""
    import importlib

    from transformers.integrations import hub_kernels

    for name in ("causal_conv1d", "mamba_ssm.ops.triton.selective_state_update", "mamba_ssm.ops.triton.ssd_combined"):
        importlib.import_module(name)
    local = {"causal-conv1d": "causal_conv1d", "mamba-ssm": "mamba_ssm"}
    found = hub_kernels.lazy_load_kernel
    hub_kernels.lazy_load_kernel = lambda name, *args, **kwargs: (
        importlib.import_module(local[name]) if name in local else found(name, *args, **kwargs))
    import transformers.integrations as integrations

    integrations.lazy_load_kernel = hub_kernels.lazy_load_kernel


def load_tested(config: dict[str, Any]):
    from transformers import AutoConfig, AutoModel

    use_local_kernels()

    settings = AutoConfig.from_pretrained(config["weights"], trust_remote_code=True)
    settings.vision_config.video_target_num_patches = PATCHES
    settings._attn_implementation = settings.llm_config._attn_implementation = "eager"  # FlashAttention2 isn't installed; its wrapper refuses sdpa
    settings.vision_config.use_flash_attn = False
    settings.video_pruning_rate = 0  # its own generate would drop 70% of the frames' tokens; the vLLM test run keeps all
    model = AutoModel.from_pretrained(config["weights"], config=settings, trust_remote_code=True, torch_dtype=DTYPE,
                                      attn_implementation="eager", device_map=device_map(settings.llm_config.num_hidden_layers))
    code = sys.modules[type(model.language_model).__module__]  # its code passes arguments transformers 5.20 renamed or dropped
    mask = code.create_causal_mask
    code.create_causal_mask = lambda *args, input_embeds=None, cache_position=None, **kwargs: mask(
        *args, inputs_embeds=input_embeds, **kwargs)
    code.NemotronHHybridDynamicCache.get_query_offset = lambda self, layer_idx=0: self.get_seq_length(layer_idx)
    build, layers = code.NemotronHHybridDynamicCache.__init__, settings.llm_config.num_hidden_layers

    def cache_on_layer_gpus(self, *args, **kwargs):  # its cache sits on one GPU; its layers don't
        build(self, *args, **kwargs)
        for states in (self.conv_states, self.ssm_states, self.key_cache, self.value_cache):
            for i in (states if isinstance(states, dict) else range(len(states))):
                states[i] = states[i].to(f"cuda:{1 + 3 * i // layers}")

    code.NemotronHHybridDynamicCache.__init__ = cache_on_layer_gpus
    processor = load_processor(config["weights"])
    return SimpleNamespace(model=model, processor=processor, tokenizer=processor.tokenizer, config=config)


def load_template(config: dict[str, Any]):
    return None, TESTED.tokenizer, None


def language_model(tested):
    return tested.model.language_model


def enable_checkpointing(language) -> None:
    def wrap(forward):
        def checkpointed(*args, **kwargs):
            if torch.is_grad_enabled() and not kwargs.get("use_cache"):
                return checkpoint(forward, *args, use_reentrant=False, **kwargs)
            return forward(*args, **kwargs)
        return checkpointed

    for layer in language.backbone.layers:
        layer.forward = wrap(layer.forward)


def train_layers(tested) -> None:
    tested.model.eval()
    tested.model.language_model.backbone.layers.train()
    tested.model.language_model.backbone.training = True


def encode(processor, row: dict[str, Any], frames: int, reply: str = "") -> dict[str, Any]:
    """A row's inputs as the test run builds them: frames, then the sound right after them, then the prompt."""
    from PIL import Image

    prompt = row["messages"][0]["content"].replace(PLACEHOLDERS, "", 1)
    text = processor.tokenizer.apply_chat_template([{"role": "user", "content": f"{VIDEO_TAG}\n{prompt}"}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
    text = text.replace(VIDEO_TAG, VIDEO_TAG + SOUND_TAG, 1) + reply
    video, metadata = read_frames(row["videos"][0], frames)
    return dict(processor(text=[text], videos=[Image.fromarray(f) for f in video], audio=[read_sound(row["audios"][0])],
                          videos_kwargs={"video_metadata": SimpleNamespace(**metadata)}, return_tensors="pt"))


def embeddings(model, ids: torch.Tensor, inputs: dict[str, Any]) -> torch.Tensor:
    """The prompt with its frames and sound in place, as its generate() builds it."""
    with torch.no_grad():
        embeds = model.language_model.get_input_embeddings()(ids).clone()
        video = model.extract_video_feature(inputs["pixel_values_videos"].to(device=model.device, dtype=DTYPE))
        extractor = model.sound_feature_extractor
        sound = extractor(inputs["sound_clips"], sampling_rate=extractor.sampling_rate, return_tensors="pt")
        device = model.sound_encoder.encoder.subsampling.linear.weight.device
        mask = sound.get("attention_mask")
        sound = model.extract_sound_feature(sound.input_features.to(dtype=DTYPE, device=device),
                                            mask.to(device) if mask is not None else None)
        size = embeds.shape[-1]
        embeds[ids == model.img_context_token_id] = video.reshape(-1, size).to(embeds.device, embeds.dtype)
        embeds[ids == model.sound_context_token_id] = sound.reshape(-1, size).to(embeds.device, embeds.dtype)
    return embeds


def loss(tested, row: dict[str, Any]) -> torch.Tensor:
    reply = row["messages"][1]["content"] + END_OF_TURN
    inputs = encode(tested.processor, row, tested.config["frames"], reply)
    language = tested.model.language_model
    first = language.get_input_embeddings().weight.device
    ids = inputs["input_ids"].to(first)
    count = len(tested.tokenizer(reply, add_special_tokens=False)["input_ids"])
    hidden = language.backbone(inputs_embeds=embeddings(tested.model, ids, inputs),
                               attention_mask=torch.ones_like(ids), use_cache=False)[0]
    shifted = hidden[:, -count - 1:-1]  # the positions that predict the reply: logits of the rest would not fit
    logits = language.lm_head(shifted.to(language.lm_head.weight.device)).float()
    return torch.nn.functional.cross_entropy(logits[0], ids[0, -count:].to(logits.device))


def answer(model, template, tokenizer, processor, example: dict[str, Any], generate_args: dict[str, Any]) -> str:
    inputs = encode(TESTED.processor, example, TESTED.config["frames"])
    first = TESTED.model.language_model.get_input_embeddings().weight.device
    ids = inputs["input_ids"].to(first)
    args = {k: v for k, v in generate_args.items() if k != "use_audio_in_video"}
    with torch.autocast("cuda", dtype=DTYPE):
        out = TESTED.model.generate(pixel_values_videos=inputs["pixel_values_videos"], sound_clips=inputs["sound_clips"],
                                    input_ids=ids, attention_mask=torch.ones_like(ids), **args)[0]
    if out.shape[0] >= ids.shape[1] and torch.equal(out[:ids.shape[1]].to(ids.device), ids[0]):
        out = out[ids.shape[1]:]  # the prompt, if generate kept it
    return tokenizer.decode(out, skip_special_tokens=True)


def merge(weights, adapter, dest) -> None:
    merge_lora(weights, adapter, dest, prefix="language_model.")
