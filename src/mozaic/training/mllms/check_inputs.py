"""Check that training's inputs equal evaluation's, for a few questions per split (CPU only).

    python -m mozaic.training.mllms.check_inputs --config qwen2_5_omni/t1_lora --subset small --limit 8

Compares the prompt token ids and every tensor: training's (qwen_omni_inputs) against evaluation's
clips, prompt and reader and the model's HF processor, as vLLM runs it.
"""

from __future__ import annotations

import logging

import torch

from mozaic.evaluation.models.mllms.qwen_omni_vllm.model import message
from mozaic.evaluation.run_inference import ask
from mozaic.training.mllms.dataset import example, load_questions
from mozaic.training.mllms.qwen_omni_inputs import EvalReaderPlugin, load_template, model_inputs
from mozaic.training.mllms.run_config import load, parse_args, prompts

log = logging.getLogger(__name__)

TENSORS = ("pixel_values_videos", "video_grid_thw", "video_second_per_grid", "input_features", "feature_attention_mask")


def evaluation_inputs(config: dict, processor, question: dict) -> tuple[list[int], dict]:
    from transformers import ProcessorMixin

    task = prompts(config["task"])
    clips, prompt, _ = ask(question, task, config, True, task.MODALITIES)
    msgs, sound_in_video = message(clips, prompt, config["frame_size"])
    text = ProcessorMixin.apply_chat_template(processor, msgs, add_generation_prompt=True, tokenize=False)
    audios, _, videos, kwargs = EvalReaderPlugin.process_mm_info(msgs, use_audio_in_video=sound_in_video,
                                                                 return_video_kwargs=True)
    # vLLM kept the frames at 224 (its predictions record 1,024 video tokens); called directly, the
    # processor would enlarge them to 336 and ignore size=, so the bounds go in as min/max_pixels
    pixels = config["frame_size"] ** 2
    video_args = {"fps": kwargs["fps"][0], "min_pixels": pixels, "max_pixels": pixels} if videos else {}  # t3's voice: {}
    out = processor(text=[text], audio=audios or None, videos=videos, return_tensors="pt", padding=True,
                    use_audio_in_video=sound_in_video, **video_args)
    return out["input_ids"][0].tolist(), out


def same(a, b) -> bool:
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    return a.shape == b.shape and torch.allclose(a.float(), b.float(), atol=1e-5)


def main() -> None:
    args = parse_args("Compare training's and evaluation's inputs for a few questions.")
    config, _ = load(args)
    template, tokenizer, processor = load_template(config)
    n, failed = args.limit or 8, 0
    for split in ("train", "val"):
        _, questions = load_questions(config["task"], config["questions"][split], None)
        for q in questions[::max(1, len(questions) // n)][:n]:  # spread over the file: both lengths, every class
            row = example(config["task"], q, config["frames"])
            ids, mm = model_inputs(template, tokenizer, processor, row)
            ref_ids, ref = evaluation_inputs(config, processor, q)
            differ = [k for k in TENSORS if (k in mm) != (k in ref) or (k in ref and not same(mm[k], ref[k]))]
            if ids != ref_ids:
                differ.insert(0, f"input_ids ({len(ids)} vs {len(ref_ids)} tokens)")
            failed += bool(differ)
            log.info("%s %s: %s", split, q["id"], "same" if not differ else "DIFFER " + ", ".join(differ))
    if failed:
        raise SystemExit(f"{failed} question(s) differ between training and evaluation")
    log.info("all the same")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main()
