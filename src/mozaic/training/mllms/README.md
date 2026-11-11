# MLLMs: SFT with LLaMA-Factory

LoRA SFT of an MLLM with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (c683e59) on a
task's train split, validated on its val split every epoch, as the classifiers. The model learns
to answer the evaluation prompt, so it is tested as the zero-shot MLLMs. The test split is never
read here.

    mllms/
    ├── dataset.py            1. question files → LLaMA-Factory dataset (<run>/data/); settings lock
    ├── train.py              2. LoRA SFT (torchrun), validated each epoch
    ├── validation.py            the per-epoch callback: the model answers val, evaluation's code scores it
    ├── merge.py              3. the best epoch merged into the full model → <run>/final/
    ├── qwen_omni_inputs.py      Qwen2.5-Omni's inputs in LLaMA-Factory, read with evaluation's reader
    ├── check_inputs.py          tool: training's inputs == evaluation's, for a few questions
    ├── run_config.py            the steps' flags, config and run folder
    └── requirements.txt

    scripts/training/train_mllm.sh    runs 1 → 2 → 3
    configs/qwen2_5_omni/{t1,t2,t3,t4}.yaml

## Models

| model | task | input | target: the reply evaluation parses | select on |
|---|---|---|---|---|
| Qwen2.5-Omni 7B | t1 | video with its sound | {"answer": gt_answer} | macro.f1 |
| Qwen2.5-Omni 7B | t2 | video, no sound | {"answer": gt_answer} | macro.f1 |
| Qwen2.5-Omni 7B | t3 | two clips: two videos, no sound (face questions), or two sounds, no frames (voice questions) | {"answer": gt_answer} | macro.f1 |
| Qwen2.5-Omni 7B | t4 | video with its sound | {"video", "audio"}: gt_target.sections | text.both.rouge_l (no judge call per epoch) |

LoRA (rank 16) on every linear layer of the language model; the vision and audio towers and the
projector stay frozen. Weights: the copy evaluation uses. Question files and filters as
evaluation's. T4 exists on MOZAIC-Small only. T3 trains on MOZAIC-Small only and is tested on the
full test split.

Qwen3-Omni-30B (`configs/qwen3_omni/t4_lora.yaml`, `VENV=llamafactory_qwen3`) differs on purpose: LoRA on the attention
layers only, because "every linear layer" would wrap 18,432 expert layers (845M LoRA parameters against 13M); and
transformers 4.57.6, the first with the model, in its own environment.

## The same as evaluation

| | |
|---|---|
| input | 32 frames, evenly spaced, 224 x 224, each clip's own frame rate; the sound inside the video at 16 kHz (t2: none; t3: per clip its label, then its video or its sound alone). Read with evaluation's reader (qwen_omni_inputs.py): LLaMA-Factory's own would floor the frame indices and give every clip one fps, putting the sound next to the wrong frames |
| prompt | evaluation's (<task>_prompts.build_prompt) and its system prompt |
| reply | the JSON evaluation's parser reads; dataset.py refuses one it would not read back |
| validation | evaluation's generation settings (eval_config: greedy; T4: qwen2_5_omni_vllm/t4/lora_trainsmall, 8192 tokens), parser, prediction rows and metrics; replies from HF generate, not vLLM |

check_inputs.py compares the prompt token ids and every tensor with evaluation's on a few questions.

## Recipe

As the classifiers: AdamW, lr 1e-4, weight decay 0.01, no schedule, batch 32 (micro-batch 1 per
GPU), bf16, seed 0, 4 GPUs; but 3 epochs: LoRA SFT of an LLM takes a few (Skyra: 5). Loss:
cross-entropy on the reply tokens.

## Run

    sbatch scripts/training/train_mllm.sh --config qwen2_5_omni/t1_lora                  # the full dataset
    sbatch scripts/training/train_mllm.sh --config qwen2_5_omni/t2_lora --subset small   # MOZAIC-Small
    sbatch scripts/training/train_mllm.sh --config qwen2_5_omni/t4_lora --subset small   # T4: Small only
    sbatch --gres=gpu:a100:4 --cpus-per-task=24 scripts/training/train_mllm.sh --config qwen2_5_omni/t3_lora --subset small   # T3: Small only, A100s for its 64 frames
    srun ... python -m mozaic.training.mllms.check_inputs --config qwen2_5_omni/t1_lora --subset small
    python -m mozaic.training.mllms.merge --config qwen2_5_omni/t1_lora --epoch 2        # another epoch as final

A rerun resumes from the last checkpoint; a validated epoch is not validated again. --overwrite
deletes the run first.

## Output

    training[_small]/qwen2_5_omni/{experiment_name}/{task}/
    ├── train_settings.json      config, question files and counts; a rerun with other settings is refused
    ├── data/                    the LLaMA-Factory dataset
    ├── checkpoints/             checkpoint-<step>/ per epoch: the LoRA (155 MB) and optimizer
    ├── val_predictions/epoch_<n>.jsonl   val replies as evaluation's prediction rows
    ├── val_scores/epoch_<n>.json, val_scores.json (the best epoch and its checkpoint)
    ├── final/, final.json       the final model: the full Omni with the best epoch's LoRA merged (21 GB)
    └── wandb/, wandb_id.txt     training loss by step, val/* by val/epoch

LLaMA-Factory saves only the LoRA, and its own export keeps only the thinker; merge.py follows its
scripts/qwen_omni_merge.py, so final/ has the released model's layout. Test: evaluation's
configs/qwen2_5_omni_vllm/<task>/lora_train<full|small>.yaml, on the test split.

## Environment

    uv venv .venv_llamafactory --python 3.11
    uv pip install --python .venv_llamafactory/bin/python torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python .venv_llamafactory/bin/python -r src/mozaic/training/mllms/requirements.txt -e .
