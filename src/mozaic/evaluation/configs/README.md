# configs

One yaml per model, task and run, read by run_inference and run_eval (--config NAME is configs/NAME.yaml, or a
path): `<model>/t<N>/<name>.yaml`, the model folder being its environment (.venv_<model>; Gemini's two folders
use .venv_gemini). The keys are listed in evaluation/README.md (Configs).

    <model>/t<N>/zeroshot.yaml               zero-shot, one per task (t1 to t3 on the test split; t4 on
                                             MOZAIC-Small's, --subset small, with a T4-length answer of 8192
                                             tokens and answer_choices off)
    <model>/t<N>/<mode>_train<full|small>.yaml   trained on MOZAIC (training/), tested on the test split: one per
                                             training run, the data trained on in its name (--subset picks the
                                             test questions); mode as in training/configs: ln, full, adapters, lora

    zero-shot models
    gemini_3_8_flash/      Gemini 3.8 Flash: Interactions API, each clip as one video (32 frames, with its sound
                           in t1), greedy, thinking low
    gemini_3_1_pro/        Gemini 3.1 Pro, the same
    qwen3_omni_vllm/       Qwen3-Omni with vLLM: 32 frames at 224, sound in the video, any split
    qwen2_5_omni_vllm/     Qwen2.5-Omni-7B with vLLM, the same input
    nemotron_omni_vllm/    Nemotron 3 Nano Omni with vLLM, the same input, reasoning off
    gemma4_vllm/           Gemma 4 12B with vLLM, the same frames and sound (frames enlarged to 384 by Gemma)
    minicpm_o_vllm/        MiniCPM-o 4.5 with vLLM, the same frames and sound, thinking off
    videollama2_av/        VideoLLaMA2.1-AV with transformers and its authors' code, the same frames and sound;
                           t1, t3 and t4 with only the clip's own sound (soundtrack: nopad), t2 (no sound) the
                           authors' default
    omnivinci/             OmniVinci with transformers and its own code from the hub, the same frames and sound
    uni_moe_2_omni/        Uni-MoE-2.0-Omni with transformers and its authors' code, the same frames and sound
    ola/                   Ola-7B with transformers and its authors' code, the same frames and sound
    vita_1_5/              VITA-1.5 with transformers and its authors' code, the same frames and sound

    trained (also in the folders above: t4/lora_trainsmall.yaml for the MLLMs fine-tuned on T4)
    qwen2_5_omni_vllm/     t1, t2: lora_trainfull, lora_trainsmall; t3, t4: lora_trainsmall (LoRA SFT)
    avhubert/              t1, t2: ln and full, each _trainfull and _trainsmall; t3: ln_trainsmall, full_trainsmall
    pe_av/                 the same
    havic/                 t1: ln and full, each _trainfull and _trainsmall
    dfd_hr/                t2: adapters_trainfull, adapters_trainsmall

**Naming a config.** `--config <model>/<run>` with `--task t<N>` reads `<model>/t<N>/<run>.yaml`; without
`--task`, every task that has `<run>.yaml` (t4 only with `--subset small`, its only test set). `<run>` left out
means `zeroshot`. A full name (`qwen3_omni_vllm/t1/zeroshot`) or a path also works. Files that differ only in
their task run as one config, so the model loads once (run_inference.load_configs):

    --config qwen3_omni_vllm --task t1                   qwen3_omni_vllm/t1/zeroshot.yaml
    --config qwen3_omni_vllm                             t1, t2, t3 (with --subset small: and t4)
    --config avhubert/ln_trainsmall --task t1            avhubert/t1/ln_trainsmall.yaml
    --config avhubert/ln_trainsmall                      t1, t2, t3: three trained runs
    --config qwen3_omni_vllm/t1/zeroshot                 the full name, as it is

A config names its question files by file name only (t1: t1_test.jsonl). The folder
comes from the run: qas/ by default, qas_small/ with --subset small, so one config
serves both. A file given as a path (with a "/") is read from the repo root instead.
T4 exists on MOZAIC-Small only.

Change experiment_name whenever a setting changes: results go to
results/{model}/{experiment_name}/{task}/, and a rerun with other settings in the same
folder is refused.
