# scripts

SLURM launchers for the cluster: each one wraps one python command from the repo
README, so a job runs exactly the command run by hand.

    scripts/
    ├── annotation/
    │   ├── build_questions.sh    T1-T3 question files and samples        core .venv
    │   └── annotate_t4.sh        T4, one step per run (Small only)       .venv_gemini
    ├── evaluation/
    │   ├── inference.sh          model → predictions (also vLLM)         .venv_<model>
    │   ├── evaluate.sh           predictions → scores                    core .venv
    │   ├── judge.sh              T4: predictions → judge                 .venv_judge
    │   └── t4_zeroshot.sh        T4: inference.sh for every open model   (bash, not sbatch)
    ├── data_generation/          examples for the GPU and long steps of data generation
    │   ├── render_facefusion.sh  render a job list (1 GPU per array task)   .venv_facefusion
    │   ├── render_s2v.sh         render a job list (1 GPU per array task)   .venv_s2v
    │   ├── render_animate.sh     render a job list (1 GPU per array task)   .venv_animate
    │   ├── convert_seed_vc.sh    converted voices (1 GPU per array task)    .venv_seed_vc
    │   ├── vocode.sh             vocoded copies of real voices (CPU)        .venv_seed_vc
    │   ├── gates.sh              the checks and keep/drop of one batch      core, .venv_facefusion, .venv_voice
    │   └── build_release.sh      one step of the release build              core (voice step: .venv_voice)
    └── training/
        └── train.sh              train + val → checkpoint (4 GPUs)       .venv_<model>

## Before the first run

- Run from the repo root, and create the log folder: `mkdir -p logs` (logs/<job>_<id>.out).
- Set the partitions in the #SBATCH lines for the cluster (`sinfo -s` lists them).
- Make the environments the script uses (repo README, Setup), and put the API keys in .env.
- API models (Gemini) need no GPU: remove the --gres line in inference.sh.
- On MOZAIC-Small, draw it once first: `python -m mozaic.common.subsets`.

## Two ways to set a run

- **The lines at the top of the script** (build_questions.sh): which task, sizes and subset. Edit them before `sbatch`.
- **Flags after the script name** (inference, evaluate, train): passed to python unchanged.

      sbatch scripts/evaluation/inference.sh --config gemini_3_8_flash --task t1 --limit 8

| Script | Top lines | Extra flags |
|---|---|---|
| build_questions.sh | TASK (t1, t2, t3), SIZES, SUBSET | none |
| annotate_t4.sh | none: always `--task t4 --subset small` | the step: `--select`, `--annotate`, `--build`, ... |
| inference.sh | none | `--config` (required), `--task`, `--subset small`, `--split` (vLLM configs only), `--limit N`, `--retry-failed`, `--overwrite`, `--env-file` |
| evaluate.sh | none | `--config` (required), `--task`, `--subset small`, `--split` (vLLM configs only) |
| judge.sh | none: always `--task t4` | `--config` (required), `--subset small`, `--split`, `--limit N`, `--retry-failed`, `--judge-model`, `--no-embedder` |
| t4_zeroshot.sh | none: `<model>/t4/zeroshot`, `--task t4 --subset small` | model names (default: every open model); `LIMIT=N` for `--limit N` |
| train.sh | none | `--config <model>/<task>` (required), `--subset small`, `--limit N`, `--overwrite` |
| train_mllm.sh | none | `--config <model>/<task>` (required), `--subset small`, `--limit N`, `--overwrite` |

No subset is the full dataset (qas/, results/); small is MOZAIC-Small (qas_small/,
results_small/). Use the same subset for building, inference and scoring. A model trained on
MOZAIC: the subset picks the test questions only; its config names the data it was trained on.

## build_questions.sh

The split files of one task, then a balanced sample of its test split for each size in SIZES.

    TASK=t1
    SIZES="40 200"     # t3: "40" (set by the counts in its config.yaml)
    SUBSET=

    sbatch scripts/annotation/build_questions.sh

## annotate_t4.sh

T4 is built on MOZAIC-Small in steps (src/mozaic/annotation/t4_explanation/README.md).
Each job runs one step, named after the script:

    sbatch scripts/annotation/annotate_t4.sh --select                                  # 1. which clips; no model
    sbatch scripts/annotation/annotate_t4.sh --annotate --golden-only --limit 8        # 2. Gemini explains, Claude verifies and rewrites: a few golden clips
    sbatch scripts/annotation/annotate_t4.sh --annotate --golden-only                  #    all golden clips
    sbatch scripts/annotation/annotate_t4.sh --annotate --golden-only --retry-failed   #    ask again the failed ones
    sbatch scripts/annotation/annotate_t4.sh --build                                   # 3. final files; no calls
    sbatch scripts/annotation/annotate_t4.sh --annotate                                # every clip, after the golden review
    sbatch scripts/annotation/annotate_t4.sh --build

- --annotate is resumable: if a job hits its time limit, submit the same command again.
- --annotate needs both keys in the .env: GEMINI_API_KEY (Gemini) and ANTHROPIC_API_KEY (Claude: verifier and rewriter).
- After each --annotate, qas_small/t4_explanation/build/stats.json says what failed and what a retry fixed.

## inference.sh

Runs a config's model on its tasks (or --task) and saves the predictions to
results/{model}/{experiment_name}/{task}/predictions.jsonl (results_small/ with --subset small).

    sbatch scripts/evaluation/inference.sh --config gemini_3_8_flash --task t1                  # the whole task
    sbatch scripts/evaluation/inference.sh --config gemini_3_8_flash --task t1 --limit 8        # a quick check first
    sbatch scripts/evaluation/inference.sh --config gemini_3_8_flash --task t1 --retry-failed   # ask again unreadable or cut-off replies

A rerun continues where the last one stopped. `--overwrite` deletes the task's
predictions, settings and scores and starts fresh.

### Qwen-Omni with vLLM

The same script, with --config qwen3_omni_vllm/t<N>/zeroshot or qwen2_5_omni_vllm/t<N>/zeroshot (one config
per task) and --split test (default; val, train) (see models/mllms/qwen_omni_vllm/README.md; ffmpeg must be on
the PATH). The predictions go to results/{model}/{experiment_name}_{split}/{task}/; score them with
evaluate.sh, the same flags. Qwen3-Omni runs on 2 GPUs (its tensor_parallel_size), so its job asks for more:

    sbatch --gres=gpu:2 --cpus-per-task=16 --mem=128G scripts/evaluation/inference.sh --config qwen3_omni_vllm --task t1 --limit 10
    sbatch scripts/evaluation/inference.sh --config qwen2_5_omni_vllm --task t1             # Qwen2.5-Omni: 1 GPU

Nemotron 3 Nano Omni the same way, with MODEL=nemotron_omni_vllm: 2 GPUs as Qwen3-Omni (see
models/mllms/nemotron_omni_vllm/README.md). Gemma 4 with MODEL=gemma4_vllm: 1 GPU (see
models/mllms/gemma4_vllm/README.md). MiniCPM-o 4.5 with MODEL=minicpm_o_vllm: 1 GPU (see
models/mllms/minicpm_o_vllm/README.md). VideoLLaMA2.1-AV with MODEL=videollama2_av: 1 GPU (see
models/mllms/videollama2_av/README.md). OmniVinci with MODEL=omnivinci: 1 GPU (see
models/mllms/omnivinci/README.md). Uni-MoE-2.0-Omni with MODEL=uni_moe_2_omni: 1 A100-80GB (see
models/mllms/uni_moe_2_omni/README.md). Ola-7B with MODEL=ola: 1 GPU (see models/mllms/ola/README.md).
VITA-1.5 with MODEL=vita_1_5: 1 GPU (see models/mllms/vita_1_5/README.md).

### Models trained on MOZAIC

One config per training run, <model>/<task>_<training>_train<full|small> (evaluation/README.md, Configs);
--subset picks only the test questions:

    sbatch scripts/evaluation/inference.sh --config avhubert/ln_trainsmall --task t1 --subset small   # Small's test split
    sbatch scripts/evaluation/inference.sh --config avhubert/ln_trainsmall --task t1                  # the full test split
    sbatch scripts/evaluation/inference.sh --config qwen2_5_omni_vllm/lora_trainfull --task t1

## evaluate.sh

Scores the saved predictions: scores.json next to predictions.jsonl. No model, no GPU;
it can also run without SLURM (bash). The same flags as the inference run:

    sbatch scripts/evaluation/evaluate.sh --config gemini_3_8_flash --task t1
    sbatch scripts/evaluation/evaluate.sh --config avhubert/ln_trainsmall --task t1 --subset small

A full test run of t1 to t3 also gives MOZAIC-Small's scores, in results_small/, from the same answers
(src/mozaic/evaluation/README.md, Flags).

## judge.sh

T4 only: the judge (gpt-5.4-mini through an OpenAI-compatible endpoint) scores each saved explanation, and the embedder
(all-MiniLM-L6-v2, local) gives the cosine, in judge.jsonl. Not part of evaluate.sh: each clip is a paid
call. evaluate.sh puts the judge in scores.json; run before the judge, it says "judge": "not run yet", so
run it again after. The same flags as the inference run:

    sbatch scripts/evaluation/judge.sh --config gemini_3_8_flash --task t4 --subset small --limit 100   # the cost per clip first
    sbatch scripts/evaluation/judge.sh --config gemini_3_8_flash --task t4 --subset small               # the rest (resumes)

## A full run

    mkdir -p logs
    sbatch scripts/annotation/build_questions.sh       # TASK=t1
    sbatch scripts/evaluation/inference.sh --config gemini_3_8_flash --task t1
    sbatch scripts/evaluation/evaluate.sh --config gemini_3_8_flash --task t1     # after inference has finished
    # T4: inference, the judge, then the scores
    sbatch scripts/evaluation/inference.sh --config gemini_3_8_flash --task t4 --subset small
    sbatch scripts/evaluation/judge.sh --config gemini_3_8_flash --task t4 --subset small
    sbatch scripts/evaluation/evaluate.sh --config gemini_3_8_flash --task t4 --subset small

Every script stops at the first error (set -e) and sets umask 002, so the files it
writes stay group-writable.
