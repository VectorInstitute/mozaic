#!/bin/bash
#SBATCH --job-name=judge
#SBATCH --gres=gpu:1                # the embedder (all-MiniLM-L6-v2); a CPU job works too, slower
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# T4 only: judge saved predictions (judge/README.md) → judge.jsonl. evaluate.sh then puts the judge in
# scores.json (run before the judge, it says "not run yet": run it again). The same --config, --subset,
# --split as the inference run (--config required).

#   sbatch scripts/evaluation/judge.sh --config gemini_3_8_flash --task t4 --subset small --limit 100   # the cost per clip first
#   sbatch scripts/evaluation/judge.sh --config qwen2_5_omni_vllm/lora_trainsmall --task t4 --subset small
#   sbatch scripts/evaluation/judge.sh --config gemini_3_8_flash --task t4 --subset small --retry-failed
# Every flag reaches run_judge as it is (--limit, --retry-failed, --judge-model, --no-embedder, ...).

set -e
umask 002
if ! echo " $* " | grep -q -- " --config"; then echo "pass --config <config>, e.g. --config gemini_3_8_flash --task t4" >&2; exit 1; fi
source .venv_judge/bin/activate   # openai and sentence-transformers (judge/README.md, Run)

python -m mozaic.evaluation.judge.run_judge --task t4 "$@"
