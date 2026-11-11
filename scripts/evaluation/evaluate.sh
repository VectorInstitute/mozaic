#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=<cpu_partition>
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# Score saved predictions. No model, no GPU. The same flags as the inference run (--config required).

#   sbatch scripts/evaluation/evaluate.sh --config gemini_3_8_flash --task t1
#   sbatch scripts/evaluation/evaluate.sh --config avhubert/ln_trainsmall --task t1 --subset small
#   bash scripts/evaluation/evaluate.sh --config qwen3_omni_vllm --task t1 --split val      # no SLURM needed

set -e
umask 002
source .venv/bin/activate

python -m mozaic.evaluation.run_eval "$@"
