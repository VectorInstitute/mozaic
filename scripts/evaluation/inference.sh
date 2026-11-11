#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --gres=gpu:1                # not needed for API models (Gemini); Qwen3-Omni with vLLM: see below
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# Run a model on a task and save its predictions. --config is required: a config in
# src/mozaic/evaluation/configs/; its first part picks the environment (avhubert/t1/ln_trainsmall: .venv_avhubert;
# gemini_3_8_flash/t1/zeroshot: .venv_gemini).
# A config given as a path (ablations/) names its environment in VENV:
#   VENV=ola sbatch scripts/evaluation/inference.sh --config ablations/modality/configs/ola_videoonly.yaml --task t1 --subset small

#   sbatch scripts/evaluation/inference.sh --config gemini_3_8_flash --task t1                  # the full dataset (qas/ → results/)
#   sbatch scripts/evaluation/inference.sh --config gemini_3_8_flash --task t4 --subset small   # MOZAIC-Small (results_small/)
#   sbatch scripts/evaluation/inference.sh --config qwen3_omni_vllm --task t1 --split val       # vLLM configs: test (default), val, train
#   sbatch scripts/evaluation/inference.sh --config avhubert/ln_trainsmall --task t1         # trained on Small, the full test split
#   sbatch scripts/evaluation/inference.sh --config gemini_3_8_flash --task t1 --limit 8        # a quick check first
# Every flag reaches run_inference as it is (--retry-failed, --overwrite, --env-file, ...).
# Qwen3-Omni and Nemotron with vLLM (their configs' tensor_parallel_size: 2):
#   sbatch --gres=gpu:2 --cpus-per-task=16 --mem=128G scripts/evaluation/inference.sh --config qwen3_omni_vllm --task t1

set -e
umask 002
CONFIG=$(echo " $* " | sed -n 's/.* --config[ =]\([^ ]*\) .*/\1/p')
if [ -z "$CONFIG" ]; then echo "pass --config <config>, e.g. --config avhubert/ln_trainsmall --task t1" >&2; exit 1; fi
ENV_NAME=${CONFIG%%/*}; case $ENV_NAME in gemini_*) ENV_NAME=gemini ;; esac   # one environment for both Gemini models
source .venv_${VENV:-$ENV_NAME}/bin/activate
# The weights are our local copies: no calls to the Hugging Face hub. spawn (vLLM only): its engine process
# starts clean (forked after the first clips were read, it hung).
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
# DataLoader workers' temp files (classifiers) on the node's own disk: on NFS their cleanup fails (.nfs... busy)
export TMPDIR=${SLURM_TMPDIR:-/tmp}

python -m mozaic.evaluation.run_inference "$@"
