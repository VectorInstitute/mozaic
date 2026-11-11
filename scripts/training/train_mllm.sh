#!/bin/bash
#SBATCH --job-name=train_mllm
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=32          # 8 DataLoader workers per GPU (train.workers in the config)
#SBATCH --mem=164G                  # the merge loads the full Omni model on the CPU
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
# LoRA SFT of an MLLM with LLaMA-Factory, validated on val each epoch. --config is required:
# <model>/<task> (a config in src/mozaic/training/configs/).
#
#   sbatch scripts/training/train_mllm.sh --config qwen2_5_omni/t1_lora                  # the full dataset
#   sbatch scripts/training/train_mllm.sh --config qwen2_5_omni/t1_lora --subset small   # MOZAIC-Small
#   sbatch scripts/training/train_mllm.sh --config qwen2_5_omni/t1_lora --limit 16       # smoke run
#   sbatch scripts/training/train_mllm.sh --config qwen2_5_omni/t1_lora --overwrite      # train fresh
#
# Every flag goes to each step. A rerun resumes where the last one stopped.

set -e
umask 002
if ! echo " $* " | grep -q -- " --config"; then echo "pass --config <model>/<task>, e.g. --config qwen2_5_omni/t1_lora" >&2; exit 1; fi
source .venv_${VENV:-llamafactory}/bin/activate   # VENV=llamafactory_qwen3 for Qwen3-Omni
export OMP_NUM_THREADS=1
# DataLoader workers' temp files on the node's own disk: on NFS their cleanup fails (.nfs... busy) every epoch
export TMPDIR=${SLURM_TMPDIR:-/tmp}
GPUS=${PROCS:-$(nvidia-smi -L | wc -l)}   # PROCS=1 for a model split over all the GPUs (Nemotron)

# 1. the LLaMA-Factory dataset (<run>/data/) and the settings lock
python -m mozaic.training.mllms.dataset "$@"
# 2. LoRA SFT, validated each epoch; --local-addr: the GPU processes meet on 127.0.0.1 (see train.sh)
torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=$GPUS -m mozaic.training.mllms.train "$@"
# 3. the best epoch merged into the full model (<run>/final/), on the CPU with all its threads
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -m mozaic.training.mllms.merge "$@"
