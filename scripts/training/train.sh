#!/bin/bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=32          # 8 DataLoader workers per GPU (train.workers in the config)
#SBATCH --mem=164G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out

# Train a classifier on a task's train split, validate on its val split. --config is required:
# <model>/<task> (a config in src/mozaic/training/configs/), and <model> picks the environment .venv_<model>.
# A config given as a path (ablations/) names its environment in VENV: VENV=avhubert sbatch ... --config ablations/....yaml

#   sbatch scripts/training/train.sh --config avhubert/t1_ln                   # the full dataset (qas/ → training/)
#   sbatch scripts/training/train.sh --config avhubert/t1_ln --subset small    # MOZAIC-Small (qas_small/ → training_small/)
#   sbatch scripts/training/train.sh --config avhubert/t2_ln --limit 64     # smoke run: a few clips
#   sbatch scripts/training/train.sh --config avhubert/t1_ln --overwrite    # delete the run's files, train fresh
# Every flag reaches run_train as it is. Fewer GPUs: sbatch --gres=gpu:a40:1 --cpus-per-task=8 ... (GPUS follows --gres).
# W&B logs online. Without internet: WANDB_MODE=offline sbatch ..., then wandb sync <run folder>/wandb/offline-run-*

set -e
umask 002
CONFIG=$(echo " $* " | sed -n 's/.* --config[ =]\([^ ]*\) .*/\1/p')
if [ -z "$CONFIG" ]; then echo "pass --config <model>/<task>, e.g. --config avhubert/t1_ln" >&2; exit 1; fi
source .venv_${VENV:-${CONFIG%%/*}}/bin/activate
export OMP_NUM_THREADS=1
# DataLoader workers' temp files on the node's own disk: on NFS their cleanup fails (.nfs... busy) every epoch
export TMPDIR=${SLURM_TMPDIR:-/tmp}
GPUS=$(nvidia-smi -L | wc -l)

# --local-addr: the GPU processes meet on 127.0.0.1. By default torchrun gives them the node's hostname,
# which does not resolve here, and they wait for each other forever.
torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=$GPUS -m mozaic.training.run_train \
    "$@"
