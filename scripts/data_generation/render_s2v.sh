#!/bin/bash
#SBATCH --job-name=render_s2v
#SBATCH --output=logs/%x_%A_%a.out

# Wan2.2-S2V renders of one job list (generators/s2v/make_jobs.py writes it), one shard per array task. Run
# from the repo root; the GPU, memory, time and array size go on the sbatch line (one 80 GB GPU per task).
# WAN_CKPT is the weights folder, WAN_OUT the build's renders folder, where the mux looks:
#   WAN_CKPT=<weights>/wan2.2-s2v-14b WAN_OUT=<data>/new_generated/renders \
#       sbatch --array=0-3 --gres=gpu:1 --cpus-per-task=6 --mem=55G --time=8:00:00 \
#       scripts/data_generation/render_s2v.sh <data>/new_generated/manifests/<build>/s2v_jobs/jobs.jsonl
# More flags after the job list reach run.py unchanged. A rerun skips finished renders.
# WAN_UPSTREAM (default external-repos/wan2.2) and WAN_VENV (default .venv_s2v): generators/s2v/README.md.

set -e
umask 002
JOBS=${1:?pass the job list, e.g. <data>/new_generated/manifests/<build>/s2v_jobs/jobs.jsonl}
: "${WAN_CKPT:?set WAN_CKPT to the Wan2.2-S2V-14B weights folder}"
: "${WAN_OUT:?set WAN_OUT to the renders folder of the build, <data>/new_generated/renders}"
# Absolute paths: generate.py runs inside the Wan2.2 clone.
export WAN_CKPT=$(realpath -ms "$WAN_CKPT") WAN_OUT=$(realpath -ms "$WAN_OUT")
export WAN_UPSTREAM=$(realpath -ms "${WAN_UPSTREAM:-external-repos/wan2.2}")
source "$(realpath -ms "${WAN_VENV:-.venv_s2v}")/bin/activate"
# run.py keeps the render offline (every weight is local)

python src/mozaic/data_generation/generators/s2v/run.py --jobs "$JOBS" \
    --shard-index "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" "${@:2}"
