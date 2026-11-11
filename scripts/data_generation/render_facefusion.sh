#!/bin/bash
#SBATCH --job-name=render_facefusion
#SBATCH --output=logs/%x_%A_%a.out

# FaceFusion face swaps of one job list (generators/facefusion/make_jobs.py writes it), one shard per array
# task. Run from the repo root; the GPU, memory, time and array size go on the sbatch line (one GPU per task,
# 24 GB is enough). FF_OUT is the build's renders folder, where the mux looks:
#   FF_OUT=<data>/new_generated/renders sbatch --array=0-3 --gres=gpu:1 --cpus-per-task=6 --mem=24G --time=4:00:00 \
#       scripts/data_generation/render_facefusion.sh <data>/new_generated/manifests/<build>/facefusion_jobs/jobs.jsonl
# More flags after the job list reach run.py unchanged. A rerun skips finished and refused swaps.
# FF_UPSTREAM (default external-repos/facefusion) and FF_VENV (default .venv_facefusion): generators/facefusion/README.md.

set -e
umask 002
JOBS=${1:?pass the job list, e.g. <data>/new_generated/manifests/<build>/facefusion_jobs/jobs.jsonl}
: "${FF_OUT:?set FF_OUT to the renders folder of the build, <data>/new_generated/renders}"
# Absolute paths: FaceFusion runs inside its clone.
export FF_OUT=$(realpath -ms "$FF_OUT")
export FF_UPSTREAM=$(realpath -ms "${FF_UPSTREAM:-external-repos/facefusion}")
FF_VENV=$(realpath -ms "${FF_VENV:-.venv_facefusion}")
source "$FF_VENV/bin/activate"
# run.py puts the venv's CUDA 12 and cuDNN 9 libraries on LD_LIBRARY_PATH for FaceFusion, and stops if the
# GPU still cannot be used.

python src/mozaic/data_generation/generators/facefusion/run.py --jobs "$JOBS" \
    --shard-index "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" "${@:2}"
