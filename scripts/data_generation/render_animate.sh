#!/bin/bash
#SBATCH --job-name=render_animate
#SBATCH --output=logs/%x_%A_%a.out

# Wan2.2-Animate renders of one job list (generators/animate/make_jobs.py writes it), one shard per array task.
# Run from the repo root; the GPU, memory, time and array size go on the sbatch line (one 80 GB GPU per task).
# The renders go where the job list says (<data>/new_generated/renders). WA_CKPT is the weights folder, WA_OUT a
# scratch folder for the preprocessing materials (WA_OUT/proc/<render_key>, deleted after each render):
#   WA_CKPT=<weights>/wan2.2-animate-14b WA_OUT=<scratch> \
#       sbatch --array=0-1 --gres=gpu:1 --cpus-per-task=6 --mem=40G --time=12:00:00 \
#       scripts/data_generation/render_animate.sh <data>/new_generated/manifests/<build>/animate_jobs/jobs.jsonl
# More flags after the job list reach run.py unchanged (--keep-proc, --proc-root). A rerun skips finished renders.
# WA_UPSTREAM (default external-repos/wan2.2), WA_VENV (default .venv_animate) and WA_ONNX_CUDA11_LIBS (only for a
# CUDA 11 onnxruntime-gpu): generators/animate/README.md.

set -e
umask 002
JOBS=${1:?pass the job list, e.g. <data>/new_generated/manifests/<build>/animate_jobs/jobs.jsonl}
: "${WA_CKPT:?set WA_CKPT to the Wan2.2-Animate-14B weights folder}"
: "${WA_OUT:?set WA_OUT to a scratch folder for the preprocessing materials}"
# Absolute paths: preprocess_data.py and generate.py run inside the Wan2.2 clone.
export WA_CKPT=$(realpath -ms "$WA_CKPT") WA_OUT=$(realpath -ms "$WA_OUT")
export WA_UPSTREAM=$(realpath -ms "${WA_UPSTREAM:-external-repos/wan2.2}")
WA_VENV=$(realpath -ms "${WA_VENV:-.venv_animate}")
source "$WA_VENV/bin/activate"
# run.py gives the preprocessing and the render the venv's CUDA libraries, torch's own and the CUDA 11 ones
# (WA_ONNX_CUDA11_LIBS, when set), and keeps them offline (every weight is local).

python src/mozaic/data_generation/generators/animate/run.py --jobs "$JOBS" \
    --shard-index "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" "${@:2}"
