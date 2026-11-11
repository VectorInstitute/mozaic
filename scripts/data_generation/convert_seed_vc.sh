#!/bin/bash
#SBATCH --job-name=convert_seed_vc
#SBATCH --gres=gpu:1                # one GPU per shard (an A40 is enough)
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-3                 # one task per shard of the job list
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out

# Seed-VC voice conversion: each array task converts its share of a build's job list
# (src/mozaic/data_generation/generators/seed_vc/README.md). Edit these lines:
JOBS=/path/to/data_generation/new_generated/manifests/<build>/seedvc_jobs/jobs.jsonl   # written by make_jobs
OUT=/path/to/data_generation/new_generated/conversions                                # the mux reads the voices here

#   sbatch scripts/data_generation/convert_seed_vc.sh
#   sbatch scripts/data_generation/convert_seed_vc.sh --dry-run     # print the jobs left, convert nothing
# Every flag reaches run.py as it is. A rerun skips the voices already made (<job_id>.done).

set -e
umask 002
source ${SEEDVC_VENV:-.venv_seed_vc}/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader || true   # the same GPU model gives the same bytes

# Absolute paths: run.py moves into the Seed-VC folder before it loads the model.
python src/mozaic/data_generation/generators/seed_vc/run.py \
    --jobs "$(realpath "$JOBS")" \
    --out-dir "$(realpath -m "$OUT")" \
    --upstream "$(realpath "${SEEDVC_UPSTREAM:-external-repos/seed-vc}")" \
    --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    "$@"
