#!/bin/bash
#SBATCH --job-name=vocode
#SBATCH --cpus-per-task=8           # no GPU: the release's vocoded copies were made on CPU
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out

# The vocoded copy of every real voice of a build: its crop's sound through Seed-VC's mel step and BigVGAN
# (src/mozaic/data_generation/generators/seed_vc/README.md). Edit this line:
JOBS=/path/to/data_generation/new_generated/manifests/<build>/vocode_jobs.jsonl   # written by make_vocode_jobs

#   sbatch scripts/data_generation/vocode.sh
#   sbatch scripts/data_generation/vocode.sh --overwrite     # make the copies again
# Every flag reaches vocode.py as it is. A rerun skips the copies already made.

set -e
umask 002
source ${SEEDVC_VENV:-.venv_seed_vc}/bin/activate

# Absolute path: vocode.py moves into the Seed-VC folder before it reads the job list.
export SEEDVC_UPSTREAM="$(realpath "${SEEDVC_UPSTREAM:-external-repos/seed-vc}")"
python src/mozaic/data_generation/generators/seed_vc/vocode.py \
    --jobs "$(realpath "$JOBS")" \
    --device cpu \
    "$@"
