#!/bin/bash
#SBATCH --job-name=annotate_t4
#SBATCH --partition=<cpu_partition>
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j.out

# T4: one step per run, on MOZAIC-Small (see t4_explanation/README.md). --annotate calls
# models through the API (Gemini explains each clip, Claude verifies and rewrites), so no GPU. The step and its flags go after the script:
#
#   sbatch scripts/annotation/annotate_t4.sh --select
#   sbatch scripts/annotation/annotate_t4.sh --annotate --golden-only --limit 8
#   sbatch scripts/annotation/annotate_t4.sh --annotate --golden-only
#   sbatch scripts/annotation/annotate_t4.sh --annotate --golden-only --retry-failed
#   sbatch scripts/annotation/annotate_t4.sh --build
#   sbatch scripts/annotation/annotate_t4.sh --annotate          # every clip, after the golden review
#
# --annotate is resumable: the same command continues where a job stopped.

set -e
umask 002
source .venv_gemini/bin/activate

python -m mozaic.annotation.run_annotation --task t4 --subset small "$@"
