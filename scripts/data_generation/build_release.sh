#!/bin/bash
#SBATCH --job-name=build_release
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# One step of the release build, on CPUs only (src/mozaic/data_generation/release/README.md).
# Run from the repo root, one step at a time, in this order:
#
#   sbatch scripts/data_generation/build_release.sh plan --exclude-inconsistent   # read-only
#   sbatch scripts/data_generation/build_release.sh copy --apply
#   sbatch scripts/data_generation/build_release.sh fix --apply
#   VENV=<env with torch + speechbrain> sbatch scripts/data_generation/build_release.sh voice
#   sbatch scripts/data_generation/build_release.sh drop --apply      # after writing _mozaic_build/drop/pairs.json
#   sbatch scripts/data_generation/build_release.sh manifests
#   sbatch scripts/data_generation/build_release.sh verify            # read-only
#
# Add the partition/account lines the cluster needs. ffmpeg 4.4 must be on PATH in the job:
# the build refuses any other version. Without --apply, copy, fix and drop only count what they would do.
# voice loads the speaker model of gates/voice_check.py from the Hugging Face Hub; without internet,
# fill the cache first and set HF_HUB_OFFLINE=1.

set -e
umask 002
source "${VENV:-.venv}/bin/activate"
if [ "$1" = voice ]; then  # the speaker model: one thread per worker process
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
fi
python -m mozaic.data_generation.release.build_mozaic "$@" --workers "${SLURM_CPUS_PER_TASK:-$(nproc)}"
