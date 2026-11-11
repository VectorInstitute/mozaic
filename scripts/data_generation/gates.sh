#!/bin/bash
#SBATCH --job-name=gates
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# The keep/drop gates of one batch (src/mozaic/data_generation/gates/README.md), CPU only: for each build
# the face, picture and voice scores, then the rule, the withdrawal of the dropped pairs and the drop ledger.
# Run from the repo root after the mux of every build has finished. Each step runs in its own environment:
#   REPO_VENV   (default .venv)              this repo, with opencv-python: gate pairs, picture, rule, withdrawal
#   FF_VENV     (default .venv_facefusion)   insightface + onnxruntime: the face scores
#   VOICE_VENV  (default .venv_voice)  torch + speechbrain: the voice scores
#   SYNCNET_VENV (default .venv_syncnet) SyncNet's environment: the lip-sync scores; SyncNet's code in
#               SYNCNET_UPSTREAM (default external-repos/syncnet_python)
# ffmpeg must be on PATH. INSIGHTFACE_ROOT (default ~/.insightface) holds the buffalo_l face models;
# HF_HOME is where speechbrain downloads its speaker model.

#   sbatch scripts/data_generation/gates.sh --batch batch_02 \
#       --pairs A=<pairs>.A.txt --pairs B=<pairs>.B.txt \
#       --build bulk_v1_ff77:A --build bulk_v1_ff153:B --build bulk_v1_s2v77:A \
#       --build bulk_v1_wan77:A --build bulk_v1_s2v153:B
#   ... --dry-run       # print every step, run nothing
# Every flag reaches run_gates as it is.

set -e
umask 002
FF_VENV=$(realpath -ms "${FF_VENV:-.venv_facefusion}")
VOICE_VENV=$(realpath -ms "${VOICE_VENV:-.venv_voice}")
SYNCNET_VENV=$(realpath -ms "${SYNCNET_VENV:-.venv_syncnet}")
export SYNCNET_UPSTREAM=$(realpath -ms "${SYNCNET_UPSTREAM:-external-repos/syncnet_python}")
source "${REPO_VENV:-.venv}/bin/activate"

python -m mozaic.data_generation.gates.run_gates \
    --face-python "$FF_VENV/bin/python" --voice-python "$VOICE_VENV/bin/python" \
    --syncnet-python "$SYNCNET_VENV/bin/python" "$@"
