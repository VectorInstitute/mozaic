#!/bin/bash
# Seeds 1-4 of every model trained on MOZAIC-Small, at the protocol's lr (seed 0: the runs in training_small/).
# Run with bash from the repo root, one stage at a time (ablations/variants.sh):
#   bash ablations/seeds/submit.sh train
#   bash ablations/seeds/submit.sh infer
#   bash ablations/seeds/submit.sh eval
variants() { echo seed1 seed2 seed3 seed4; }
source ablations/variants.sh
