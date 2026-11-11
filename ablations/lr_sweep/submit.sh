#!/bin/bash
# The learning rate of every model trained on MOZAIC-Small, seed 0 (1e-4: the runs in training_small/).
# Run with bash from the repo root, one stage at a time (ablations/variants.sh):
#   bash ablations/lr_sweep/submit.sh train
#   bash ablations/lr_sweep/submit.sh infer
#   bash ablations/lr_sweep/submit.sh eval
variants() { echo lr1e-6 lr3e-6 lr1e-5 lr3e-5 lr3e-4; }   # every model and tuning mode, the same grid
source ablations/variants.sh
