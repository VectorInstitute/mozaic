#!/bin/bash
# T1 with one video generator left out of training (train and val questions), every trained T1 model on
# MOZAIC-Small, seed 0; tested on the whole test set (the runs in training_small/: every generator seen).
# Run with bash from the repo root, one stage at a time (ablations/variants.sh):
#   bash ablations/unseen_generators/submit.sh train
#   bash ablations/unseen_generators/submit.sh infer
#   bash ablations/unseen_generators/submit.sh eval
variants() {
  case $1 in
    */t1_*) echo no-facefusion no-s2v no-animate ;;
  esac
}
source ablations/variants.sh
