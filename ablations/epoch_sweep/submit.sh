#!/bin/bash
# The epoch budget of every model trained on MOZAIC-Small, seed 0, each run from scratch (10 epochs, Qwen 3: the
# runs in training_small/). Run with bash from the repo root, one stage at a time (ablations/variants.sh):
#   bash ablations/epoch_sweep/submit.sh train
#   bash ablations/epoch_sweep/submit.sh infer
#   bash ablations/epoch_sweep/submit.sh eval
variants() {
  case $1 in
    qwen2_5_omni/*) echo ep1 ep2 ep5 ep6 ;; # LoRA: 3 epochs; 6 = 2x, as classifiers' 20
    *)              echo ep5 ep15 ep20 ;;   # classifiers: 10 epochs
  esac
}
job_time() {  # at least twice the measured time, rounded up to a day or 3 days (variants.sh)
  case $1_$2 in
    dfd_hr/t2_adapters_ep15|dfd_hr/t2_adapters_ep20|pe_av/t1_ln_ep20|pe_av/t3_*|qwen2_5_omni/t[34]_lora_ep[56]) echo 3-00:00:00 ;;
    *) echo 1-00:00:00 ;;
  esac
}
source ablations/variants.sh
