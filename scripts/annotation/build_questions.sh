#!/bin/bash
#SBATCH --job-name=build_questions
#SBATCH --partition=<cpu_partition>
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Build a task's question file and samples. Edit these lines:
TASK=t1           # t1, t2, t3 (t4: annotate_t4.sh)
SIZES="40 200"    # samples to draw; t3: "40" (set by the counts in its config.yaml)
SUBSET=           # empty: the full dataset (qas/); "small": MOZAIC-Small -> qas_small/
                  # (results then go to results_small/)

# sbatch scripts/annotation/build_questions.sh

set -e
umask 002
source .venv/bin/activate

python -m mozaic.annotation.run_annotation --task $TASK ${SUBSET:+--subset $SUBSET}
for SIZE in $SIZES; do
    python -m mozaic.annotation.run_annotation --task $TASK --size $SIZE ${SUBSET:+--subset $SUBSET}
done
