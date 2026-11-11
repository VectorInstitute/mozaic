# Ablations

One folder per ablation in the paper: its submit.sh (the Slurm jobs), collect.py (the results as CSV) and README.

| Folder | What changes | Models |
|---|---|---|
| frame_budget/ | frames per clip: 8, 16, 32, 64 | zero-shot MLLMs, T1 |
| modality/ | video only or audio only | zero-shot MLLMs, T1 |
| seeds/ | the seed: 0-4 | every model trained on MOZAIC-Small, every task |
| lr_sweep/ | the learning rate: 1e-6 to 3e-4 | the same |
| epoch_sweep/ | the epoch budget: 0.5x to 2x | the same |
| unseen_generators/ | one video generator left out of training | every model trained on MOZAIC-Small, T1 |

seeds/, lr_sweep/, epoch_sweep/ and unseen_generators/ share variants.sh (the jobs) and variants.py (the results).

## Cluster settings

Every submit.sh takes these from the environment; each is optional:

| Variable | Use |
|---|---|
| GPU_QOS | the QOS of the A100 jobs |
| CPU_PARTITION | the partition of the scoring jobs (evaluate.sh) |
| EXCLUDE | nodes no job may use (sbatch --exclude) |

    GPU_QOS=<qos> CPU_PARTITION=<cpu_partition> EXCLUDE=<nodes> bash ablations/seeds/submit.sh train
