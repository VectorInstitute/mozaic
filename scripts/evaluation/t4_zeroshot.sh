#!/bin/bash
# T4 zero-shot answers for the open models: submits inference.sh once per model, with its <model>/t4_zeroshot
# config on MOZAIC-Small's test split, and the GPUs that model needs. Run with bash (not sbatch), from the repo
# root. Judge and score each model afterwards (judge.sh, then evaluate.sh, the same --config).
#
#   bash scripts/evaluation/t4_zeroshot.sh                         # every open model, all 3,334 test clips
#   LIMIT=8 bash scripts/evaluation/t4_zeroshot.sh                 # a quick check: the first 8 clips
#   bash scripts/evaluation/t4_zeroshot.sh gemma4_vllm ola         # some models
#
# Account, QOS and partition: sbatch reads them from SBATCH_ACCOUNT, SBATCH_QOS and SBATCH_PARTITION, e.g.
#   export SBATCH_ACCOUNT=<account> SBATCH_QOS=<qos> SBATCH_PARTITION=<gpu_partition>
# Each model needs its .venv_<model> (its README, Setup). Rerunning continues where a run stopped.
set -e
MODELS=${*:-"qwen3_omni_vllm nemotron_omni_vllm qwen2_5_omni_vllm gemma4_vllm minicpm_o_vllm omnivinci videollama2_av vita_1_5 ola uni_moe_2_omni"}
EXTRA=${LIMIT:+--limit $LIMIT}
mkdir -p logs
for m in $MODELS; do
  case $m in
    qwen3_omni_vllm|nemotron_omni_vllm) gpu="--gres=gpu:2 --cpus-per-task=16 --mem=128G" ;;  # tensor_parallel_size: 2
    *)                                  gpu="--gres=gpu:1 --cpus-per-task=8 --mem=64G" ;;
  esac
  job=$(sbatch --parsable $gpu --job-name=t4_$m --time=8:00:00 \
        scripts/evaluation/inference.sh --config $m/t4/zeroshot --subset small $EXTRA)
  echo "$m: job $job (logs/t4_${m}_$job.out)"
done
