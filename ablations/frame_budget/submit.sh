#!/bin/bash
# T1 on MOZAIC-Small at 8 / 16 / 64 frames (32: the zero-shot runs) for Gemini 3.8 Flash, Nemotron, Ola and
# MiniCPM-o, each followed by its scoring. Run with bash from the repo root; answers a run already has are not asked again.
# MODELS and FRAMES pick other runs, e.g. MODELS=qwen3_omni_vllm FRAMES=64 bash ablations/frame_budget/submit.sh
# Cluster settings GPU_QOS, CPU_PARTITION, EXCLUDE: ablations/README.md.
set -e
site="${EXCLUDE:+--exclude=$EXCLUDE}"; qos="${GPU_QOS:+--qos=$GPU_QOS}"; cpu="${CPU_PARTITION:+--partition=$CPU_PARTITION}"
E=ablations/frame_budget
C=src/mozaic/evaluation/configs
mkdir -p $E/configs logs
for n in ${FRAMES:-8 16 64}; do
  for m in ${MODELS:-gemini_3_8_flash nemotron_omni_vllm ola minicpm_o_vllm}; do
    cfg=$E/configs/${m}_${n}f.yaml
    case $m in
      gemini_3_8_flash) sed -e "s|^frames: 32 |frames: $n |" -e "s|^experiment_name: zeroshot_32f_|experiment_name: zeroshot_${n}f_|" $C/$m/t1/zeroshot.yaml > $cfg
                        venv=gemini; res="--gres=gpu:a100:1"; time=20:00:00 ;;
      qwen3_omni_vllm|nemotron_omni_vllm) sed -e "s|^frames: 32 |frames: $n |" -e "s|^experiment_name: zeroshot_32f_224 |experiment_name: zeroshot_${n}f_224 |" $C/$m/t1/zeroshot.yaml > $cfg
                        venv=$m; res="--gres=gpu:a100:2 --cpus-per-task=16 --mem=128G"; time=8:00:00 ;;
      *)                sed -e "s|^frames: 32 |frames: $n |" -e "s|^experiment_name: zeroshot_32f_224 |experiment_name: zeroshot_${n}f_224 |" $C/$m/t1/zeroshot.yaml > $cfg
                        venv=$m; res="--gres=gpu:a100:1"; time=8:00:00 ;;
    esac
    grep -q "^frames: $n " $cfg && grep -q "^experiment_name: zeroshot_${n}f_" $cfg || { echo "sed missed $cfg" >&2; exit 1; }
    inf=$(VENV=$venv sbatch --parsable $res $qos $site --time=$time --job-name=fb_${m}_${n}f scripts/evaluation/inference.sh --config $cfg --task t1 --subset small)
    ev=$(sbatch --parsable --dependency=afterok:$inf $cpu --gres=none $site --job-name=fbeval_${m}_${n}f scripts/evaluation/evaluate.sh --config $cfg --task t1 --subset small)
    echo "$m ${n}f: inference $inf, evaluate $ev"
  done
done
