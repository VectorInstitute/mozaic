#!/bin/bash
# 8 runs: 4 models x {video only, audio only}, T1 on MOZAIC-Small, each followed by its scoring. bash, from the repo root.
# MODELS picks other models, e.g. MODELS=qwen3_omni_vllm bash ablations/modality/submit.sh
# Each config is written here from the model's zero-shot config (configs/, git-ignored): modalities under t1 and
# its own experiment_name.
# Cluster settings GPU_QOS, CPU_PARTITION, EXCLUDE: ablations/README.md.
set -e
site="${EXCLUDE:+--exclude=$EXCLUDE}"; qos="${GPU_QOS:+--qos=$GPU_QOS}"; cpu="${CPU_PARTITION:+--partition=$CPU_PARTITION}"
E=ablations/modality
C=src/mozaic/evaluation/configs
mkdir -p $E/configs logs
for mod in video audio; do
  for m in ${MODELS:-nemotron_omni_vllm ola minicpm_o_vllm gemini_3_8_flash}; do
    cfg=$E/configs/${m}_${mod}only.yaml
    case $mod in
      video) line="    modalities: [video]             # the frames only, no sound (run_inference.clip_modalities)" ;;
      audio) line="    modalities: [audio]             # the sound only, no frames (run_inference.clip_modalities)" ;;
    esac
    case $m in   # the experiment gets _<mod>only; modalities at the end of t1 (the config's only task)
      gemini_3_8_flash) sed -e "s|^experiment_name: zeroshot_32f_3_8_flash_test |experiment_name: zeroshot_32f_3_8_flash_${mod}only_test |" $C/$m/t1/zeroshot.yaml > $cfg; echo "$line" >> $cfg ;;
      *)                sed -e "s|^experiment_name: zeroshot_32f_224 |experiment_name: zeroshot_32f_224_${mod}only |" $C/$m/t1/zeroshot.yaml > $cfg; echo "$line" >> $cfg ;;
    esac
    grep -q "^experiment_name: zeroshot_32f_.*_${mod}only" $cfg && grep -q "^    modalities: \[$mod\]" $cfg || { echo "sed missed $cfg" >&2; exit 1; }
    case $m in
      gemini_3_8_flash)   venv=gemini; res="--gres=gpu:a100:1"; time=12:00:00 ;;
      qwen3_omni_vllm|nemotron_omni_vllm) venv=$m; res="--gres=gpu:a100:2 --cpus-per-task=16 --mem=128G"; time=8:00:00 ;;
      *)                  venv=$m; res="--gres=gpu:a100:1"; time=8:00:00 ;;
    esac
    inf=$(VENV=$venv sbatch --parsable $res $qos $site --time=$time --job-name=mod_${m}_${mod} scripts/evaluation/inference.sh --config $cfg --task t1 --subset small)
    ev=$(sbatch --parsable --dependency=afterok:$inf $cpu --gres=none $site --job-name=modeval_${m}_${mod} scripts/evaluation/evaluate.sh --config $cfg --task t1 --subset small)
    echo "$m ${mod} only: inference $inf, evaluate $ev"
  done
done
