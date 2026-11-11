# Shared by seeds/, lr_sweep/, epoch_sweep/ and unseen_generators/: sourced by their submit.sh, which defines variants <training config>.
# One run per variant of each model in TRAINED (training/configs), trained on MOZAIC-Small, tested as its _trainsmall eval
# config. A variant is seed<N> (train.seed), lr<value> (train.lr), ep<N> (train.epochs) or no-<generator> (that video
# generator's clips left out of the train and val questions; the test set is unchanged); its run and results are
# the original's experiment_name + _<variant>. Stages, each submitted once the previous one has finished (no job waits on another):
#   configs the generated configs only, nothing submitted (a dry run)
#   train   training jobs on the GPUs seed 0 trained on (on_a100; A100 nodes have 24 CPUs); finished runs are skipped
#   infer   test predictions of every finished run, one A100 each (T1-T3: the full test split; T4: Small's)
#   eval    scores of every run with predictions (CPU); T1-T3 also score MOZAIC-Small from the full answers
# A job queued or running under the same name is not submitted again. Cluster settings GPU_QOS, CPU_PARTITION,
# EXCLUDE: ablations/README.md.
set -e
STAGE=${1:?pass configs, train, infer or eval}
E=$(dirname "$0")
TC=src/mozaic/training/configs
EC=src/mozaic/evaluation/configs
R=$(.venv/bin/python -c "from mozaic.common.paths import data_root; print(data_root())")
mkdir -p $E/configs logs

key() { sed -n "s/^$1: *\([^ #]*\).*/\1/p" "$2"; }   # a top-level key's value
# A training job's limit: at least twice its measured time (epochs x slowest epoch), rounded up to a day or 3 days
# (a longer limit fits fewer partitions). PE-AV T3: about 1.6 h an epoch on 4 A40s, 0.8 h on 4 A100s. A caller
# may define its own job_time <training config> <variant>.
declare -F job_time > /dev/null || job_time() { case $1 in pe_av/t3_*) echo 3-00:00:00 ;; *) echo 1-00:00:00 ;; esac; }
queued() { squeue -h -u $USER -n "$1" -o %i | grep -q .; }
site="${EXCLUDE:+--exclude=$EXCLUDE}"                 # added to every job
# A100s for PE-AV Full (out of memory on an A40: its weights and optimizer state) and Qwen2.5-Omni T3 (two clips
# of 32 frames per question); A40s for the rest
on_a100() { case $1 in pe_av/t1_full|pe_av/t3_full|qwen2_5_omni/t3_lora) return 0 ;; *) return 1 ;; esac; }

# The models in the ablations: a fixed list, so a model trained later (e.g. the other MLLMs' T4 LoRA) joins only
# when added here
TRAINED="avhubert/t1_ln avhubert/t1_full avhubert/t2_ln avhubert/t2_full avhubert/t3_ln avhubert/t3_full dfd_hr/t2_adapters
  havic/t1_ln havic/t1_full pe_av/t1_ln pe_av/t1_full pe_av/t2_ln pe_av/t2_full pe_av/t3_ln pe_av/t3_full
  qwen2_5_omni/t1_lora qwen2_5_omni/t2_lora qwen2_5_omni/t3_lora qwen2_5_omni/t4_lora"

for name in $TRAINED; do
  train=$TC/$name.yaml
  eval=$(grep -l "^  config: $name\( \|$\)" $EC/*/*/*_trainsmall.yaml)
  model=$(key model $train); task=$(key task $train); emodel=$(key model $eval)
  sub=$([ $task = t4 ] && echo "--subset small" || true)
  for v in $(variants $name); do
    tcfg=$E/configs/${name/\//_}_$v.yaml
    ecfg=$E/configs/${name/\//_}_${v}_eval.yaml
    job=${E##*/}_${name/\//_}_$v
    lines=2                                             # the lines a generated training config changes
    case $v in
      seed*) edit="s|^  seed: 0\b|  seed: ${v#seed}|" ;;
      lr*)   edit="s|^  lr: 1.0e-4\b|  lr: $(echo ${v#lr} | sed 's/e/.0e/')|" ;;   # 3e-5 -> 3.0e-5: YAML reads it as a number
      ep*)   edit="s|^  epochs: [0-9]*\b|  epochs: ${v#ep}|" ;;
      no-*)  keep=$(printf '%s\n' real facefusion s2v animate | grep -vx ${v#no-} | paste -sd, | sed 's/,/, /g')
             edit="s|^\(      arms: \[vocoded\].*\)$|\1\n      video_methods: [$keep]   # ${v#no-} held out|"
             lines=3 ;;                                 # after arms: in train and in val, and experiment_name
    esac
    sed -e "$edit" -e "s|^experiment_name: \([^ ]*\)|experiment_name: \1_$v|" $train > $tcfg
    sed -e "s|^  config: $name\b|  config: $tcfg|" -e "s|^experiment_name: \([^ ]*\)|experiment_name: \1_$v|" $eval > $ecfg
    [ "$(diff $train $tcfg | grep -c '^>')" = $lines ] || { echo "sed missed a line in $tcfg" >&2; exit 1; }
    [ "$(diff $eval $ecfg | grep -c '^>')" = 2 ] || { echo "sed missed a line in $ecfg" >&2; exit 1; }

    epochs=$(sed -n 's/^  epochs: *\([0-9]*\).*/\1/p' $tcfg)   # the variant's own
    time=$(job_time $name $v)
    run=$R/training_small/$model/$(key experiment_name $tcfg)/$task
    results=$R/results${sub:+_small}/$emodel/$(key experiment_name $ecfg)/$task
    trained() { [ -d $run/final ] || [ "$(cat $run/history.jsonl 2>/dev/null | wc -l)" = "$epochs" ]; }
    if [ $STAGE = configs ]; then echo "$job: $tcfg -> $run"; continue; fi
    if queued $job; then echo "$job: queued or running"; continue; fi
    case $STAGE in
      train)
        if trained; then echo "$job: trained"; continue; fi
        gpus=$(on_a100 $name && echo "--gres=gpu:a100:4 --cpus-per-task=24 ${GPU_QOS:+--qos=$GPU_QOS}" || true)  # else the script's a40:4
        script=$([ $model = qwen2_5_omni ] && echo train_mllm.sh || echo train.sh)
        id=$(VENV=$model sbatch --parsable --job-name=$job $site --time=$time $gpus \
             scripts/training/$script --config $tcfg --subset small) ;;
      infer)
        if ! trained; then echo "$job: not trained yet"; continue; fi
        if [ -f $results/scores.json ]; then echo "$job: scored"; continue; fi
        id=$(VENV=$emodel sbatch --parsable --job-name=$job --gres=gpu:a100:1 ${GPU_QOS:+--qos=$GPU_QOS} $site --time=1-00:00:00 \
             scripts/evaluation/inference.sh --config $ecfg --task $task $sub) ;;
      eval)
        if [ ! -f $results/predictions.jsonl ]; then echo "$job: no predictions yet"; continue; fi
        id=$(sbatch --parsable --job-name=$job ${CPU_PARTITION:+--partition=$CPU_PARTITION} --gres=none --time=1-00:00:00 $site \
             scripts/evaluation/evaluate.sh --config $ecfg --task $task $sub) ;;
      *) echo "unknown stage $STAGE: configs, train, infer or eval" >&2; exit 1 ;;
    esac
    echo "$job: $STAGE $id"
  done
done
