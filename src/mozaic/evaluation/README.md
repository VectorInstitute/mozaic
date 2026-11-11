# Evaluation

Runs a model on a task's questions (inference), then scores its predictions (eval).
The two are separate steps: inference needs the model's environment, eval needs only
the core one, so scores can be redone any time without the model.

    evaluation/
    ├── run_inference.py     questions → predictions.jsonl (model environment)
    ├── run_eval.py          predictions.jsonl → scores.json (core environment)
    ├── base.py              BaseMLLM and BaseClassifier, what a model implements
    ├── configs/             one yaml per model and experiment
    ├── predictions.py       answer or class probabilities → pred_* fields, per task
    ├── prompts/             per task, MLLMs only: prompt and reply parsing
    ├── metrics/             per task: scores, plus shared stats.py
    ├── models/
    │   ├── api/             API models (gemini/)
    │   ├── mllms/            open MLLMs (qwen_omni_vllm/, nemotron_omni_vllm/, gemma4_vllm/, minicpm_o_vllm/,
    │   |                     videollama2_av/, omnivinci/, uni_moe_2_omni/, ola/, vita_1_5/)
    │   └── classifiers/     classifiers trained on MOZAIC (trained/: training's best.pt)
    └── visualization/       cross-model tables and plots

 
## Run

    python -m mozaic.evaluation.run_inference --config gemini_3_8_flash --task t1
    python -m mozaic.evaluation.run_inference --config gemini_3_8_flash --task t1 --limit 4
    python -m mozaic.evaluation.run_inference --config gemini_3_8_flash --task t1 --retry-failed
    python -m mozaic.evaluation.run_eval --config gemini_3_8_flash --task t1
    python -m mozaic.evaluation.run_inference --config gemini_3_8_flash --task t4 --subset small   # MOZAIC-Small
    python -m mozaic.evaluation.judge.run_judge --config gemini_3_8_flash --task t4 --subset small # T4: the judge (judge/README.md; scripts/evaluation/judge.sh)
    python -m mozaic.evaluation.run_eval --config gemini_3_8_flash --task t4 --subset small

## Flags

    run_inference
    --config NAME     <model>/<run> (zeroshot if left out), a full config name, or a path (required;
                      configs/README.md, Naming a config)
    --task t1         only this task (default: every task in the config)
    --limit N         only the first N questions
    --retry-failed    read unreadable saved replies again with the current parser (no call), then ask again
                      the ones still unreadable or cut off
    --overwrite       delete the task's predictions.jsonl, inference_settings.json and
                      scores.json, then run fresh (not with --retry-failed)
    --env-file PATH   env file with API keys (overrides the config)
    --subset small    run on MOZAIC-Small: questions from qas_small/, results to results_small/
                      (t4 exists on Small only)
    --split val       for configs whose files say {split} (t1_{split}.jsonl): that split (default
                      test), results in {experiment_name}_{split}

    run_eval
    --config NAME     same config as the inference run (required)
    --task t1         only this task
    --subset small    score the run made with --subset small
    --split val       score the run made with --split val

Every Small question is a copy of the full one with its id, so the model's Small answers are its full run's
for those ids: they are copied, never asked again (an API model would be paid twice). Scoring a full run of t1,
t2 or t3 copies them to results_small/ and scores Small once every Small question has an answer; a
--subset small run copies them before asking anything, and asks the model only what the full run has no
answer to. A copied answer has copied_from (the full run's folder), its inference_settings.json derived_from;
a copy follows the full answer if that changes (--retry-failed there). Only from the full run of the same
config and settings; an answer the model gave in a --subset small run is never replaced.

## Choosing the questions

Each task in a config names a question file and, optionally, filters that pick a
subset of it (arms, frames, labels, video_methods; see common/questions.py):

    tasks:
      t1:
        file: t1_test.jsonl
        filters:
          frames: [77]             # 77-frame clips only
      t2: t2_test.jsonl            # short form: no filters

- Filters are applied before --limit and saved in inference_settings.json, so a run
  with other filters in the same folder is refused.
- run_eval applies the same filters, so questions filtered out never count as missing.
  scores.json records them next to question_file.
- Put the filter in experiment_name (e.g. zeroshot_32f_77f).
- Only compare models run on the same question file and filters.
- The T1, T3 and T4 files are already the vocoded view (every voice has been through a
  vocoder), so arms is not needed on them. It is about real voices only: fake-voice clips
  and T3 face questions (no sound) are always kept.

## Model families

Each config sets family. run_inference picks the loop by it:

| family | Base class | Input | Output |
|---|---|---|---|
| mllm | BaseMLLM | we sample frames and audio and build the prompt; generate() returns text | parsed into pred_*, score null |
| classifier | BaseClassifier | predict_proba() gets a batch of questions and reads the clips itself | class probabilities in score, pred_* from the top class (t1 to t3; t4 is answered in text, MLLMs only) |

Both write the same prediction rows, so run_eval scores them the same way. With a
classifier's probabilities, AUROC, EER and AP use them (score_source: model); for an
MLLM they come from the hard answer (score_source: hard_label). See metrics/README.md.

## Questions with two clips (t3)

The task's prompts file says per clip what to send (clip_inputs): face questions get
the frames of both clips, voice questions the sound of both clips. run_inference reads
each clip and calls generate_clips() with a list of ClipInput (label "Clip 1:", frames,
sound), then the prompt. BaseMLLM.generate_clips() passes a one-clip question to
generate(), so a model needs its own generate_clips() only for two-clip tasks.

## Output

    results/{model}/{experiment_name}/{task}/          (results_small/ with --subset small)
    ├── inference_settings.json   settings of the run, written once
    ├── predictions.jsonl         one row per answered question
    └── scores.json               written by run_eval

A prediction row:

    id, model, task
    pred_answer, pred_label / pred_target ...   mirror the question's gt_* fields (t4: pred_answer is the
                                                two explanations joined, pred_target.sections each one)
    score                                       class probabilities, or null
    raw, finish_reason, usage                   the model's reply (MLLM only)
    time

## Guards

    Resume           a rerun skips questions already in predictions.jsonl
    Failures         a failed call is not saved, so the next run asks it again; the
                     run stops after stop_after_failures failures in a row
    Settings         inference_settings.json holds model, question file, frames,
                     filters, prompt hash (MLLM) or checkpoint (classifier). A rerun with other
                     settings is refused: use a new experiment_name
    Same input       MLLMs get frames sampled uniformly by common/frame_sampler.py and
                     audio at one rate, the same for every MLLM
    Append           each prediction is appended and flushed as it comes
    Permissions      every file and folder written is group-writable
    T4               runs with --subset small only; refuses the annotator, verifier and rewriter
                     models (they wrote the reference explanations) and t4_golden.jsonl (human review only)

## Configs

    model: gemini                  name in load_model and in results/
    family: mllm                   mllm or classifier
    experiment_name: zeroshot_32f  results folder; change it when a setting changes
    frames: 32
    audio_sample_rate: 16000       mllm
    stop_after_failures: 3
    tasks:                         question file per task, in qas/{task folder}/ (qas_small/ with --subset small)
      t1: t1_test.jsonl            t1_40.jsonl for a pilot, t1_test.jsonl for final numbers
                                   or {file: ..., filters: {...}}, see Choosing the questions

An mllm config also has model_version, generation and retry, and may have batch_size (default 1):
the questions run_inference gives the model per call (generate_batch). A model with reads_clips
gets each clip's file instead of its frames and sound (ClipInput.path). A classifier config has
batch_size, micro_batch_size and workers.

A model trained on MOZAIC (training/) has one config per training run, named for it:
configs/<model>/<task>_<training>_train<data>.yaml (avhubert/t1/ln_trainsmall, avhubert/t1/full_trainfull,
qwen2_5_omni_vllm/t1/lora_trainsmall). Its trained: names the run, {config: avhubert/t1_full, data: full},
and the weights come from it: best.pt for a classifier, final/ (the merged model) for an MLLM, in
training/ (data: full) or training_small/ (data: small). --subset picks only the test questions, so a
model trained on Small is tested on Small's test split or the full one; experiment_name ends in
_train<data>, so the two never share a results folder. It has the run's one task, on the test split
only: run_inference refuses any other file, since its train and val questions were seen in training
(Small's splits sit inside the full ones, with no clip or identity across them). The folder is the
model's environment (avhubert/: .venv_avhubert).
An mllm that takes videos may have frames_as: video (default images): read_clips then also
gives each clip the real rate of its frames (ClipInput.fps).

Video only or audio only (t1, t4): a task's entry may have modalities, both by default:

    tasks:
      t1:
        file: t1_test.jsonl
        modalities: [video, audio]   # keep one: [video] sends the frames, no sound; [audio] the sound, no frames

The prompt's first line then says what is not given (t1_prompts.FIRST_LINES); the question, options and reply
format stay, so a t4 reply still explains both parts. Give the run its own experiment_name. The modality
ablation: ablations/modality/.

## Generation

The same answer settings for every MLLM (its config's generation), so the answers compare:

    T1-T3   greedy (temperature 0), at most 256 output tokens: the answer is a short JSON
    T4      greedy (temperature 0), at most 8192 output tokens: the answer is a written explanation

Thinking models (Gemini) count their hidden thinking in the limit, so they keep their own settings:
gemini_3_8_flash/t1/zeroshot.yaml and gemini_3_1_pro/t1/zeroshot.yaml, the runs compared with the open models, are greedy
(temperature 0) with thinking low (the lowest Gemini 3.1 Pro takes) and 2048 tokens.

## Adding a model

1. A folder in models/api/, models/mllms/ or models/classifiers/ with model.py,
   holding a class Model that subclasses BaseMLLM or BaseClassifier, a
   requirements.txt and a README.
2. A branch for it in load_model (run_inference.py). The import stays inside the
   branch, because each model has its own environment.
3. A config in configs/ with the model's name and family.
4. Its own environment: uv venv .venv_{model}, then uv pip install -e . and its
   requirements.txt.

## Adding a task

1. prompts/{task}_prompts.py and metrics/{task}_metrics.py, like t1 to t4. Take options and
   regions from the task's annotation code, never a copy.
2. A branch for the task in to_prediction and from_probs (predictions.py), in
   load_prompts (run_inference.py) and in eval_task (run_eval.py).
