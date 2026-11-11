# Qwen-Omni with vLLM

Qwen3-Omni-30B-A3B-Instruct and Qwen2.5-Omni-7B with vLLM, in their own environment (vLLM 0.17.1,
torch 2.10, transformers 4.57.6). They run through run_inference and run_eval like every model; run_inference gives
them batch_size questions at a time, which vLLM answers together.

    qwen_omni_vllm/
    ├── model.py           reads each clip with qwen_omni_utils (reads_clips), answers a batch with vLLM
    └── requirements.txt   everything it needs, at the versions it was tested with

Settings live in evaluation/configs/qwen3_omni_vllm.yaml and qwen2_5_omni_vllm/t1/zeroshot.yaml; the models
fine-tuned on MOZAIC (training/mllms/) in qwen2_5_omni_vllm/<task>/lora_train<full|small>.yaml.

## Input

Each clip is read by Qwen's helper qwen_omni_utils with its defaults, except two config settings: 32
frames evenly spaced from the first to the last (frames: FPS_MIN_FRAMES = FPS_MAX_FRAMES = 32), kept at
the clips' own 224 x 224 (frame_size). The sound is at its default 16 kHz. Per clip its label ("Clip 1:"
in t3), its video or its sound, then the prompt: the prompts of evaluation/prompts/, as for every model.
t1 sends each clip as one video with its sound inside (use_audio_in_video); t2 and t3's face questions
send silent videos; t3's voice questions the two sounds. Answers are greedy.

- **The frames' real rate goes in twice.** qwen_omni_utils gives it (32 / frames in the clip x 25 fps:
  10.4 for a 77-frame clip, 5.2 for 153). vLLM needs it as fps (Qwen's processor: the frames' times) and
  as second_per_grid_ts (vLLM itself: where the sound goes between frame pairs). Without the second, vLLM
  assumes 2 s per pair and puts the sound next to the wrong frames; the model card's example has this.
- **224, not enlarged.** By default qwen_omni_utils and Qwen's processor enlarge a 224 frame until it
  gives 128 tokens per frame pair: to 384 x 384 for Qwen3-Omni, 336 for Qwen2.5-Omni. Enlarging adds no
  detail and costs about 3 times the tokens; frame_size keeps 224.

## Did the sound arrive?

Every prediction's usage counts the tokens the model got of each kind:

                    Qwen3-Omni                   Qwen2.5-Omni
    audio_tokens    40 per 77-frame clip,        77 per 77-frame clip,
                    80 per 153-frame clip        153 per 153-frame clip
    video_tokens    784 per clip (49 per pair)   1,024 per clip (64 per pair)

## Environment

Its own environment, as every model's (vLLM 0.17.1 needs torch 2.10; the transformers Qwen environments
have torch 2.9):

    uv venv .venv_qwen_omni_vllm --python 3.12.12 --managed-python
    uv pip install --python .venv_qwen_omni_vllm/bin/python --torch-backend cu128 -e . \
        -r src/mozaic/evaluation/models/mllms/qwen_omni_vllm/requirements.txt
    ln -s .venv_qwen_omni_vllm .venv_qwen3_omni_vllm      # inference.sh opens .venv_<config>:
    ln -s .venv_qwen_omni_vllm .venv_qwen2_5_omni_vllm    # one environment, both configs

uv's own Python (--managed-python): vLLM compiles a small C helper when it starts, which needs Python's
header files, and a system Python may lack them on the GPU nodes (ours did). The versions are those
this was first run with (vLLM 0.17.1). qwen_omni_utils reads the sound with the
ffmpeg program, which must be on the PATH (ffmpeg 4.4.4 and 6.1.1 give the same sound for our clips).
model.py always reads videos with torchvision, and turns vLLM's media cache off, as with it a clip seen
twice loses its sound-in-video and fails.

## Run

The global scripts, as every model (--config, --task, --split, --subset; no --task: every task):

    sbatch --gres=gpu:2 --cpus-per-task=16 --mem=128G scripts/evaluation/inference.sh --config qwen3_omni_vllm --task t1 --limit 10
    sbatch scripts/evaluation/inference.sh --config qwen2_5_omni_vllm --task t1 --limit 10        # Qwen2.5-Omni: 1 GPU
    sbatch scripts/evaluation/inference.sh --config qwen2_5_omni_vllm/lora_trainfull --task t1  # fine-tuned on MOZAIC
    sbatch scripts/evaluation/evaluate.sh --config qwen2_5_omni_vllm --task t1                    # scores; the same flags

GPUs: Qwen3-Omni on 2 A40s (its config's tensor_parallel_size: 2; 1 on an A100 80 GB), Qwen2.5-Omni on
1 A40; --gres must match. A100s work too (Qwen3-Omni then fits one: a copy of its config with
tensor_parallel_size: 1). Run each task on one kind of GPU from start to end: inference_settings.json
saves the GPU and the vllm settings, so a run resumed on another GPU or split is refused (either can
change a few answers). Results: results/{model}/{experiment_name}_{split}/{task}/ (results_small/ with
SUBSET=small). A rerun continues where the last one stopped.
