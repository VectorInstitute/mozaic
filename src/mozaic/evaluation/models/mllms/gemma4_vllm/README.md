# Gemma 4 with vLLM

Google's Gemma-4-12B-it (Apache-2.0) with vLLM, in its own environment (vLLM 0.28.0, the same versions as
nemotron_omni_vllm/). A third model family next to Qwen and Nemotron: it takes video frames and sound in
one question. It runs through run_inference and run_eval like every model, batch_size questions at a time,
which vLLM answers together (as qwen_omni_vllm/ and nemotron_omni_vllm/).

    gemma4_vllm/
    ├── model.py           reads each clip with vLLM's readers (reads_clips), answers a batch with vLLM
    └── requirements.txt   everything it needs, at the versions it was tested with (vLLM 0.28.0)

Settings live in evaluation/configs/gemma4_vllm.yaml.

## Input

Each clip is read by vLLM's own readers, as for Nemotron: 32 frames evenly spaced from the first to the last
(its video reader's num_frames: frames) and the sound at 16 kHz. Per clip its label ("Clip 1:" in t3), its
video and/or its sound, then the prompt: the prompts of evaluation/prompts/, as for every model. Answers are
greedy; Gemma's thinking is off by default.

- **The sound comes as its own part.** Gemma cannot take the sound inside a video, so t1 sends the clip's
  video, then its sound, then the prompt (as Gemini gets them). t2 and t3's face questions send silent
  videos; t3's voice questions the two sounds.
- **Frames at 384, not 224.** Gemma cuts a frame into pieces of 48 x 48 pixels, and vLLM gives each video
  frame at most 70 of them; a 224 x 224 frame is enlarged to 384 x 384 (64 tokens). 224 is not possible
  (not a multiple of 48), and frames resized by us first are enlarged again. Enlarging adds no detail.
- **Each frame has its time written before it** ("00:01 ..."), from the clip's 25 fps and the frames'
  numbers.
- **Triton attention.** Gemma has layers 512 wide; on A40s vLLM's FlashAttention stops at 256, and its next
  choice, FlashInfer, compiles with CUDA's nvcc, which our GPU nodes do not have. The config picks vLLM's
  Triton attention. model.py also turns off vLLM's FlashInfer sampler, for the same reason (as for Nemotron).

## Did the sound arrive?

Every prediction's usage counts the tokens the model got of each kind:

    audio_tokens    77 per 77-frame clip, 153 per 153-frame clip (25 per second)
    video_tokens    2,048 per clip: 32 frames x 64

## Environment

The same versions as Nemotron's, so one environment serves both (inference.sh opens .venv_<config>):

    ln -s .venv_nemotron_omni_vllm .venv_gemma4_vllm

or its own, from requirements.txt:

    uv venv .venv_gemma4_vllm --python 3.12.12 --managed-python
    uv pip install --python .venv_gemma4_vllm/bin/python --torch-backend cu129 \
        --extra-index-url https://wheels.vllm.ai/0.28.0/cu129 -e . \
        -r src/mozaic/evaluation/models/mllms/gemma4_vllm/requirements.txt

Download the weights into models_weights/ (24 GB):

    hf download google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
        --local-dir models_weights/gemma-4-12b-it

## Run

The global scripts, as every model (MODEL, TASK, SPLIT, SUBSET at their top; TASK empty: every task):

    sbatch scripts/evaluation/inference.sh --config gemma4_vllm --task t1 --limit 10   # 1 GPU (an A40 is enough)
    sbatch scripts/evaluation/evaluate.sh --config gemma4_vllm --task t1               # scores; the same flags

As for the other vLLM models: run each task on one kind of GPU from start to end, results in
results/{model}/{experiment_name}_{split}/{task}/, and a rerun continues where the last one stopped.
