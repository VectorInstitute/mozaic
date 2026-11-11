# Nemotron 3 Nano Omni with vLLM

NVIDIA's Nemotron-3-Nano-Omni-30B-A3B-Reasoning (BF16, 2026-04-28, NVIDIA Open Model Agreement) with vLLM,
in its own environment (vLLM 0.28.0). An open omni model not built on Qwen that takes a video with its
own sound: its LLM is Nemotron 3 Nano (a Mamba-Transformer mixture of experts, 31B with about 3B active),
its vision encoder C-RADIOv4-H, its sound encoder Parakeet. It runs through run_inference and run_eval like
every model, batch_size questions at a time, which vLLM answers together (as qwen_omni_vllm/).

    nemotron_omni_vllm/
    ├── model.py           reads each clip with vLLM's readers (reads_clips), answers a batch with vLLM
    └── requirements.txt   everything it needs, at the versions it was tested with (vLLM 0.28.0)

Settings live in evaluation/configs/nemotron_omni_vllm.yaml.

## Input

Each clip is read by vLLM's own readers, as vLLM reads it for this model when serving: 32 frames evenly
spaced from the first to the last (its video reader's num_frames: frames), kept at the clips' own
224 x 224 (frame_size), and the sound at 16 kHz. Per clip its label ("Clip 1:" in t3), its video or its
sound, then the prompt: the prompts of evaluation/prompts/, as for every model. t1 sends each clip as one
video with its sound (use_audio_in_video); t2 and t3's face questions send silent videos; t3's voice
questions the two sounds. Answers are greedy.

- **Reasoning off.** Nemotron reasons before it answers by default. chat_template_kwargs
  enable_thinking: false makes it answer directly (the model card's switch), so a T1-T3 answer fits in
  256 tokens, as for every model.
- **The sound comes after the frames, as one piece.** Nemotron cuts sound into pieces of 30 seconds (its
  paper, arXiv 2604.24954), so the sound of a 3 or 6 s clip is one piece. With use_audio_in_video, vLLM
  puts it right after the clip's 32 frames (NVIDIA's own code also keeps the sound as one block). Each frame pair has its time
  written next to it ("Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.08 seconds: ..."), the
  sound has none. Qwen instead puts small bits of sound between the frame pairs, in time order. So
  Nemotron gets the same frames and sound, but no fine timing between them.
- **224, not enlarged.** Our clips are 224 x 224. By default Nemotron enlarges each frame to 512 x 512
  (1,024 patches of 16 x 16): about 5 times more tokens (256 per frame pair instead of 49), and no new
  detail. model.py keeps 224 (video_target_num_patches = (224 / 16)^2 = 196), as we did for Qwen.
- **The labels stay with their clips.** Nemotron's chat template moves media sent as separate parts to
  the front of the message; model.py writes the tags (`<video>`, `<so_embedding>`) into the text instead,
  each after its clip's label.
- **Every frame token kept.** vLLM's video token pruning is off unless asked for (the model card offers it
  to save memory).

## Did the sound arrive?

Every prediction's usage counts the tokens the model got of each kind:

    audio_tokens    about 39 per 77-frame clip, 77 per 153-frame clip (from vLLM's formula; the first test confirms)
    video_tokens    784 per clip: 16 frame pairs x 49 (vLLM sends a video's frames as <image> tokens)

## Environment

Its own environment, as every model's (vLLM 0.28.0 needs torch 2.13; the Qwen vLLM environment has 2.10):

    uv venv .venv_nemotron_omni_vllm --python 3.12.12 --managed-python
    uv pip install --python .venv_nemotron_omni_vllm/bin/python --torch-backend cu129 \
        --extra-index-url https://wheels.vllm.ai/0.28.0/cu129 -e . \
        -r src/mozaic/evaluation/models/mllms/nemotron_omni_vllm/requirements.txt

vLLM's CUDA 12.9 build: PyPI's vLLM 0.28.0 comes with torch 2.13 for CUDA 13, which needs a newer GPU
driver than CUDA 12 (the Qwen environment runs CUDA 12.8 on our GPUs). uv's own Python (--managed-python),
as for Qwen: vLLM compiles a small C helper when it starts, which needs Python's header files. model.py
turns off vLLM's FlashInfer sampler: it compiles a kernel with CUDA's nvcc when vLLM starts, and our GPU
nodes have no nvcc (vLLM's PyTorch sampler gives the same greedy answers).

Download the weights into models_weights/ (66 GB):

    hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
        --revision e5e9932441de940c9a62185c870ea5bcd4cd24e2 \
        --local-dir models_weights/nemotron-3-nano-omni-30b-a3b

## Run

The global scripts, as every model (MODEL, TASK, SPLIT, SUBSET at their top; TASK empty: every task):

    sbatch --gres=gpu:2 --cpus-per-task=16 --mem=128G scripts/evaluation/inference.sh --config nemotron_omni_vllm --task t1 --limit 10
    sbatch scripts/evaluation/evaluate.sh --config nemotron_omni_vllm --task t1   # scores; the same flags

GPUs: 2 A40s (its config's tensor_parallel_size: 2) or 1 A100 80 GB (a copy of its config with
tensor_parallel_size: 1); --gres must match. As for qwen_omni_vllm/: run each task on one kind of GPU from
start to end, results in results/{model}/{experiment_name}_{split}/{task}/, and a rerun continues where
the last one stopped.
