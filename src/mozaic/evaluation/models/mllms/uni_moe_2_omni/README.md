# Uni-MoE-2.0-Omni

HIT-TMG's Uni-MoE-2.0-Omni (November 2025, [arXiv 2511.12609](https://arxiv.org/abs/2511.12609), Apache-2.0): a
Qwen2.5-7B language model with mixture-of-experts layers, a SigLIP picture encoder (384 x 384) and a Whisper sound
encoder. It is in neither transformers nor vLLM, so it runs with transformers 4.51.3 and the authors' own code
([HITsz-TMG/Uni-MoE](https://github.com/HITsz-TMG/Uni-MoE), folder `Uni-MoE-2`, commit
`6f18a7aedfdc3ade10cd1a992e28b29d1650db38`), cloned into `external-repos/uni_moe_2` (git-ignored, never pushed; `code`
in the config). It runs through run_inference and run_eval like every model, one question at a time.

    uni_moe_2_omni/
    ├── model.py           our frames and sound through the authors' steps, then their generate()
    └── requirements.txt   the authors' versions (torch 2.5.1, transformers 4.51.3, deepspeed 0.15.4), on Python 3.11

Settings live in evaluation/configs/uni_moe_2_omni.yaml.

## Input

As every model gets it: 32 frames per clip, evenly spaced, at the clips' own 224 x 224, and the sound at 16 kHz;
per clip its label ("Clip 1:" in t3), then the prompt (evaluation/prompts/). model.py reads each clip with the same
reader as run_inference and turns it into Uni-MoE's input with the authors' own steps, as their processor would from
the clip's file:

- **Frames**: each frame enlarged to 384 x 384 (its picture encoder takes only that size); the model makes 144 pieces
  of each (4,608 per clip). 32 frames, as every model: its own reader takes 1 a second, at least 8 (8 for our clips),
  and allows up to 32 for a video with its sound. Each frame's time is the time between our frames, as its reader
  gives it; the model puts frames on half-second time steps, so about 5 of our frames share a step in a 3 s clip.
- **Sound**: its Whisper encoder takes exactly 30 s (transformers stops otherwise), so each sound is the clip's own
  sound, then silence to 30 s; the model sums each 30 s window up in 200 pieces, each reading the window's 1,500
  encoder steps. `silence: cut` (the config): the 200 pieces read only the steps of the clip's own sound (154 for
  3 s, 306 for 6 s), so no silence reaches them; model.py calls their summary step with those steps
  (`summarize`), nothing in their code is changed. `silence: kept`: every step, the authors' way. Tested both ways
  on the 30 test questions: t2 and t3 the same answers (t3's 6 voice questions too), t1 7 of 10 the same (2 right
  with kept, 3 with cut).
- **Order**: t1 sends a clip's video, then its sound, as the authors' example asks about a video with its sound; t2
  and t3's face questions send silent videos, t3's voice questions the two sounds.
- **Text**: per clip its label and its tags (`<video>`, `<audio>`), each on its own line, then the prompt, in its
  chat template (system line "You are a helpful assistant."), as the authors' example writes it.
- **Answers**: greedy, at most 256 tokens, in bfloat16 with FlashAttention 2, as the authors' examples run it, with
  no repetition penalty (its generation_config.json sets none). `answer_choices: true` (as MiniCPM-o): the reply can
  only be `{"answer": "<one of the question's options>"}`; without it, it wrote the option alone ("face swap") in 5
  of the 10 t2 test questions, which the prompts can't read.

Their own readers are not used: their sound reader cannot open our .mov clips, and their video reader takes 8
frames. Checked before the GPU test: for each kind of question, our input equals what their processor makes as the
authors' example asks it (with their reader set to 32 frames): the same token ids, pictures, sound and frame times.

## Did the sound arrive?

Every prediction's usage counts what the language model got (the model checks it made as many pieces as places):

    video_tokens    4,608 per video clip (32 frames of 144 pieces)
    audio_tokens    200 per sound (one 30 s window); t3 voice 400
    audio_steps     encoder steps the sound's 200 pieces read: 154 per 3 s sound, 306 per 6 s (cut; kept: 1,500)

## Speed

About 2.8 s per question on one A100 (t2: 2.6 s); loading takes about 2 minutes (65 GB). It needs up to 73 GB of
GPU memory, so one A100-80GB (an A40 is too small); about 39 GB of RAM.

## Environment

Its own, with the authors' versions, on uv's own Python 3.11:

    git clone https://github.com/HITsz-TMG/Uni-MoE external-repos/uni_moe_2
    git -C external-repos/uni_moe_2 checkout 6f18a7aedfdc3ade10cd1a992e28b29d1650db38
    uv venv .venv_uni_moe_2_omni --python 3.11 --managed-python
    uv pip install --python .venv_uni_moe_2_omni/bin/python --torch-backend cu121 \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 setuptools==80.9.0 wheel
    uv pip install --python .venv_uni_moe_2_omni/bin/python --torch-backend cu121 --no-build-isolation-package deepspeed \
        --no-build-isolation-package clip -e . -r src/mozaic/evaluation/models/mllms/uni_moe_2_omni/requirements.txt

deepspeed (its mixture-of-experts layers) asks a CUDA compiler for its version when imported on a GPU node, and ours
have none (the pip CUDA packages hold no nvcc): CUDA 12.1's nvcc from NVIDIA's conda channel, torch's CUDA, set by
`cuda_home` in the config. Nothing is compiled.

    micromamba create -p models_weights/cuda-nvcc-12.1 -c nvidia cuda-nvcc=12.1.105

The weights (65 GB without its image- and speech-generation files, which answering does not use); one pattern per `--exclude`:

    hf download HIT-TMG/Uni-MoE-2.0-Omni --revision e4bdc0346ccd025b4994f3b61fe19567f5fe80cb \
        --local-dir models_weights/uni-moe-2.0-omni \
        --exclude "visual_gen/*" --exclude "speech_gen/*" --exclude "imgs/*"

## Run

The global scripts, as every model (MODEL, TASK, SPLIT, SUBSET at their top; TASK empty: every task):

    sbatch --gres=gpu:a100:1 scripts/evaluation/inference.sh --limit 10   # 1 A100-80GB
    sbatch scripts/evaluation/evaluate.sh                               # scores; the same MODEL, TASK, SPLIT

Results go to results/uni_moe_2_omni/{experiment_name}_{split}/{task}/; a rerun continues where the last one
stopped, on the same kind of GPU.
