# Ola-7B

Tsinghua University, Tencent Hunyuan and NTU's Ola-7B (February 2025, [arXiv 2502.04328](https://arxiv.org/abs/2502.04328),
Apache-2.0): a Qwen2.5-7B language model with an any-size SigLIP picture encoder and two sound encoders read together
(Whisper-large-v3 and BEATs). It is in neither transformers nor vLLM, so it runs with transformers 4.43.4 and the
authors' own code ([Ola-Omni/Ola](https://github.com/Ola-Omni/Ola), commit
`f681820b32db398e62d430c4e5b4f3dd962bbed4`), cloned into `external-repos/ola` (git-ignored, never pushed; `code` in the
config). Weights: [THUdyh/Ola-7b](https://huggingface.co/THUdyh/Ola-7b), revision
`d2fcaa533792480ce748321bea5101fed29d857e`. It runs through run_inference and run_eval like every model, one question
at a time.

    ola/
    ├── model.py           our frames and sound through the authors' steps, then their generate()
    └── requirements.txt   the authors' versions (torch 2.1.2, transformers 4.43.4, deepspeed 0.12.6), on Python 3.10

Settings live in evaluation/configs/ola.yaml.

## Input

As every model gets it: 32 frames per clip, evenly spaced, at the clips' own 224 x 224, and the sound at 16 kHz;
per clip its label ("Clip 1:" in t3), then the prompt (evaluation/prompts/). model.py reads each clip with the same
reader as run_inference and turns it into Ola's input with the authors' own steps (their `inference/infer.py`):

- **Frames**: each frame through their picture step (`process_anyres_video`), which enlarges it to 256 x 256 (their
  inference settings enlarge a frame to at least 288 x 288 in area, rounded down to a multiple of 64; their training
  settings would make 288). The model makes 73 pieces of each (16 x 16 patches merged 2 x 2, a row end per row, a
  separator), 2,338 per clip with the 2 around each video. 32 frames, as every model; their example takes 64. No frame
  times: the frames go in one after the other.
- **All 32 frames reach the language model**: model.py passes the videos to their `generate()` as a list, as their
  training does. Their inference scripts (`inference/infer.py`, their Hugging Face demo, lmms-eval's `ola.py`) pass a
  video as one block, and their input step then takes one entry per video from it: only the first frame gets through
  (75 pieces instead of 2,338, checked on the GPU). They also pick their 64 frames twice (GitHub issue #44), keeping
  54 of a 3 s clip's and 27 of a 6 s clip's.
- **Sound**: its two sound encoders read 30 s windows (Whisper stops otherwise), so each sound is the clip's own sound,
  then silence to 30 s, as their `load_audio` makes it (`sound()` in model.py). Their projector makes a piece of every
  10 encoder steps: 150 a window, 153 with its 3 marks. `silence: cut` (the config): only the encoder steps of the
  clip's own sound go on (154 for 3 s, 306 for 6 s; their projector drops the 4 or 6 steps left over after its groups
  of 10: 15 and 30 pieces, 18 and 33 with the marks), so no silence reaches the language model, as every model hears
  only the clip's sound; a hook in model.py cuts their encoders' output (`cut`), nothing in their code is changed.
  `silence: kept`: the whole window, the authors' way. Tested both ways on the 30 test questions: the same answer in
  28 (the 2 others: t3 voice questions of 6 s, one right each way), 4 right each way.
- **Order**: t1 sends a clip's video and its sound as the authors' example does (`<speech><image>`); their input step
  puts the video's pieces first, then the sound's. t2 and t3's face questions send silent videos (`<image>`, as their
  training's videos without sound; their silent sound goes along but has no place in the text, so none of it reaches
  the language model); t3's voice questions the two sounds (`<speech>`, with their blank picture, also never placed).
- **Text**: per clip its label and its tags, each on its own line, then the prompt, in their chat template
  (`qwen_1_5`, system line "You are a helpful assistant."), as their example writes it.
- **Answers**: greedy, at most 256 tokens, in bfloat16, as the authors' example runs it (their loader: the language
  model with PyTorch's attention, the picture encoder with FlashAttention 2, BEATs in float32), with no repetition
  penalty (its generation_config.json samples at temperature 0.7 with a repetition penalty of 1.05).
  `answer_choices: true` (as MiniCPM-o and Uni-MoE-2.0-Omni): the reply can only be `{"answer": "<one of the question's
  options>"}`; without it, it wrote the option alone in quotes (`"the video"`) in 17 of the 30 test questions, which the
  prompts can't read. transformers 4.43's own filter (`prefix_allowed_tokens_fn`) cannot start from an empty reply,
  which Ola's is (its prompt goes in as embeddings), so model.py passes the same filter as a logits processor
  (`only_choices`).

Their own readers are not used: they pick 64 frames (then 54 or 27, see above), and their sound goes through moviepy
and librosa. Checked before the GPU test: for each kind of question, our input equals what their own steps make from
our frames and sound (the same token ids, pictures, spectrograms and samples), and decord gives the same frames as our
reader.

## Did the sound arrive?

Every prediction's usage counts what the language model got (all its input at its first step, and the pieces of each
video at the picture projector):

    video_tokens    2,338 per video clip (32 frames of 73 pieces, and 2)
    audio_tokens    18 per 3 s sound, 33 per 6 s (cut; kept: 153 per sound)
    audio_steps     encoder steps passed on: 154 per 3 s sound, 306 per 6 s (cut; kept: 1,500)

## Speed

About 1.4 s per question on one A40 (t3: 1.5 s); loading takes about 2 minutes. It needs up to 24 GB of GPU memory
and about 10 GB of RAM.

## Environment

Its own, with the authors' versions, on uv's own Python 3.10:

    git clone https://github.com/Ola-Omni/Ola external-repos/ola
    git -C external-repos/ola checkout f681820b32db398e62d430c4e5b4f3dd962bbed4
    uv venv .venv_ola --python 3.10 --managed-python
    R=src/mozaic/evaluation/models/mllms/ola/requirements.txt
    uv pip install --python .venv_ola/bin/python --torch-backend cu121 -r <(grep -v '^deepspeed' $R)
    uv pip install --python .venv_ola/bin/python --torch-backend cu121 --no-build-isolation-package deepspeed -r $R
    uv pip install --python .venv_ola/bin/python --no-deps -e .

deepspeed (imported by their picture encoder's file) asks a CUDA compiler for its version when imported on a GPU
node, and ours have none (the pip CUDA packages hold no nvcc): CUDA 12.1's nvcc from NVIDIA's conda channel, torch's
CUDA, set by `cuda_home` in the config. Nothing is compiled.

    micromamba create -p models_weights/cuda-nvcc-12.1 -c nvidia cuda-nvcc=12.1.105

The weights (21.5 GB, with Whisper's `large-v3.pt` and BEATs' file; their config names those two from the working
folder, so model.py gives their full path). Their picture encoder
also asks the hub for CLIP's picture settings, so that one file goes into the HF cache (inference runs with HF_HUB_OFFLINE=1: no hub calls):

    hf download THUdyh/Ola-7b --revision d2fcaa533792480ce748321bea5101fed29d857e \
        --local-dir models_weights/ola-7b
    hf download openai/clip-vit-large-patch14 preprocessor_config.json

## Run

The global scripts, as every model:

    sbatch --gres=gpu:a40:1 scripts/evaluation/inference.sh --config ola --task t1 --limit 10   # 1 A40
    sbatch scripts/evaluation/evaluate.sh --config ola --task t1                               # scores

Results go to results/ola/{experiment_name}_{split}/{task}/; a rerun continues where the last one stopped, on the same
kind of GPU.
