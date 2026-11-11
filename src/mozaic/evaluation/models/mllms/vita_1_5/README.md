# VITA-1.5

VITA-1.5 (January 2025, [arXiv 2501.01957](https://arxiv.org/abs/2501.01957); Tencent's VITA1.5 licence: academic,
research and education use only): a Qwen2.5-7B language model with an InternViT-300M picture encoder (448 x 448) and
its own speech encoder. It is in neither transformers nor vLLM, so it runs with the authors' own code
([VITA-MLLM/VITA](https://github.com/VITA-MLLM/VITA), commit `35d064a6542a5d812136fcd66fa93d9beb27b03c`), cloned into
`external-repos/vita` (git-ignored, never pushed; `code` in the config). Weights:
[VITA-MLLM/VITA-1.5](https://huggingface.co/VITA-MLLM/VITA-1.5), revision `e42b47fa9f390f8bee2c852bf3d4af0281c532bc`,
with the two encoders its config names: [OpenGVLab/InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)
(`fe158954b3b9bbc92a37fd24a78912173910558a`) and [VITA-MLLM/VITA-1.5_AudioEnc](https://huggingface.co/VITA-MLLM/VITA-1.5_AudioEnc)
(`d8bf08396c832f395d70c2344f245d4452ea10cb`). It runs through run_inference and run_eval like every model, one
question at a time.

    vita_1_5/
    ├── model.py           our frames and sound through the authors' steps, then their generate()
    └── requirements.txt   the authors' versions (torch 2.3.1; transformers 4.44.2, see Environment), on Python 3.10

Settings live in evaluation/configs/vita_1_5.yaml.

## Input

As every model gets it: 32 frames per clip, evenly spaced, at the clips' own 224 x 224, and the sound at 16 kHz;
per clip its label ("Clip 1:" in t3), then the prompt (evaluation/prompts/). model.py reads each clip with the same
reader as run_inference and turns it into VITA's input with the authors' own steps (their `video_audio_demo.py`):

- **Frames**: each frame through its picture encoder's processor, which enlarges it to 448 x 448 (the only size it
  takes); the model makes 256 pieces of each, 8,192 per clip. 32 frames, as every model; their reader takes 1 a
  second, at most 16 (4 or 7 for our clips). Each frame has its own `<image>` tag, one after the other, as their
  example writes a video.
- **No input cut**: their code cuts the input at 6,200 pieces (their training length), also when answering, so 32
  frames (8,192 pieces; t3's face questions 16,384) would lose the question. `input_cap: null` (the config) lifts it;
  its language model takes 32,768. It was trained on at most 16 frames.
- **Sound**: their sound step (a Kaldi filterbank of the clip's own sound, no padding: no silence reaches the model);
  38 pieces for 3 s, 76 for 6 s. It adds random noise (dither, from their training settings): `seed` (the config)
  seeds it per question, so a rerun gives the same answer (checked: the 20 test questions with sound asked again gave
  the same replies).
- **What its sound channel is for**: VITA was trained to hear the user's spoken question there (their demo takes a
  question in text or in sound, never both); here it carries the clip's sound, as every model hears it, with the
  question in text. Each answer starts with a mark of the question it heard (☜ text, ☞ spoken, ☟ a sound to ignore),
  which their evaluations drop: model.py drops it too and keeps it in the usage (`answer_mark`). In the test it marked
  every question as a text question (☜), those with the clip's sound too.
- **Order**: per clip its frames' tags, then its sound's (`<audio>`), then, on a new line, the question, as every model
  gets it (their example puts the sound after the question, being the question). t2 and t3's face questions send
  silent videos (their blank sound goes along but has no place in the text, so it never reaches the model); t3's
  voice questions the two sounds (with their blank picture, also never placed).
- **Text**: their chat template (`qwen2p5_instruct`), whose system line changes with the input: for a video ("...you
  are seeing the video, not the image") or with no picture (t3 voice).
- **Answers**: greedy, at most 256 tokens, in float16, as the authors' example runs it, with no repetition penalty
  (its generation_config.json samples at temperature 0.7 with a repetition penalty of 1.05). `answer_choices: true`
  (as MiniCPM-o, Uni-MoE-2.0-Omni and Ola): the reply can only be its mark, then `{"answer": "<one of the question's
  options>"}`; without it, it wrote the option alone (`"neither"`) in 3 of the 30 test questions, which the prompts
  can't read (the chosen options did not change). transformers 4.44's own filter (`prefix_allowed_tokens_fn`) cannot
  start from an empty reply, which VITA's is (its prompt goes in as embeddings), so model.py passes the same filter
  as a logits processor (`only_choices`).

Their own readers are not used. Checked before the GPU test: for each kind of question, their reader's frames (1 a
second) through their steps equal ours at the same frame numbers, their sound step gives the same filterbank from our
sound with the same seed, and the token ids hold one tag per frame and per sound.

## Did the sound arrive?

Every prediction's usage counts what the language model got (all its input at its first step, and the pieces of the
frames at the picture projector):

    video_tokens    8,192 per video clip (32 frames of 256 pieces)
    audio_tokens    38 per 3 s sound, 76 per 6 s
    answer_mark     the question it heard: ☜ text, ☞ spoken, ☟ a sound to ignore

## Speed

About 2.9 s per question on one A40; loading takes under a minute. It needs up to 28 GB of GPU memory (t3's face
questions are 16,800 tokens) and about 5 GB of RAM.

## Environment

Its own, with the authors' versions, on uv's own Python 3.10.
Their requirements.txt says transformers 4.41.1, but their code (`vita_qwen2.py`) reads the `cache_position` that 4.42
and later pass; 4.44.2 is the version their checkpoint was saved with (its config.json), so no change to their code.

    git clone https://github.com/VITA-MLLM/VITA external-repos/vita
    git -C external-repos/vita checkout 35d064a6542a5d812136fcd66fa93d9beb27b03c
    uv venv .venv_vita_1_5 --python 3.10 --managed-python
    uv pip install --python .venv_vita_1_5/bin/python --torch-backend cu121 \
        -r src/mozaic/evaluation/models/mllms/vita_1_5/requirements.txt
    uv pip install --python .venv_vita_1_5/bin/python --no-deps -e .

The weights (18.2 GB without its speech-output files, which answering does not use) and the two encoders its config
names by hub id (read when the model is built; runs set HF_HUB_OFFLINE=1, so model.py gives our copies' paths); one pattern per `--exclude`:

    M=models_weights
    hf download VITA-MLLM/VITA-1.5 --revision e42b47fa9f390f8bee2c852bf3d4af0281c532bc --local-dir $M/vita-1.5 \
        --exclude "vita_tts_ckpt/*"
    hf download OpenGVLab/InternViT-300M-448px --revision fe158954b3b9bbc92a37fd24a78912173910558a \
        --local-dir $M/internvit-300m-448px --exclude "mlp_projector/*"
    hf download VITA-MLLM/VITA-1.5_AudioEnc --revision d8bf08396c832f395d70c2344f245d4452ea10cb \
        --local-dir $M/vita-1.5-audioenc

## Run

The global scripts, as every model:

    sbatch --gres=gpu:a40:1 scripts/evaluation/inference.sh --config vita_1_5 --task t1 --limit 10   # 1 A40
    sbatch scripts/evaluation/evaluate.sh --config vita_1_5 --task t1                               # scores

Results go to results/vita_1_5/{experiment_name}_{split}/{task}/; a rerun continues where the last one stopped, on
the same kind of GPU.
