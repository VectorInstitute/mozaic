# OmniVinci

NVIDIA's OmniVinci (October 2025, [arXiv 2510.15870](https://arxiv.org/abs/2510.15870)): a 7B Qwen2-type language
model with a SigLIP picture encoder (448 x 448) and a sound encoder from Audio Flamingo 3, made for questions about a
video and its sound. It is in neither transformers nor vLLM; its code comes with its weights on the hub
([nvidia/omnivinci](https://huggingface.co/nvidia/omnivinci), revision `7b3777b1c1f7c4e85f7bdef7b765dee1e76c1b7f`,
code also on [GitHub](https://github.com/NVlabs/OmniVinci)), and transformers 4.46.0 loads it with
`trust_remote_code`, so there is nothing to clone. Licence: NVIDIA OneWay Noncommercial (its README; the hub's tag
says Apache-2.0). It runs through run_inference and run_eval like every model, one question at a time.

    omnivinci/
    ├── model.py           our frames and sound through its own steps, then its generate()
    └── requirements.txt   the authors' versions (torch 2.3.0, transformers 4.46.0), on Python 3.10

Settings live in evaluation/configs/omnivinci.yaml.

## Input

As every model gets it: 32 frames per clip, evenly spaced, at the clips' own 224 x 224, and the sound at 16 kHz;
per clip its label ("Clip 1:" in t3), then the prompt (evaluation/prompts/). model.py reads each clip with the same
reader as run_inference (it needs each frame's number too), and turns it into OmniVinci's input with its own steps
(its media code), as its processor would from the clip's file:

- **Frames**: its picture processor enlarges each frame to 448 x 448 (its picture encoder takes only that size). 32
  frames, also its own default. Each frame gets its time (frame number / 25 fps). Inside the model every 8
  neighbouring frames are averaged into one picture, as it was trained: 4 pictures per clip, 256 pieces each.
- **Sound**: its sound encoder takes 30 s windows only, so each sound is the clip's own sound once, then silence to
  30 s (750 pieces); the language model gets only the pieces that cover the clip's own sound (25 a second: 77 for a
  3 s clip, 153 for 6 s), no silence:
  - t1: a clip's frames and sound go together, as one video with its soundtrack. The model keeps the soundtrack's
    pieces itself, and puts them after the pictures: pictures, sound, then the question.
  - t3's voice questions send the two sounds alone. Here the model would keep all 750 pieces (27 s of silence for a
    3 s clip), so model.py keeps only the clip's own pieces, by the model's own rule for a soundtrack: a hook on its
    sound encoder (`keep_own_sound`); nothing in its code is changed.
  - t2 and t3's face questions send silent videos.
- **Text**: per clip its label and its tags (`<sound><vila/video>`, `<vila/video>` or `<sound>`), then the prompt,
  joined as its processor joins a message's parts, with nothing in between (the authors' example: the video, then
  the question right after it), in its chat template (system line "You are a helpful assistant"), tokenized by its
  own helper.
- **Answers**: greedy, at most 256 tokens, in float16, as the authors' example runs it, with no repetition penalty,
  as every model answers (its language model's generation_config.json says sampling and 1.05; the config sets
  greedy and 1.0).

Two things of its code model.py works around, without changing it: its media code reads clip files itself (OpenCV
frames, decord sound) and quietly drops the sound when that fails, so model.py gives it our frames and sound instead;
and its generate stops on two sounds in one question as its processor lists them, so model.py lists each sound on
its own (the same sounds, in the same order).

## Did the sound arrive?

Every prediction's usage counts what the language model got: `prompt_tokens` all of it (text, pictures, sound),
`video_tokens` the pictures' pieces at the video encoder, `audio_tokens` the rest after the text:

    video_tokens    1,024 per video clip (4 pictures of 256): t1 and t2 1,024, t3 face 2,048
    audio_tokens    77 per 3 s sound, 153 per 6 s: t1 77 or 153, t3 voice 154 or 306 (two voices; 1,500 uncut)

All as above in the 30 test questions, with the text making up the rest of prompt_tokens exactly. Loading reports
no unused or missing weights.

## Speed

About 1 s per question on one A40 to answer (t3 face, two videos: 1.8 s), about 2 s with reading the clips;
loading takes about 25 s (17.5 GB). The 30 test questions took a minute in all.

## Environment

Its own, with the authors' versions (Python 3.10: their FlashAttention 2 wheel is for 3.10):

    uv venv .venv_omnivinci --python 3.10
    uv pip install --python .venv_omnivinci/bin/python --torch-backend cu121 -e . \
        -r src/mozaic/evaluation/models/mllms/omnivinci/requirements.txt

The authors' environment_setup.sh also replaces one file of transformers (modeling_utils.py); its only change from
4.46.0 lets weight files without a header load, and all of OmniVinci's have one, so it is not used.

The weights and code (17.5 GB):

    hf download nvidia/omnivinci --revision 7b3777b1c1f7c4e85f7bdef7b765dee1e76c1b7f \
        --local-dir models_weights/omnivinci

## Run

The global scripts, as every model (MODEL, TASK, SPLIT, SUBSET at their top; TASK empty: every task):

    sbatch scripts/evaluation/inference.sh --limit 10           # 1 GPU (an A40 is enough)
    sbatch scripts/evaluation/evaluate.sh                       # scores; the same MODEL, TASK, SPLIT

Results go to results/omnivinci/{experiment_name}_{split}/{task}/; a rerun continues where the last one stopped, on
the same kind of GPU.
