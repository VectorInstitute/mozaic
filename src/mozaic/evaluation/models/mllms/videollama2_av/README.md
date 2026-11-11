# VideoLLaMA2.1-AV

DAMO's VideoLLaMA2.1-7B-AV (Apache-2.0): Qwen2-7B with a SigLIP picture encoder and a BEATs sound encoder,
made for questions about a video and its sound, and the audio-video model most deepfake benchmarks test. It
is in neither transformers nor vLLM, so it runs with transformers 4.42.3 and the authors' own code
([DAMO-NLP-SG/VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2), branch `audio_visual`, commit
`8cedfbda23e779246112cb963416b95415448a91`), cloned into `external-repos/videollama2` (git-ignored, never
pushed; `code` in the config). It runs through run_inference and run_eval like every model, one question at
a time.

    videollama2_av/
    ├── model.py           our frames and sound through the authors' steps, then their generate()
    └── requirements.txt   the authors' versions (torch 2.2.0, transformers 4.42.3), on Python 3.10

Settings live in evaluation/configs/videollama2_av.yaml.

## Input

As every model gets it: run_inference reads 32 frames per clip, evenly spaced, at the clips' own 224 x 224,
and the sound at 16 kHz; per clip its label ("Clip 1:" in t3), then the prompt (evaluation/prompts/).
model.py turns them into VideoLLaMA2's input with the authors' own steps:

- **Frames**: its picture processor (SigLIP) enlarges each frame to 384 x 384. 32 frames, as every model
  (VideoLLaMA2's default is 8; its code takes at most 32).
- **Sound**: t1 sends a clip's frames and sound together, as one video with its soundtrack (VideoLLaMA2's
  way). t3's voice questions send the two sounds alone: each clip's own sound once, padded with silence to
  30 s (the authors' sound-only step), then a 128-band spectrogram. t2 and t3's face questions send silent
  videos. The sound three ways (`soundtrack` in the config), all kept:
  - `pieces` (videollama2_av/t2/zeroshot.yaml): the authors' audio-visual step, 8 overlapping pieces of 2 s spread over
    the clip (16 s), then 14 s of silence. From our samples: their own function reads the sound with ffmpeg
    and gives silence when that fails.
  - `plain` (t1): the clip's sound once, then silence, as t3's voice questions and every other model get it.
    A check, not in the results (experiment zeroshot_32f_224_plainsound).
  - `nopad` (videollama2_av/t1/zeroshot.yaml, t1 and t3; videollama2_av/t4/zeroshot.yaml, t4): only the clip's own sound (3 s or 6 s), no silence and no
    repeats, for t1's soundtrack and t3's voice clips alike: the authors' sound-only spectrogram without its
    padding to 30 s. This is the run reported for VideoLLaMA2.1-AV on t1, t3 and t4 (experiment
    zeroshot_32f_224_nopad); t2 has no sound and comes from videollama2_av/t2/zeroshot.yaml.

  Checked on 1,000 val t1 questions (seed 7): 32.8% right with the pieces, 33.0% plain, 30 answers changed,
  McNemar p = 0.75 (no clear difference); the model answers "the video" about 85% of the time either way.
- **Text**: per clip its label and its tag (`<video>` or `<audio>`), then the prompt, in the model's chat
  template, as the authors' mm_infer writes it. The model puts each clip's features where its tag is.
- **Answers**: greedy, at most 256 tokens, in float16, as mm_infer runs it, with no repetition penalty, as
  every model answers. The checkpoint's generation_config.json would add one (1.05), so the config sets it to
  1.0 (off).

## Did the sound arrive?

Every prediction's usage counts the features the language model got of each kind, at the two projectors
(one turns a clip's frames into its input, one its sound):

    video_tokens    2,704 per video clip (its 32 frames)
    audio_tokens    1,496 per sound (its 30 s window)

The checkpoint also holds a copy of the picture encoder, which loading skips (448 tensors "not used"): the
authors' loader takes SigLIP itself, and the copy equals SigLIP rounded to bfloat16 (all 448 tensors checked).

## Speed

About 2 s per question on one A40 (t2: 1.6 s), plus reading the clips; loading takes about a minute (17 GB).
The 30 test questions took 2.5 minutes in all.

## Environment

Its own, with the authors' versions (Python 3.10: their decord 0.6.0 has no newer wheels):

    git clone --branch audio_visual https://github.com/DAMO-NLP-SG/VideoLLaMA2 external-repos/videollama2
    git -C external-repos/videollama2 checkout 8cedfbda23e779246112cb963416b95415448a91
    uv venv .venv_videollama2_av --python 3.10
    uv pip install --python .venv_videollama2_av/bin/python --torch-backend cu118 -e . \
        -r src/mozaic/evaluation/models/mllms/videollama2_av/requirements.txt

The weights (17.3 GB) and its picture encoder (3.5 GB):

    hf download DAMO-NLP-SG/VideoLLaMA2.1-7B-AV --revision d944d4298cb0188c3d5406eff42929cb67731361 \
        --local-dir models_weights/videollama2.1-7b-av
    hf download google/siglip-so400m-patch14-384 --revision 9fdffc58afc957d1a03a25b10dba0329ab15c2a3 \
        --local-dir models_weights/siglip-so400m-patch14-384

The checkpoint names its picture encoder by its hub id; model.py loads our copy instead (`vision_tower` in
the config), so nothing is fetched from the hub at run time. Nothing in the authors' code is changed.

## Run

The global scripts, as every model (MODEL, TASK, SPLIT, SUBSET at their top; TASK empty: every task):

    sbatch scripts/evaluation/inference.sh --limit 10           # 1 GPU (an A40 is enough)
    sbatch scripts/evaluation/evaluate.sh                       # scores; the same MODEL, TASK, SPLIT

Results go to results/videollama2_av/{experiment_name}_{split}/{task}/; a rerun continues where the last one
stopped, on the same kind of GPU.
