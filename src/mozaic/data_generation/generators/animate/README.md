# Wan2.2-Animate

Wan-AI's [Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) (Apache-2.0): from one picture of a
person and a driving video, it makes a video of that person moving as the driver moves. It runs with the
authors' code ([Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)) in two steps, preprocess_data.py (pose and
face tracking, segmentation) then generate.py. One of MOZAIC's three video generators: a build renders with one
of them, named in its manifest (video_method `animate`, the default).

    animate/
    ├── make_jobs.py    the build's job list (core environment)
    └── run.py          per render: face photo, preprocess, generate, re-stamp to 25 fps, one shard per GPU
                        task (Animate's environment; imports nothing from mozaic)

Settings live in data_generation/configs/animate.yaml.

## What it does to a clip

- **Input**: the driver's crop (crop/: 224 x 224, 25 fps) and one face photo of the other person: a frame of
  their own VoxCeleb2 clip, at the time reference_frame.py picks (as for the other video generators; run.py
  extracts it).
- **Replace mode** (the cross renders, render_key `cross__<crop_key>__face_<id>`): meant to put the other
  person into the driver's video. On our tight 224 px face crops the person is the whole frame, so it redraws
  every pixel from the one photo, copying the driver's head motion, expression and lips frame by frame, and
  re-lights the person with its relighting LoRA (on, as MOZAIC rendered). Preprocessing segments the person with
  sam2 (replace-mode flags `--iterations 3 --k 7 --w_len 1 --h_len 1`).
- **Animate mode** (the self-drive control, `self__<crop_key>`): the driver re-made from their own photo (a
  frame of the same clip) with pose retargeting (`--retarget_flag`), so a quality drop there is the generator's,
  not the swap's.
- **Size**: `--resolution_area 832 480` keeps the crop's square shape: Animate works at 624 x 624. The mux in
  final_clips/ resizes every clip through 624 px to 224 x 224.
- **77 frames only**: Animate renders 77-frame chunks; a second chunk drifts back to the driver's look, so
  MOZAIC renders one chunk, on 77-frame crops only (frame_num 77, refert_num 1).
- **Frame rate**: preprocessing runs at `--fps 25`, so the pose video keeps the crop's 77 frames (its default,
  30, would retime the driver into more frames). generate.py stamps its output 30 fps; run.py rescales the
  timestamps to 25 fps without re-encoding (ffmpeg -itsscale 1.2, stream copy) and checks that every frame
  still decodes, that there are 77 and that the rate is 25/1. Anything else fails the render.
- **One render, two clips**: FVRA and FVFA of a pair share the render; the mux gives FVRA the real sound and
  FVFA the converted voice. The render has no sound.
- **Failures**: generate is retried once; a render that still fails gets `<render_key>.failed.json` and its
  whole pair is dropped (settings.refused_pair_ids). Its preprocessing materials are kept, and reused on the next run.

## Settings

From configs/animate.yaml, the same for every render:

    resolution_area      832 480 (624 x 624 here)
    frame_num            77 (one chunk)
    refert_num           1
    sample_steps         20
    sample_shift         5.0
    sample_guide_scale   1.0
    prompt               "A photorealistic video of a person speaking directly to the camera in a television
                         interview, natural skin texture, sharp focus, even natural lighting, steady framing,
                         realistic detail" (left out, generate.py would use its generic demo prompt)
    base_seed            per render: blake2b of "42:<render_key>", below 2**31 (generate.py draws a random one
                         when it is left out)
    model                --convert_model_dtype, --offload_model True; --size 1280*720 is a placeholder that
                         generate.py requires, the real size comes from preprocessing

## Code and weights

The code: the same Wan2.2 clone as S2V, in the git-ignored external-repos/, at the commit MOZAIC used:

    git clone https://github.com/Wan-Video/Wan2.2 external-repos/wan2.2
    git -C external-repos/wan2.2 checkout 42bf4cfaa384bc21833865abc2f9e6c0e67233dc

Its own environment, from requirements.txt in this folder: S2V's packages plus the preprocessing ones
(onnxruntime-gpu, sam2, decord, moviepy). flash-attn is required; numpy must stay below 2.

    uv venv .venv_animate --python 3.12
    uv pip install --python .venv_animate/bin/python -r src/mozaic/data_generation/generators/animate/requirements.txt

The preprocessing runs its ONNX pose and detector models with onnxruntime-gpu. A CUDA 11 build of it (such as
1.18.1) needs CUDA 11 libraries the venv lacks, or those models run on the CPU, silently: unzip the `nvidia/*/lib`
folders of the nvidia-*-cu11 wheels (cublas 11.11.3.6, cudnn 8.9.2.26, curand 10.3.0.86, cufft 10.9.0.58,
cuda_runtime 11.8.89) into WA_ONNX_CUDA11_LIBS, which run.py adds to LD_LIBRARY_PATH. A CUDA 12 build
(1.19 or later; MOZAIC's environment had 1.22.0) needs only the venv's own libraries.

The weights (about 62 GB), at the revision MOZAIC used; the Flax, TensorFlow and ONNX copies of xlm-roberta-large
are never read:

    hf download Wan-AI/Wan2.2-Animate-14B --revision cb93a225fbaf1ca100f54e79da8f994995b689b3 \
        --local-dir <weights>/wan2.2-animate-14b --exclude "*.msgpack" "*.h5" "*/onnx/model.onnx_data"

    diffusion_pytorch_model-0000{1..4}-of-00004.safetensors   the 14B model
    models_t5_umt5-xxl-enc-bf16.pth                           text encoder (umT5-XXL), google/umt5-xxl/ its tokenizer
    models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth   CLIP picture encoder, xlm-roberta-large/ its tokenizer
    relighting_lora/, relighting_lora.ckpt                    the relighting LoRA (replace mode)
    Wan2.1_VAE.pth                                            video VAE
    process_checkpoint/det/yolov10m.onnx                      person detector       (preprocessing)
    process_checkpoint/pose2d/vitpose_h_wholebody.onnx/       body, face and hands  (preprocessing)
    process_checkpoint/sam2/sam2_hiera_large.pt               segmentation, replace mode only

WA_CKPT names that folder. Set HF_HOME to a folder with room for the download.

## Run

One 80 GB GPU per render (run.py stops before the first render on a smaller card; --skip-gpu-check skips that).
ffmpeg and ffprobe must be on the PATH (MOZAIC was made with ffmpeg 4.4.4). After the crops (crop/), in the core
environment, the job list:

    python -m mozaic.data_generation.generators.animate.make_jobs --build-id <build>
    python -m mozaic.data_generation.generators.animate.make_jobs --build-id <build> --pair-ids-file <pair_ids.txt>

It refuses a build whose manifest has no PASSing validate_report.json, checks every input and writes the list
to <data>/new_generated/manifests/<build>/animate_jobs<suffix>/jobs.jsonl (suffix `_pairs<N>_<sha8>` for a
--pair-ids-file). Each job names its face photo (<data>/new_generated/refs/<render_key>.png, written by
run.py) and its render (<data>/new_generated/renders/<render_key>.mp4). Options for tests only, none used for
MOZAIC: `--limit N` (N jobs taken in turn from each mode, `_micro<N>`), `--only-mode`, `--force-mode`, `--prompt`,
`--sample-steps`, `--refert-num`, `--no-relighting`, `--jobs-suffix`, `--ref-mode fixed`.

Then the renders (about 6 minutes each), WA_OUT a scratch folder for the preprocessing materials
(WA_OUT/proc/<render_key>, deleted after each render; `--proc-root` sets another, `--keep-proc` keeps them), and
WA_ONNX_CUDA11_LIBS too with a CUDA 11 onnxruntime-gpu (the script makes every path absolute: both steps run
inside the clone):

    mkdir -p logs
    WA_CKPT=<weights>/wan2.2-animate-14b WA_OUT=<scratch> \
        sbatch --array=0-1 --gres=gpu:1 --cpus-per-task=6 --mem=40G --time=12:00:00 \
        scripts/data_generation/render_animate.sh <data>/new_generated/manifests/<build>/animate_jobs/jobs.jsonl

Per render: `<render_key>.mp4`, its sidecar `<render_key>.json` (settings, seed, frame counts, peak GPU memory,
time per step, the clips it serves) and `<render_key>.done`, written last. A rerun skips finished renders, and
redoes one whose sidecar names another frame count.

Last, the mux (final_clips/make_jobs.py, then final_clips/mux.py) puts the sound on and
gives every clip, real or fake, the same size, picture levels, loudness and container.
