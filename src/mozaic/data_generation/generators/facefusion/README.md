# FaceFusion

[FaceFusion](https://github.com/facefusion/facefusion) 3.7.1 (OpenRAIL-AS), running InsightFace's
inswapper_128 face swapper (fp16; its licence: non-commercial). It puts one person's face on the person in the
driver's clip and keeps everything else of the clip. One of MOZAIC's three video generators: a build renders
with one of them, named in its manifest (video_method `facefusion`).

    facefusion/
    ├── make_jobs.py    the build's job list, and each render's face photo (core environment)
    └── run.py          the swaps, one shard per GPU task (FaceFusion's environment; imports nothing from mozaic)

Settings live in data_generation/configs/facefusion.yaml.

## What it does to a clip

- **Input**: the driver's crop (crop/: 224 x 224, 25 fps, 77 or 153 frames) and one face photo of the other
  person: a frame of their own VoxCeleb2 clip, picked by reference_frame.py (of 24 frames spread over the clip,
  the best score of sharpness x eyes found, on a frontal face).
- **Output**: the same crop with only the face swapped. Head motion, expression, lips, background and light
  are the driver's; the size, frame count and frame rate are the crop's. The sound of the file is the crop's
  own (FaceFusion re-encodes it to AAC); the mux in final_clips/ replaces it anyway.
- **One render, two clips**: FVRA and FVFA of a pair share the render (render_key `ffcross__<crop_key>__face_<id>`);
  the mux gives FVRA the real sound and FVFA the converted voice. The self-drive control
  (`ffself__<crop_key>`) swaps the driver's own face (a frame of the same clip) onto their crop.
- **Refusals**: FaceFusion runs a content check on every target video before it swaps: one frame per second
  (frames 0, 25, 50 and 75 of a 77-frame crop) through three NSFW classifiers; when more than 10% of those
  frames are flagged (by two of the three), it refuses the whole video, so one flagged frame is enough. It
  checks its own analyser code at start-up, so the check cannot be turned off. A refused render gets
  `<render_key>.refused.json` and is never tried again, and its whole pair is dropped: no class of it
  ships (settings.refused_pair_ids). Any other failure is retried once, then gets `<render_key>.failed.json`, with the
  same effect.

## Settings

From configs/facefusion.yaml, the same for every render:

    face swapper           inswapper_128_fp16
    face detector          yolo_face (yoloface_8n), score 0.5
    face enhancer          none (a second model would sharpen the fakes only)
    pixel boost            256x256 (the 256 px face is split into four 128 px images of every second pixel,
                           each swapped by the 128 px swapper, then woven back together)
    output video quality   95
    output sound           aac (replaced by the mux)
    execution              CUDA, 4 threads

Every other setting is FaceFusion's default (its facefusion.ini is left empty). Each render has its own
--jobs-path and --temp-path, so shards running at once do not collide. FaceFusion takes no seed: nothing in the
swap is drawn at random.

## Code and weights

The code, cloned into the git-ignored external-repos/ at the commit MOZAIC used:

    git clone https://github.com/facefusion/facefusion external-repos/facefusion
    git -C external-repos/facefusion checkout 3f81a8a78454089d720b8f318a12ae1702c4633b

Its own environment, from requirements.txt in this folder (FaceFusion needs numpy 2, the Wan models numpy 1):

    uv venv .venv_facefusion --python 3.12
    uv pip install --python .venv_facefusion/bin/python \
        -r src/mozaic/data_generation/generators/facefusion/requirements.txt

onnxruntime-gpu takes CUDA 12 and cuDNN 9 from the venv's nvidia-*-cu12 wheels; run.py puts their folders on
LD_LIBRARY_PATH for FaceFusion and stops if the GPU still cannot be used (on the CPU the swaps would differ).

The weights: FaceFusion downloads its own models from its release pages (GitHub facefusion/facefusion-assets,
or the facefusion/models-* repos on Hugging Face) into external-repos/facefusion/.assets/models. Fetch the whole set
once (about 39 GB), so jobs read local copies:

    cd external-repos/facefusion && ../../.venv_facefusion/bin/python facefusion.py force-download

The swap uses inswapper_128_fp16.onnx, yoloface_8n.onnx, 2dfan4.onnx and fan_68_5.onnx (landmarks),
arcface_w600k_r50.onnx (face embedding), fairface.onnx, nsfw_1/2/3.onnx (the content check), and
xseg_1.onnx and bisenet_resnet_34.onnx (face masks), which FaceFusion's start-up check also needs.

## Run

ffmpeg and ffprobe must be on the PATH (MOZAIC was made with ffmpeg 4.4.4). After the crops (crop/), in the
core environment, the job list; FF_UPSTREAM names the clone, whose commit goes into every job:

    export FF_UPSTREAM=external-repos/facefusion
    python -m mozaic.data_generation.generators.facefusion.make_jobs --build-id <build>
    python -m mozaic.data_generation.generators.facefusion.make_jobs --build-id <build> --pair-ids-file <pair_ids.txt>

It refuses a build whose manifest has no PASSing validate_report.json, checks every input, writes each render's
face photo to <data>/new_generated/refs/<render_key>.png and the list to
<data>/new_generated/manifests/<build>/facefusion_jobs<suffix>/jobs.jsonl (suffix `_pairs<N>_<sha8>` for a
--pair-ids-file).

Then the swaps: one GPU per array task (24 GB is enough, about 25 s per render; the release's swaps were
made on A100 GPUs, a few on A40: the same bytes need the same GPU type), FF_OUT the build's renders
folder, where the mux looks (the script makes every path absolute: FaceFusion runs inside its clone):

    mkdir -p logs
    FF_OUT=<data>/new_generated/renders sbatch --array=0-3 --gres=gpu:1 --cpus-per-task=6 --mem=24G --time=4:00:00 \
        scripts/data_generation/render_facefusion.sh <data>/new_generated/manifests/<build>/facefusion_jobs/jobs.jsonl

Per render: `<render_key>.mp4`, its sidecar `<render_key>.json` (settings, command, face photo, upstream
commit, the clips it serves) and `<render_key>.done`, written last. A rerun skips finished and refused renders.

Last, the mux (final_clips/make_jobs.py, then final_clips/mux.py) puts the sound on and
gives every clip, real or fake, the same size, picture levels, loudness and container.
