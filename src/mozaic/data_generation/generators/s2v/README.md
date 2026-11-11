# Wan2.2-S2V

Wan-AI's [Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) (Apache-2.0), a speech-to-video model:
one photo of a person and one speech recording become a video of that person saying it, with lip sync and head
motion. It runs with the authors' code ([Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2), generate.py).
One of MOZAIC's three video generators: a build renders with one of them, named in its manifest (video_method
`s2v`).

    s2v/
    ├── make_jobs.py    the build's job list, and each render's face photo (core environment)
    └── run.py          the renders and the retime to 25 fps, one shard per GPU task (S2V's environment;
                        imports nothing from mozaic)

Settings live in data_generation/configs/s2v.yaml.

## What it does to a clip

- **Input**: one face photo of the other person (a frame of their own VoxCeleb2 clip, picked by
  reference_frame.py, as for the other video generators) and the driver's real speech (the crop's wav, 77 or
  153 frames long). Nothing of the driver's video is used: the face, lips, head motion and background of the
  render all come from the model, from the photo and the sound.
- **Size**: S2V keeps the photo's shape and works at a multiple of 64 within the --size area (832*480 is only
  the largest area allowed), so our 224 x 224 photo is enlarged to 256 x 256, and the render is 256 x 256. The
  mux in final_clips/ resizes every clip to 224 x 224.
- **Length and frame rate**: S2V makes 80-frame chunks at 16 fps (5 s each) and as many chunks as the job asks
  for: enough to cover the crop (1 for a 77-frame crop of 3.08 s, 2 for 153 frames). run.py keeps that file as
  `<render_key>__native16.mp4` and retimes it to 25 fps by repeating frames (ffmpeg fps=25: nothing is invented,
  the same input gives the same output), keeps the first 77 (or 153) frames, video only, x264 crf 10. The
  repeated frames leave a 16-in-25 cadence; it is written in each render's sidecar.
- **One render, two clips**: FVRA and FVFA of a pair share the render (render_key `s2vcross__<crop_key>__face_<id>`);
  the mux gives FVRA the real sound it was made from and FVFA the converted voice of the same words. The
  self-drive control (`s2vself__<crop_key>`) is the driver's own photo with their own speech.
- **Failures**: a failed render is retried once, then gets `<render_key>.failed.json` and its whole pair is
  dropped (settings.refused_pair_ids). A render whose retime failed redoes only the retime on the next run.

## Settings

From configs/s2v.yaml, the same for every render:

    size              832*480 (the area limit; 256 x 256 here)
    infer_frames      80 per chunk, at 16 fps
    sample_steps      40
    guide_scale       4.5
    start_from_ref    false (true would put the real photo as frame 0)
    prompt            "A person speaking directly to the camera in a medium close-up shot. Natural facial
                      expressions and lip movements synchronized to the speech, subtle head motion, steady
                      lighting, realistic skin texture, static background."
    base_seed         per render: blake2b of "42:<render_key>", below 2**31 (generate.py draws a random one
                      when it is left out)
    model             --convert_model_dtype, --offload_model True (the parts take turns on the GPU)

## Code and weights

The code, cloned into the git-ignored external-repos/ at the commit MOZAIC used (Wan-Animate uses the same clone):

    git clone https://github.com/Wan-Video/Wan2.2 external-repos/wan2.2
    git -C external-repos/wan2.2 checkout 42bf4cfaa384bc21833865abc2f9e6c0e67233dc

Its own environment, from requirements.txt in this folder. flash-attn is required: the S2V modules call it
directly, with no fallback. numpy must stay below 2 for the Wan modules.

    uv venv .venv_s2v --python 3.12
    uv pip install --python .venv_s2v/bin/python -r src/mozaic/data_generation/generators/s2v/requirements.txt

The weights (about 46 GB), at the revision MOZAIC used; the Flax and TensorFlow copies are never read:

    hf download Wan-AI/Wan2.2-S2V-14B --revision dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b \
        --local-dir <weights>/wan2.2-s2v-14b --exclude "*.msgpack" "*.h5"

    diffusion_pytorch_model-0000{1..4}-of-00004.safetensors   the 14B model (bf16)
    models_t5_umt5-xxl-enc-bf16.pth                           text encoder (umT5-XXL), google/umt5-xxl/ its tokenizer
    wav2vec2-large-xlsr-53-english/                           sound encoder
    Wan2.1_VAE.pth                                            video VAE

WAN_CKPT names that folder. Set HF_HOME to a folder with room for the download.

## Run

One 80 GB GPU per render (the 14B model on one GPU). ffmpeg and ffprobe must be on the PATH (MOZAIC was made
with ffmpeg 4.4.4). After the crops (crop/), in the core environment, the job list:

    python -m mozaic.data_generation.generators.s2v.make_jobs --build-id <build>
    python -m mozaic.data_generation.generators.s2v.make_jobs --build-id <build> --pair-ids-file <pair_ids.txt>

It refuses a build whose manifest has no PASSing validate_report.json or is not an s2v build, checks every
input, writes each render's face photo to <data>/new_generated/refs/<render_key>.png and the list to
<data>/new_generated/manifests/<build>/s2v_jobs<suffix>/jobs.jsonl (suffix `_pairs<N>_<sha8>` for a
--pair-ids-file; `--limit N` keeps the first N jobs, `_first<N>`).

Then the renders (about 4.5 minutes per chunk), WAN_OUT the build's renders folder, where the mux looks (the
script makes every path absolute: generate.py runs inside the clone):

    mkdir -p logs
    WAN_CKPT=<weights>/wan2.2-s2v-14b WAN_OUT=<data>/new_generated/renders \
        sbatch --array=0-3 --gres=gpu:1 --cpus-per-task=6 --mem=55G --time=8:00:00 \
        scripts/data_generation/render_s2v.sh <data>/new_generated/manifests/<build>/s2v_jobs/jobs.jsonl

Per render: `<render_key>.mp4` (25 fps), `<render_key>__native16.mp4`, its sidecar `<render_key>.json`
(settings, command, seed, retime facts, the clips it serves) and `<render_key>.done`, written last. A rerun
skips finished renders.

Last, the mux (final_clips/make_jobs.py, then final_clips/mux.py) puts the sound on and
gives every clip, real or fake, the same size, picture levels, loudness and container.
