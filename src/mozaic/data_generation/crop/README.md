# crop

Cuts every driver clip of a build to its fixed window. This is the first step of every class:
the real classes ship the crop itself, the video generators start from its video, and the voice
steps (Seed-VC and the vocoder) start from its audio.

    crop/
    └── crop_clips.py   pair list -> one video file and one audio file per driver clip

## Run

    python -m mozaic.data_generation.crop.crop_clips --build-id <build_id>
    python -m mozaic.data_generation.crop.crop_clips --build-id <build_id> --pair-ids-file pairs.txt   # only these pairs
    python -m mozaic.data_generation.crop.crop_clips --build-id <build_id> --shard-index 0 --num-shards 4
    python -m mozaic.data_generation.crop.crop_clips --build-id <build_id> --limit 5                  # a quick try

Other options: `--manifest` (another pair list), `--concurrency` (ffmpeg processes at once,
default 8), `--overwrite` (cut again even when a good crop exists).

## Inputs and outputs

Paths are relative to the `data` folder of configs/paths.yaml; the source clips are in its
`voxceleb2` folder.

    in   <voxceleb2>/dev/mp4/<id>/<video>/<clip>.mp4            the driver clip
         new_generated/manifests/<build_id>/manifest.jsonl      the pair list: crop_key, driver_clip_id, crop_frames
    out  new_generated/crops/<crop_key>.mp4                     video only
         new_generated/crops/<crop_key>.wav                     audio only
         new_generated/_logs/crop_<build_id>.jsonl              one row per clip: ok, frames, samples, error
         new_generated/_logs/crop_<build_id>.done               a run over the whole build had no failure

The crop name is the clip id with `/` replaced by `_`, plus `__f153` for a 153-frame crop.

## What a crop is

- Video: exactly `crop_frames` frames (77 or 153, from the pair-list row) at 25 fps, starting at
  0.0 s, at the clip's own size (VoxCeleb2 clips are 224 x 224), no audio. Lossless H.264
  (`-qp 0`, yuv420p): the mux encodes the final clip, and this file is shared by the real and the
  generated path, so its encoding cannot become a class cue.
- Audio: exactly `crop_frames` x 640 samples, 16 kHz mono, 16-bit PCM wav. 16000 / 25 = 640 samples
  per frame, so the frame count and the sample count end at the same instant.
- Both are cut by count (`-frames:v`, and `atrim` by sample index), never by a float duration, and
  both are probed again before they are kept. A crop with the wrong count is logged as a failure.
- The source must be 25 fps; a clip at another rate is refused, never resampled.
- Video and audio stay in two files, so the one mux (final_clips/) joins them the same way for every class.

## Re-running

- A crop that exists and probes to the right counts is skipped; `--overwrite` cuts it again.
- Each process writes its own temp files (`<crop_key>.mp4.<pid>.tmp`) and renames them into
  place, so a killed run never leaves a crop that looks finished.
- The `.done` marker is written only by a run over the whole build (no shards, `--limit` or
  `--pair-ids-file`) with no failure; such a build is skipped next time unless `--overwrite`.
- Shards append to the log; a single run rewrites it. The exit code is 1 when any clip failed.

## Settings

configs/run.yaml: `clips` (fps 25, size 224, crop_frames 77, crop_start_s 0.0,
crop_concurrency 8) and `audio` (sample_rate_hz 16000, channels 1); `inventory.ffprobe_timeout_s`
for the probes. A build sets `crop_frames: 153` on its rows for the long clips.
