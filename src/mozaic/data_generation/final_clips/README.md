# final_clips

The last step of every clip. One mux makes every final `.mov`, real and fake alike; then a pair
that lost one of its clips is taken out whole.

    final_clips/
    ├── make_jobs.py                   pair list -> job list: the video and the audio of each final clip
    ├── mux.py                         the one mux: length, picture, loudness, encode, and a check of every output
    └── withdraw_incomplete_pairs.py   moves aside a pair that shipped only some of its classes

## Run

Paths are relative to the `data` folder of configs/paths.yaml.

    python -m mozaic.data_generation.final_clips.make_jobs --build-id <build_id> --out-root <data>/new_generated/final_<build_id>
    python -m mozaic.data_generation.final_clips.mux --jobs <data>/new_generated/manifests/<build_id>/terminal_jobs.jsonl
    python -m mozaic.data_generation.final_clips.withdraw_incomplete_pairs --build-id <build_id> --final-dir <data>/new_generated/final_<build_id> --apply

Each build is muxed into its own folder, as above: sample ids carry no build id, so two builds in
one folder would overwrite each other. Without `--out-root` the clips go to `new_generated/final/`.
A run on part of a build passes the same `--pair-ids-file` to all three.

Other options:

- make_jobs: `--classes`, `--arms` (only these), `--out` (job list path),
  `--allow-missing` (write jobs even when an input is missing; not for real runs).
- mux: `--overwrite`, `--no-audit` (skip the extra audio-length probe),
  `--out-dir` (root for relative `out` paths), `--refine-passes N` (this run only); one clip with
  `--video V --audio A --out O [--frames N]`; `--null-test --video V --audio A [--workdir D]`.
- withdraw_incomplete_pairs: without `--apply` it only reports; `--withdraw-pairs FILE --reason TEXT`
  withdraws the listed pairs whole (every arm), e.g. pairs the selection rule rejected.

## Inputs and outputs

make_jobs

    in   new_generated/manifests/<build_id>/manifest.jsonl
         new_generated/manifests/<build_id>/validate_report.json     must say PASS
    out  new_generated/manifests/<build_id>/terminal_jobs.jsonl      one job per final clip
         new_generated/manifests/<build_id>/terminal_jobs_skipped.jsonl
                                                                      rows of pairs that lost a render or a voice

The row decides which two streams make a clip (its video_status and audio_variant), never a file name:

| clip | video | audio |
|---|---|---|
| RVRA | crops/<crop_key>.mp4 | crops/<crop_key>.wav |
| RVFA | crops/<crop_key>.mp4 | conversions/<conversion_key>.wav |
| FVRA | renders/<render_key>.mp4 | crops/<crop_key>.wav |
| FVFA | renders/<render_key>.mp4 | conversions/<conversion_key>.wav |
| RVRA, FVRA vocoded | as above | vocoded/<crop_key>.wav |
| FVRA self-drive | renders/<render_key>.mp4 (the driver re-made as themselves) | crops/<crop_key>.wav |

(all under `new_generated/`). A render or voice that its generator refused or failed on twice
has a `.refused.json` or `.failed.json` beside it; that pair ships in no class. Every input is
checked to exist before the mux runs.

mux

    out  <final_dir>/<class>/<sample_id>.mov     the final clip
         <final_dir>/<class>/<sample_id>.json    its record: the settings, and this clip's own
                                                 measurements, corrections and checks
         <jobs>_dropped.jsonl                    clips dropped by design, with the reason

withdraw_incomplete_pairs

    out  <final_dir>/_withdrawn/<pair_id>/       the pair's clips and records, moved (never deleted),
                                                 with reason.json naming the clips that never arrived

## What the mux does

Every clip of every class goes through the same mux. There is no branch on the class: the same
filters, encoder settings and number of passes for every clip. Only numbers measured from the clip
itself (its brightness, colour and loudness) change from clip to clip.

1. Length. The video must already have exactly 77 or 153 frames (the row's crop_frames); the mux
   checks it and never cuts video. The audio may be up to 20 ms off (Seed-VC comes out ~1.5 ms
   short) and is conformed to exactly frames x 640 samples: every clip is cut 160 samples (10 ms)
   short and padded back with silence, so every clip ends in the same 160-sample tail of silence,
   whatever made its audio. No silence is trimmed at the start.
2. Picture. Every clip is resized to 624 x 624 and then to 224 x 224 (bicubic), so real and
   generated clips share one resampling history. Brightness: a gamma on Y brings the mean luma
   (raw 0-255) to 72, tolerance 2. Colour: a gain on U/V brings the mean colour strength,
   sqrt((U-128)^2 + (V-128)^2), to 17, tolerance 0.75. Both are solved from all of the clip's own
   frames, then refined twice against what the encoder really produced.
3. Loudness. Measured with EBU R128 (loudnorm, used only as a meter), then one constant gain
   towards -23 LUFS, capped so the true peak stays at or under -2 dBTP, and at most +30 dB. A clip
   the cap holds back lands under the target, and the shortfall is recorded. No compression.
4. Encode. H.264 (libx264, crf 18, preset medium, yuv420p), 25 fps, 224 x 224; 16 kHz mono 16-bit
   PCM audio in 1024-sample packets; a `.mov` file (bit-exact flags, fast start).
5. Checks. Each output is measured again: frames, 25 fps, 224 x 224, exact sample count, loudness,
   true peak, audio onset (a gain cannot move it, so any shift is an error), brightness and colour.

Dropped, failed or short:

- Dropped by design: the output misses the brightness or colour target although its correction
  could reach it; the true peak ends above the ceiling (by more than the meter's 0.5 dB); a render
  has the wrong number of frames. No clip is written, the drop is recorded and the step still
  exits 0. withdraw_incomplete_pairs then moves the rest of that pair aside, so no pair ships
  only some of its classes (that would let the driver's identity give the label away).
- Kept, short of the target: a correction at its limit (gamma 0.2 to 5, colour gain 0.2 to 4), or
  a colour the picture cannot reach (a source with colour strength under 8, or less than 90% of
  the colour gain taking effect). The shortfall is recorded.
- Failed (exit 1, no clip): anything else, e.g. a crop with the wrong frame count, audio more than
  20 ms off, an encode error, a moved audio onset.

A clip that already exists and opens is skipped and keeps its record; one that exists but cannot be
opened is made again. Each process writes its own temp file (`<clip>.mov.<host>.<pid>.tmp`) and
renames it into place, so two runs over one job list never mix their files.

`--null-test` checks that the mux cannot see the label: the same inputs made twice must match to
the byte, and a slightly changed input must come out different.

The release was made with ffmpeg 4.4.4 (libx264) on PATH; another version can give different bytes.

## Settings

configs/run.yaml, the `mux` section: video_codec, crf, preset, pixel_format, scale_flags,
resample_via; luma (target, tolerance, gamma_min, gamma_max); chroma (target, tolerance,
source_floor, gain_min, gain_max); refine_passes; loudness (integrated_lufs, true_peak_ceiling_db,
max_gain_db); audio (codec, bitrate, frame_samples, tail_pad_samples); container (format, ext).
Also `clips` (fps, size, crop_frames) and `audio.sample_rate_hz`. `luma.probe_frames` is not used
here: the mux measures every frame.
