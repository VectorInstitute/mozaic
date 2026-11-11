# inputs

What generation starts from: the list of VoxCeleb2 clips, the split of people into train, val and
test, and the pair draw of each build (its pair list, or manifest), plus the voice clips that
the voice conversion reads.

    inputs/
    ├── list_clips.py                one shard of people -> one ffprobe line per clip
    ├── join_clip_lists.py           shard lines + gender -> the clip list
    ├── make_identity_split.py       clip list -> each person's split
    ├── draw_pairs.py                split + clip list -> one build's pair list and its summary
    ├── check_pairs.py               checks a pair list before any GPU time is spent
    ├── make_audio_extract_list.py   pair list -> the voice clips its conversions need
    └── extract_voices.py            those clips' sound -> wav files

## Run

In this order. Paths are relative to the `data` folder of configs/paths.yaml; the VoxCeleb2
download is its `voxceleb2` folder. ffmpeg and ffprobe must be on the PATH, and these steps need
pandas and pyarrow.

To re-make the release, skip steps 1 and 2 and start from the saved clip list and split (see
"Facts" below).

1. The clip list. Shards are runs of whole people and can run at the same time (60 here); then
   join them:

       python -m mozaic.data_generation.inputs.list_clips --shard-index 0 --num-shards 60   # ... up to 59
       python -m mozaic.data_generation.inputs.join_clip_lists --num-shards 60 --strict

2. The split:

       python -m mozaic.data_generation.inputs.make_identity_split

3. The five pair lists of the release draw. bulk_v1_wan77 draws the people; the other four take
   its rings (`--rings-from`), so all five have the same people:

       D="--splits train val test --rings-per-split-gender train=70 val=15 test=15 \
          --reserve-rings-per-split-gender train=14 val=3 test=3 --eligibility-frames 153 77"
       M="python -m mozaic.data_generation.inputs.draw_pairs"
       $M --build-id bulk_v1_wan77  $D --crop-frames 77  --video-method animate
       $M --build-id bulk_v1_ff77   $D --crop-frames 77  --video-method facefusion --rings-from bulk_v1_wan77
       $M --build-id bulk_v1_s2v77  $D --crop-frames 77  --video-method s2v        --rings-from bulk_v1_wan77
       $M --build-id bulk_v1_ff153  $D --crop-frames 153 --video-method facefusion --rings-from bulk_v1_wan77
       $M --build-id bulk_v1_s2v153 $D --crop-frames 153 --video-method s2v        --rings-from bulk_v1_wan77

   Seed (42), ring size (5) and the arm fractions are the run.yaml defaults (`--seed`,
   `--ring-size`, `--same-speaker-frac`, `--self-drive-frac`, `--vocoded-real-frac` change them).
   `--out-dir` writes somewhere else. A written pair list is never overwritten: a change is a new
   build id.

4. Check each pair list; it must say PASS before any later step:

       python -m mozaic.data_generation.inputs.check_pairs --build-id bulk_v1_wan77   # and the other four

   `--skip-duration-check` skips C8 (the only check that reads the clip list); `--manifest` and
   `--out` check another file.

5. The voice clips of each build: list them, then take their sound out of the mp4s (a wav that
   exists is kept, so the five builds can share them):

       python -m mozaic.data_generation.inputs.make_audio_extract_list --build-id bulk_v1_wan77
       python -m mozaic.data_generation.inputs.extract_voices \
           --clip-ids-file <data>/new_generated/manifests/bulk_v1_wan77/voice_ref_clips.txt

   `make_audio_extract_list --pair-ids-file F` lists only the pairs in F (a run on a batch of
   pairs); `--missing-only` leaves out the wavs that exist. `extract_voices` also takes
   `--clip-ids ID ...`, or `--shard-index I --num-shards N` for every usable clip of a shard of
   people.

## Inputs and outputs

    in   <voxceleb2>/dev/mp4/<id>/<video>/<clip>.mp4                    the VoxCeleb2 dev clips
         <voxceleb2>/meta/vox2_meta.csv                                 each person's gender
    out  voxceleb2/_logs/inventory_shard<i>_of_<n>.jsonl (+ .done)      one ffprobe line per clip
         voxceleb2/_reports/inventory.parquet                           the clip list: one row per clip, with
                                                                        duration_s, gender and drop_reason
         voxceleb2/split_map.json                                       person -> train, val or test ("groups")
         new_generated/manifests/<build_id>/manifest.jsonl              one row per clip to make
         new_generated/manifests/<build_id>/manifest_summary.json       counts, and the draw: every ring, built and reserve
         new_generated/manifests/<build_id>/validate_report.json        PASS, or every violation
         new_generated/manifests/<build_id>/voice_ref_clips.txt         the voice clips the conversions open
         voxceleb2/audio/<id>/<video>/<clip>.wav                        16 kHz mono 16-bit PCM
         voxceleb2/_logs/audio_shard<i>_of_<n>.jsonl (+ .done)          shard mode only

manifest_summary.json also has five fields that are always empty here (`excluded_driver_clips_of`,
`reserve_file`, `reserve_sha256`, `parent_build`, `pairs_from`) and `draw.role` = `primary`: they
belong to build modes this code leaves out, and stay so the file is the same as the release's.

## Settings

From configs/run.yaml:

- `seed` (42): the split and every random choice of the draw.
- `inventory`: `min_clip_duration_s` (3.0), `ffprobe_timeout_s` (also the ffmpeg time limit of
  the audio step), `concurrency`.
- `split`: `ratio` (0.70, 0.10, 0.20 for train, val, test), `stratify_key` (gender).
- `audio`: `sample_rate_hz` (16000), `channels` (1), `extract_concurrency`.
- `pairs`: `ring_size` (5), `ring_genders` (m, f), `folders_per_celeb` (4), `clips_per_folder` (2),
  `min_driver_clip_margin_s` (0.5), `same_speaker_arm_frac` (0.10), `self_drive_control_frac`
  (0.10), `vocoded_real_frac` (1.0).
- `clips`: `fps` (25), `crop_frames` (77, the default length), `crop_start_s` (0.0).

From the generator files: `generator_version` and `render_key_prefixes` are written into the rows;
animate.yaml `frame_num` and `refert_num` and facefusion.yaml `sec_per_render` only feed the cost
estimates.

## Facts

- **Usable clip**: ffprobe reads it, it has sound and it lasts at least 3.0 s. Other clips stay in
  the clip list with a drop_reason (`corrupt`, `too_short`, `no_audio`) and are never used.
- **The split** is made once, before any pair, with seed 42. Whole people: a person is in one split
  only. The splits are filled with 70/10/20 of the usable clips of each gender; people with no
  usable clip are left out. Every face and voice a clip borrows comes from its own split.
- **Rings of 5 people of the same gender**, inside one split (a cross-gender pair could be told
  apart by pitch alone). Each person gives 4 folders (recording sessions) x 2 clips = 8 pairs per
  build. Person i's folder f borrows from ring-mate (i + f + 1) mod 5, so each person borrows from
  each of the other four once, and never from themselves. The borrowed person's voice clips come
  from their other folders, never from the folder of the face photo; up to 40 are listed, and
  Seed-VC uses the first.
- **The release draw**: 70, 15 and 15 rings per gender in train, val and test, so 200 rings =
  1,000 people (700, 150, 150) and 8,000 pairs per build. The same one random sample per (split,
  gender) also gives 14, 3 and 3 reserve rings per gender = 200 more people, recorded in the
  summary. The release never used a reserve ring, but keep the counts: the built rings are the
  first part of that one sample, and Python's random.sample picks its method from the sample
  size, so in some cells another reserve count would also draw other built rings.
- **Two lengths**: 77 frames (3.08 s) or 153 frames (6.12 s) at 25 fps, from 0.0 s; a driver clip
  must be 0.5 s longer than its crop (3.58 s, 6.62 s). With `--eligibility-frames 153 77` a folder
  counts only if it can give 2 clips to each length (2 clips of at least 6.62 s and 4 of at least
  3.58 s), and a person needs 4 such folders. The 153-frame clips are picked first, the 77-frame
  clips from the rest: the 77- and 153-frame builds have the same people, borrowed person, face
  clip and voice clips per folder, but never the same driver clip. Wan2.2-Animate makes 77-frame
  clips only.
- **Rows per pair**: the 4 classes (RVRA, RVFA, FVRA, FVFA), plus vocoded twins of RVRA and FVRA
  (the real voice through the vocoder; every pair), plus a same-speaker RVFA (the driver's own
  voice from another folder) and a self-drive FVRA (the driver re-made as themselves), each in 10%
  of the pairs. A release build has 49,600 rows: 32,000 main, 8,000 + 8,000 vocoded, 800 + 800.
- **Same bytes**: the build id is in every row and the rows follow the `--splits` order, so use the
  build ids and the commands above exactly (`--splits train val test`, `--rings-from` given as the
  build id).
- **Start from the saved clip list.** list_clips writes the clips in the order its ffprobe
  threads finish, so every new listing of VoxCeleb2 has a new row order. The split (its seeded
  shuffle starts from the order people first appear) and the draw (voice clips are shuffled from
  the clip order) both depend on that order. A new listing gives a valid split and draw, but not
  the release's: to get the release's, start from its inventory.parquet and split_map.json.
