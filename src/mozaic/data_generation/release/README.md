# release

Turns the finished generation run into the MOZAIC release folder. Nothing in `new_generated/`
is moved, changed or deleted: every released clip is a new file, written from its source clip
and checked against it. Each step leaves a record (plan, journals, reports) in `_mozaic_build/`.

The three folders sit side by side in the `data:` folder of `configs/paths.yaml`:
`new_generated/` (the run), `mozaic/` (the release) and `_mozaic_build/` (the build record).

## The steps

Run them one at a time, in this order. A step refuses to start while the one it builds on is
incomplete (copy needs a READY plan, fix a complete copy, voice a complete fix, manifests all
of them), and copy, fix and voice can be stopped and run again: they go on where they stopped.

1. **plan** (changes no clip). Finds every pair each generator kept, and decides every release file.
   - A pair is kept if its gate decision accepts it and all its clips are there: shipped in its
     family's final folder, or set aside whole in `_withdrawn/<pair>/` by a step other than the
     gates. (In the first batches a matching step set aside pairs that had passed their own
     checks; the release puts them back.)
   - Every kept pair is checked again against the keep/drop limits (face, voice, picture),
     without trusting the gate's own pass flag.
   - Every source clip is fingerprinted: an MD5 of every decoded frame and sound packet, with
     its timestamps.
   - The design rules are checked on those fingerprints. Within a pair: FVRA main, FVRA vocoded
     and FVFA show one video; RVRA main, RVRA vocoded and RVFA show one video; RVRA main and
     FVRA main share the natural voice; the two vocoded clips share the vocoded voice; RVFA and
     FVFA share the fake voice. Across generators: a pair's real clips (RVRA, RVFA) are the same
     in every generator that kept it, since they are stored once.
   - With `--exclude-inconsistent`, each pair-generator entry that breaks a rule is left out;
     without it the plan is BLOCKED. For the release this left out 12 entries: one FaceFusion
     153-frame pair whose three fake-face clips did not show one video, and eleven S2V pairs
     (nine 77-frame, two 153-frame) whose real clips differed from FaceFusion's copy. Those
     pairs keep their other generators.
   - Writes `_mozaic_build/plan/`: every file to release with its source and fingerprint, the
     pairs, the split of every person, the rule breaks, the errors, and the status READY or BLOCKED.

2. **copy --apply**. Writes each planned clip into `mozaic/` under its release name, without
   the encoder's version text. The picture and sound streams are copied as they are (no
   re-encoding); the H.264 SEI messages (where x264 writes its version and settings) and the
   container's metadata are removed, and the `.mov` is written in bit-exact mode. Each new file
   is checked: it decodes to exactly the picture, sound and timestamps of its source, it has the
   standard format (H.264 224x224 at 25 fps, yuv420p; PCM 16 kHz mono; `.mov`), and no version
   text is left.

3. **fix --apply**. Changes every clip in place in the same two ways, so neither carries the label:
   - the same 1:1 pixel aspect ratio tag, in the H.264 stream and in the `.mov` header (before,
     real clips had one and generated clips none);
   - the same start of sound: the first 50 ms are silent, then the sound fades in over 10 ms
     (converted voices used to fade in on their own, real voices did not).

   The video is copied, not re-encoded; the sound is written back as PCM in the same 1024-sample
   packets. Each clip is checked: its picture is exactly its source's, its sound is exactly the
   source sound with this start, its sound packets are the standard ones, both headers say 1:1,
   and no version text is left.

4. **voice** (does not change `mozaic/`). Measures every pair's three voice scores again, on
   the released sound, with the gate's own speaker model and code (ECAPA, SpeechBrain;
   `gates/voice_check.py`): `voice_target` (fake voice vs the target's reference recording),
   `voice_source` (fake voice vs the source's own recording) and `voice_vocoded` (vocoded real
   voice vs the source's recording). 300 pairs (drawn with seed 42) are also scored on the sound
   the gate checked: they must give the gate's numbers within 0.001, or the step fails. The
   report lists the pairs now below the voice floor (0.2496 at 77 frames, 0.3048 at 153 frames).

5. **drop --apply**. Moves the clips of the pairs listed in `_mozaic_build/drop/pairs.json` out
   of `mozaic/` into `_mozaic_build/drop/clips/` (kept, not deleted). From then on every step
   reads the plan without those pairs. The list is written by hand from the voice report
   (`below_floor_pairs` in `_mozaic_build/voice/run.json`), with the reason; only its `pairs`
   field is read. For the release: the 15 pairs whose `voice_target` fell below the voice floor
   on the released sound (141 clips).

       {"reason": "voice_target below the voice floor on the released sound", "pairs": ["<pair_id>", ...]}

6. **manifests**. Writes the dataset's files into `mozaic/` (below), with the voice scores of
   step 4, and `SOURCES.csv` (every released file, its source file and their checksums) into
   `_mozaic_build/`. The dataset card, `README.md` and `DATASHEET.md`, is written by hand, not by
   this code. If it is in `mozaic/` when this step runs, `SHA256SUMS` lists it, so after editing
   the card run manifests (and verify) again.

7. **verify** (changes no clip). Checks the whole release again from the files on disk: every
   clip from scratch against its source (as in steps 2 and 3), the file list (nothing missing,
   nothing extra; the dataset card may be there or not), every `metadata.jsonl` row against the
   plan and `pairs.jsonl`, `metadata_table.parquet` against `metadata.jsonl`, the splits, the
   counts, the voice scores and `SHA256SUMS`. Writes `_mozaic_build/verify/` with PASS or FAIL.

## What comes out

`mozaic/`:

| File | What it holds |
|---|---|
| `videos/<label>/<generator>/<length>/<arm>/*.mov` | the clips |
| `metadata.jsonl` | one row per clip: labels, generators, length, arm, split, whose face and voice, pair, scores, sha256 |
| `metadata_table.parquet` | the same rows, with typed columns |
| `pairs.jsonl` | one row per pair: the two people, the source clip, the target's face and voice clips, the generators that kept it, its scores |
| `splits.json` | person -> train, val or test |
| `summary.json` | the labels and arms, the source, encode settings, generator versions, every count |
| `SHA256SUMS` | the sha256 of every other file; check a copy with `sha256sum -c SHA256SUMS` |

`_mozaic_build/`: `plan/`, `copy/` (journal, and a DONE file per folder), `fix/`, `voice/`,
`drop/`, `SOURCES.csv`, `manifests_run.json` and `verify/`. Every step writes a run record
(`run.json`): when and where it ran, the sha256 of every file in this folder, its counts, and
the ffmpeg used by the steps that run it.

## What it reads

From `new_generated/`:

- `manifests/bulk_v1_<family>/manifest.jsonl`: every clip of the five families, `ff77`, `s2v77`,
  `wan77`, `ff153`, `s2v153` (generator and length).
- `final_bulk_v1_<family>/`: the finished clips, each with a `.json` record of its encode
  settings; `_withdrawn/<pair>/` holds the pairs set aside whole.
- `_reports/bulk_v1_<family>_selection_<batch>/`: each batch's keep/drop decision for every pair
  (a folder renamed with a suffix was set aside and is not read), and
  `_reports/bulk_v1_<family>_selection/`, the first batch's joint decision; plus the gate folder
  each decision names, for the gate's own voice scores (`sv_report.jsonl`).
- `conversions/<key>.json` (the reference recording Seed-VC used), `crops/<key>.wav` (the
  source's own recording) and the VoxCeleb2 clips: for the voice step.
- `manifests/bulk_v1/CODE_COMMIT`: the commit of the code that made the clips, kept in the
  plan's run record (`null` without the file).

The limits come from `settings.GATES` (`configs/run.yaml`, `gates:`): per length (`A` = 77
frames, `B` = 153 frames) `arcface_target_abs_min`, `arcface_target_floor`,
`arcface_excess_cap` and `sv_floor`; then `picture` (`flicker_max` per length,
`melt_ratio_max`, `sharp_ratio_min`) and `family_exceptions` (the checks a generator skips).

## Commands

From the repo root, with ffmpeg 4.4 on PATH (one ffmpeg build writes every release file: the
build refuses any other version, and copy and fix refuse an ffmpeg other than the plan's), in an
environment with numpy, pyyaml and pyarrow:

    python -m mozaic.data_generation.release.build_mozaic plan --exclude-inconsistent
    python -m mozaic.data_generation.release.build_mozaic copy --apply
    python -m mozaic.data_generation.release.build_mozaic fix --apply
    python -m mozaic.data_generation.release.build_mozaic voice            # needs torch + speechbrain
    # write _mozaic_build/drop/pairs.json from the voice report, then:
    python -m mozaic.data_generation.release.build_mozaic drop --apply
    python -m mozaic.data_generation.release.build_mozaic manifests
    python -m mozaic.data_generation.release.build_mozaic verify

- `--workers N`: processes to use (default: every CPU available).
- Without `--apply`, copy, fix and drop only say what they would do.
- On a cluster, each step as a CPU job: `sbatch scripts/data_generation/build_release.sh <step> [options]`.

## Files

| File | What it does |
|---|---|
| `build_mozaic.py` | the seven steps |
| `mozaic_layout.py` | the release rules: folders, file names, ids, `metadata.jsonl` fields, the start-of-sound treatment, the fingerprint, the limits re-check |
