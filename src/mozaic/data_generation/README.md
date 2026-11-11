# data_generation

The code that made the MOZAIC clips from VoxCeleb2.

For each pair of two same-gender people, one real clip of the first person (the driver) is
turned into four clips:

| Label | Video | Voice |
|---|---|---|
| RVRA | the driver's real clip | the driver's real voice |
| RVFA | the driver's real clip | the driver's words in the other person's voice (Seed-VC) |
| FVRA | the other person's face put into the clip (FaceFusion, Wan2.2-S2V or Wan2.2-Animate) | the driver's real voice |
| FVFA | the same fake video as FVRA | the same fake voice as RVFA |

Every real voice also gets a vocoded copy (through BigVGAN, the vocoder Seed-VC ends in), so
real and fake voices share the vocoder's trace.

## Folders

    data_generation/
    ├── run_generation.py   one command: prints (or runs) every step in order
    ├── settings.py         every setting and file name, read from configs/
    ├── configs/            run.yaml (the release's settings), one yaml per generator, paths.example.yaml
    ├── inputs/             VoxCeleb2 clip list, identity split, pair draw and its checks, voice references
    ├── crop/               cuts each clip to exactly 77 or 153 frames
    ├── generators/         facefusion/, s2v/, animate/ (fake video) and seed_vc/ (fake voice, vocoded copies)
    ├── final_clips/        the one mux that makes every final clip; removes half-made pairs
    ├── gates/              face, voice, picture and lip-sync checks; the keep/drop rule
    └── release/            turns the kept clips into the release folder

Each folder has its own README with the details of its step. The files the code writes keep the
names of the release run (for example `terminal_jobs.jsonl`, `identity_gate.json`), so the run's
folders read the same.

## Setup

1. Copy `configs/paths.example.yaml` to `configs/paths.yaml` and set two folders: the VoxCeleb2
   download and the folder generation writes to.
2. This repo's environment needs `requirements.txt` (numpy, opencv-python, pandas, pyarrow,
   pyyaml) and ffmpeg 4.4 on PATH.
3. Each generator runs in its own environment, made from its `requirements.txt` (the exact
   versions that made the clips; the install command is at the top of each file, with uv); its
   code goes into the git-ignored `external-repos/` at a fixed commit, and its weights are
   downloaded as its README says. Model code and weights are not part of this repo.
4. The gates' voice check runs in one more environment, made from `gates/requirements_voice.txt`
   (default `.venv_voice`), and their lip-sync check in another, made from
   `gates/requirements_syncnet.txt` (default `.venv_syncnet`) with SyncNet's code
   (`gates/README.md`); their face check uses the FaceFusion one.

## Running it

    python -m mozaic.data_generation.run_generation prepare              # clip list, split, the 5 pair lists
    python -m mozaic.data_generation.run_generation batch \
        --build-id bulk_v1_ff77 --pair-ids-file pairs.txt                # one batch of one build
    python -m mozaic.data_generation.release.build_mozaic plan           # then copy, fix, voice, drop, manifests, verify

Without `--run` the commands are only printed. A batch runs: crop, voice references, render,
voice conversion, vocoded copies, mux, removal of half-made pairs, gates (`--batch` names the gate
decisions; default: the pair list's file name). Render and voice conversion need a GPU and the
variables that name each generator's code and weights (its README); `scripts/data_generation/`
has example SLURM jobs for them. Every step skips work it has already done, so a stopped batch is
resumed by running it again.

The release has five builds, one per video generator and length: `bulk_v1_ff77`,
`bulk_v1_s2v77`, `bulk_v1_wan77` (77 frames, 3.08 s) and `bulk_v1_ff153`, `bulk_v1_s2v153`
(153 frames, 6.12 s; Wan2.2-Animate only at 77 frames). They share the same people and
pairs (seed 42).

## What is exact, and what is not

- The clip list, the split, the five pair lists and their checks come out byte for byte the
  same as the release's.
- Each render and voice conversion has its own fixed seed, taken from the job's name, so a
  job gives the same output in any batch.
- Crops and the mux are byte-identical with ffmpeg 4.4 (x264 core 164). Renders and voices made
  from these requirements files, the pinned model code and the same weights come out
  byte-identical on the GPU type that made them (A100 for FaceFusion, S2V and Animate; A40 for
  Seed-VC). FaceFusion picks its GPU kernels by timing, so even there a swap can differ very
  slightly between runs (about 55 dB PSNR). On another GPU type renders differ slightly (a
  FaceFusion swap on an A40: about 46 dB PSNR from the A100 one), which can move a few pairs near
  a limit. A vocoded copy's samples are exact, but its WAV header holds the time it was written.
- The face and picture scores of the gates come out exact. The voice and lip-sync scores run on the CPU: on
  another CPU type, or with another ffmpeg build decoding the AAC sound, they can differ slightly (voice: at most
  0.0003 in a re-check of five release batches; lip sync: at most 0.001, with the same offsets); no keep/drop
  decision changed.
- Some steps of the original run depended on the machine they ran on:
  - Face photos: when no frame of a clip can be scored, the photo is the frame at 1.0 s.
    Re-running the job makers today picks a scored frame for 537 of the 8,800 Wan2.2-Animate
    renders (6%) where the original run fell back, because its busy machine could not read the
    frames; 57 of 17,160 S2V photos also differ; every FaceFusion photo is the same.
  - 45 released voices were made on a CPU because a GPU was faulty; the vocoded copies were all
    made on a CPU.
