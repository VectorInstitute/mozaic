# gates

Every generated pair is checked before it can enter MOZAIC. A pair is the four clips of one
driver clip: RVRA (real video, real voice), RVFA (real video, fake voice), FVRA (fake video,
real voice) and FVFA (fake video, fake voice). Each video family (Wan-Animate, FaceFusion,
Wan-S2V) and clip length is its own build, and each build is judged on its own.

**Any failed check drops the whole four-class pair for that generator**, so a pair never
ships with only some of its classes. A check that could not be measured (for example no
face was found) is reported as "not measured" and does not drop the pair.

## The checks and their limits

Two clip lengths have their own limits: stratum A (77 frames, 3.08 s) and B (153 frames,
6.12 s). Similarities are cosines between embeddings.

| check | what is compared | keep the pair if |
|---|---|---|
| face: target | ArcFace similarity of the fake face (the FVRA clip, 8 frames) to the target's photo (the still the generator was given) | at least 0.30, at least the floor 0.1442, and higher than the fake face's similarity to the driver |
| face: leftover driver | how much more the fake face resembles the driver than a perfect swap would: (fake face vs real driver) minus (target's photo vs real driver) | at most 0.0977 (FaceFusion skips this check: it swaps only the face, so hair and head shape stay the driver's) |
| voice: target | ECAPA similarity of the converted voice (the shipped RVFA clip) to the target's reference recording | at least 0.2496 (A) / 0.3048 (B) |
| voice: vocoded real | ECAPA similarity of the vocoded real voice (the vocoded RVRA clip) to the natural sound of the same driver clip: the vocoder must keep the speaker | the same limit, 0.2496 (A) / 0.3048 (B) |
| lip sync (a sample) | SyncNet confidence of every clip of the batch's first 20 pairs (10% of a 200-pair batch; the face detected, tracked and cropped by SyncNet's own steps, scored with the clip's own sound) | every scored clip at least 3.411 (A) / 4.1096 (B), and each fake clip at most 3.761 (A) / 3.8225 (B) under its own pair's real clip; the other pairs are not measured for lip sync |
| picture: flicker | brightness pulsing of each fake-video clip (FVRA, FVFA): mean frame-to-frame change of the frame's mean brightness after a 5-frame moving average | at most the real clips' 99th percentile: 0.993 (A) / 0.843 (B) |
| picture: melting | per-pixel frame-to-frame change of the blurred face region, divided by the same number for the same pair's real clip | at most 2.0 times the real clip (S2V skips this check: it invents its own head motion, so a change against the real clip measures motion, not melting) |
| picture: sharpness | Laplacian variance of the blurred face region, divided by the same number for the same pair's real clip | at least 0.5 times the real clip |
| complete group | every main clip of the pair shipped | all four are there |
| refusal | the generator refused the input, or failed on it twice (a `.refused.json` or `.failed.json` next to the render) | no refusal |

Each picture number compares a generated clip with the real clip of the same pair, so
camera motion, focus and sensor grain cancel out. The face check is on the cross render that
FVRA and FVFA share. RVFA keeps the real footage, so its picture is not judged.

The limits come from real clips only, never from a wanted rejection rate. The face limits and
the 77-frame voice and lip-sync limits come from calibration sets made before the main run (240
pairs at 77 frames, 80 at 153 frames: 95th / 5th percentiles); the 153-frame voice and lip-sync
limits and the flicker limits from the real clips of the first batch (5th / 95th / 99th
percentiles). A lip-sync floor is the 5th percentile of real clips' confidence, its margin the 95th
percentile of the difference between two real clips of the same person. Lip sync is a sample
because SyncNet's face detector runs over every frame on the CPU: a whole batch would take hours. Three are judgement
calls: the 0.30 face minimum, and the 2.0 melting and 0.5 sharpness ratios, chosen by looking at
the first batch's clips. They are fixed values here: `configs/run.yaml`, `gates:`.

## The files

| file | what it does | runs in |
|---|---|---|
| `face_check.py` | face scores of every cross render (ceiling, floor, target, leak) | an environment with insightface + onnxruntime (the FaceFusion one) |
| `make_voice_check_list.py` | the voice-gate input: each conversion and a real clip of the same pair | this repo's |
| `make_lip_sync_list.py` | the lip-sync input: every shipped clip of the batch's first 20 pairs | this repo's |
| `picture_check.py` | flicker, melting and sharpness of every shipped main clip | this repo's (needs opencv-python) |
| `voice_check.py` | voice scores: converted clip vs target and vs driver | an environment with speechbrain + torch (`requirements_voice.txt`) |
| `lip_sync_check.py` | lip-sync scores (SyncNet offset and confidence) of each listed clip; a script, run with its environment's python | the SyncNet environment (`requirements_syncnet.txt`, CPU) with SyncNet's code |
| `keep_or_drop.py` | the rule above: keep or drop each pair of one build | this repo's |
| `run_gates.py` | the gate stage of one batch: all of the above, then the withdrawal and the drop ledger | this repo's |

ffmpeg must be on PATH (picture, voice and lip sync). The face models (buffalo_l) are read from
`$INSIGHTFACE_ROOT`, default `~/.insightface`. SyncNet's code, at the commit MOZAIC used, and its two
weight files go into the git-ignored `external-repos/` (`$SYNCNET_UPSTREAM`):

    git clone https://github.com/joonson/syncnet_python external-repos/syncnet_python
    git -C external-repos/syncnet_python checkout 907c0b579c2e2d83f0eae1b2ac9e720cde4e5623
    (cd external-repos/syncnet_python && bash download_model.sh)   # syncnet_v2.model and the S3FD face detector

## Running it

The gate stage of one batch, after the mux of every build has finished (as a job:
`scripts/data_generation/gates.sh`):

    python -m mozaic.data_generation.gates.run_gates --batch batch_02 \
        --pairs A=<pairs>.A.txt --pairs B=<pairs>.B.txt \
        --build bulk_v1_ff77:A --build bulk_v1_ff153:B --build bulk_v1_s2v77:A \
        --build bulk_v1_wan77:A --build bulk_v1_s2v153:B \
        --face-python <face env>/bin/python --voice-python <voice env>/bin/python

In order: the four scorers on every build (a build already scored is skipped), the rule on
every build, then for every build the withdrawal of each dropped pair's clips to
`final_<build>/_withdrawn/<pair>/` (`final_clips/withdraw_incomplete_pairs.py`) and one ledger line
per dropped pair. `--dry-run` prints every step and runs nothing.

Each step alone, on one build (G = the gate folder), from the repo root; in an environment
that does not have this repo installed, set `PYTHONPATH=src` (run_gates does this for its steps):

    <face env>/bin/python -m mozaic.data_generation.gates.face_check --build-id B --final-dir F --pair-ids-file P --out G/identity_gate.json
    python -m mozaic.data_generation.gates.make_voice_check_list --build-id B --final-dir F --pair-ids-file P --out-dir G
    python -m mozaic.data_generation.gates.picture_check --build-id B --final-dir F --pair-ids-file P --out G/picture_report.jsonl
    <voice env>/bin/python -m mozaic.data_generation.gates.voice_check --pairs G/sv_pairs.jsonl --out G/sv_report.jsonl
    python -m mozaic.data_generation.gates.make_lip_sync_list --build-id B --final-dir F --pair-ids-file P --out-dir G
    SYNCNET_UPSTREAM=external-repos/syncnet_python <SyncNet env>/bin/python src/mozaic/data_generation/gates/lip_sync_check.py --pairs G/sync_pairs.jsonl --out G/sync_report.jsonl
    python -m mozaic.data_generation.gates.keep_or_drop --build-id B --final-dir F --stratum A --gates-dir G --pair-ids-file P --out-dir O

## What is written

Under `new_generated/`:

| where | what |
|---|---|
| `_reports/<build>_pairs<N>_<hash>_gates/` | the scores of one pair list: `identity_gate.json` (and `<build>_identity_excluded.txt` when a render is below 0.30), `sv_pairs.jsonl`, `sync_pairs.jsonl`, `gate_provenance.json`, `picture_report.jsonl`, `sv_report.jsonl`, `sync_report.jsonl` |
| `_reports/<build>_selection_<batch>/` | the decisions: `pair_ids_accepted.txt`, `pair_ids_rejected.txt` (pair and the checks it failed), `selection_pairs.jsonl` (every check's value, limit and verdict per pair), `selection_summary.json`, `selection_report.md` |
| `_reports/gate_stage/` | `<batch>.drops.jsonl` (the drop ledger: pair, build, why, where its clips went), `<batch>.gate_summary.json`, `<batch>.GATE_DONE`, `withdraw/<batch>.<build>.withdraw.txt` |
| `final_<build>/_withdrawn/<pair>/` | the clips of a dropped pair, moved, never deleted |
