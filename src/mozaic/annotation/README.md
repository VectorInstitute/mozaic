# Annotation

Builds the question files for each task into the dataset root. Reads metadata.jsonl;
never calls a model for the closed-ended tasks (T1-T3). T4 adds an annotate step: Gemini explains,
Claude verifies.

    annotation/
    ├── run_annotation.py        one runner for every task
    ├── sampling.py              balanced and counted samples, option shuffling, shared
    ├── writer.py                writes the question files, shared by every task
    ├── t1_spot_the_fake/        which part is fake: video, audio, both, neither
    │   ├── config.yaml
    │   └── build_questions.py
    ├── t2_manipulation_type/    how the video was made (video only)
    │   ├── config.yaml
    │   └── build_questions.py
    ├── t3_identity_check/       two clips: is clip 1 real or fake, and clip 2's person
    │   ├── config.yaml
    │   └── build_questions.py
    └── t4_explanation/          what is fake, where, and the evidence (facts from metadata, explained by Gemini)
        ├── config.yaml, build_questions.py, annotate.py
        ├── facts/       metadata facts, signs, claims, artifact categories
        ├── models/      annotator (Gemini, sees the clip), verifier and rewriter (Claude, text only)
        ├── prompts/     what each model is told
        └── checks/      the code checks

## Run

    python -m mozaic.annotation.run_annotation --task t1              # train, val, test files
    python -m mozaic.annotation.run_annotation --task t1 --size 200   # sample of the test split
    python -m mozaic.annotation.run_annotation --task t2 --size 200
    python -m mozaic.annotation.run_annotation --task t3 --size 40
    python -m mozaic.annotation.run_annotation --task t1 --subset small               # MOZAIC-Small -> qas_small/
    python -m mozaic.annotation.run_annotation --task t4 --subset small --select      # t4: Small only, in steps (t4_explanation/README.md)

## Flags

    --task t1        which task: t1, t2, t3, t4 (required)
    --size N         write a balanced sample of the test split instead of the split files
                     (t3: the counts in config.yaml set the sample; N is their total)
    --seed S         sample seed (default in the task config)
    --out DIR        write to this folder instead of qas/{task folder} (experiments)
    --subset small   build on MOZAIC-Small: reads metadata_small.jsonl, writes qas_small/ (required for t4)
    t4 only          one step per run: --select, --annotate [--golden-only] [--retry-failed] [--limit N],
                     --build (no --size or --seed; --out DIR runs the step in DIR, for trials)

## Output

    qas/{task folder}/
    ├── t1_train.jsonl      training
    ├── t1_val.jsonl        validation
    ├── t1_test.jsonl       evaluation: final numbers
    ├── t1_200.jsonl        sample of the test split, default seed (pilots)
    └── t1_200_s7.jsonl     sample, seed 7

With --subset small the same files go to qas_small/{task folder}/.

Splits come from metadata.jsonl (identity-disjoint: a person is in one split only).
Which clips go into the files is set under filters: in the task's config.yaml
(T1 keeps every clip; T2 keeps each video once, 77f only).

A sample is a smaller question file in the same format. To run a model on it, point
the model config at it.

## Guards

    Label check      every answer is checked against face_real, voice_real and the
                     label; if any disagree, nothing is written
    Atomic write     written to a temp file, then renamed: never half a file
    Samples          equal per answer, spread over each answer's kinds, one clip per
                     pair; size must be a multiple of 4; same seed, same sample
    Option order     shuffled per clip (t3: per question), the same every build
                     (options_seed + clip or question id)
    Second path      t3: every row is worked out again another way before writing;
                     if any row disagrees, nothing is written
    Counted samples  t3: an exact number per kind (config.yaml), one clip per pair,
                     the same number of each answer per part
    Permissions      every file and folder written is group-writable

## Adding a task

1. A folder {task folder}/ with build_questions.py exposing build(size, seed, out),
   and a config.yaml.
2. The folder name under tasks: in common/config.yaml, and a branch for the task in
   run_annotation.py.
3. A README in the task folder: what is asked, where answers come from, the record.

## Tasks annotated by a model

A task whose answers come from a model call (for example, explanations written by
an LLM) needs more than a build step. Split it into steps that talk through files,
one step per run, as T4 does in run_annotation.py:

    --select          pick the clips and write their facts, no model calls
    --annotate        call the model and save each raw reply, untouched
    --build           build the question files from the saved replies, no model calls
    --retry-failed    with --annotate: call again only for missing or invalid replies

The saved raw replies are the checkpoint: the question file can be rebuilt any time
without paying for another call. Follow the same guards as run_inference: append each
reply as it comes, skip what is already done on a rerun, stop after a few failed
calls in a row, and refuse to mix runs with different settings. The task's README
says what counts as an invalid reply.