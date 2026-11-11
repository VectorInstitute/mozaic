# common

Shared by every stage: annotation, training and evaluation.

    common/
    ├── config.yaml        shared settings: dataset root, env file, subset names, task folders, sampling defaults
    ├── config.py          loads config.yaml and the env file (API keys)
    ├── paths.py           paths inside the dataset root: metadata, qas, training, results; the subset switch
    ├── io.py              reads and writes json and jsonl; everything it creates is group-writable
    ├── metadata.py        loads metadata.jsonl keyed by file_name, filters clips, checks labels
    ├── questions.py       picks a subset of question records by filters, shared by training and evaluation
    ├── subsets.py         MOZAIC-Small: draws it and writes its files next to the full ones
    └── frame_sampler.py   uniform frame sampling and audio extraction, the same for every model

## Dataset root

The repo holds code only. Videos, metadata, question files, checkpoints and results
live under one dataset root: the absolute path in config.yaml (`data_root`).
MOZAIC_ROOT overrides it. `data/` is a place to put the download. Model weights live in `models_weights/`, one folder per model.

    data/
    ├── videos/                                    shared, never copied
    ├── metadata.jsonl, pairs.jsonl, summary.json, metadata_table.parquet, splits.json
    ├── metadata_small.jsonl, pairs_small.jsonl, summary_small.json, metadata_table_small.parquet
    ├── qas/          qas_small/          question files, one folder per task
    ├── training/     training_small/     checkpoints, per model and experiment
    └── results/      results_small/      predictions and scores, per model, experiment and task

Set MOZAIC_ROOT in the environment to use a different root.

## MOZAIC-Small

A smaller MOZAIC for training and quick evaluation: about 10,000 / 2,000 / 5,000 clips
(train / val / test), built by subsets.py:

    python -m mozaic.common.subsets

- Whole pairs, each Small split from the same full split, so a person stays in one split.
- The full split's mix is kept: pairs per length, generators that kept the pair and gender.
- Every person is the source (A) of at least one pair; nobody of more than ceil(pairs / people).
- Seed 7: the same every build.

Small is a choice of rows, not a copy: it writes four files next to the full ones and
shares videos/ (needs pyarrow for the parquet table):

    metadata_small.jsonl            Small's clips, rows copied unchanged from metadata.jsonl
    pairs_small.jsonl               Small's pairs, rows copied unchanged from pairs.jsonl
    summary_small.json              summary.json counted on Small, plus a subset block: seed,
                                    method, the full files' sha256, the commit, per-split counts
    metadata_table_small.parquet    metadata_small.jsonl as the full table's schema

Then every stage runs on Small with the same flag, and writes next to the full results:

    python -m mozaic.annotation.run_annotation --task t1 --subset small        # qas_small/
    python -m mozaic.evaluation.run_inference --config gemini_3_8_flash --task t1 --subset small   # results_small/
    python -m mozaic.evaluation.run_eval --config gemini_3_8_flash --task t1 --subset small

- Leave --subset out and everything is the full dataset, exactly as before.
- paths.py decides: metadata_path(), summary_path(), qas_dir(), results_dir() and
  training_dir() add _small when --subset is given (use_subset() in each entry point).
  metadata_path(full=True) always names the full file.
- T3 takes clip 1 from Small and clip 2 from the full metadata, so a Small T3 question is
  the same as in the full build. T1 to T3 split files on Small are the full rows of Small's
  pairs; a --size sample is drawn from Small's test split.
- T4 is built on Small only and refuses to run without --subset small.
- Use the same filters as on Full.
- Subset names are listed in config.yaml (subsets:); a typo fails early.

## Writing files

Always write through io.py. Folders it creates are group-writable and keep the
folder's group, so everyone on the team can read, rerun and append.

## Sampling

Every model gets the same input: frames sampled uniformly from first to last, and
audio as 16 kHz mono. Both defaults are in config.yaml and a model config can
override them.