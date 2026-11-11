# T1: Spot the Fake

Which part of the clip is AI-generated: the video, the audio, both, or neither?
The model sees the frames and hears the audio.

    t1_spot_the_fake/
    ├── config.yaml          filters, option shuffle seed, sample settings
    └── build_questions.py   metadata.jsonl → t1_train / t1_val / t1_test, or a sample

## Answers

The answer comes from the clip's label, so no model is used.

| Label | Answer |
|---|---|
| RVRA | neither |
| RVFA | the audio |
| FVRA | the video |
| FVFA | both |

Vocoded clips keep their label: a real voice passed through a vocoder is still real.

Options are shuffled per clip (seeded by options_seed and the clip id), so the same
clip always gets the same order and no position gives the answer away. What each
option means is explained in the prompt (evaluation/prompts/t1_prompts.py).

## Run

    python -m mozaic.annotation.run_annotation --task t1              # split files
    python -m mozaic.annotation.run_annotation --task t1 --size 200   # sample
    python -m mozaic.annotation.run_annotation --task t1 --subset small   # MOZAIC-Small

Output goes to qas/t1_spot_the_fake/ (qas_small/t1_spot_the_fake/ with --subset small):

    t1_train.jsonl      training
    t1_val.jsonl        validation
    t1_test.jsonl       evaluation: final numbers
    t1_200.jsonl        sample of the test split (default seed)
    t1_200_s7.jsonl     sample, seed 7

The files are the vocoded view, as in T3 and T4 (config.yaml arms): a real voice is its
vocoded copy only (RVRA, FVRA vocoded), a converted voice its one version (RVFA, FVFA
main), already vocoded inside Seed-VC. So every voice has been through a vocoder and its
sound alone gives nothing away. All labels and both lengths (77f, 153f) are kept; every
length has all four answers.

## Samples

Drawn from the test split. Equal per label, spread evenly over each label's kinds
(video method, length), every clip from a different pair. Any first part of the
file is still balanced, so a run that stops early is still fair.

## Record

    {
      "id": "t1_FVFA__s2v__153f__main__id00012_21Uxsk56VDQ_00001__id06949",
      "task": "task1_spot_the_fake",
      "split": "test",
      "clips": ["videos/FVFA/s2v/153f/main/FVFA__s2v__153f__main__id00012_21Uxsk56VDQ_00001__id06949.mov"],
      "question": "Which part of this audio-visual clip, if any, is AI-generated or manipulated? ...",
      "options": ["both", "the video", "neither", "the audio"],
      "gt_answer": "both",
      "gt_label": "FVFA",
      "gt_target": {"video_fake": 1, "audio_fake": 1},
      "meta": {"video_method": "s2v", "frames": 153, "arm": "main",
               "source_id": "id00012", "target_id": "id06949", "pair_id": "..."}
    }

clips is relative to the dataset root. meta is for analysis only; a model never sees it.