# T2: Manipulation Type

Which part of the video, if any, was generated or altered by AI? The question is only
about the video, so the model sees the frames and no sound.

    t2_manipulation_type/
    ├── config.yaml          filters, option shuffle seed, sample settings
    └── build_questions.py   metadata.jsonl → t2_train / t2_val / t2_test, or a sample

## Answers

The answer comes from the tool that made the video, so no model is used.

| gt_target | gt_target_name | Option | Video made by |
|---|---|---|---|
| 0 | real | real | camera |
| 1 | face_swap | face swap | FaceFusion |
| 2 | fully_generated | fully generated | Wan2.2-S2V |
| 3 | full_body_swap | full-body swap | Wan2.2-Animate (whole frame redrawn from one photo, copying the original movement) |

Options are short, as in T1. What each one means is explained in the prompt
(evaluation/prompts/t2_prompts.py), and the model answers with the option text.
Options name what changed in the frame, not how it was made (the model hears no sound,
so "audio-driven" would mean nothing to it). gt_target_name is the option as an id, in the
same order (TARGET_NAMES); scores take the class from the index, gt_target.

Options are shuffled per clip (seeded by options_seed and the clip id), so the same
clip always gets the same order and no position gives the answer away.

## Run

    python -m mozaic.annotation.run_annotation --task t2              # split files
    python -m mozaic.annotation.run_annotation --task t2 --size 200   # sample
    python -m mozaic.annotation.run_annotation --task t2 --subset small   # MOZAIC-Small

Output goes to qas/t2_manipulation_type/ (qas_small/t2_manipulation_type/ with --subset small):

    t2_train.jsonl      training
    t2_val.jsonl        validation
    t2_test.jsonl       evaluation: final numbers
    t2_200.jsonl        sample of the test split (default seed)

## Which clips

Set in config.yaml:

    labels    RVRA, FVRA    T2 is about the video only: RVFA has the same video as
                            RVRA, and FVFA as FVRA, so each video is asked once
    arms      main          a vocoded clip has the same video as its main clip
    lengths   77f, 153f     different videos, so both are asked

Every clip gets the same four options. Wan2.2-Animate (full-body swap) exists
only at 77f, so at 153f that option is offered but is never the right answer. The
scores report accuracy per length (per_frames), so the two can be read apart.

## Samples

Drawn from the test split, 1/4 per answer, every clip from a different pair, 77f only
(set under sample: in config.yaml). At 153f the answer is never "full-body
swap", so mixing lengths in a small sample would give that away. Set the sample
filter to null to sample both lengths.

## Record

    {
      "id": "t2_FVRA__facefusion__77f__main__id00012_21Uxsk56VDQ_00001__id06949",
      "task": "task2_manipulation_type",
      "split": "test",
      "clips": ["videos/FVRA/facefusion/77f/main/FVRA__facefusion__77f__main__...mov"],
      "question": "Which part of this video, if any, was generated or altered by AI? ...",
      "options": ["full-body swap", "real", "face swap", "fully generated"],
      "gt_answer": "face swap",
      "gt_label": "FVRA",
      "gt_target": 1,
      "gt_target_name": "face_swap",
      "meta": {"video_method": "facefusion", "audio_method": "real", "frames": 77, "arm": "main",
               "source_id": "id00012", "target_id": "id06949", "pair_id": "..."}
    }

A check runs before writing: the video is real exactly when face_real is true and the
label starts with R. If any clip disagrees, nothing is written.