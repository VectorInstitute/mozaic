# T3: Identity Check

Is clip 1 real or fake, and is it the same person as clip 2? Clip 2 is always a real
recording of one of the two people of clip 1's pair. A face question gives the model the
frames of both clips and no sound; a voice question gives the sound of both clips and no
frames.

    t3_identity_check/
    ├── config.yaml          clip 2 seed, option shuffle seed, sample counts
    └── build_questions.py   metadata.jsonl → t3_train / t3_val / t3_test, or the sample

## Answers

A is the person in the original recording (source_id), B the person the fake turns
them into (target_id). The answer comes from the label and from whose face (face_id)
and whose voice (voice_id) clip 1 was made to show, so no model is used.

| gt_target_name | Option | gt_target |
|---|---|---|
| real_same_person | real, same person | {"fake": 0, "same_person": 1} |
| real_different_person | real, different person | {"fake": 0, "same_person": 0} |
| fake_this_person | fake of this person | {"fake": 1, "same_person": 1} |
| fake_different_person | fake of a different person | {"fake": 1, "same_person": 0} |

| Part | Clip 1 | Clip 2 of A | Clip 2 of B |
|---|---|---|---|
| face | real video (RVRA main) | real, same person | real, different person |
| face | fake video (FVRA main: facefusion, s2v, animate) | fake of a different person | fake of this person |
| voice | real voice (RVRA main) | real, same person | real, different person |
| voice | real voice through the vocoder (RVRA vocoded) | real, same person | real, different person |
| voice | converted voice (RVFA main) | fake of a different person | fake of this person |

The vocoded voice was only processed, so it still counts as A's real voice. FaceFusion
swaps only the inner face and keeps A's hair and head, so the face question is about
the inner face only. What each option means is explained in the prompt
(evaluation/prompts/t3_prompts.py).

Options are shuffled per question (seeded by options_seed and the question id), so the
same question always gets the same order and no position gives the answer away.

## Run

    python -m mozaic.annotation.run_annotation --task t3             # split files
    python -m mozaic.annotation.run_annotation --task t3 --size 40   # the sample
    python -m mozaic.annotation.run_annotation --task t3 --subset small   # MOZAIC-Small

Output goes to qas/t3_identity_check/: t3_train.jsonl, t3_val.jsonl, t3_test.jsonl
and t3_40.jsonl (sample of the test split, seed 7). With --subset small, the same files
go to qas_small/t3_identity_check/: clip 1 comes from Small, clip 2 from the full
metadata, so each Small question is the same as in the full build.

## Which clips

Each picture and each sound of a pair is asked once with clip 2 = A and once with
clip 2 = B: the real video and each generator's video (face), the vocoded real voice
and the converted voice (voice). The pair's other clips repeat one
of these, so they are not asked again and no FVFA clip appears.

Clip 2 is an RVRA main clip of A or B with the same length as clip 1, from a YouTube
video the pair never used (not the original recording's, nor those of B's face
picture and voice reference), so a shared background or recording cannot give the
answer away. It is drawn at random once per pair and person, seeded by clip2_seed.

Pairs where A or B has no such clip 2 are left out.

Every converted voice is asked, and its answer follows what it was made to be: B's
voice. In 218 pairs (1.45%) the converted voice scored at least as close to A's voice as
to B's (voice_source >= voice_target in metadata.jsonl), since the quality gate only
asked that it sound enough like B; these are kept with the same answers.

Raw or vocoded voice: the files are the vocoded view, as in T1 and T4. The real voice
asked is the vocoded copy only, so every voice in clip 1 has been through the vocoder,
real or converted, and its sound gives nothing away. Clip 2 is always the raw real
recording.

## Samples

Set in config.yaml as an exact number of questions per kind (part, what clip 1 shows,
frames, clip 2's person), drawn from the test split, every question from a different
pair. The 40 sample has, per part, 5 questions per answer, so 10 with clip 2 = A and 10
with B; the fakes cover every generator, and the real voices are vocoded ones. Rows are
mixed so that any first part holds about its share of every kind.

## Guards

    Second path      before writing, every row is worked out again another way: the
                     answer from face_real / voice_real and face_id / voice_id instead
                     of the label, clip 1s from face_real, voice_real and kept_by,
                     clip 2 drawn again
    Balance          each split and part has as many clip 2 = A as B
    Pairs            pairs per length match summary.json (summary_small.json on Small)
    Sample           exact counts per kind, one pair per question, the same number of
                     each answer and as many A as B per part

If any check fails, nothing is written.

## Record

    {
      "id": "t3_face_A_FVRA__animate__77f__main__id00024_3ysHAigPe68_00005__id00330",
      "task": "task3_identity_check",
      "split": "test",
      "clips": ["videos/FVRA/animate/77f/main/FVRA__animate__77f__main__id00024_3ysHAigPe68_00005__id00330.mov",
                "videos/RVRA/real/77f/main/RVRA__real__77f__main__id00024_q_q9EPRJcMo_00090__id06425.mov"],
      "question": "Clip 2 is a real recording of one person. Look only at the faces ...",
      "options": ["fake of a different person", "real, different person", "fake of this person", "real, same person"],
      "part": "face",
      "gt_answer": "fake of a different person",
      "gt_label": "FVRA",
      "gt_target": {"fake": 1, "same_person": 0},
      "gt_target_name": "fake_different_person",
      "meta": {"video_method": "animate", "audio_method": "none", "frames": 77, "arm": "main",
               "source_id": "id00024", "target_id": "id00330",
               "pair_id": "id00024_3ysHAigPe68_00005__id00330", "clip2_person": "A"}
    }

clips[0] is clip 1, clips[1] is clip 2; both are relative to the dataset root. meta
describes clip 1 and is for analysis only; a model never sees it.
