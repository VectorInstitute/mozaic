# metrics

Turns predictions into scores. Ground truth always comes from the question file,
never from the prediction.

    metrics/
    ├── stats.py          shared helpers, pure numpy
    ├── t1_metrics.py     T1: spot the fake
    ├── t2_metrics.py     T2: manipulation type
    ├── t3_metrics.py     T3: identity check, yes or no
    ├── t4_metrics.py     T4: explanation: text overlap, embedding cosine and the judge
    └── text_metrics.py   ROUGE-L, BLEU-4, CIDEr-D (as pycocoevalcap), cosine

run_eval picks the task's file (eval_task in run_eval.py) and calls score(questions, predictions).

## stats.py

    accuracy()            accuracy with 95% Wilson interval
    binary()              every metric for fake = 1 vs real = 0
    multiclass()          per-class and averaged metrics, kappa, confusion matrix
    auroc(), eer(), average_precision()
    by_group()            accuracy per group, e.g. per video method

## T1 scores

    4-way answer
    ├── accuracy, ci95
    ├── per_class          precision, recall, F1, AUROC per label
    ├── macro, weighted    averages of the above
    ├── balanced_accuracy
    ├── cohen_kappa
    └── confusion          gold label → predicted label ("none" = unreadable)

    real_vs_fake           RVRA vs the rest
    video                  video part only (FV vs RV)
    audio                  audio part only (FA vs RA)
    └── each has: accuracy, precision, recall, specificity, fpr, fnr, f1,
        balanced_accuracy, mcc, auroc, eer, ap, confusion (tp, fp, tn, fn)

    per_method, per_arm, per_frames, per_label    accuracy per group
    pred_labels                                   answer bias
    pred_positions, gt_positions                  order bias: which option position (1 to 4)
                                                  the model picks vs where the right answer was
    tokens                                        token totals
    questions, answered, missing, unreadable

## T2 scores

    4-way type             real, face_swap, fully_generated, full_body_swap
    ├── accuracy, ci95
    ├── per_class          precision, recall, F1, AUROC per type
    ├── macro, weighted
    ├── balanced_accuracy
    ├── cohen_kappa
    └── confusion

    real_vs_fake           any manipulation counts as fake; same fields as T1's binary
                           views. fpr = real videos called fake, recall = fakes caught

    per_method, per_frames, per_arm    accuracy per group
    pred_classes                       answer bias
    pred_positions, gt_positions       order bias: which option position (1 to 4) the
                                       model picks vs where the right answer was
    tokens, questions, answered, missing, unreadable

With class probabilities in score ({"real": p, "face_swap": p, ...}), real_vs_fake
uses 1 - real as the fake score.

## T3 scores

    4-way answer           real_same_person, real_different_person, fake_this_person,
    ├── accuracy, ci95     fake_different_person
    ├── per_class, macro, weighted, balanced_accuracy, cohen_kappa, confusion

    fake                   real or fake half only: fake = 1
    same_person            same person as clip 2 or not, half only: same = 1
    └── each has the same fields as T1's binary views

    per_part                        face, voice
    per_clip1                       what clip 1 shows: face real / facefusion / s2v /
                                    animate, voice real / vocoded / converted
    per_label, per_clip2_person, per_right_answer, per_frames    accuracy per group
    pred_answers                    answer bias
    pred_positions, gt_positions    order bias: which option position (1 to 4) the
                                    model picks vs where the right answer was
    tokens
    prompt_tokens                   prompt tokens per part and length: every voice
                                    question of one length should cost the same, and
                                    more at 153 frames than at 77; if not, the sound
                                    did not arrive

## T4 scores

The answer is an explanation per part (pred_target.sections: video, audio), scored
against the reference (gt_target.sections). A reply that could not be read counts as
an empty text: 0 on every score.

    parse_rate             replies with a video and an audio explanation (rate, ci95): the format only
    text                   per part and both: rouge_l (mean), bleu_4 (corpus), cider_d (mean);
                           word overlap, computed here (text_metrics.py, equal to pycocoevalcap)
    judge                  per part: the mean 0-10; both: the mean of the two parts. From judge.jsonl
    cosine                 per part: mean cosine of the reference's and the prediction's embeddings
    judged, not_judged, failed, judge_per_cell
    per_cell, per_frames   parse_rate per label x video method, and per length
    tokens, questions, answered, missing, unreadable

judge and cosine come from judge/run_judge.py (gpt-5.4-mini through an OpenAI-compatible endpoint, all-MiniLM-L6-v2 here), run
before run_eval; until then they say "not run yet". Word overlap rewards the reference's
words, not its meaning: read it next to the judge.

## Unreadable replies

A reply that is not one of the options counts as wrong everywhere.

## AUROC, EER and AP

These need a score per answer. A prediction can carry class probabilities:

    "score": {"RVRA": 0.1, "RVFA": 0.2, "FVRA": 0.3, "FVFA": 0.4}

Then fake probabilities are derived as follows:

    real_vs_fake   1 - RVRA
    video          FVRA + FVFA
    audio          RVFA + FVFA

Models that give only an answer (most MLLMs) get these computed from the hard answer,
marked score_source: hard_label. AUROC then equals balanced accuracy, and EER is a
rough figure. Compare AUROC and EER only between models that give scores.

## Checked

stats.py matches scikit-learn on random data for AUROC, AP, F1, MCC, balanced
accuracy, macro and weighted F1, and Cohen's kappa. EER is interpolated where the
false positive and false negative rates cross.