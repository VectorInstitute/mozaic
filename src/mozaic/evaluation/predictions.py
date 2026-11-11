"""Answer to prediction fields, shared by MLLMs and classifiers.

pred_* fields mirror the question's gt_* fields, so metrics compare them directly.
"""

from __future__ import annotations

from typing import Any

from mozaic.annotation.t1_spot_the_fake.build_questions import ANSWERS as T1_ANSWERS
from mozaic.annotation.t2_manipulation_type.build_questions import OPTIONS as T2_OPTIONS
from mozaic.annotation.t2_manipulation_type.build_questions import TARGET_NAMES as T2_TARGET_NAMES
from mozaic.annotation.t3_identity_check.build_questions import OPTIONS as T3_OPTIONS
from mozaic.annotation.t3_identity_check.build_questions import TARGET_NAMES as T3_TARGET_NAMES
from mozaic.annotation.t3_identity_check.build_questions import TARGETS as T3_TARGETS

T1_LABELS = {answer: label for label, answer in T1_ANSWERS.items()}


def to_prediction(task: str, answer: str | None) -> dict[str, Any]:
    """pred_* fields for an answer (one of the options), or all None if unreadable."""
    if task == "t1":
        if answer is None:
            return {"pred_answer": None, "pred_label": None, "pred_target": None}
        label = T1_LABELS[answer]
        return {
            "pred_answer": answer,
            "pred_label": label,
            "pred_target": {"video_fake": int(label[0] == "F"), "audio_fake": int(label[2] == "F")},
        }
    if task == "t2":
        if answer is None:
            return {"pred_answer": None, "pred_target": None, "pred_target_name": None}
        target = T2_OPTIONS.index(answer)
        return {"pred_answer": answer, "pred_target": target, "pred_target_name": T2_TARGET_NAMES[target]}
    if task == "t3":
        if answer is None:
            return {"pred_answer": None, "pred_target": None, "pred_target_name": None}
        i = T3_OPTIONS.index(answer)
        fake, same_person = T3_TARGETS[i]
        return {"pred_answer": answer, "pred_target": {"fake": fake, "same_person": same_person},
                "pred_target_name": T3_TARGET_NAMES[i]}
    if task == "t4":  # answer: the parsed reply {video, audio}; as gt_answer and gt_target.sections
        if answer is None:
            return {"pred_answer": None, "pred_target": None}
        return {"pred_answer": " ".join(answer.values()), "pred_target": {"sections": answer}}
    raise ValueError(f"unknown task {task}")


def from_probs(task: str, probs: dict[str, float]) -> dict[str, Any]:
    """pred_* fields from a classifier's class probabilities; the top class is the answer.

    t1 classes: RVRA, RVFA, FVRA, FVFA. t2 classes: real, face_swap,
    fully_generated, full_body_swap. t3 classes: real_same_person,
    real_different_person, fake_this_person, fake_different_person.
    """
    top = max(probs, key=probs.get)
    if task == "t1":
        answer = T1_ANSWERS[top]
    elif task == "t2":
        answer = T2_OPTIONS[T2_TARGET_NAMES.index(top)]
    elif task == "t3":
        answer = T3_OPTIONS[T3_TARGET_NAMES.index(top)]
    elif task == "t4":
        raise ValueError("t4 is answered in text (a video and an audio explanation): MLLMs only")
    else:
        raise ValueError(f"unknown task {task}")
    return {**to_prediction(task, answer), "score": probs}
