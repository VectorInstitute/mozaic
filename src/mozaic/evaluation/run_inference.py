"""Run a model on a task's questions and save its predictions. No scoring here.

    python -m mozaic.evaluation.run_inference --config gemini_3_8_flash --task t1
    python -m mozaic.evaluation.run_inference --config gemini_3_8_flash --task t1 --limit 10
    python -m mozaic.evaluation.run_inference --config gemini_3_8_flash --task t4 --subset small   # qas_small/ -> results_small/
    python -m mozaic.evaluation.run_inference --config qwen3_omni_vllm --task t1 --split val      # files named t1_{split}.jsonl

Two families of models, set by family in the model config:
  mllm        answers in text: we sample frames and audio, build a prompt, parse the reply;
              the config's batch_size (default 1) questions per call
  classifier  outputs class probabilities: it reads the clips itself, in batches

A rerun skips questions already answered. A failed call is not saved, so the next
run asks it again. The run stops after stop_after_failures failed calls in a row.
--retry-failed first reads unreadable saved replies again with the current parser (fixed in place, no call), then
asks again the ones still unreadable or cut off.
--overwrite deletes the task's predictions, settings and scores first, then runs fresh; on a full run, also its
Small result when every answer there is a copy of the full run's (remove_small_copy).
With --subset small, answers the full run already has are copied, not asked (fill_from_full).

A model trained on MOZAIC (the config's trained: its training config and the data it was trained on)
takes its weights from that run, on the run's one task, and is tested on the test split only: its
train and val splits were seen in training. --subset picks the test questions only, so a model
trained on Small can be tested on the full test split.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from mozaic.common.config import REPO_ROOT
from mozaic.common.frame_sampler import (
    load_audio,
    pcm_to_wav,
    sample_frames,
    sampled_rate,
)
from mozaic.common.io import (
    append_jsonl,
    iter_jsonl,
    read_jsonl,
    write_json,
    write_jsonl,
)
from mozaic.common.paths import (
    qas_dir,
    resolve,
    results_dir,
    subset,
    training_dir,
    use_subset,
)
from mozaic.common.questions import select, task_entry, task_modalities
from mozaic.evaluation.base import ClipInput
from mozaic.evaluation.predictions import from_probs, to_prediction

CONFIG_DIR = Path(__file__).with_name("configs")
TRAIN_CONFIG_DIR = Path(__file__).parents[1] / "training" / "configs"
FAMILIES = ("mllm", "classifier")
ONE_MODALITY_TASKS = ("t1", "t4")  # one clip with its frames and sound: a config may send only one of them

log = logging.getLogger(__name__)


def load_model_config(name_or_path: str) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    return yaml.safe_load(path.read_text())


def load_configs(name: str, task: str | None = None) -> list[dict[str, Any]]:
    """--config and --task as the configs to run, each with only the tasks to run (configs/README.md, Naming a config).

    A full name (qwen3_omni_vllm/t1/zeroshot) or a path is read as it is; --task picks one of its tasks.
    <model>/<run> reads <model>/<task>/<run>.yaml for --task, else every task that has <run>.yaml (t4 only with
    --subset small: T4 exists on MOZAIC-Small only); <run> left out is zeroshot. Files that differ only in their
    task make one config, so the model is loaded once.
    """
    if Path(name).exists() or (CONFIG_DIR / f"{name}.yaml").exists():
        config = load_model_config(name)
        if task:
            if task not in config["tasks"]:
                raise SystemExit(f"{name} has no {task} (its tasks: {', '.join(config['tasks'])})")
            config["tasks"] = {task: config["tasks"][task]}
        return [config]
    model, _, run = name.partition("/")
    run = run or "zeroshot"
    if "/" in run or not (CONFIG_DIR / model).is_dir():
        raise SystemExit(f"no config {name}: pass <model>/<run> (e.g. avhubert/ln_trainsmall), a full name or a path")
    tasks = [task] if task else sorted(p.parent.name for p in (CONFIG_DIR / model).glob(f"t*/{run}.yaml"))
    if not task and subset() is None:
        tasks = [t for t in tasks if t != "t4"]  # T4: --subset small only
    names = [f"{model}/{t}/{run}" for t in tasks if (CONFIG_DIR / model / t / f"{run}.yaml").exists()]
    if not names:
        raise SystemExit(f"no config {model}/{task or 't*'}/{run}.yaml")
    groups: list[dict[str, Any]] = []
    for n in names:
        config = load_model_config(n)
        rest = {k: v for k, v in config.items() if k != "tasks"}
        same = next((g for g in groups if {k: v for k, v in g.items() if k != "tasks"} == rest), None)
        if same:
            same["tasks"].update(config["tasks"])
        else:
            groups.append(config)
    return groups


def use_split(config: dict[str, Any], split: str | None) -> None:
    """A config whose question files say {split} (t1_{split}.jsonl) takes that split's files, test by
    default, and saves to {experiment_name}_{split}. Other configs name their files: no --split."""
    tasks = config["tasks"]
    if not any("{split}" in task_entry(config, task)[0] for task in tasks):
        if split:
            raise SystemExit("--split is for configs whose question files say {split}, e.g. t1_{split}.jsonl")
        return
    split = split or "test"
    config["experiment_name"] = f"{config['experiment_name']}_{split}"
    for task, entry in tasks.items():
        if isinstance(entry, str):
            tasks[task] = entry.format(split=split)
        else:
            entry["file"] = entry["file"].format(split=split)


def trained_weights(config: dict[str, Any]) -> str:
    """The weights of the training run named by trained: its config (in training/configs) and the data it
    was trained on (data: full, training/; data: small, training_small/), whatever --subset is tested.
    A classifier's best.pt; an MLLM's final/ (the best epoch merged). A config given as a .yaml path is read
    from the repo root (ablations/)."""
    name, data = config["trained"]["config"], config["trained"]["data"]
    path = REPO_ROOT / name if name.endswith(".yaml") else TRAIN_CONFIG_DIR / f"{name}.yaml"
    trained = yaml.safe_load(path.read_text())
    if list(config["tasks"]) != [trained["task"]]:
        raise SystemExit(f"{name} was trained on {trained['task']}: the config's tasks must be that one")
    run = training_dir(trained["model"], trained["experiment_name"], on=None if data == "full" else data) / trained["task"]
    weights = run / ("best.pt" if config["family"] == "classifier" else "final")
    if not weights.exists():
        raise SystemExit(f"{weights} not found: train {name} on {data} first (training/README.md)")
    return str(weights)


def load_model(config: dict[str, Any]):
    """Imported here, not at the top: each model has its own environment and packages."""
    if config["model"] == "gemini":
        from mozaic.evaluation.models.api.gemini.model import Model
    elif config["model"] in ("qwen3_omni_vllm", "qwen2_5_omni_vllm"):
        from mozaic.evaluation.models.mllms.qwen_omni_vllm.model import Model
    elif config["model"] == "nemotron_omni_vllm":
        from mozaic.evaluation.models.mllms.nemotron_omni_vllm.model import Model
    elif config["model"] == "gemma4_vllm":
        from mozaic.evaluation.models.mllms.gemma4_vllm.model import Model
    elif config["model"] in ("avhubert", "dfd_hr", "havic", "pe_av"):
        from mozaic.evaluation.models.classifiers.trained.model import Model
    elif config["model"] == "minicpm_o_vllm":
        from mozaic.evaluation.models.mllms.minicpm_o_vllm.model import Model
    elif config["model"] == "videollama2_av":
        from mozaic.evaluation.models.mllms.videollama2_av.model import Model
    elif config["model"] == "omnivinci":
        from mozaic.evaluation.models.mllms.omnivinci.model import Model
    elif config["model"] == "uni_moe_2_omni":
        from mozaic.evaluation.models.mllms.uni_moe_2_omni.model import Model
    elif config["model"] == "ola":
        from mozaic.evaluation.models.mllms.ola.model import Model
    elif config["model"] == "vita_1_5":
        from mozaic.evaluation.models.mllms.vita_1_5.model import Model
    else:
        raise ValueError(f"unknown model {config['model']}")
    return Model(config)


def load_prompts(task: str):
    """The task's prompt and reply parsing (MLLMs only)."""
    if task == "t1":
        from mozaic.evaluation.prompts import t1_prompts as prompts
    elif task == "t2":
        from mozaic.evaluation.prompts import t2_prompts as prompts
    elif task == "t3":
        from mozaic.evaluation.prompts import t3_prompts as prompts
    elif task == "t4":
        from mozaic.evaluation.prompts import t4_prompts as prompts
    else:
        raise ValueError(f"unknown task {task}")
    return prompts


def clip_modalities(config: dict[str, Any], task: str, prompts) -> tuple[str, ...]:
    """What the model gets of the clip: the task's own modalities, or in t1 and t4 the config's [video] or [audio]
    (tasks: {t1: {file: ..., modalities: [video]}})."""
    modalities = task_modalities(config, task, prompts.MODALITIES)
    if set(modalities) == set(prompts.MODALITIES):
        return prompts.MODALITIES
    if task not in ONE_MODALITY_TASKS or modalities not in (("video",), ("audio",)):
        raise SystemExit(f"{task}: modalities {list(modalities)}: "
                         f"only {', '.join(ONE_MODALITY_TASKS)} take [video] or [audio]")
    return modalities


def read_clips(q: dict[str, Any], prompts, config: dict[str, Any], reads_clips: bool = False,
               modalities: tuple[str, ...] | None = None) -> list[ClipInput]:
    """The frames and sound of a question's clips, as its task's prompts ask for them.

    One clip (t1, t2, t4): its frames, plus its sound if the task uses audio; modalities (clip_modalities) may
    keep only one of them. Several clips
    (t3): the prompts' clip_inputs() gives each clip's label and what to send. With
    frames_as: video in the config, each clip also gets the real rate of its frames.
    reads_clips: the model reads the clips itself, so each clip is its file and what to send.
    """
    if hasattr(prompts, "clip_inputs"):
        wanted = prompts.clip_inputs(q)
    else:
        modalities = modalities or prompts.MODALITIES
        wanted = [(None, "video" in modalities, "audio" in modalities)]
    if reads_clips:
        return [ClipInput(label=label, path=str(resolve(path)), send_frames=use_frames, send_audio=use_audio)
                for path, (label, use_frames, use_audio) in zip(q["clips"], wanted, strict=True)]
    rate = config["audio_sample_rate"]
    as_video = config.get("frames_as") == "video"
    clips = []
    for path, (label, use_frames, use_audio) in zip(q["clips"], wanted, strict=True):
        video = resolve(path)
        frames = sample_frames(video, config["frames"]) if use_frames else []
        fps = sampled_rate(video, config["frames"]) if use_frames and as_video else None
        audio = pcm_to_wav(load_audio(video, rate), rate) if use_audio else None
        clips.append(ClipInput(label=label, frames=frames, audio_wav=audio, fps=fps))
    return clips


def check_settings(out_dir: Path, settings: dict[str, Any]) -> None:
    """Save the run's settings once. Refuse to mix a run with other settings (derived_from aside: where answers were
    copied from, fill_from_full)."""
    path = out_dir / "inference_settings.json"
    if not path.exists():
        write_json(path, settings)
        return
    old = json.loads(path.read_text())
    differ = sorted(k for k in set(old) | set(settings) if k != "derived_from" and old.get(k) != settings.get(k))
    if differ:
        raise SystemExit(f"{out_dir} was run with other settings ({', '.join(differ)}). Use another experiment_name.")


SMALL = "small"  # MOZAIC-Small: its questions are copies of full questions, with their ids


def fill_from_full(model: str, experiment: str, task: str, questions: list[dict[str, Any]],
                   retry_failed: bool = False, settings: dict[str, Any] | None = None) -> int:
    """With --subset small: copy the full run's answers to Small questions into results_small/ instead of asking the
    model again (an API model would be paid twice). Returns how many were copied.

    Only from the full run of the same config (results/, the same settings but question_file) and the same question
    file name, and only for a question identical in both. A copied answer is the full one plus copied_from (the full
    run's folder). Missing answers are filled, and a copy is replaced when the full run's answer has changed since (a
    --retry-failed there); an answer the model gave in a Small run is never replaced. With retry_failed, an unreadable
    or cut-off full answer is not copied, so it is asked again. settings: the Small run's, when it has none saved yet
    (written only if something is copied)."""
    if subset() != SMALL:
        return 0
    small_dir = results_dir(model, experiment, task)
    made = small_dir / "inference_settings.json"
    if made.exists():
        settings = json.loads(made.read_text())
    if settings is None:
        return 0
    file = Path(settings["question_file"]).name
    use_subset(None)
    try:
        full_dir, full_file = results_dir(model, experiment, task), qas_dir(task) / file
    finally:
        use_subset(SMALL)
    if not all(p.exists() for p in (full_dir / "inference_settings.json", full_dir / "predictions.jsonl", full_file)):
        return 0
    full_settings = json.loads((full_dir / "inference_settings.json").read_text())
    same = lambda s: {k: v for k, v in s.items() if k not in ("question_file", "derived_from")}
    if Path(full_settings["question_file"]).name != file or same(full_settings) != same(settings):
        log.info("[%s] nothing copied from %s: it was run with other settings", task, full_dir)
        return 0

    full_questions = {q["id"]: q for q in iter_jsonl(full_file)}
    answers = {r["id"]: r for r in iter_jsonl(full_dir / "predictions.jsonl")}  # the last answer per question
    copy = lambda i: {**answers[i], "copied_from": str(full_dir)}
    pred_path = small_dir / "predictions.jsonl"
    rows = read_jsonl(pred_path) if pred_path.exists() else []
    refreshed = [copy(r["id"]) if r.get("copied_from") and r["id"] in answers else r for r in rows]
    have = {r["id"] for r in rows}
    new = [copy(q["id"]) for q in questions if q["id"] not in have and q["id"] in answers
           and full_questions.get(q["id"]) == q and not (retry_failed and is_failed(answers[q["id"]]))]
    changed = sum(a != b for a, b in zip(rows, refreshed))
    if not new and not changed:
        return 0
    if changed:
        write_jsonl(pred_path, refreshed + new)
    else:
        for r in new:
            append_jsonl(pred_path, r)
    write_json(made, {**settings, "derived_from": str(full_dir)})
    log.info("[%s] copied %d answers from the full run %s (not asked)%s", task, len(new), full_dir,
             f"; {changed} copies changed since, copied again" if changed else "")
    return len(new) + changed


def is_failed(pred: dict[str, Any]) -> bool:
    """A saved reply worth asking again: unreadable, or cut off before the end."""
    return pred["pred_answer"] is None or pred.get("finish_reason") not in ("stop", None)


def reread_failed(pred_path: Path, task: str, questions: list[dict[str, Any]]) -> int:
    """Read the saved text of unreadable replies again with the current parser: one it now reads is fixed in place,
    with no model call (the model is greedy: asking again gives the same reply). Cut-off replies are left to be
    asked again. Returns how many were fixed."""
    if not pred_path.exists():
        return 0
    parse, options = load_prompts(task).parse, {q["id"]: q["options"] for q in questions}
    rows, fixed = read_jsonl(pred_path), 0
    for i, r in enumerate(rows):
        if r["pred_answer"] is not None or r.get("finish_reason") not in ("stop", None) or not r.get("raw") \
                or r["id"] not in options:
            continue
        answer = parse(r["raw"], options[r["id"]])
        if answer is not None:
            rows[i] = {**r, **to_prediction(task, answer), "reread": True}  # reread: read again from its raw text
            fixed += 1
    if fixed:
        write_jsonl(pred_path, rows)
    return fixed


def drop_failed(pred_path: Path) -> int:
    """Remove failed replies from the predictions file, so they are asked again."""
    if not pred_path.exists():
        return 0
    rows = read_jsonl(pred_path)
    keep = [r for r in rows if not is_failed(r)]
    write_jsonl(pred_path, keep)
    return len(rows) - len(keep)


def remove_results(out_dir: Path, task: str) -> None:
    """Delete the task's predictions, settings and scores, nothing else."""
    for name in ("predictions.jsonl", "inference_settings.json", "scores.json"):
        path = out_dir / name
        if path.exists():
            path.unlink()
            log.info("[%s] --overwrite: removed %s", task, path)


def remove_small_copy(model: str, experiment: str, task: str, full_dir: Path) -> None:
    """--overwrite on a full run: a Small result made only of its copied answers goes too, so it is copied again from
    the new answers (a copy is refreshed only while the settings match, and a rerun, e.g. retrained, changes them).
    A Small result with any answer of its own is kept."""
    use_subset(SMALL)
    try:
        small_dir = results_dir(model, experiment, task)
    finally:
        use_subset(None)
    pred_path = small_dir / "predictions.jsonl"
    if pred_path.exists() and all(r.get("copied_from") == str(full_dir) for r in iter_jsonl(pred_path)):
        remove_results(small_dir, task)


def start_run(config: dict[str, Any], task: str, limit: int | None, retry_failed: bool, overwrite: bool,
              settings: dict[str, Any]) -> tuple[list[dict[str, Any]], Path]:
    """Load the task's questions, check the run's settings, return what is left to ask."""
    file, filters = task_entry(config, task)
    if config.get("trained") and not file.endswith("_test.jsonl"):  # its train and val questions were seen in training
        raise SystemExit(f"{file}: a model trained on MOZAIC is tested on the test split only")
    question_file = REPO_ROOT / file if "/" in file else qas_dir(task) / file  # a path: from the repo root (ablations/)
    selected = select(read_jsonl(question_file), filters)
    total = len(selected)
    questions = selected[:limit]

    out_dir = results_dir(config["model"], config["experiment_name"], task)
    if overwrite:
        remove_results(out_dir, task)
        if subset() != SMALL:
            remove_small_copy(config["model"], config["experiment_name"], task, out_dir)
    if config.get("trained"):
        settings = {"weights": config["weights"], **settings}
    check_settings(out_dir, {"model": config["model"], "task": task, "question_file": str(question_file),
                             "filters": filters, **settings})

    pred_path = out_dir / "predictions.jsonl"
    if retry_failed:
        log.info("[%s] read again %d saved replies the current parser reads (no call)", task,
                 reread_failed(pred_path, task, selected))
        log.info("[%s] asking again %d unreadable or cut-off replies", task, drop_failed(pred_path))
    fill_from_full(config["model"], config["experiment_name"], task, questions, retry_failed)  # --subset small only
    done = {r["id"] for r in iter_jsonl(pred_path)} if pred_path.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    log.info("[%s] %s questions after filters %s (running %d); %d done, asking %d",
             task, f"{total:,}", filters, len(questions), len(questions) - len(todo), len(todo))
    return todo, pred_path


def check_t4(config: dict[str, Any], task: str) -> None:
    """T4 exists on MOZAIC-Small only. Its reference explanations were written by models (the annotator, the
    verifier and the rewriter): never score one of them on T4, and never run on the golden set, which is for human review only."""
    if task != "t4":
        return
    from mozaic.annotation.t4_explanation.build_questions import CONFIG as T4

    if subset() != T4["subset"]:
        raise SystemExit(f"T4 exists on the {T4['subset']} subset only: add --subset {T4['subset']}")
    if config.get("model_version") in (T4[m]["model_version"] for m in ("annotator", "verifier", "rewriter")):
        raise SystemExit(f"{config['model_version']} wrote the T4 reference explanations; test another model on t4.")
    if task_entry(config, task)[0] == "t4_golden.jsonl":
        raise SystemExit("t4_golden.jsonl is for human review only; run t4 on t4_test.jsonl.")


def ask(q: dict[str, Any], prompts, config: dict[str, Any], reads_clips: bool,
        modalities: tuple[str, ...]) -> tuple[list[ClipInput], str, int]:
    """A question's clips and prompt, and the frames per clip the model gets."""
    clips = read_clips(q, prompts, config, reads_clips, modalities)
    frames = config["frames"] if clips[0].send_frames else len(clips[0].frames)
    if modalities == prompts.MODALITIES:  # every task's prompt
        return clips, prompts.build_prompt(q, frames), frames
    return clips, prompts.build_prompt(q, frames, modalities), frames  # t1, t4: video only or audio only


def run_mllm(model, config: dict[str, Any], task: str, limit: int | None, retry_failed: bool, overwrite: bool) -> None:
    check_t4(config, task)
    prompts = load_prompts(task)
    modalities = clip_modalities(config, task, prompts)
    prompt = prompts.PROMPT if modalities == prompts.MODALITIES else prompts.template(modalities)
    todo, pred_path = start_run(config, task, limit, retry_failed, overwrite, {
        "model_version": config.get("model_version"),
        "frames": config["frames"],
        "audio_sample_rate": config["audio_sample_rate"],
        "modalities": list(modalities),
        "generation": config.get("generation"),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        **model.settings(),
    })

    failed_in_a_row = 0
    batch_size = config.get("batch_size", 1)  # questions per call: a model with generate_batch answers them together
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        asked = [ask(q, prompts, config, model.reads_clips, modalities) for q in batch]

        try:
            gens = model.generate_batch([(clips, prompt) for clips, prompt, _ in asked])
        except Exception as e:  # noqa: BLE001  any failed call: not saved, so the next run asks it again
            failed_in_a_row += 1
            log.warning("%s: failed: %s", batch[0]["id"] if len(batch) == 1 else f"batch at {start}", e)
            if failed_in_a_row >= config["stop_after_failures"]:
                log.error("STOPPED after %d failures in a row. Run the same command to continue.", failed_in_a_row)
                return
            continue
        failed_in_a_row = 0

        for i, (q, (clips, _, frames), gen) in enumerate(zip(batch, asked, gens, strict=True), start + 1):
            answer = prompts.parse(gen.raw, q["options"])
            audio = any(c.audio_wav is not None or c.send_audio for c in clips)
            preprocessing = {"frames": frames, "audio_sample_rate": config["audio_sample_rate"] if audio else None}
            if len(clips) > 1:
                preprocessing["clips"] = len(clips)
            append_jsonl(pred_path, {
                "id": q["id"],
                "model": config["model"],
                "task": q["task"],
                **to_prediction(task, answer),
                "score": None,
                "raw": gen.raw,
                "finish_reason": gen.finish_reason,
                "usage": gen.usage,
                "preprocessing": preprocessing,
                "time": dt.datetime.now().isoformat(timespec="seconds"),
            })
            gt = q["gt_target"]["regions"] if task == "t4" else q["gt_answer"]  # t4: gt_answer is a paragraph
            log.info("[%d/%d] %s  gt=%r  pred=%r", i, len(todo), q["id"], gt, answer)

    log.info("[%s] predictions in %s", task, pred_path)


def run_classifier(model, config: dict[str, Any], task: str, limit: int | None, retry_failed: bool, overwrite: bool) -> None:
    if task == "t4":
        raise SystemExit("t4 is answered in text (a video and an audio explanation): run it with an mllm config")
    todo, pred_path = start_run(config, task, limit, retry_failed, overwrite, model.settings())

    failed_in_a_row = 0
    batch_size = config["batch_size"]
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        try:
            probs = model.predict_proba(batch)
        except Exception as e:  # noqa: BLE001  any failed call: not saved, so the next run asks it again
            failed_in_a_row += 1
            log.warning("batch at %d: failed: %s", start, e)
            if failed_in_a_row >= config["stop_after_failures"]:
                log.error("STOPPED after %d failures in a row. Run the same command to continue.", failed_in_a_row)
                return
            continue
        failed_in_a_row = 0

        for i, (q, p) in enumerate(zip(batch, probs, strict=True), start + 1):
            pred = from_probs(task, p)
            append_jsonl(pred_path, {
                "id": q["id"],
                "model": config["model"],
                "task": q["task"],
                **pred,
                "raw": None,
                "finish_reason": None,
                "time": dt.datetime.now().isoformat(timespec="seconds"),
            })
            log.info("[%d/%d] %s  gt=%r  pred=%r", i, len(todo), q["id"], q["gt_answer"], pred["pred_answer"])

    log.info("[%s] predictions in %s", task, pred_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Run a model and save predictions.")
    p.add_argument("--config", required=True, help="<model>/<run> (zeroshot if left out), a full config name, or a path")
    p.add_argument("--task", help="run only this task (default: every task the config has)")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--retry-failed", action="store_true", help="ask again saved replies that were unreadable or cut off")
    p.add_argument("--overwrite", action="store_true", help="delete the task's predictions, settings and scores, then run fresh")
    p.add_argument("--env-file", help="env file with API keys (overrides the config)")
    p.add_argument("--subset", help="run on a subset, e.g. small (default: the full dataset)")
    p.add_argument("--split", help="the questions' split, for configs whose files say {split}: test (default), val, train")
    args = p.parse_args()
    use_subset(args.subset)
    if args.overwrite and args.retry_failed:
        p.error("--overwrite and --retry-failed cannot be used together")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for name in ("httpx", "httpx2"):  # openai >= 3 sends requests through httpx2
        logging.getLogger(name).setLevel(logging.WARNING)

    for config in load_configs(args.config, args.task):
        use_split(config, args.split)
        run_config(config, args)


def run_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    """One config's tasks, with the model loaded once."""
    if config["family"] not in FAMILIES:
        raise ValueError(f"unknown family {config['family']}, expected one of {FAMILIES}")
    if args.env_file:
        config["env_file"] = args.env_file
    tasks = list(config["tasks"])

    if config.get("trained"):
        config["weights"] = trained_weights(config)
        log.info("weights %s (trained by %s on %s)", config["weights"], config["trained"]["config"], config["trained"]["data"])

    model = load_model(config)
    model.load()
    try:
        for task in tasks:
            if config["family"] == "mllm":
                run_mllm(model, config, task, args.limit, args.retry_failed, args.overwrite)
            elif config["family"] == "classifier":
                run_classifier(model, config, task, args.limit, args.retry_failed, args.overwrite)
    finally:
        model.unload()


if __name__ == "__main__":
    main()