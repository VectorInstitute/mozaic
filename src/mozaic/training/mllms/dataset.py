"""Step 1: the task's train and val questions → a LLaMA-Factory dataset in <run>/data/.

One sharegpt row per question: the evaluation prompt in, the reply evaluation parses out.
t3 has two clips per question: their two videos (face) or their two sounds (voice).
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from mozaic.common.io import read_jsonl, write_json, write_text
from mozaic.common.paths import qas_dir, resolve, subset
from mozaic.common.questions import select
from mozaic.training.common import store
from mozaic.training.common.tracking import ID_FILE
from mozaic.training.mllms.run_config import load, parse_args, prompts, with_sound

log = logging.getLogger(__name__)

LLAMAFACTORY_COMMIT = "c683e59"  # requirements.txt pins the same
# --limit keeps each class equally
CLASS_KEY = {"t1": "gt_label", "t2": "gt_target_name", "t3": "gt_target_name", "t4": "gt_label"}


def load_questions(task: str, entry: str | dict[str, Any], limit: int | None) -> tuple[str, list[dict[str, Any]]]:
    """A split's file and rows, filtered as in evaluation; with limit, about limit rows, equal per class and part."""
    file, filters = (entry, None) if isinstance(entry, str) else (entry["file"], entry.get("filters"))
    rows = select(read_jsonl(qas_dir(task) / file), filters)
    if not limit:
        return file, rows
    by_class: dict[tuple[str | None, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_class.setdefault((row.get("part"), row[CLASS_KEY[task]]), []).append(row)
    per_class = -(-limit // len(by_class))
    return file, [row for c in sorted(by_class) for row in by_class[c][:per_class]]


def reply(task: str, question: dict[str, Any]) -> str:
    """The target: the JSON the prompt asks for, checked with evaluation's parser."""
    if task in ("t1", "t2", "t3"):
        answer = {"answer": question["gt_answer"]}
        ok = prompts(task).parse(json.dumps(answer), question["options"]) == question["gt_answer"]
    elif task == "t4":
        answer = question["gt_target"]["sections"]  # {video, audio}: the reference explanation per part
        ok = prompts(task).parse(json.dumps(answer), question["options"]) == answer
    else:
        raise ValueError(f"no SFT data for task {task}")
    if not ok:
        raise ValueError(f"{question['id']}: evaluation would not read this reply")
    return json.dumps(answer, ensure_ascii=False)


def placeholders(sound: bool) -> str:
    return "<video><audio>" if sound else "<video>"


def two_clips(question: dict[str, Any]) -> tuple[str, dict[str, list[str]]]:
    """t3, as evaluation sends it: what goes before the prompt (per clip its label, then its video or its sound), and
    the row's files. Both columns are always there, the unused one empty: one dataset holds face and voice questions."""
    clips, wanted = [str(resolve(c)) for c in question["clips"]], prompts("t3").clip_inputs(question)
    face = question["part"] == "face"
    before = "".join(label + ("<video>" if face else "<audio>") for label, _, _ in wanted)
    return before, {"videos": clips if face else [], "audios": [] if face else clips}


def example(task: str, question: dict[str, Any], frames: int) -> dict[str, Any]:
    if task == "t3":
        before, files = two_clips(question)
    else:
        clip, sound = str(resolve(question["clips"][0])), with_sound(task)
        before = placeholders(sound)
        files = {"videos": [clip]} | ({"audios": [clip]} if sound else {})  # the sound is read from the same file
    prompt = before + prompts(task).build_prompt(question, frames)
    return {"id": question["id"],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply(task, question)}],
            **files}


def build(config: dict[str, Any], out: Path, limit: int | None) -> dict[str, Any]:
    """Writes <out>/data/ and its dataset_info.json; returns the files and counts, for the settings."""
    task, data = config["task"], out / "data"
    info, written = {}, {}
    for split in ("train", "val"):
        file, rows = load_questions(task, config["questions"][split], limit)
        name = f"{task}_{split}"
        write_json(data / f"{name}.json", [example(task, r, config["frames"]) for r in rows])
        any_sound = "audio" in prompts(task).MODALITIES  # t1, t3, t4
        columns = {"messages": "messages", "videos": "videos"} | ({"audios": "audios"} if any_sound else {})
        info[name] = {"file_name": f"{name}.json", "formatting": "sharegpt", "columns": columns,
                      "tags": {"role_tag": "role", "content_tag": "content",
                               "user_tag": "user", "assistant_tag": "assistant"}}
        written |= {f"{split}_file": file, f"{split}_questions": len(rows)}
    write_json(data / "dataset_info.json", info)
    return written


def main() -> None:
    args = parse_args("Step 1: build the LLaMA-Factory dataset.")
    config, out = load(args)
    if args.overwrite:
        store.remove_run(out)
        (out / "final.json").unlink(missing_ok=True)
        for name in ("data", "checkpoints", "final", "val_predictions"):
            shutil.rmtree(out / name, ignore_errors=True)
    written = build(config, out, args.limit)
    store.check_settings(out, {**config, **written, "limit": args.limit, "subset": subset(),
                               "llamafactory_commit": LLAMAFACTORY_COMMIT})
    if not (out / ID_FILE).exists():  # fixed here, so a resumed run logs into the same W&B run
        write_text(out / ID_FILE, uuid.uuid4().hex[:8] + "\n")
    log.info("%s on %s: %d train, %d val questions → %s", config["model"], config["task"],
             written["train_questions"], written["val_questions"], out / "data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main()
