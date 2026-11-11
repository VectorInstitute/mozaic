"""Check that the sound reaches Gemini through its endpoint. Two calls, nothing scored.

A gateway has dropped sound before without an error (input_audio parts). This takes the
first voice question of each length (77 and 153 frames) in the task's question file,
sends the sound of both clips, asks for the words in each, and checks:

    - no call failed, and no reply says NO AUDIO;
    - the 153-frame call has 152 more prompt tokens (2 clips x 76 more sound tokens).

    python -m mozaic.evaluation.models.api.gemini.sound_check --config gemini_3_8_flash --task t3

Needs internet (the endpoint in base_url). Writes results/<model>/<experiment>/<task>/sound_check_<time>.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from mozaic.common.io import read_jsonl, write_json
from mozaic.common.paths import qas_dir, results_dir
from mozaic.common.questions import select, task_entry
from mozaic.evaluation.models.api.gemini.model import Model
from mozaic.evaluation.run_inference import load_configs, load_prompts, read_clips

SOUND_CHECK = """Above are two audio files: clip 1, then clip 2. Write down the words spoken in each. If you cannot hear a file, write NO AUDIO for it.

Reply in JSON only: {"clip 1": "<the words>", "clip 2": "<the words>"}"""
EXTRA_TOKENS = 152  # 153 vs 77 frames: 76 more sound tokens per clip, 2 clips


def main() -> int:
    p = argparse.ArgumentParser(description="Check that the sound reaches Gemini.")
    p.add_argument("--config", default="gemini_3_8_flash", help="<model>/<run> (zeroshot if left out), a full config name, or a path")
    p.add_argument("--task", default="t3", choices=["t3"], help="a task with voice questions")
    p.add_argument("--env-file", help="env file with API keys (overrides the config)")
    args = p.parse_args()

    config = load_configs(args.config, args.task)[0]
    if args.env_file:
        config["env_file"] = args.env_file
    file, filters = task_entry(config, args.task)
    question_file = qas_dir(args.task) / file
    questions = select(read_jsonl(question_file), filters)
    prompts = load_prompts(args.task)
    picks = [next(q for q in questions if q["part"] == "voice" and q["meta"]["frames"] == f) for f in (77, 153)]

    model = Model(config)
    model.load()
    tests = []
    for q in picks:
        clips = read_clips(q, prompts, config)  # voice: the sound of both clips, no frames
        try:
            gen = model.generate_clips(clips, SOUND_CHECK)
            reply, usage, finish, error = gen.raw, gen.usage, gen.finish_reason, None
        except Exception as e:
            reply, usage, finish, error = "", {}, None, str(e)
        tests.append({"id": q["id"], "frames": q["meta"]["frames"], "clips": q["clips"],
                      "prompt_tokens": usage.get("prompt_tokens"), "reply": reply, "finish_reason": finish,
                      "usage": usage, "error": error})
    model.unload()

    tokens = [t["prompt_tokens"] for t in tests]
    problems = [f"{t['id']}: call failed: {t['error']}" for t in tests if t["error"]]
    problems += [f"{t['id']}: the reply says NO AUDIO" for t in tests if "NO AUDIO" in t["reply"].upper()]
    if None in tokens or tokens[1] - tokens[0] != EXTRA_TOKENS:
        problems.append(f"prompt tokens {tokens}: expected the 153-frame call to be {EXTRA_TOKENS} tokens longer")
    record = {
        "model": config["model"],
        "model_version": config.get("model_version"),
        "generation": config.get("generation"),
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "question_file": str(question_file),
        "prompt": SOUND_CHECK,
        "tests": tests,
        "problems": problems,
        "verdict": "PASS: the sound arrived" if not problems else "STOP",
    }
    out_dir = results_dir(config["model"], config["experiment_name"], args.task)
    path = out_dir / f"sound_check_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    write_json(path, record)
    for t in tests:
        print(f"{t['id']} ({t['frames']} frames): prompt tokens {t['prompt_tokens']}, reply {t['reply']!r}")
    print(f"{record['verdict']} {problems}; saved {path}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
