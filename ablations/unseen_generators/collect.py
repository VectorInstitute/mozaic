"""T1 test accuracy per video generator with one generator left out of training, seed 0, as CSV.

One row per run: "none" is the original run (every generator seen). acc_<generator> is the accuracy on that
generator's test clips (per_method in scores.json; real: the real-video clips); the held-out one is unseen.

    python ablations/unseen_generators/collect.py                  # the full test split
    python ablations/unseen_generators/collect.py --subset small   # Small's test split
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
from variants import scores  # ablations/variants.py

GENERATORS = ["facefusion", "s2v", "animate", "real"]
small = "--subset" in sys.argv

runs: dict[tuple[str, str], dict[str, str]] = {}  # (model, the original's experiment) -> {held out: experiment}
for path in sorted((Path(__file__).parent / "configs").glob("*_eval.yaml")):
    config = yaml.safe_load(path.read_text())
    held_out = path.stem.removesuffix("_eval").rsplit("_", 1)[1].removeprefix("no-")
    base = config["experiment_name"].removesuffix(f"_no-{held_out}")
    runs.setdefault((config["model"], base), {"none": base})[held_out] = config["experiment_name"]

print("model,experiment,held_out,n,accuracy," + ",".join(f"acc_{g}" for g in GENERATORS))
for (model, base), by_held_out in runs.items():
    for held_out, experiment in by_held_out.items():
        s = scores(model, experiment, "t1", small)
        if s is None:
            print(f"{model},{base},{held_out}," + "," * (1 + len(GENERATORS)))
            continue
        per = [f"{s['per_method'][g]['accuracy']:.4f}" if g in s["per_method"] else "" for g in GENERATORS]
        print(f"{model},{base},{held_out},{s['n']},{s['accuracy']:.4f}," + ",".join(per))
