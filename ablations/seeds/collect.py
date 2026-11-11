"""Test metrics over seeds 0-4 of every model trained on MOZAIC-Small: mean and sample std of each, as CSV.

    python ablations/seeds/collect.py                  # T1-T3: the full test split; T4: Small's
    python ablations/seeds/collect.py --subset small   # every task on Small's test split
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from variants import METRICS, runs  # ablations/variants.py

print("model,experiment,task,seeds," + ",".join(f"{m}_mean,{m}_std" for m in METRICS))
for (model, experiment, task), by_seed in runs(Path(__file__).parent, "--subset" in sys.argv).items():
    scored = [s for s in [by_seed["base"], *(by_seed.get(f"seed{n}") for n in range(1, 5))] if s]
    row = []
    for m in METRICS:
        values = [s[m] for s in scored if s.get(m) is not None]
        row += [f"{statistics.mean(values):.4f}" if values else "", f"{statistics.stdev(values):.4f}" if len(values) > 1 else ""]
    print(f"{model},{experiment},{task},{len(scored)}," + ",".join(row))
