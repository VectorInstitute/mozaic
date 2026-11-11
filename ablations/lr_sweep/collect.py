"""Test metrics at each learning rate, seed 0, one row per run, as CSV (1e-4: the protocol's run).

    python ablations/lr_sweep/collect.py                  # T1-T3: the full test split; T4: Small's
    python ablations/lr_sweep/collect.py --subset small   # every task on Small's test split
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from variants import METRICS, cells, runs  # ablations/variants.py

LRS = {"lr1e-6": "1e-6", "lr3e-6": "3e-6", "lr1e-5": "1e-5", "lr3e-5": "3e-5", "base": "1e-4", "lr3e-4": "3e-4"}

print("model,experiment,task,lr," + ",".join(METRICS))
for (model, experiment, task), by_lr in runs(Path(__file__).parent, "--subset" in sys.argv).items():
    for variant, lr in LRS.items():
        print(f"{model},{experiment},{task},{lr}," + cells(by_lr.get(variant)))
