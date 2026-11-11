"""Test metrics at each epoch budget, seed 0, one row per run, as CSV (10 epochs, Qwen2.5-Omni 3: the protocol's run).

    python ablations/epoch_sweep/collect.py                  # T1-T3: the full test split; T4: Small's
    python ablations/epoch_sweep/collect.py --subset small   # every task on Small's test split
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from variants import METRICS, cells, runs  # ablations/variants.py

PROTOCOL = {"qwen2_5_omni_vllm": 3}  # the original runs' epochs; 10 for every classifier

print("model,experiment,task,epochs,budget," + ",".join(METRICS))
for (model, experiment, task), by_budget in runs(Path(__file__).parent, "--subset" in sys.argv).items():
    protocol = PROTOCOL.get(model, 10)
    budgets = sorted([protocol, *(int(v[2:]) for v in by_budget if v != "base")])
    for epochs in budgets:
        run = by_budget["base"] if epochs == protocol else by_budget.get(f"ep{epochs}")
        print(f"{model},{experiment},{task},{epochs},{epochs / protocol:.2f}x," + cells(run))
