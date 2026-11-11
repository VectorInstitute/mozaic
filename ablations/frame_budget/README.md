# T1 frame budget

T1 zero-shot on MOZAIC-Small at 8, 16, 32 and 64 frames per clip.

Each config is the model's zero-shot config with `frames` and `experiment_name` changed; submit.sh writes them to
configs/.

    bash ablations/frame_budget/submit.sh
    python ablations/frame_budget/collect.py
