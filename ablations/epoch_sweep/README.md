# Epoch budget

Every model trained on MOZAIC-Small for other numbers of epochs, seed 0, each run from scratch; 10 (Qwen2.5-Omni: 3)
is the run in training_small/. Classifiers: 5, 15, 20; LoRA: 1, 2, 5, 6. Each run picks its best epoch on val.

Each config is the original (training/configs, evaluation/configs `*_trainsmall`) with `train.epochs` and
`experiment_name` + `_ep<N>`; `bash ablations/epoch_sweep/submit.sh configs` writes them to configs/.

    bash ablations/epoch_sweep/submit.sh train    # then infer, then eval, each once the previous stage has finished
    python ablations/epoch_sweep/collect.py       # every test metric at each epoch budget
