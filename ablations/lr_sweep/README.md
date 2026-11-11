# Learning rate

Every model trained on MOZAIC-Small at other learning rates, seed 0; 1e-4 (the protocol's) is the run in
training_small/. Every model and tuning mode: 1e-6, 3e-6, 1e-5, 3e-5, 3e-4.

Each config is the original (training/configs, evaluation/configs `*_trainsmall`) with `train.lr` and
`experiment_name` + `_lr<value>`; `bash ablations/lr_sweep/submit.sh configs` writes them to configs/.

    bash ablations/lr_sweep/submit.sh train    # then infer, then eval, each once the previous stage has finished
    python ablations/lr_sweep/collect.py       # every test metric at each learning rate
