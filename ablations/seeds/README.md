# Seeds

Every model trained on MOZAIC-Small with seeds 1-4 at the protocol's lr; seed 0 is the run in training_small/.

Each config is the original (training/configs, evaluation/configs `*_trainsmall`) with `train.seed: N` and
`experiment_name` + `_seedN`; `bash ablations/seeds/submit.sh configs` writes them to configs/.

    bash ablations/seeds/submit.sh train    # then infer, then eval, each once the previous stage has finished
    python ablations/seeds/collect.py       # every test metric over seeds 0-4: mean and sample std
