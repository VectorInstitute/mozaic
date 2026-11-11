# classifiers
The classifiers trained on MOZAIC (training/classifiers/), tested on a task's test split.

    trained/model.py    every one of them: best.pt's training config rebuilds the network with
                        training's own code, the trained weights go on top, and training's dataset
                        and processor read the clips (the same input as in training; T3: two
                        clips per question, training's common/pairs.py)

Its configs: configs/<model>/<task>_<training>_train<full|small>.yaml, with trained: (evaluation/README.md, Configs).
It runs in the model's training environment (.venv_<model>).

    sbatch scripts/evaluation/inference.sh --config avhubert/t1/ln_trainsmall                  # the full test split
    sbatch scripts/evaluation/inference.sh --config avhubert/t1/ln_trainsmall --subset small   # Small's
    sbatch scripts/evaluation/evaluate.sh --config avhubert/t1/ln_trainsmall                   # the same flags
