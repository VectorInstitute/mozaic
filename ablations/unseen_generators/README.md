# Unseen generators

T1 with one video generator (FaceFusion, Wan2.2-S2V or Wan2.2-Animate) left out of the train and val questions,
for every model trained on MOZAIC-Small, seed 0; tested on the whole test set. The runs in training_small/ saw
every generator.

Each config is the original T1 config with `video_methods: [the other three]` added to the train and val filters
and `experiment_name` + `_no-<generator>`; `bash ablations/unseen_generators/submit.sh configs` writes them to
configs/.

    bash ablations/unseen_generators/submit.sh train    # then infer, then eval, each once the previous stage has finished
    python ablations/unseen_generators/collect.py       # T1 accuracy per generator, held out or seen
