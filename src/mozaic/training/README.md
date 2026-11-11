# Training

Trains the benchmark's classifier baselines on a task's train split, selected on its val
split. The test split is never read here: evaluation scores it. MLLMs are fine-tuned with
LLaMA-Factory (mllms/, its own README): LoRA SFT on the evaluation prompt, validated each
epoch, the best merged as the final model.

    training/
    ├── run_train.py      config → checkpoint + val scores (torchrun)
    ├── base.py           BaseProcessor, BaseClassifier: what a model implements
    ├── configs/          <model>/<task>.yaml
    ├── common/           data, head, tuning modes, training loop, W&B, run folder; T3's two clips (pairs.py)
    ├── classifiers/      one folder per model
    └── mllms/            MLLM SFT with LLaMA-Factory: dataset, train (validated each epoch), merge the best

## Models

| model | task | backbone, checkpoint | tuning | its input format |
|---|---|---|---|---|
| AV-HuBERT | t1 | LARGE, large_vox_iter5.pt (self-supervised, LRS3 + VoxCeleb2) | layernorm | grayscale 88 x 88; one step per clip frame, the 32 frames at their own steps; log filterbank, 4 per step |
| AV-HuBERT | t2 | the same | layernorm | the same video, no audio: the video branch alone (audio features zero, as when pretraining dropped the audio) |
| AV-HuBERT | t3 | the same (pair.py) | layernorm | two clips per question: a face question's videos as t2's; a voice question's sounds through the audio branch alone (video features zero), every step of the audio |
| HAVIC | t1 | ViT-B video + audio encoders and interaction, pt_model.200.pth (self-supervised on real videos) | layernorm, plus its token reducers and pools (not in the checkpoint) and the 3 video tensors stretched 8 → 16 segments | RGB 224; 32 frames as 16 video segments; fbank 128 resized to 1024 steps |
| PE-AV | t1 | Large (pe-av-large; contrastive, about 100M videos, a foundation model): audio, video (PE-L per frame) and audio-video encoders | layernorm (and the Transformers' RMSNorms) | RGB 336; 32 frames; the 48 kHz waveform, one step per audio token (25 per second) |
| PE-AV | t2 | the same: its video encoder alone | layernorm (and RMSNorms) | the same video, no audio |
| PE-AV | t3 | the same: its video encoder (face questions) or audio encoder (voice questions) alone (pair.py) | layernorm | two clips per question: a face question's videos as t2's; a voice question's waveforms, no frames |
| DFD-HR | t2 | CLIP ViT-L/14 (ViT-L-14.pt), with its MoE adapters and routers | frozen: CLIP frozen, its adapters, token routers and query attention train (not in CLIP's checkpoint) | 448 per frame: a 224 global view and 2 x 2 tiles of 224 |

Each model folder (`classifiers/<model>/`): **model.py** (main: Model, a BaseClassifier),
preprocess.py (Processor: the protocol's clip → its input), requirements.txt, README.md
(checkpoint, input, what differs from the original).

## Classes

The head has one output per class; index i is the class below. The order comes from the task's
annotation code (common/data.py, CLASSES), the same keys evaluation's from_probs and metrics use.

| index | t1 (from gt_label) | t1 answer | t2 (from gt_target_name) | t2 answer | t3 (from gt_target_name) | t3 answer |
|---|---|---|---|---|---|---|
| 0 | RVRA | neither | real | real | real_same_person | real, same person |
| 1 | RVFA | the audio | face_swap | face swap | real_different_person | real, different person |
| 2 | FVRA | the video | fully_generated | fully generated | fake_this_person | fake of this person |
| 3 | FVFA | both | full_body_swap | full-body swap | fake_different_person | fake of a different person |

The probabilities go to evaluation as score ({class: p}); the top class is the answer.
T1 gets video and audio, T2 the video only (common/data.py, MODALITIES), as in evaluation. T3 has two clips
per question: a face question gets their frames, a voice question their sound (common/pairs.py), as in evaluation.

## Protocol: the same for every model

| | |
|---|---|
| input | 32 frames, evenly spaced, full frame (no crops), and the complete audio at 16 kHz: what every evaluated model gets. A model's processor only converts it to its format. |
| alignment | the audio is never cut to the frames. Each sampled frame keeps its index in the clip (its time), and the processor lines it up with the audio at that time where the model joins them per step: AV-HuBERT puts frame i at step i. HAVIC joins audio and video by cross-attention over all tokens, so it needs no step alignment: its frames keep their order, the audio is resized to 1024 steps. PE-AV lines the frames up with its audio tokens itself (nearest in time); a token is 1920 samples at 48 kHz, one clip frame. |
| padding | to the longest clip in the batch; masked in the backbone, pooling and loss |
| head | ours: masked mean over time, then an mlp to the task's classes. A model's own classifier is dropped. DFD-HR keeps its own per-frame scheme (per_step): the mlp on each frame, the clip's logits the mean of the frames' logits. T3 (common/pairs.py): each clip pooled on its own and layer-normed, then an mlp per part on both clips' [a, b, \|a - b\|, a * b]. |
| questions | the train and val files (qas/; MOZAIC-Small's with --subset small), with the same filters as evaluation (t1, t3: arms vocoded; t2: none) |
| loss | cross-entropy only, weighted by class from the train counts (T1 is 2:1; T2's full-body swap is 10%). DFD-HR: on every frame, with its video's label, as released. |
| training | AdamW, lr 1e-4, weight decay 0.01, 10 epochs, batch 32 (micro-batches for memory), bf16, seed 0, 4 A40 (T3: 4 A100, the gradients shared every micro-step) |
| selection | best.pt by val macro F1, scored by evaluation's own metrics each epoch. Not AUROC: an MLLM has no probabilities, so its AUROC is only balanced accuracy. |

`tuning.mode`: frozen (head), layernorm (head + LayerNorms, and RMSNorms), full. A model's parts not in its
checkpoint always train; a backbone not fully trained stays in eval mode (BatchNorm, dropout).

## Run

    sbatch scripts/training/train.sh --config avhubert/t1_ln --limit 64    # smoke run
    sbatch scripts/training/train.sh --config avhubert/t1_ln               # full run, 4 GPUs, the full dataset
    sbatch scripts/training/train.sh --config avhubert/t1_ln --subset small  # on MOZAIC-Small
    sbatch --gres=gpu:a100:4 --cpus-per-task=24 scripts/training/train.sh --config pe_av/t3_full --subset small  # T3: MOZAIC-Small only, an A100 node
    torchrun --nproc_per_node=4 -m mozaic.training.run_train --config avhubert/t1_ln [--limit N] [--overwrite] [--subset small]

A rerun resumes after the last finished epoch. A rerun with other settings is refused: change
experiment_name, or --overwrite.

## Output

    training_small/{model}/{experiment_name}/{task}/      (training/ on the full dataset)
    ├── train_settings.json      the locked settings
    ├── history.jsonl            per epoch: losses, accuracy, macro F1, balanced accuracy, kappa
    ├── val_scores/epoch_N.json  evaluation's full scores
    ├── val_scores.json          the best epoch's
    ├── best.pt                  trained weights only; load on top of the original checkpoint (test: evaluation, configs/<model>/<task>_<training>_train<full|small>.yaml)
    ├── last.pt                  to resume; removed once every epoch is done
    └── wandb/                   W&B's local files (the run is logged online)

## Adding a model

1. `classifiers/<model>/` with model.py, preprocess.py, requirements.txt, README.md.
2. A branch in load_model (run_train.py), importing inside the branch.
3. `configs/<model>/<task>.yaml`, and its environment `.venv_<model>`.
4. For T3: `classifiers/<model>/pair.py` (PairProcessor and PairClassifier from common/pairs.py) and its branch in load_model.

A new task: its classes in CLASSES and label() (common/data.py), and load_metrics (common/loop.py).
