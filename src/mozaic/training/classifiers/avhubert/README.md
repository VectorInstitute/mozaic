# AV-HuBERT

Shi et al., "Learning Audio-Visual Speech Representation by Masked Multimodal Cluster
Prediction", ICLR 2022. Code: [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert).

A T1, T2 and T3 backbone: audio and video are encoded per step (a ResNet-18 for video, a linear layer for
audio), joined per step, and passed through a Transformer. We take its per-step features and
put our head on top.

    avhubert/
    ├── model.py           main: Model, the backbone through fairseq, extract_finetune, tunable
    ├── preprocess.py      Processor: the protocol's clip → video and audio steps
    ├── pair.py            T3: Model and Processor for two clips per question (../../common/pairs.py)
    ├── requirements.txt
    └── README.md

    configs/avhubert/t1_ln.yaml    video + audio    sbatch scripts/training/train.sh --config avhubert/t1_ln
    configs/avhubert/t2_ln.yaml    video only       sbatch scripts/training/train.sh --config avhubert/t2_ln
    configs/avhubert/t3_ln.yaml        two clips                 sbatch --gres=gpu:a100:4 --cpus-per-task=24 scripts/training/train.sh --config avhubert/t3_ln --subset small
    configs/avhubert/t3_full.yaml   two clips, tuning: full   sbatch --gres=gpu:a100:4 --cpus-per-task=24 scripts/training/train.sh --config avhubert/t3_full --subset small

## What differs from the original

| | original | here | why |
|---|---|---|---|
| video | mouth crops (88 x 88), 25 fps, every frame | full frame resized to 88 x 88; 32 frames, evenly spaced | the protocol's input |
| frames vs audio | every frame has its frame | the 32 frames at their own steps, the steps between empty; the audio fills every step | complete audio, aligned with each frame |
| output | their fine-tuning decoders (ASR, VSR) | per-step features, masked mean, our mlp head (4 classes) | the protocol's head |
| training | full fine-tuning | LayerNorms + head | tuning: layernorm |

## Checkpoint

`large_vox_iter5.pt`: LARGE (24 layers, 1024-d, 325M parameters), self-supervised pretraining on
LRS3 + VoxCeleb2 (1,759 h), no fine-tuning. VoxCeleb2 is also where MOZAIC's real speakers come
from, so the backbone has heard them in pretraining.

    https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt

It sits in $MOZAIC_ROOT/weights/avhubert/ (backbone.checkpoint in the config).

## Input

The protocol's clip (32 frames, evenly spaced, full frame; the complete audio at 16 kHz),
in AV-HuBERT's format (preprocess.py):

| | |
|---|---|
| steps | one per frame of the clip (77 or 153), the rate AV-HuBERT joins audio and video at |
| video | the 32 frames at their own steps, the other steps empty; grayscale (its first conv takes 1 channel), resized to 88 x 88, mean 0.421, std 0.165 |
| audio | log filterbank (26 bins, 100 per second), 4 stacked per step (104), fitted to the frame count, layer norm per step |

Features come from extract_finetune, which never drops a modality: its forward() drops one at
random in training mode. T2 has no sound: the processor gives no audio and extract_finetune runs
the video branch alone, with the audio features zero, as when pretraining dropped the audio
(modality dropout).

T3 has two clips per question (pair.py): a face question's two videos as T2's; a voice question's two
sounds with no frames, so extract_finetune runs the audio branch alone, with the video features zero
(as when pretraining dropped the video), and every step of the audio is kept (25 per second: the
clips' frame rate, so the same steps as T1's). Each clip is pooled on its own and layer-normed, and
our head per part compares the two (../../common/pairs.py).

## Training

tuning layernorm: the head and the backbone's LayerNorms train (about 0.1M parameters); the rest
stays frozen and in eval mode (its ResNet's BatchNorm statistics, dropout, layer drop).

## Environment

Their fairseq fork, on Python 3.10 and torch 2.3.1:

    git clone https://github.com/facebookresearch/av_hubert external-repos/av_hubert
    git -C external-repos/av_hubert submodule update --init    # fairseq at afc77bd
    uv venv .venv_avhubert --python 3.10
    uv pip install --python .venv_avhubert/bin/python torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    uv pip install --python .venv_avhubert/bin/python -r src/mozaic/training/classifiers/avhubert/requirements.txt -e .
    .venv_avhubert/bin/python -m pip install --no-build-isolation -e external-repos/av_hubert/fairseq

backbone.code in the config points at the clone; the model imports its avhubert/ package from there.
