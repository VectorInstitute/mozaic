# PE-AV

Vyas et al., "Pushing the Frontier of Audiovisual Perception with Large-Scale Multimodal
Correspondence Learning", 2025 (arXiv 2512.19687). Code: [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models);
weights: [facebook/pe-av-large](https://huggingface.co/facebook/pe-av-large) (Apache 2.0), through transformers.

A T1, T2 and T3 backbone, and a foundation model, not a deepfake detector: Meta's Perception Encoder
extended to audio, trained contrastively on about 100M videos (audio, video, audio-video and
their captions). We take its per-step features and put our head on top.

    pe_av/
    ├── model.py           main: Model, the audio-video encoder (t1) or the video encoder (t2), tunable
    ├── preprocess.py      Processor: the protocol's clip → frames (32, 3, 336, 336) and waveform steps (T, 1920)
    ├── pair.py            T3: Model and Processor for two clips per question (../../common/pairs.py)
    ├── requirements.txt
    └── README.md

    configs/pe_av/t1_ln.yaml    video + audio    sbatch scripts/training/train.sh --config pe_av/t1_ln
    configs/pe_av/t2_ln.yaml    video only       sbatch scripts/training/train.sh --config pe_av/t2_ln
    configs/pe_av/t1_full.yaml, t2_full.yaml   tuning: full (t1 with gradient checkpointing)
    configs/pe_av/t3_ln.yaml        two clips                 sbatch --gres=gpu:a100:4 --cpus-per-task=24 scripts/training/train.sh --config pe_av/t3_ln --subset small
    configs/pe_av/t3_full.yaml   two clips, tuning: full   sbatch --gres=gpu:a100:4 --cpus-per-task=24 scripts/training/train.sh --config pe_av/t3_full --subset small

## Network

| part | |
|---|---|
| video encoder | PE-L (ViT-L/14, 336) on each frame, attention-pooled to one vector per frame; a 4-layer Transformer over the frames |
| audio encoder | DAC-VAE (frozen, as released) on the 48 kHz waveform: 25 tokens per second; a 28-layer Transformer |
| audio-video encoder | the frames aligned to the audio tokens (nearest), joined per token, a 6-layer Transformer |

T1 takes the audio-video encoder's output, one step per audio token (25 per second, the clips' frame
rate). T2 has no sound: the video encoder alone (the same weights as inside the audio-video
encoder), one step per frame. The text encoder and the contrastive heads are not kept.

T3 has two clips per question (pair.py): a face question's two videos go through the video encoder,
as T2's; a voice question's two sounds through the audio encoder alone (the same weights as inside
the audio-video encoder), one step per audio token. Each clip is pooled on its own and layer-normed
(a few of PE-AV's features are near 100), and our head per part compares the two (../../common/pairs.py).

## What differs from the original

| | original | here | why |
|---|---|---|---|
| video | every frame (pe-av-large) | 32 frames, evenly spaced, full frame, resized to 336 | the protocol's input |
| audio | 48 kHz | the clips' 16 kHz resampled to 48 kHz on load | the protocol's audio, at the model's rate |
| output | pooled embeddings, contrastive with text | per-step features, masked mean, our mlp head | the protocol's head |
| training | contrastive pretraining | RMSNorms and LayerNorms + head | tuning: layernorm |

## Checkpoint

`facebook/pe-av-large`, the largest released: in $MOZAIC_ROOT/weights/pe_av/pe-av-large/
(backbone.weights in the config), without the tokenizer.

    hf download facebook/pe-av-large --local-dir $MOZAIC_ROOT/weights/pe_av/pe-av-large --exclude 'tokenizer*' --exclude special_tokens_map.json

## Input

The protocol's clip in PE-AV's format (preprocess.py), checked against its processor:

| | |
|---|---|
| video | RGB, resized to 336 x 336 (bilinear), mean 0.5, std 0.5 |
| audio | the complete audio at 48 kHz in [-1, 1], reflect-padded to a multiple of 1920 samples, one step per 1920 (an audio token) |

## Training

tuning layernorm: the head and the backbone's norms (PE-L's LayerNorms, the Transformers'
RMSNorms) train. micro_batch_size 1: 32 frames through PE-L at 336 per clip. T3 (t3, and t3_full with full
tuning): gradient checkpointing, micro_batch_size 1: a face question is 2 x 32 frames.

## Environment

    uv venv .venv_pe_av --python 3.11
    uv pip install --python .venv_pe_av/bin/python torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python .venv_pe_av/bin/python -r src/mozaic/training/classifiers/pe_av/requirements.txt -e .
