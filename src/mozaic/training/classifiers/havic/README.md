# HAVIC

HAVIC, CVPR 2026 (arXiv 2603.23960). Code: [tuffy-studio/HAVIC](https://github.com/tuffy-studio/HAVIC)
(MIT, at 66f4908); weights: [JielunPeng/HAVIC](https://huggingface.co/JielunPeng/HAVIC).

A T1 backbone: video and audio ViT-B encoders, an audio-visual interaction layer, and HAVIC's
aggregation; our head on top.

    havic/
    ├── model.py           main: Model, loads pt_model.200.pth (video stretched 8 → 16 segments), encode, tunable
    ├── preprocess.py      Processor: the protocol's clip → frames (32, 3, 224, 224) and fbank (1024, 128)
    ├── vendor/            HAVIC's model code and LICENSE, as released except for the segment count
    ├── requirements.txt
    └── README.md

    configs/havic/t1_ln.yaml       sbatch scripts/training/train.sh --config havic/t1_ln
    configs/havic/t1_full.yaml  tuning: full

## Network

| part | |
|---|---|
| video encoder | ViT-B (12 layers, 768-d), tubelets of 2 x 16 x 16; one cls token per time segment |
| audio encoder | ViT-B, 16 x 16 patches of the fbank; one cls token per segment. Its 8 segments are mel bands: patches are ordered band-major (8 bands x 64 time steps) |
| interaction | one layer of audio→video and video→audio cross-attention over all tokens, on the last layer |
| aggregation | a token reducer (learned attention pooling) on layers 3, 6, 9, 12 of each encoder and on each interaction output; a learned weighted pool over the 4 layers; concatenated: audio, video, audio-interaction, video-interaction (3072) |

The checkpoint is HAVIC's self-supervised pretraining (MAE and contrastive, on real videos only):
its encoders and interaction layer. The token reducers and pools are not in it (HAVIC trains them
at fine-tuning), so they start from scratch.

## What differs from the original

| | original | here | why |
|---|---|---|---|
| video | 16 face crops (FaceX-Zoo), 224 | 32 frames, evenly spaced, full frame, resized to 224 | the protocol's input |
| video segments | 8 | 16: pos_embedding (1568 → 3136), cls_tokens and cls_pos_embed (8 → 16) interpolated along time on load | 32 frames in one pass; every other weight loads unchanged (strict) |
| audio | fbank resized to 1024 steps | the same, from the complete audio | the protocol's input; no change needed |
| heads | 3 BCE classifiers (overall, audio, video) | dropped; our mlp head on the 3072-d feature, 4 classes | the protocol's head |
| loss | 3 BCE terms | CE on the 4 classes | the protocol's loss |
| augmentation | flip, crop, SpecAugment | none | the protocol's input |
| training | full fine-tuning, lr 1e-5 (heads x10) | LayerNorms + head + the new parts + the stretched video tensors | tuning: layernorm; the protocol's recipe |

## Training

tuning layernorm: the encoders' and interaction layer's LayerNorms train with the head. Always
trained (tunable()["new"]): the token reducers and pools (not in the checkpoint), and the three
video tensors stretched to 16 segments. Memory: 3,152 video tokens per clip, so micro_batch_size 4.

As released, each interaction block also runs a self-attention on video and audio whose output is
never returned, so its LayerNorms get no gradient; DDP looks for such parameters (dynamic_graph).

## Checkpoint

`pt_model.200.pth` (973 MB) in $MOZAIC_ROOT/weights/havic/ (backbone.checkpoint in the config).

## Environment

    uv venv .venv_havic --python 3.10
    uv pip install --python .venv_havic/bin/python torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    uv pip install --python .venv_havic/bin/python -r src/mozaic/training/classifiers/havic/requirements.txt -e .
