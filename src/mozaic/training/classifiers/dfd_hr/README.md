# DFD-HR

Jiamu Sun, Zhiyuan Yan, Ke-Yue Zhang, Taiping Yao, Shouhong Ding. "DFD-HR: Generalizable Deepfake
Detection via Hierarchical Routing Learning", CVPR 2026. Code: [Disguiser15/DFD-HR](https://github.com/Disguiser15/DFD-HR).
Ported from the deepfake-moe re-implementation (models/dfd_hr/), as released.

A T2 backbone (video only): frozen CLIP ViT-L/14 with MoE adapters and routing, per frame.

    dfd_hr/
    ├── model.py           main: Model, the routed CLIP pass per frame, tunable
    ├── clip.py            OpenAI CLIP ViT whose blocks expose their sublayers; loads ViT-L-14.pt
    ├── routing.py         MoE adapter, token and layer routers, Gumbel sigmoid
    ├── preprocess.py      Processor: the protocol's 32 frames → (32, 3, 448, 448) in [-1, 1]
    ├── requirements.txt
    └── README.md

    configs/dfd_hr/t2_adapters.yaml    sbatch scripts/training/train.sh --config dfd_hr/t2_adapters

## Network (as released)

| part | |
|---|---|
| encoder | OpenAI CLIP ViT-L/14 at 224, frozen |
| views | each 448 frame as one 224 global view and 2 x 2 tiles of 224 (5 CLIP passes per frame) |
| adapters | an MoE adapter (4 experts, all active, bottleneck 0.25) after the attention and the MLP of all 24 blocks |
| routing | in blocks 21 to 23, a layer router may stop a view and a token router passes the top 75% of tokens, re-weighted by their softmaxed scores |
| frame feature | the projected global class token and a 4-head query-attention summary of the 4 tile class tokens (1536) |

As released, the layer routers get no gradient (their exit decision acts only through indexing),
so they keep their initial weights; they are frozen explicitly.

## What differs from the original

| | original | here | why |
|---|---|---|---|
| input | face crops or frames at 448, their sampling | 32 frames, evenly spaced, full frame, resized to 448 | the protocol's input |
| head | linear on each frame's feature | our mlp on each frame's feature (4 classes) | the protocol's head |
| clip prediction | mean of the frame logits | the same | DFD-HR's own (per_step) |
| loss | per-frame CE (each frame with its video's label) + 0.1 x token-ranking term | per-frame CE only | DFD-HR's own CE. The ranking term comes from argsort ranks, so in the released code it carries no gradient: dropping it changes no weight |
| classes | 2 (real, fake) | 4 (t2: real, face_swap, fully_generated, full_body_swap) | the task |

## Training

tuning frozen: CLIP stays frozen and in eval mode. The adapters, token routers and query attention
are not in CLIP's checkpoint, so they train with the head (tunable()["new"]): DFD-HR's own
parameter-efficient training. A layer router may stop a view, so DDP looks for parameters without a
gradient (dynamic_graph).

Memory: 5 CLIP-L passes per frame, 32 frames per clip, with gradient checkpointing per block;
micro_batch_size 2 per GPU, accumulated to batch 32.

## Checkpoint

OpenAI's ViT-L-14.pt (sha256 b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836), in
$MOZAIC_ROOT/weights/clip/ (backbone.checkpoint in the config):

    https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt

## Environment

    uv venv .venv_dfd_hr --python 3.10
    uv pip install --python .venv_dfd_hr/bin/python torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    uv pip install --python .venv_dfd_hr/bin/python -r src/mozaic/training/classifiers/dfd_hr/requirements.txt -e .
