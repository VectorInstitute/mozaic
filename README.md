<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mozaic_logo_dark.png">
    <img src="docs/assets/mozaic_logo.png" alt="MOZAIC logo" width="220">
  </picture>
</p>

<h1 align="center">MOZAIC: An audio-video deepfake benchmark for telling <em>what</em> is fake</h1>

<p align="center">
  Shaina Raza<sup>1</sup>, Mohamed Hafez<sup>1</sup>, Ahmed Radwan<sup>1</sup>, Khalid Elgazzar<sup>2</sup>,
  Kathryn Hume<sup>1</sup><br>
  <sup>1</sup>Vector Institute &nbsp; <sup>2</sup>Ontario Tech University
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.10-blue.svg" alt="Python ≥ 3.10">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/code-Apache%202.0-blue.svg" alt="code license">
  </a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/data-CC%20BY--NC%204.0-97ca00.svg" alt="data license">
  </a>
</p>

---

## What is MOZAIC

MOZAIC is an audio-video deepfake benchmark built on VoxCeleb2. Every clip pairs a real or fake face with a real or
fake voice, so a model is tested on *what* is fake, not only *whether* something is.

- **Four-way labels.** Every source clip comes as RVRA, RVFA, FVRA and FVFA (real or fake video × real or fake
  audio), plus a vocoded copy of each real voice, so a score splits into video and audio.
- **Three video generators and one voice converter.** FaceFusion (inner-face swap), Wan2.2-S2V (whole frame from
  the voice), Wan2.2-Animate (whole frame from the motion) and Seed-VC.
- **Identity-disjoint splits.** 137,745 clips from 15,014 pairs of 1,000 people; nobody in the test split appears
  in training. MOZAIC-Small is a pair-level subset for training runs, ablations and T4.
- **Twelve zero-shot MLLMs and five trained detectors**, evaluated with the same input and greedy answers.

## Tasks

| Task | Question | Input |
|---|---|---|
| **T1** Audio-video authenticity | Which part is fake: the video, the audio, both, or neither? | frames + audio |
| **T2** Manipulation type | How was the video made: real, face swap, full-body swap, or fully generated? | frames |
| **T3** Identity verification | Is the query real, and is it the same person as the reference? | 2 clips, face or voice |
| **T4** Explanation | What is AI-generated, and what evidence supports it? | frames + audio (MOZAIC-Small) |

## Repository layout

```
mozaic/
├── src/mozaic/
│   ├── data_generation/  1. makes the clips from VoxCeleb2
│   ├── annotation/       2. builds the question files
│   ├── evaluation/       3-4. runs models (inference) and scores them (eval)
│   ├── training/         5. trains the baselines
│   └── common/           config, paths, io, metadata, frame and audio sampling
├── scripts/              Slurm launchers for every step
├── ablations/            the paper's ablations
├── external-repos/       third-party code the models run, cloned here (git-ignored)
├── docs/                 the project page
├── data/                 the dataset, downloaded from Hugging Face (git-ignored)
├── models_weights/       model weights, one folder per model (git-ignored)
└── tests/                PYTHONPATH=src python -m pytest tests
```

Each step has its own README: [data generation](src/mozaic/data_generation/README.md),
[annotation](src/mozaic/annotation/README.md), [evaluation](src/mozaic/evaluation/README.md),
[training](src/mozaic/training/README.md). Shared paths: [common](src/mozaic/common/README.md).
Slurm: [scripts](scripts/README.md). Ablations: [ablations](ablations/README.md).

The repo holds code only. Download the dataset from [Hugging Face](https://huggingface.co/datasets/vector-institute/Mozaic)
into `data/`, or anywhere else, and set `data_root` in `src/mozaic/common/config.yaml` to that absolute path
(`MOZAIC_ROOT` overrides it). Download each model's weights into `models_weights/`, as its README says.

```
data/
├── videos/                                     clips, shared by the full dataset and MOZAIC-Small
├── metadata.jsonl       metadata_small.jsonl   one row per clip
├── qas/                 qas_small/             question files, one folder per task
│   ├── t1_spot_the_fake/
│   ├── t2_manipulation_type/
│   ├── t3_identity_check/
│   └── t4_explanation/                         T4 is MOZAIC-Small only (qas_small/)
├── training/            training_small/        checkpoints, written by training
└── results/             results_small/         predictions and scores, written by evaluation
```

`videos/`, the metadata files and `qas/` come with the download. Every file is described in
[common](src/mozaic/common/README.md).

## Quick start

```bash
git clone https://github.com/VectorInstitute/mozaic.git
cd mozaic
uv sync                     # the core environment (Python ≥ 3.10, uv)
cp .env.example .env        # then add the API keys
```

Each model runs in its own environment, set up as its README says (under
[`evaluation/models`](src/mozaic/evaluation/models) or [`training/classifiers`](src/mozaic/training/classifiers)).

## Usage

The pipeline has five steps. To evaluate a model, start at step 3: download the dataset, clips and question files
included, from [Hugging Face](https://huggingface.co/datasets/vector-institute/Mozaic) and set `data_root` to that
folder. Steps 1 and 2 rebuild it from VoxCeleb2.

**1. Data** ([`data_generation`](src/mozaic/data_generation/README.md)). The clip list, the splits and the pairs, then one
batch per generator, then the release folder. Rendering needs GPUs and each generator's code and weights.

```bash
python -m mozaic.data_generation.run_generation prepare
python -m mozaic.data_generation.run_generation batch --build-id bulk_v1_ff77 --pair-ids-file pairs.txt
python -m mozaic.data_generation.release.build_mozaic plan
```

**2. Questions** ([`annotation`](src/mozaic/annotation/README.md)). One question file per task and split; T4's
written references have their own steps ([`t4_explanation`](src/mozaic/annotation/t4_explanation/README.md)).

```bash
python -m mozaic.annotation.run_annotation --task t1
```

**3. Inference.** A model answers a task's questions; a rerun continues where the last one stopped. A config is
named `<model>/<run>`: `--config gemini_3_8_flash --task t1` reads `gemini_3_8_flash/t1/zeroshot.yaml`, and
`--config avhubert/ln_trainsmall` a trained run ([`evaluation/configs`](src/mozaic/evaluation/configs/README.md)).

```bash
python -m mozaic.evaluation.run_inference --config gemini_3_8_flash --task t1
python -m mozaic.evaluation.run_inference --config gemini_3_8_flash --task t1 --retry-failed   # ask unreadable replies again
```

**4. Evaluation.** Scores from the saved answers, no model needed ([`evaluation/metrics`](src/mozaic/evaluation/metrics/README.md)).
T4 is judged first ([`evaluation/judge`](src/mozaic/evaluation/judge/README.md)).

```bash
python -m mozaic.evaluation.run_eval --config gemini_3_8_flash --task t1

python -m mozaic.evaluation.run_inference --config gemini_3_8_flash --task t4 --subset small
python -m mozaic.evaluation.judge.run_judge --config gemini_3_8_flash --task t4 --subset small
python -m mozaic.evaluation.run_eval --config gemini_3_8_flash --task t4 --subset small
```

Results go to `results/<model>/<experiment>/<task>/` (`results_small/` with `--subset small`).

**5. Training** (optional; [`training`](src/mozaic/training/README.md)). A baseline on MOZAIC-Small, validated each
epoch; then steps 3 and 4 test it.

```bash
torchrun --nproc_per_node=4 -m mozaic.training.run_train --config avhubert/t1_ln --subset small
```

**On Slurm**, every step has a launcher that passes its flags through ([`scripts`](scripts/README.md)), e.g.
`sbatch scripts/evaluation/inference.sh --config gemini_3_8_flash --task t1`. **Ablations**: frames per clip, video
or audio only, seeds, learning rate, epoch budget and a held-out generator ([`ablations`](ablations/README.md)).

<details>
<summary><b>Models</b></summary>

| Zero-shot (32 frames at 224 × 224, sound at 16 kHz, greedy) | Trained on MOZAIC |
|---|---|
| Gemini 3.8 Flash, Gemini 3.1 Pro, Qwen3-Omni, Qwen2.5-Omni-7B, Nemotron 3 Nano Omni, Gemma 4 12B, MiniCPM-o 4.5, VideoLLaMA2.1-AV, OmniVinci, Uni-MoE-2.0-Omni, Ola-7B, VITA-1.5 | AV-HuBERT, PE-AV and HAVIC (LayerNorm-only or full fine-tuning), DFD-HR (adapters), Qwen2.5-Omni-7B (LoRA, T1–T4); T4 LoRA for the open MLLMs |

</details>

## License

- **Code:** Apache License 2.0 ([`LICENSE`](LICENSE)).
- **Dataset** (clips, metadata and question files, on [Hugging Face](https://huggingface.co/datasets/vector-institute/Mozaic)):
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Citation

```bibtex
@inproceedings{TO FILL,
  title     = {{MOZAIC}: An Audio-Video Deepfake Benchmark for Telling What Is Fake},
  author    = {Raza, Shaina and Hafez, Mohamed and Radwan, Ahmed and Elgazzar, Khalid and Hume, Kathryn},
  booktitle = {TO FILL},
  year      = {TO FILL}
}
```

## Acknowledgements

Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of
Canada through CIFAR, and companies sponsoring the Vector Institute
([vectorinstitute.ai/#partners](https://vectorinstitute.ai/#partners)).

This research was funded by the European Union's Horizon Europe research and innovation programme under the AIXPERT
project (Grant Agreement No. 101214389), which aims to develop an agentic, multi-layered, GenAI-powered framework for
creating explainable, accountable, and transparent AI systems.

## Contact

Shaina Raza, [shaina.raza@vectorinstitute.ai](mailto:shaina.raza@vectorinstitute.ai)
