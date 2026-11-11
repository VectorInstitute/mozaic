# external-repos

Third-party code the models run, cloned here (git-ignored: only this README and .gitkeep are tracked). Each
config's `code:` (or AV-HuBERT's `backbone.code`) points at its clone, relative to the repo root. The steps per
model are in its README; the commits are the ones the reported runs used.

| Folder | Repository | Commit | Used by | Steps |
|---|---|---|---|---|
| av_hubert/ | github.com/facebookresearch/av_hubert (and its fairseq submodule, afc77bd) | submodule as cloned | AV-HuBERT training and evaluation | training/classifiers/avhubert/README.md |
| ola/ | github.com/Ola-Omni/Ola | f681820 | Ola-7B | evaluation/models/mllms/ola/README.md |
| uni_moe_2/ | github.com/HITsz-TMG/Uni-MoE | 6f18a7a | Uni-MoE-2.0-Omni (code: uni_moe_2/Uni-MoE-2) | evaluation/models/mllms/uni_moe_2_omni/README.md |
| videollama2/ | github.com/DAMO-NLP-SG/VideoLLaMA2, branch audio_visual | 8cedfbd | VideoLLaMA2.1-AV | evaluation/models/mllms/videollama2_av/README.md |
| vita/ | github.com/VITA-MLLM/VITA | 35d064a | VITA-1.5 | evaluation/models/mllms/vita_1_5/README.md |
| wan2.2/ | github.com/Wan-Video/Wan2.2 | 42bf4cf | data generation: Wan2.2-S2V and Wan2.2-Animate | data_generation/generators/s2v/README.md |
| facefusion/ | github.com/facefusion/facefusion | 3f81a8a | data generation: FaceFusion | data_generation/generators/facefusion/README.md |
| seed-vc/ | github.com/Plachtaa/seed-vc | 51383ef | data generation: Seed-VC | data_generation/generators/seed_vc/README.md |
| syncnet_python/ | github.com/joonson/syncnet_python | 907c0b5 | data generation: the lip-sync check | data_generation/gates/README.md |

AV-HuBERT's fairseq is an editable install from av_hubert/fairseq: after moving a clone, install it again
(training/classifiers/avhubert/README.md, Setup). Paths under src/mozaic/ in those READMEs are relative to the
repo root.
