# T1 modality ablation

T1 zero-shot on MOZAIC-Small with one modality removed: video only (the 32 frames, no sound) and audio only (the
sound, no frames). The both-modalities numbers are the existing 32-frame runs. Models: the ablation plan's Gemini
3.8 Flash, Nemotron 3 Nano Omni, Ola and MiniCPM-o 4.5, and Qwen3-Omni
(`MODELS=qwen3_omni_vllm bash ablations/modality/submit.sh`).

Each config (written by submit.sh into configs/, git-ignored) is the model's zero-shot config with `modalities: [video]` or `[audio]` under t1 and its own
experiment_name (<experiment>_videoonly / _audioonly). run_inference then sends only that modality, and the
prompt's first line says what is not given ("The clip's soundtrack is not provided." / "The clip's frames are not
provided."); question, options and reply format are unchanged (evaluation/README.md, Configs).

    bash ablations/modality/submit.sh
    python ablations/modality/collect.py
