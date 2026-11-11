# mllms
One folder per open MLLM, each with its own requirements.txt.

    qwen_omni_vllm/      Qwen3-Omni and Qwen2.5-Omni with vLLM, in its own environment, questions in batches
    nemotron_omni_vllm/  Nemotron 3 Nano Omni (not Qwen) with vLLM, the same way
    gemma4_vllm/         Gemma 4 12B (Google) with vLLM, the same way
    minicpm_o_vllm/      MiniCPM-o 4.5 (OpenBMB) with vLLM, the same way
    videollama2_av/      VideoLLaMA2.1-AV (DAMO) with transformers and its authors' code, one question at a time
    omnivinci/           OmniVinci (NVIDIA) with transformers and its own code from the hub, one question at a time
    uni_moe_2_omni/      Uni-MoE-2.0-Omni (HIT-TMG) with transformers and its authors' code, one question at a time
    ola/                 Ola-7B (Tsinghua, Tencent, NTU) with transformers and its authors' code, one question at a time
    vita_1_5/            VITA-1.5 (VITA-MLLM) with transformers and its authors' code, one question at a time
    answer_choices.py    the token filter that keeps a reply to one of the options (Uni-MoE-2.0-Omni,
                         Ola, VITA-1.5: answer_choices: true)
