# MiniCPM-o 4.5 with vLLM

OpenBMB's MiniCPM-o-4_5 (9.4B, Apache-2.0) with vLLM, in its own environment (vLLM 0.28.0, the same
versions as nemotron_omni_vllm/). An omni model (SigLIP2 vision, Whisper sound, Qwen3-8B language model)
that takes video frames and sound in one question. It runs through run_inference and run_eval like every
model, batch_size questions at a time, which vLLM answers together (as gemma4_vllm/).

    minicpm_o_vllm/
    ├── model.py           reads each clip with vLLM's readers (reads_clips), answers a batch with vLLM
    └── requirements.txt   everything it needs, at the versions it was tested with (vLLM 0.28.0)

Settings live in evaluation/configs/minicpm_o_vllm.yaml.

## Input

Each clip is read by vLLM's own readers, as for Nemotron and Gemma: 32 frames evenly spaced from the first
to the last (its video reader's num_frames: frames) and the sound at 16 kHz. Per clip its label ("Clip 1:"
in t3), its video and/or its sound, then the prompt: the prompts of evaluation/prompts/, as for every model.
Answers are greedy.

- **Thinking off.** MiniCPM-o thinks before it answers by default. chat_template_kwargs enable_thinking:
  false makes it answer directly, so a T1-T3 answer fits in 256 tokens, as for every model.
- **The sound comes as its own part.** vLLM takes MiniCPM-o's sound as its own item (its "omni mode",
  where every second of frames and sound comes together, exists only in its own transformers code). So t1
  sends the clip's video, then its sound, then the prompt (as Gemma and Gemini get them). t2 and t3's face
  questions send silent videos; t3's voice questions the two sounds.
- **The tags are written into the text.** MiniCPM-o's chat template takes plain text, so model.py writes
  the tags vLLM swaps for the clips (`(<video>./</video>)`, `(<audio>./</audio>)`), each after its clip's label.
- **Frames at the model's own size.** MiniCPM-o turns each frame into 64 tokens itself, whatever its size.
- **The reply is held to the options.** Asked for JSON only, MiniCPM-o often writes a sentence around it
  ('Based on the video, the answer is {"answer": "neither"}.'), which the prompts' parse() can't read (30 of
  30 in our first test). With answer_choices: true, vLLM lets it write only `{"answer": "<option>"}` with one
  of the question's options (vLLM's structured outputs, `choice`), so every reply is read. MiniCPM-o still
  picks the option itself. Only this model does this; the prompts and parse() are the same as for every model.

## Did the sound arrive?

Every prediction's usage counts the tokens the model got of each kind (between their start and end marks):

    audio_tokens    30 per 77-frame clip, 61 per 153-frame clip (10 per second, in 1-second pieces)
    video_tokens    2,048 per clip: 32 frames x 64

## Environment

The same versions as Nemotron's, so one environment serves both (inference.sh opens .venv_<MODEL>):

    ln -s .venv_nemotron_omni_vllm .venv_minicpm_o_vllm

or its own, from requirements.txt:

    uv venv .venv_minicpm_o_vllm --python 3.12.12 --managed-python
    uv pip install --python .venv_minicpm_o_vllm/bin/python --torch-backend cu129 \
        --extra-index-url https://wheels.vllm.ai/0.28.0/cu129 -e . \
        -r src/mozaic/evaluation/models/mllms/minicpm_o_vllm/requirements.txt

model.py turns off vLLM's FlashInfer sampler, as for Nemotron and Gemma: it compiles a kernel with CUDA's
nvcc when vLLM starts, and our GPU nodes have no nvcc (vLLM's PyTorch sampler gives the same greedy answers).

Download the weights into models_weights/ (19 GB):

    hf download openbmb/MiniCPM-o-4_5 --revision d59117d19ddb8a28a62f1b5f4e3b34ac48d510a4 \
        --local-dir models_weights/minicpm-o-4_5

One fix in the downloaded code: vLLM (0.28 up to 0.32) reads the processor's sound part as
`feature_extractor`, but MiniCPM-o's processing_minicpmo.py calls it `audio_processor`, so vLLM stops when it
starts. In class MiniCPMOProcessor, right after its `__init__`, add (keep the original as
processing_minicpmo.py.orig):

    @property
    def feature_extractor(self):  # vLLM reads this name; this code calls it audio_processor
        return self.audio_processor

It is only a second name for the same sound processor: nothing the model gets changes.

## Run

The global scripts, as every model (MODEL, TASK, SPLIT, SUBSET at their top; TASK empty: every task):

    sbatch scripts/evaluation/inference.sh --limit 10           # 1 GPU (an A40 is enough)
    sbatch scripts/evaluation/evaluate.sh                       # scores; the same MODEL, TASK, SPLIT

As for the other vLLM models: run each task on one kind of GPU from start to end, results in
results/{model}/{experiment_name}_{split}/{task}/, and a rerun continues where the last one stopped.
