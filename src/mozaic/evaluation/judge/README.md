# T4 judge

Scores T4 explanations (the video and the audio part of each reply) against the reference
explanations, with a model that reads text only. run_eval then puts the scores in scores.json
next to the word-overlap scores (metrics/t4_metrics.py).

    judge/
    ├── config.yaml        the judge (gpt-5.4-mini), the embedder, workers, retries
    ├── prompt.py          the judge's system and user prompt, and how its reply is read
    ├── run_judge.py       per prediction: one judge call and the embeddings -> judge.jsonl
    └── requirements.txt   openai, sentence-transformers

## Per clip

| step | model | sees | gives |
|---|---|---|---|
| judge | gpt-5.4-mini | the reference and the predicted explanation, per part | a score 0-10 and a justification per part |
| cosine | all-MiniLM-L6-v2 (sentence-transformers, local weights) | the same four texts | the cosine of reference and prediction per part |

A prediction that could not be read gets 0 per part, with no call. The judge never sees the
clip, and the model being judged is refused as its own judge.

## Run

The judge has its own environment (openai and sentence-transformers, which pulls in torch);
.venv_gemini has openai only. Once (e.g. in a CPU job):

    uv venv .venv_judge --python 3.10
    uv pip install --python .venv_judge/bin/python -e . -r src/mozaic/evaluation/judge/requirements.txt

After run_inference, with `.venv_judge` and `OPENAI_API_KEY`
in `.env`. In a Slurm job (the embedder is a model: never on the login node); a CPU job is enough, a GPU
is used when there is one. `scripts/evaluation/judge.sh` runs it (then `evaluate.sh` for the scores):

    sbatch scripts/evaluation/judge.sh --config qwen3_omni_vllm --task t4 --subset small --limit 100

By hand:

    # 100 first: the tokens it logs at the end give the cost per clip
    python -m mozaic.evaluation.judge.run_judge --config qwen3_omni_vllm --task t4 --subset small --limit 100
    # the rest (resumable: the same command continues)
    python -m mozaic.evaluation.judge.run_judge --config qwen3_omni_vllm --task t4 --subset small
    # replies that were not the judge's JSON, and the 0s of predictions that read now (inference's --retry-failed)
    python -m mozaic.evaluation.judge.run_judge --config qwen3_omni_vllm --task t4 --subset small --retry-failed
    # then the scores, with .venv
    python -m mozaic.evaluation.run_eval --config qwen3_omni_vllm --task t4 --subset small

If the endpoint's key cannot use the judge model (403 "Project does not have access to this model"),
pick another, or leave the cosine out, without editing config.yaml:

    python -m mozaic.evaluation.judge.run_judge --config qwen3_omni_vllm --task t4 --subset small \
        --judge-model <another model> --no-embedder

Judge scores from different judges are not comparable: judge every model with the same one
(judge_settings.json records it, and a folder judged by another is refused).

Check first which models the endpoint serves (a key may still be refused some of them):

    python -c "from openai import OpenAI; import os; from mozaic.common.config import load_env; load_env(); \
      print(sorted(m.id for m in OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url='https://api.openai.com/v1').models.list()))"

A failed call (the API down) is not saved, so the next run asks again; the run stops after 3
failed clips in a row. A folder judged with other settings (judge, embedder, generation or
prompt) is refused: judge_settings.json records them.

## Files

    results_small/<model>/<experiment>/t4/
    ├── predictions.jsonl, inference_settings.json   run_inference
    ├── judge.jsonl                                  one row per prediction: id, ok, scores
    │                                                {video, audio: score, justification}, cosine
    │                                                {video, audio}, raw, usage
    ├── judge_settings.json                          judge, embedder, generation, prompt_sha256
    └── scores.json                                  run_eval
