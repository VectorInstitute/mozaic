# tests

Made-up data only: no videos are read and no API is called (a fake client stands in for
Gemini).

    tests/
    ├── made_up.py           a made-up release: metadata.jsonl and summary.json for 7 pairs
    ├── test_annotation.py   T1, T2, T3 questions, their guards and samples
    ├── test_evaluation.py   run_inference, run_eval, the Gemini message and the sound check
    ├── conftest.py          gives data-generation tests an empty paths file when none is set
    ├── test_data_generation_inputs.py    clip list, identity split, pair draw and its checks
    ├── test_data_generation_final_clips.py  the mux and the removal of half-made pairs
    ├── test_data_generation_gates.py     the checks and the keep/drop rule
    ├── test_data_generation_generators.py  the GPU libraries and offline use the generator runners set up
    ├── test_data_generation_release.py   the release build's steps and files
    └── test_subsets.py      MOZAIC-Small's draw (every person, the mix, the per-person limit, the seed),
                             its four _small files, the --subset paths, and Small's T1 and T3 rows
                             equal to the full build's rows of Small's pairs

Run from the repo root, in an environment with pytest, pyyaml, numpy and pillow
(openai for the Gemini tests, which are skipped without it; the data-generation tests also need
src/mozaic/data_generation/requirements.txt):

    PYTHONPATH=src python -m pytest tests
