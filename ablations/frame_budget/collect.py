"""T1 accuracy and balanced accuracy at 8 / 16 / 32 / 64 frames on MOZAIC-Small, from each run's scores.json."""
import json

from mozaic.common.paths import results_dir, use_subset

RUNS = [("Gemini 3.8 Flash", "gemini", "zeroshot_{n}f_3_8_flash_test"),
        ("Qwen3-Omni", "qwen3_omni_vllm", "zeroshot_{n}f_224_test"),
        ("Nemotron 3 Nano Omni", "nemotron_omni_vllm", "zeroshot_{n}f_224_test"),
        ("Ola", "ola", "zeroshot_{n}f_224_test"),
        ("MiniCPM-o 4.5", "minicpm_o_vllm", "zeroshot_{n}f_224_test")]
use_subset("small")
print("model,frames,n,accuracy,lo,hi,balanced_accuracy")
for name, model, exp in RUNS:
    for n in (8, 16, 32, 64):
        path = results_dir(model, exp.format(n=n), "t1") / "scores.json"
        if not path.exists():
            print(f"{name},{n},,,,,")
            continue
        d = json.loads(path.read_text())
        print(f"{name},{n},{d['n']},{d['accuracy']:.4f},{d['ci95'][0]:.4f},{d['ci95'][1]:.4f},{d['balanced_accuracy']:.4f}")
