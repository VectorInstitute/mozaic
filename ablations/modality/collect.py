"""T1 accuracy on MOZAIC-Small with both modalities, video only and audio only, from each run's scores.json.

video_bal and audio_bal: balanced real/fake accuracy of that stream, the mean of fakes called fake and reals called
real (chance 50%), as in the paper's Table 8. Plain video_acc and audio_acc follow the label mix (2/3 of videos are
fake), so a model that always says "fake video" scores high on video_acc with no frames at all."""
import json

from mozaic.common.paths import results_dir, use_subset

RUNS = [("Gemini 3.8 Flash", "gemini", "zeroshot_32f_3_8_flash{m}_test"),
        ("Qwen3-Omni", "qwen3_omni_vllm", "zeroshot_32f_224{m}_test"),
        ("Nemotron 3 Nano Omni", "nemotron_omni_vllm", "zeroshot_32f_224{m}_test"),
        ("Ola", "ola", "zeroshot_32f_224{m}_test"),
        ("MiniCPM-o 4.5", "minicpm_o_vllm", "zeroshot_32f_224{m}_test")]
use_subset("small")


def bal(stream):
    """Balanced accuracy of one stream: the mean of recall (fakes called fake) and specificity (reals called real)."""
    return (stream["recall"] + stream["specificity"]) / 2


print("model,input,n,accuracy,lo,hi,balanced_accuracy,macro_f1,video_acc,audio_acc,video_bal,audio_bal")
for name, model, exp in RUNS:
    for label, m in (("both", ""), ("video only", "_videoonly"), ("audio only", "_audioonly")):
        path = results_dir(model, exp.format(m=m), "t1") / "scores.json"
        if not path.exists():
            print(f"{name},{label}" + "," * 10)
            continue
        d = json.loads(path.read_text())
        print(f"{name},{label},{d['n']},{d['accuracy']:.4f},{d['ci95'][0]:.4f},{d['ci95'][1]:.4f},"
              f"{d['balanced_accuracy']:.4f},{d['macro']['f1']:.4f},{d['video']['accuracy']:.4f},"
              f"{d['audio']['accuracy']:.4f},{bal(d['video']):.4f},{bal(d['audio']):.4f}")
