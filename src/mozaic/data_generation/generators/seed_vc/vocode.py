"""The vocoded copy of a real voice: its sound through BigVGAN, the vocoder Seed-VC ends in (.venv_seed_vc).

The copy keeps the same words, voice and timing and only gains the trace the vocoder leaves. It stays
labelled real (audio_variant vocoded_real). Every converted voice went through this vocoder, so without
vocoded real voices a detector could call a voice fake just for having been vocoded.

The front end is Seed-VC's own: this file imports modules.audio.mel_spectrogram and modules.bigvgan from
the clone ($SEEDVC_UPSTREAM) and reads the mel settings from Seed-VC's own config file, so a copy goes
through the same mel step and vocoder as a converted voice. A hand-written mel step that differed in
center, win_size or fmax would leave a trace the converted voices do not have.

    16 kHz real wav
      -> resample to 22,050 Hz        (Seed-VC's rate)
      -> mel: n_fft and window 1024, hop 256, 80 bands, fmin 0, fmax none, center off
      -> BigVGAN nvidia/bigvgan_v2_22khz_80band_256x
      -> 22,050 Hz wav
      -> resample to 16 kHz           (the clips' rate; the mux gives every class its exact length)

--report prints the rate at every stage, so the chain can be checked instead of assumed. Like run.py,
this file imports nothing from mozaic.

    python vocode.py --jobs vocode_jobs.jsonl                # every job; a copy that exists is skipped
    python vocode.py --report --in-wav crop.wav              # print every stage's rate
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

TARGET_SR = 16000  # the clips' rate, in and out


def load_seedvc_env() -> tuple[Any, Any, dict[str, Any]]:
    """Import Seed-VC's OWN mel step and vocoder, and read its OWN mel settings.

    Returns (mel_spectrogram_fn, bigvgan_model, mel_kwargs). Nothing here is re-derived: the point is the
    identical front end.
    """
    upstream = os.environ.get("SEEDVC_UPSTREAM")
    if not upstream:
        msg = "$SEEDVC_UPSTREAM is not set — point it at the Seed-VC clone (external-repos/seed-vc)"
        raise SystemExit(msg)
    # Seed-VC's checkpoint loader uses a relative ./checkpoints, so work from the clone (as run.py does).
    os.chdir(upstream)
    sys.path.insert(0, upstream)

    import yaml
    from hf_utils import load_custom_model_from_hf
    from modules.audio import mel_spectrogram
    from modules.bigvgan import bigvgan

    # The same checkpoint and config as the wrapper's default path (no pitch input), so the mel settings
    # below are the ones Seed-VC's voices were made from.
    _, config_path = load_custom_model_from_hf(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
    )
    config = yaml.safe_load(Path(config_path).read_text())
    spect = config["preprocess_params"]["spect_params"]
    mel_kwargs = {
        "n_fft": spect["n_fft"],
        "win_size": spect["win_length"],
        "hop_size": spect["hop_length"],
        "num_mels": spect["n_mels"],
        "sampling_rate": config["preprocess_params"]["sr"],
        "fmin": 0,
        "fmax": None,
        "center": False,
    }
    model = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)
    model.remove_weight_norm()
    return mel_spectrogram, model, mel_kwargs


def vocode(
    in_wav: Path, out_wav: Path, mel_fn: Any, model: Any, mel_kwargs: dict[str, Any], device: Any
) -> dict[str, Any]:
    """Run one clip through resample -> mel -> BigVGAN -> resample.

    Records the sample rate at every stage.
    """
    import torch
    import torchaudio

    vocoder_sr = int(mel_kwargs["sampling_rate"])
    wav, sr = torchaudio.load(str(in_wav))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    stages: dict[str, Any] = {"input_sr": int(sr), "input_samples": int(wav.shape[-1])}

    up = torchaudio.functional.resample(wav, sr, vocoder_sr) if sr != vocoder_sr else wav
    stages["vocoder_sr"] = vocoder_sr
    stages["vocoder_samples_in"] = int(up.shape[-1])

    with torch.no_grad():
        mel = mel_fn(up.to(device).float(), **mel_kwargs)
        stages["mel_bins"], stages["mel_frames"] = int(mel.shape[-2]), int(mel.shape[-1])
        out = model(mel).squeeze(0).cpu()
    stages["vocoder_samples_out"] = int(out.shape[-1])

    down = torchaudio.functional.resample(out, vocoder_sr, TARGET_SR)
    stages["output_sr"] = TARGET_SR
    stages["output_samples"] = int(down.shape[-1])
    # The length is NOT fixed here: the mux brings every class to the exact length with one filter, and a
    # pad or trim here would be a second length edit on this arm only.
    stages["length_delta_samples"] = stages["output_samples"] - stages["input_samples"]

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # A temp name private to this process: builds can share a vocoded copy and run at the same time. Both
    # write the same bytes, and the rename into place is atomic.
    tmp = out_wav.with_name(f"{out_wav.stem}.{socket.gethostname()}.{os.getpid()}.tmp.wav")
    # format is explicit: torchaudio guesses it from the extension, and a ".tmp" in the name makes it fail.
    torchaudio.save(str(tmp), down, TARGET_SR, format="wav")
    tmp.replace(out_wav)
    return stages


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="BigVGAN pass-through for the vocoded-real control arm.")
    p.add_argument("--jobs", type=Path, help="JSONL: {job_id, source_wav, out_wav}")
    p.add_argument("--in-wav", type=Path, help="single-file mode")
    p.add_argument("--out-wav", type=Path, help="single-file mode")
    p.add_argument("--report", action="store_true", help="print every stage's sample rate and exit")
    p.add_argument("--device", default=None, help="cuda/cpu (default: cuda if available)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    """Vocode a job list, or one file, printing the per-stage rates."""
    args = parse_args()
    mel_fn, model, mel_kwargs = load_seedvc_env()

    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.eval().to(device)
    print(f"[vocode] device={device}  mel={mel_kwargs}", flush=True)

    if args.report:
        if not args.in_wav:
            print("--report needs --in-wav", file=sys.stderr)
            return 2
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            stages = vocode(args.in_wav, Path(d) / "probe.wav", mel_fn, model, mel_kwargs, device)
        print("[vocode] PASS-THROUGH CHAIN")
        for k, v in stages.items():
            print(f"  {k:24}= {v}")
        return 0

    if args.in_wav and args.out_wav:
        stages = vocode(args.in_wav, args.out_wav, mel_fn, model, mel_kwargs, device)
        print(json.dumps(stages, indent=2))
        return 0

    if not args.jobs:
        print("need --jobs, or --in-wav/--out-wav, or --report", file=sys.stderr)
        return 2

    failures = 0
    jobs = [json.loads(x) for x in args.jobs.read_text().splitlines() if x.strip()]
    for n, job in enumerate(jobs, start=1):
        out = Path(job["out_wav"])
        if out.exists() and not args.overwrite:
            print(f"[vocode] {n}/{len(jobs)} {job['job_id']} — skip (exists)", flush=True)
            continue
        started = time.time()
        try:
            stages = vocode(Path(job["source_wav"]), out, mel_fn, model, mel_kwargs, device)
        except Exception as exc:  # noqa: BLE001  a failed copy is counted, the rest still run
            print(f"[vocode] {job['job_id']} FAILED: {exc}", file=sys.stderr, flush=True)
            failures += 1
            continue
        record = {
            "job_id": job["job_id"],
            "vocoder": "nvidia/bigvgan_v2_22khz_80band_256x",
            "mel": mel_kwargs,
            "stages": stages,
            "elapsed_s": round(time.time() - started, 2),
        }
        out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"[vocode] {n}/{len(jobs)} {job['job_id']} ok  delta={stages['length_delta_samples']} samples", flush=True
        )
    if failures:
        print(f"[vocode] {failures} failure(s)", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
