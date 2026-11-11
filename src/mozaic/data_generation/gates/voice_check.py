"""Voice check: did the converted voice move to the target speaker?

SpeechBrain's pretrained ECAPA-TDNN speaker model (speechbrain/spkrec-ecapa-voxceleb,
trained on VoxCeleb, like our clips) gives the cosine between a converted clip and
  (a) the target's reference recording: should be high, it is the voice asked for;
  (b) the driver's own recording: should be lower, or the voice did not move.
timbre_moved is target_cosine > source_cosine. This file only scores; the keep/drop
limit is applied by keep_or_drop (see README.md).

One triple, to try it:

    python -m mozaic.data_generation.gates.voice_check --converted conv.wav --target target_ref.wav --source source.wav

A batch (make_voice_check_list writes the pairs file):

    python -m mozaic.data_generation.gates.voice_check --pairs <gates>/sv_pairs.jsonl --out <gates>/sv_report.jsonl

Runs in an environment with speechbrain and torch; ffmpeg must be on PATH. It reads
no setting, so it needs nothing else from this repo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


@contextmanager
def as_wav(path: str) -> Iterator[str]:
    """Yield a path SpeechBrain can load: the file itself if it is a .wav, else a temp .wav.

    SpeechBrain reads audio through soundfile, which cannot decode the sound of the
    shipped clips (.mov) or of the VoxCeleb2 recordings (.mp4). The shipped clip is
    what must be scored, so ffmpeg converts it to a temporary wav first.
    """
    if Path(path).suffix.lower() == ".wav":
        yield path
        return
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", path, tmp.name],
            check=True,
        )
        yield tmp.name


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Voice gate: SV cosine (converted vs target, converted vs source).")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pairs", type=Path, help="JSONL: {job_id, converted_wav, target_ref_wav, source_wav} per line")
    group.add_argument("--converted", type=Path, help="single-pair mode: converted wav")
    p.add_argument("--target", type=Path, help="single-pair mode: target reference wav")
    p.add_argument("--source", type=Path, help="single-pair mode: original source wav")
    p.add_argument("--out", type=Path, default=None, help="write JSONL report here (batch mode)")
    return p.parse_args()


def run_pair(verifier: Any, converted_wav: str, target_ref_wav: str, source_wav: str) -> dict[str, Any]:
    """Cosine(converted, target) and cosine(converted, source) for one triple."""
    with as_wav(converted_wav) as conv_wav, as_wav(target_ref_wav) as tgt_wav, as_wav(source_wav) as src_wav:
        target_score, _ = verifier.verify_files(conv_wav, tgt_wav)
        source_score, _ = verifier.verify_files(conv_wav, src_wav)
    target_cosine = float(target_score)
    source_cosine = float(source_score)
    return {
        "target_cosine": target_cosine,
        "source_cosine": source_cosine,
        "timbre_moved": target_cosine > source_cosine,
    }


def main() -> int:
    """Score one triple or a batch; print or write JSON."""
    args = parse_args()
    if args.converted and not (args.target and args.source):
        sys.exit("--target and --source are required with --converted")

    from speechbrain.inference.speaker import SpeakerRecognition

    print(f"[voice_check] loading {MODEL_SOURCE} ...")
    verifier = SpeakerRecognition.from_hparams(source=MODEL_SOURCE)

    if args.converted:
        result = run_pair(verifier, str(args.converted), str(args.target), str(args.source))
        print(json.dumps(result, indent=2))
        return 0

    rows = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    out_fh = args.out.open("w") if args.out else None
    try:
        for row in rows:
            result = run_pair(verifier, row["converted_wav"], row["target_ref_wav"], row["source_wav"])
            result["job_id"] = row.get("job_id")
            result["scored_clip"] = row["converted_wav"]
            print(
                f"[voice_check] {row.get('job_id')}: target={result['target_cosine']:.3f} "
                f"source={result['source_cosine']:.3f} moved={result['timbre_moved']}"
            )
            if out_fh:
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        if out_fh:
            out_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
