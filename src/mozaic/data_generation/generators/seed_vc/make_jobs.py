"""Seed-VC job list: one job per converted voice, from a pairs file (make_conversion_pairs).

Writes <out-dir>/jobs.jsonl for run.py. A pair's own source_wav and target_ref_wav win; a pair without
them (one written by hand) gives clip ids, read from the extracted whole clips in voxceleb2/audio/.
Seed-VC takes ONE short reference clip of the target voice.

A pair line; the four ids are required, the rest optional:

    {"job_id": "...", "source_clip_id": "id00012/xyz/00001", "target_identity_id": "id00015",
     "target_ref_clip_id": "id00015/abc/00001", "length_adjust": 1.0, "diffusion_steps": 10,
     "inference_cfg_rate": 0.7}

    python -m mozaic.data_generation.generators.seed_vc.make_jobs \
        --pairs <manifests>/<build>/conversion_pairs.jsonl --out-dir <manifests>/<build>/seedvc_jobs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S

# Used only by a pair that does not set its own (the build's pairs always do).
DEFAULT_LENGTH_ADJUST = 1.0
DEFAULT_DIFFUSION_STEPS = 10
DEFAULT_INFERENCE_CFG_RATE = 0.7  # SeedVCWrapper.convert_voice()'s own default


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Build a Seed-VC job list from a pairs manifest.")
    p.add_argument("--pairs", type=Path, required=True, help="pairs JSONL (see module docstring for schema)")
    p.add_argument("--out-dir", type=Path, required=True, help="where jobs.jsonl is written")
    return p.parse_args()


def wav_path_for_clip(clip_id: str) -> Path:
    """The extracted wav of a clip_id (id/video_id/utterance)."""
    return S.audio_wav_path(f"{clip_id}.mp4")


def main() -> int:
    """Resolve every pair's clip_ids to wav paths and write jobs.jsonl."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = args.out_dir / "jobs.jsonl"

    written = 0
    skipped = 0
    with args.pairs.open() as pf, jobs_file.open("w") as out:
        for line in pf:
            if not line.strip():
                continue
            pair: dict[str, Any] = json.loads(line)

            # Explicit paths win: the build's source is the CROPPED wav (cut to the video's frame count)
            # while its reference is a whole clip, so both cannot come from the whole-clip tree.
            source_wav = (
                Path(pair["source_wav"]) if pair.get("source_wav") else wav_path_for_clip(pair["source_clip_id"])
            )
            target_ref_wav = (
                Path(pair["target_ref_wav"])
                if pair.get("target_ref_wav")
                else wav_path_for_clip(pair["target_ref_clip_id"])
            )
            missing = [str(p) for p in (source_wav, target_ref_wav) if not p.exists()]
            if missing:
                print(f"[jobs] {pair['job_id']}: missing extracted wav(s), skipping: {missing}", file=sys.stderr)
                skipped += 1
                continue

            out.write(
                json.dumps(
                    {
                        "job_id": pair["job_id"],
                        "source_wav": str(source_wav),
                        "target_ref_wav": str(target_ref_wav),
                        "source_clip_id": pair["source_clip_id"],
                        "target_identity_id": pair["target_identity_id"],
                        "target_ref_clip_id": pair["target_ref_clip_id"],
                        "length_adjust": pair.get("length_adjust", DEFAULT_LENGTH_ADJUST),
                        "diffusion_steps": pair.get("diffusion_steps", DEFAULT_DIFFUSION_STEPS),
                        "inference_cfg_rate": pair.get("inference_cfg_rate", DEFAULT_INFERENCE_CFG_RATE),
                        # Carried through so a converted wav names its seed and the samples it serves
                        # (RVFA and FVFA share one conversion).
                        "base_seed": pair.get("base_seed", 0),
                        "conversion_key": pair.get("conversion_key"),
                        "consumer_sample_ids": pair.get("consumer_sample_ids"),
                    },
                    ensure_ascii=False,
                )
                + "\n",
            )
            written += 1

    print(f"[jobs] wrote {written} jobs to {jobs_file} ({skipped} skipped for missing wavs)")
    print(f"[jobs] convert with: sbatch scripts/data_generation/convert_seed_vc.sh (JOBS={jobs_file})")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
