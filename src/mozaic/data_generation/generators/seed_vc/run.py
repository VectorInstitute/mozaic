"""Seed-VC voice conversion over a job list (make_jobs.py), one shard per task, in .venv_seed_vc.

Each job turns the driver's own words (source_wav, the cropped sound) into the other person's voice,
learnt from ONE short reference clip (target_ref_wav), with the same timing (length_adjust 1.0), so the
lips in the video stay in sync. This file imports nothing from mozaic: only Seed-VC's own code, from its
clone (--upstream or $SEEDVC_UPSTREAM).

    {"job_id": "conv__<crop_key>__voice_<id>",
     "source_wav": ".../crops/<crop_key>.wav",          # the words and their timing
     "target_ref_wav": ".../audio/<id>/.../00001.wav",   # the voice: ONE short clip
     "length_adjust": 1.0, "diffusion_steps": 5, "inference_cfg_rate": 0.3, "base_seed": ...}

Per job, in the output folder: <job_id>.wav (22,050 Hz, Seed-VC's rate), <job_id>.json (how it was made)
and <job_id>.done, written last. A job with a .done is skipped, so a killed or requeued shard resumes. A
job that fails twice writes <job_id>.failed.json instead: that voice, and so its pair, is dropped.

Two things in Seed-VC's code this works around:
1. Its checkpoint loader (hf_utils.load_custom_model_from_hf) downloads into a RELATIVE ./checkpoints
   folder, so the script moves into the clone before it loads the model.
2. SeedVCWrapper.convert_voice() holds a `yield`, which makes it a generator even with
   stream_output=False: calling it runs nothing, and the audio comes back as StopIteration.value
   (run_convert below).

    python run.py --jobs jobs.jsonl --shard-index 0 --num-shards 1 --upstream <clone> --out-dir <dir>
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


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Run Seed-VC voice conversion over a JSONL job list.")
    p.add_argument("--jobs", type=Path, required=True, help="JSONL job file")
    p.add_argument("--shard-index", type=int, default=0, help="0-based index of this shard")
    p.add_argument("--num-shards", type=int, default=1, help="total number of shards")
    p.add_argument("--upstream", type=Path, default=None, help="Seed-VC clone (default $SEEDVC_UPSTREAM)")
    p.add_argument("--out-dir", type=Path, default=None, help="output dir (default $SEEDVC_OUT)")
    p.add_argument("--device", default=None, help="cuda/cpu (default: cuda if available)")
    p.add_argument("--dry-run", action="store_true", help="print job resolution without generating")
    return p.parse_args()


def resolve_dir(explicit: Path | None, env_var: str) -> Path:
    """Return ``explicit`` if given, else the directory named by ``env_var``."""
    if explicit is not None:
        return explicit
    value = os.environ.get(env_var)
    if not value:
        msg = f"neither the CLI flag nor ${env_var} is set"
        raise SystemExit(msg)
    return Path(value)


def load_shard(jobs_file: Path, shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    """Load the job lines belonging to this shard (every num_shards-th line)."""
    if not 0 <= shard_index < num_shards:
        msg = f"shard-index {shard_index} out of range for {num_shards} shards"
        raise SystemExit(msg)
    jobs: list[dict[str, Any]] = []
    with jobs_file.open() as fh:
        for i, raw in enumerate(fh):
            line = raw.strip()
            if line and i % num_shards == shard_index:
                jobs.append(json.loads(line))
    return jobs


def run_convert(
    wrapper: Any, source: str, target: str, length_adjust: float, diffusion_steps: int, inference_cfg_rate: float = 0.7
) -> Any:
    """Drive SeedVCWrapper.convert_voice() to its end and return its audio.

    convert_voice() is a generator even with stream_output=False: its `return full_audio` arrives as
    StopIteration.value, not as a normal return (module docstring, point 2). inference_cfg_rate defaults
    to the wrapper's own default (0.7).
    """
    gen = wrapper.convert_voice(
        source,
        target,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        stream_output=False,
    )
    try:
        next(gen)
    except StopIteration as stop:
        return stop.value
    msg = "convert_voice yielded instead of returning — stream_output=False path changed upstream?"
    raise RuntimeError(msg)


def write_provenance(path: Path, job: dict[str, Any], elapsed: float, sr: int) -> None:
    """Record how one voice was made, next to the voice itself."""
    record = {
        "job_id": job["job_id"],
        "generator": "seed-vc/Plachtaa",
        "manipulation": "voice_conversion",
        "streams_faked": ["audio"],
        "source_clip_id": job.get("source_clip_id"),
        "target_identity_id": job.get("target_identity_id"),
        "target_ref_clip_id": job.get("target_ref_clip_id"),
        "length_adjust": job["length_adjust"],
        "diffusion_steps": job["diffusion_steps"],
        "inference_cfg_rate": job.get("inference_cfg_rate", 0.7),
        # The seed of the noise draw that made this voice.
        "base_seed": job.get("base_seed"),
        "f0_condition": False,  # Seed-VC's default path: no pitch input, 22,050 Hz
        "sample_rate_hz": sr,
        "elapsed_seconds": round(elapsed, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def convert_one(wrapper: Any, job: dict[str, Any], out_dir: Path, sr: int, device: str) -> bool:
    """One conversion, retried once; True when the wav is written.

    On a second failure it writes ``<job_id>.failed.json`` and returns False: a documented drop of the pair,
    like a refused render. The mux skips that pair, and the steps after this shard still run.
    """
    # Heavy imports, only when converting.
    import soundfile as sf
    import torch

    job_id = job["job_id"]
    out_wav = out_dir / f"{job_id}.wav"
    started = time.time()
    for attempt in (1, 2):
        try:
            # Seed right before each conversion, not once per shard: Seed-VC's starting noise
            # (modules/flow_matching.py) is its only random draw, and nothing upstream seeds it. A seed
            # per job also makes a voice independent of its place in the shard, so re-sharding changes
            # nothing.
            seed = int(job.get("base_seed", 0))
            torch.manual_seed(seed)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            audio = run_convert(
                wrapper,
                job["source_wav"],
                job["target_ref_wav"],
                job["length_adjust"],
                job["diffusion_steps"],
                job.get("inference_cfg_rate", 0.7),
            )
            # Write to a temp name private to this process, then rename: a job killed mid-write leaves no
            # half-written wav under the final name, and two processes converting the same voice never
            # share a temp file. The rename is atomic.
            tmp_wav = out_wav.with_name(f"{out_wav.stem}.{socket.gethostname()}.{os.getpid()}.tmp.wav")
            sf.write(str(tmp_wav), audio, sr)
            tmp_wav.replace(out_wav)
        except Exception as exc:  # noqa: BLE001  any failure: retried once, then a documented drop
            elapsed = time.time() - started
            if attempt == 1:
                # One retry: a passing CUDA or disk error costs one conversion, not the whole shard.
                msg = f"[seedvc] {job_id} failed after {elapsed:.1f}s ({exc}); retrying once"
                print(msg, file=sys.stderr, flush=True)
                continue
            print(f"[seedvc] {job_id} FAILED after {elapsed:.1f}s: {exc}", file=sys.stderr, flush=True)
            (out_dir / f"{job_id}.failed.json").write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "error": str(exc),
                        "attempts": attempt,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    indent=2,
                )
                + "\n"
            )
            return False
        else:
            (out_dir / f"{job_id}.failed.json").unlink(missing_ok=True)
            write_provenance(out_dir / f"{job_id}.json", job, time.time() - started, sr)
            (out_dir / f"{job_id}.done").touch()
            print(f"[seedvc] {job_id} ok in {time.time() - started:.1f}s -> {out_wav}", flush=True)
            return True
    return False


def main() -> int:
    """Convert every job in this shard, skipping the ones already done."""
    args = parse_args()
    upstream = resolve_dir(args.upstream, "SEEDVC_UPSTREAM")
    out_dir = resolve_dir(args.out_dir, "SEEDVC_OUT")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_shard(args.jobs, args.shard_index, args.num_shards)
    print(f"[seedvc] shard {args.shard_index}/{args.num_shards}: {len(jobs)} jobs", flush=True)

    pending = [job for job in jobs if not (out_dir / f"{job['job_id']}.done").exists()]
    for job in jobs:
        if job not in pending:
            print(f"[seedvc] {job['job_id']} — skip (done)", flush=True)
    if not pending:
        print("[seedvc] nothing to do", flush=True)
        return 0
    if args.dry_run:
        for job in pending:
            print(json.dumps(job))
        return 0

    # Module docstring, point 1: the checkpoint loader uses a relative ./checkpoints, so work from the clone.
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))

    import torch
    from seed_vc_wrapper import SeedVCWrapper

    device = args.device or ("cuda" if torch.cuda.is_available() else "")
    if not device:
        # On CPU a voice takes minutes instead of seconds: say so and stop, unless asked (--device cpu).
        raise SystemExit(
            f"[seedvc] no usable CUDA device on {socket.gethostname()} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}); pass --device cpu to run on CPU on purpose"
        )
    print(f"[seedvc] device={device}  loading SeedVCWrapper ...", flush=True)
    # SeedVCWrapper keeps a given device as it is, and its torch.autocast(device_type=self.device.type)
    # fails on a plain string: pass a torch.device.
    wrapper = SeedVCWrapper(device=torch.device(device))
    sr = wrapper.sr  # 22050 on the default path (no pitch input) used here

    failures = 0
    for n, job in enumerate(pending, start=1):
        print(f"[seedvc] {n}/{len(pending)} {job['job_id']}", flush=True)
        if not convert_one(wrapper, job, out_dir, sr, device):
            failures += 1

    if failures:
        print(
            f"[seedvc] shard finished with {failures} documented failure(s) after a retry; the mux skips those pairs",
            file=sys.stderr,
        )
    # A documented failure is a recorded drop of the pair, like a refused render: exit 0 so the steps
    # queued after this shard run.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
