"""One command for MOZAIC generation: prints every step's command in order, and runs them with --run.

    prepare   the clip list (from finished shard lists), the identity split, the five pair lists
    batch     every step for one build and one list of pairs: crop, voice references, render,
              voice conversion, vocoded copies, mux, removal of half-made pairs, keep/drop

The generators run inside their own venvs: set FF_VENV, WAN_VENV, WA_VENV, SEEDVC_VENV (defaults
.venv_facefusion, .venv_s2v, .venv_animate, .venv_seed_vc in the repo). The gates use the
FaceFusion venv for the face scores, VOICE_VENV (default .venv_voice) for the voice scores and
SYNCNET_VENV (default .venv_syncnet, with SyncNet's code in SYNCNET_UPSTREAM) for the lip-sync scores.
Each generator also needs its model code and weights, named as its README says (FF_UPSTREAM;
WAN_UPSTREAM, WAN_CKPT; WA_UPSTREAM, WA_CKPT; SEEDVC_UPSTREAM). Each step skips work it has
already done, so a stopped batch is resumed by running it again.

Usage:
    python -m mozaic.data_generation.run_generation prepare [--run]
    python -m mozaic.data_generation.run_generation batch --build-id bulk_v1_ff77 --pair-ids-file pairs.txt \
        [--batch NAME] [--run]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from mozaic.data_generation import settings as S

REPO = Path(__file__).resolve().parents[3]
PKG = "mozaic.data_generation"
GENERATORS = Path(__file__).resolve().parent / "generators"
# video method of a build -> generator folder and its venv variable
VIDEO = {
    S.VIDEO_METHOD_FACEFUSION: ("facefusion", "FF_VENV"),
    S.VIDEO_METHOD_S2V: ("s2v", "WAN_VENV"),
    S.VIDEO_METHOD_ANIMATE: ("animate", "WA_VENV"),
}
VENV_DEFAULT = {"FF_VENV": ".venv_facefusion", "WAN_VENV": ".venv_s2v", "WA_VENV": ".venv_animate",
                "SEEDVC_VENV": ".venv_seed_vc", "VOICE_VENV": ".venv_voice", "SYNCNET_VENV": ".venv_syncnet"}
# clip length -> the gates' stratum (each has its own limits, configs/run.yaml gates:)
STRATUM = {77: "A", 153: "B"}


def venv_python(var: str) -> str:
    """Python of a generator's venv: $VAR, else the default venv folder in the repo."""
    return str(Path(os.environ.get(var) or REPO / VENV_DEFAULT[var]) / "bin" / "python")


def module(name: str, *args: str) -> list[str]:
    return [sys.executable, "-m", f"{PKG}.{name}", *args]


def prepare_commands() -> list[list[str]]:
    """Clip list, identity split, and the release's five pair lists with their checks."""
    draw = S.RUN["draw"]
    common = ["--splits", *draw["splits"], "--eligibility-frames", *map(str, draw["eligibility_frames"]),
              "--rings-per-split-gender", *(f"{k}={v}" for k, v in draw["rings_per_split_gender"].items()),
              "--reserve-rings-per-split-gender",
              *(f"{k}={v}" for k, v in draw["reserve_rings_per_split_gender"].items())]
    cmds = [module("inputs.join_clip_lists", "--num-shards", "60"), module("inputs.make_identity_split")]
    for b in draw["builds"]:
        extra = []
        if b.get("rings_from"):  # the first build draws the rings, the others reuse them
            extra = ["--rings-from", b["rings_from"], "--video-method", b["video_method"],
                     "--crop-frames", str(b["crop_frames"])]
        cmds.append(module("inputs.draw_pairs", "--build-id", b["build_id"], *common, *extra))
    cmds += [module("inputs.check_pairs", "--build-id", b["build_id"]) for b in draw["builds"]]
    return cmds


def build_info(build_id: str) -> tuple[str, int]:
    """The build's video generator and clip length, read from its pair list."""
    with S.new_gen_manifest_path(build_id).open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("needs_render") and row.get("video_method") not in (None, S.VIDEO_METHOD_NONE):
                return row["video_method"], S.row_crop_frames(row)
    raise SystemExit(f"{build_id}: no rendered rows in its pair list")


def batch_commands(build_id: str, pair_ids_file: Path, batch: str | None = None) -> list[list[str]]:
    """Every step of one batch, in order."""
    sfx = S.batch_suffix_for(set(pair_ids_file.read_text().split()))
    ids_path = str(pair_ids_file.resolve())
    pairs = ["--pair-ids-file", ids_path]
    mdir = S.new_gen_manifest_dir(build_id)
    final = S.NEW_GEN_ROOT / f"final_{build_id}"
    method, frames = build_info(build_id)
    name, var = VIDEO[method]
    stratum = STRATUM[frames]
    voice_refs = mdir / f"voice_ref_clips{sfx}.txt"
    conv_pairs, seedvc_jobs = mdir / f"conversion_pairs{sfx}.jsonl", mdir / f"seedvc_jobs{sfx}"
    vocode_jobs, mux_jobs = mdir / f"vocode_jobs{sfx}.jsonl", mdir / f"terminal_jobs{sfx}.jsonl"
    render_jobs = mdir / f"{name}_jobs{sfx}" / "jobs.jsonl"
    # renders go to the renders folder; Animate names each output in its job and needs a folder for its
    # preprocessing (as in the release run)
    render_out = (["--proc-root", str(S.NEW_GEN_ROOT / "_proc")] if name == "animate"
                  else ["--out-dir", str(S.new_gen_render_dir())])
    seedvc = venv_python("SEEDVC_VENV")
    return [
        module("crop.crop_clips", "--build-id", build_id, *pairs),
        module("inputs.make_audio_extract_list", "--build-id", build_id, "--out", str(voice_refs), *pairs),
        module("inputs.extract_voices", "--clip-ids-file", str(voice_refs), "--concurrency", "8"),
        module(f"generators.{name}.make_jobs", "--build-id", build_id, *pairs),
        [venv_python(var), str(GENERATORS / name / "run.py"), "--jobs", str(render_jobs), *render_out],
        module("generators.seed_vc.make_conversion_pairs", "--build-id", build_id, "--out", str(conv_pairs), *pairs),
        module("generators.seed_vc.make_jobs", "--pairs", str(conv_pairs), "--out-dir", str(seedvc_jobs)),
        [seedvc, str(GENERATORS / "seed_vc" / "run.py"), "--jobs", str(seedvc_jobs / "jobs.jsonl"),
         "--out-dir", str(S.new_gen_conversion_dir())],
        module("generators.seed_vc.make_vocode_jobs", "--build-id", build_id, "--out", str(vocode_jobs), *pairs),
        [seedvc, str(GENERATORS / "seed_vc" / "vocode.py"), "--jobs", str(vocode_jobs), "--device", "cpu"],
        module("final_clips.make_jobs", "--build-id", build_id, "--out-root", str(final), "--out", str(mux_jobs),
               *pairs),
        module("final_clips.mux", "--jobs", str(mux_jobs)),
        module("final_clips.withdraw_incomplete_pairs", "--build-id", build_id, "--final-dir", str(final), *pairs,
               "--apply"),
        module("gates.run_gates", "--batch", batch or pair_ids_file.stem, "--pairs", f"{stratum}={ids_path}",
               "--build", f"{build_id}:{stratum}", "--face-python", venv_python("FF_VENV"),
               "--voice-python", venv_python("VOICE_VENV"), "--syncnet-python", venv_python("SYNCNET_VENV")),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    sub.add_parser("prepare", help="clip list, identity split, the five pair lists")
    b = sub.add_parser("batch", help="every step for one build and one list of pairs")
    b.add_argument("--build-id", required=True)
    b.add_argument("--pair-ids-file", type=Path, required=True, help="the batch's pairs, one pair id per line")
    b.add_argument("--batch", default=None, help="name of the batch; names its gate decisions (default: the "
                   "pair list's file name)")
    for s in sub.choices.values():
        s.add_argument("--run", action="store_true", help="run the commands (default: print them)")
    args = p.parse_args()

    if args.what == "prepare":
        cmds = prepare_commands()
    else:
        cmds = batch_commands(args.build_id, args.pair_ids_file, args.batch)
    for cmd in cmds:
        print(shlex.join(cmd), flush=True)
        if args.run and subprocess.run(cmd, check=False).returncode != 0:
            print(f"[run_generation] stopped: step failed: {shlex.join(cmd)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
