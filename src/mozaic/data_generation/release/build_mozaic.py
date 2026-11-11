"""Build the MOZAIC release from the finished bulk_v1 run.

Copies only: nothing in new_generated/ is moved, changed or deleted. Seven steps, run
one at a time (``scripts/data_generation/build_release.sh <step>`` queues each as a CPU job):

    plan       read-only. Decide every release file, fingerprint every source clip,
               check the design rules, write the plan and a report to
               _mozaic_build/plan/. Status READY or BLOCKED.
    copy       --apply. Rewrite each planned clip into mozaic/ without encoder
               version text, check it against its source, journal it. Resumable:
               a file under its final name has passed every check.
    fix        --apply. After copy, in place: give every clip the same aspect-ratio
               tag (1:1) and the same sound start (mozaic_layout.treat_start), check
               it against its source, journal it in _mozaic_build/fix/. Resumable.
    voice      After fix, read-only: re-score every pair's voice on the released sound
               with the gate's own speaker code; journal it in _mozaic_build/voice/.
               Needs torch and speechbrain. manifests then writes these voice scores.
    drop       --apply. Move the clips of the pairs in _mozaic_build/drop/pairs.json out
               of mozaic/ into _mozaic_build/drop/clips/ (kept, not deleted); every step
               reads the plan without those pairs.
    manifests  write metadata.jsonl (one row per clip) and metadata_table.parquet,
               pairs.jsonl, splits.json, summary.json and SHA256SUMS into mozaic/, and
               SOURCES.csv into the build. README.md and DATASHEET.md (the dataset card)
               are written by hand; SHA256SUMS lists them when they are there.
    verify     read-only. Re-check every released file and manifest.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from mozaic.data_generation import settings as S  # noqa: N812
from mozaic.data_generation.release import mozaic_layout as L  # noqa: N812


NEW_GEN = S.NEW_GEN_ROOT
RELEASE = S.RELEASE_ROOT
BUILD = S.BUILD_ROOT
REPORTS = NEW_GEN / "_reports"
CODE_COMMIT = NEW_GEN / "manifests" / "bulk_v1" / "CODE_COMMIT"  # the code that made the clips, for the plan's record
HERE = Path(__file__).resolve().parent

FFMPEG_VERSION = "ffmpeg version 4.4"  # every release file is rewritten by one ffmpeg build
BIGVGAN = "nvidia/bigvgan_v2_22khz_80band_256x"  # the vocoder Seed-VC uses, also for the vocoded arm
REWRITE = ["-map", "0", "-c", "copy", "-bsf:v", "filter_units=remove_types=6", "-map_metadata", "-1",
           "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact", "-f", "mov"]  # fmt: skip
# fix step: real clips carried a 1:1 aspect-ratio tag and fake ones none (a perfect
# label); this sets it in the H.264 and mov headers of every clip, without re-encoding.
VIDEO_FIX = ["-bsf:v", "h264_metadata=sample_aspect_ratio=1/1", "-aspect", "1:1"]
VOICE_KEYS = L.VOICE_SCORES
VOICE_CONTROL = 300  # pairs also re-scored on the sound the gate checked: must reproduce its numbers
VOICE_CONTROL_TOL = 0.001
DROP = BUILD / "drop" / "pairs.json"  # pairs removed after the build, with the reason
SUMS_FILE = "SHA256SUMS"
# The files manifests writes next to videos/.
DATA_FILES = ("summary.json", "splits.json", "pairs.jsonl", "metadata.jsonl", "metadata_table.parquet", SUMS_FILE)
# The dataset card, written by hand, not by this code: never required, never unexpected,
# and listed in SHA256SUMS when it is in mozaic/ as manifests runs.
DOC_FILES = ("README.md", "DATASHEET.md")
# Earlier builds wrote these; metadata.jsonl (top level) and summary.json replace them.
# manifests moves any left in mozaic/ into the build history, with the per-folder metadata.jsonl.
OLD_MANIFESTS = ("clips.jsonl", "metadata.json")
OLD_MANIFESTS_HISTORY = BUILD / "history" / "manifests_before_rename"
SETTINGS_KEYS = ("video_codec", "crf", "preset", "pixel_format", "scale_flags", "resample_via", "container", "fps",
                 "resolution", "audio_codec", "audio_bitrate", "audio_sample_rate", "luma", "chroma", "loudness",
                 "refine_passes_configured", "noise_strength", "silence_trim")  # fmt: skip


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def now() -> str:
    """Local time, minutes."""
    return f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"


def log(msg: str) -> None:
    """Print with a timestamp."""
    print(f"[mozaic {now()}] {msg}", flush=True)


def run(cmd: list[str]) -> str:
    """Run a command, return stdout; raise with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.strip()[-400:]}")
    return proc.stdout


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``path`` through a temporary file and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    """Pretty JSON, atomically."""
    write_text_atomic(path, json.dumps(obj, indent=1, sort_keys=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """One JSON object per line, atomically."""
    write_text_atomic(path, "".join(json.dumps(r, separators=(", ", ": ")) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file."""
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def tools() -> dict[str, str]:
    """Return ffmpeg/ffprobe paths and version; refuse any other FFMPEG_VERSION."""
    out = {}
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        if path is None:
            raise SystemExit(f"{name} not found on PATH")
        out[name] = str(Path(path).resolve())
    version = run([out["ffmpeg"], "-version"]).splitlines()[0]
    if not version.startswith(FFMPEG_VERSION):
        raise SystemExit(f"refusing: {version!r}; the release is written with {FFMPEG_VERSION}.x only")
    out["version"] = version
    return out


def code_hashes() -> dict[str, str]:
    """Return sha256 of every build-code file (content pins the uncommitted code)."""
    return {p.name: sha256_file(p) for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def code_commit() -> str | None:
    """Return the commit of the code that made the clips (CODE_COMMIT); None if it was not recorded."""
    return CODE_COMMIT.read_text().strip() if CODE_COMMIT.exists() else None


def gates_sha256() -> str:
    """sha256 of the keep/drop limits (S.GATES as sorted JSON), for the plan's record."""
    return hashlib.sha256(json.dumps(S.GATES, sort_keys=True, default=str).encode()).hexdigest()


def run_record(step: str, started: str, extra: dict[str, Any]) -> dict[str, Any]:
    """Record what ran, where, with which code and tools."""
    return {
        "step": step,
        "started": started,
        "finished": now(),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "user": os.environ.get("USER"),
        "python": sys.version.split()[0],
        "new_generated": str(NEW_GEN.resolve()),
        "release": str(RELEASE.resolve()),
        "code_sha256": code_hashes(),
        **extra,
    }


# --------------------------------------------------------------------------- #
# per-clip media work (run in worker processes)
# --------------------------------------------------------------------------- #
def fingerprint(path: Path, ffmpeg: str) -> dict[str, Any]:
    """Return the decoded picture + sound + timestamps fingerprint of a clip."""
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0", "-f", "framemd5", "-"]
    return L.parse_framemd5(run(cmd))


def format_problems(path: Path, ffprobe: str, length: str) -> list[str]:
    """Container and stream properties every released clip must have."""
    info = json.loads(run([ffprobe, "-v", "error", "-show_entries",
                           "format=format_name:format_tags:stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,"
                           "sample_rate,channels:stream_tags", "-of", "json", str(path)]))  # fmt: skip
    problems = []
    tags = [info.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in info.get("streams", [])]
    problems += [f"encoder tag {t['encoder']!r} left" for t in tags if "encoder" in t]
    streams = info.get("streams", [])
    if [s.get("codec_type") for s in streams] != ["video", "audio"]:
        problems.append(f"streams {[s.get('codec_type') for s in streams]}")
        return problems
    v, a = streams
    want_v = {"codec_name": "h264", "width": 224, "height": 224, "r_frame_rate": "25/1", "pix_fmt": "yuv420p"}
    want_a = {"codec_name": "pcm_s16le", "sample_rate": "16000", "channels": 1}
    problems += [f"video {k}={v.get(k)!r}" for k, x in want_v.items() if v.get(k) != x]
    problems += [f"audio {k}={a.get(k)!r}" for k, x in want_a.items() if a.get(k) != x]
    if "mov" not in info.get("format", {}).get("format_name", ""):
        problems.append(f"format {info.get('format')}")
    return problems


def describe_source(task: tuple[str, str, str], ffmpeg: str) -> dict[str, Any]:
    """Plan worker: fingerprint, sha256 and encode settings of one source clip."""
    key, rel, length = task
    path = NEW_GEN / rel
    out: dict[str, Any] = {"key": key, "problems": []}
    try:
        out["fingerprint"] = fingerprint(path, ffmpeg)
        out["sha256"] = sha256_file(path)
        side = json.loads(path.with_suffix(".json").read_text())
        out["settings"] = json.dumps({k: side.get(k) for k in SETTINGS_KEYS}, sort_keys=True)
        fp = out["fingerprint"]
        if fp["frames"] != L.LENGTH_FRAMES[length] or fp["audio_samples"] != L.expected_audio_samples(length):
            out["problems"].append(f"length {fp['frames']} frames / {fp['audio_samples']} samples")
    except Exception as exc:  # noqa: BLE001
        out["problems"].append(f"unreadable: {exc}")
    return out


def copy_one(entry: dict[str, Any], tool: dict[str, str]) -> dict[str, Any]:
    """Copy worker: rewrite one clip without version text, check it, then name it."""
    final = RELEASE / entry["release_path"]
    rec: dict[str, Any] = {"release_path": entry["release_path"], "source_path": entry["source_path"]}
    try:
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():  # final name but no journal line: check it, never overwrite it
            problems = clip_problems(final, entry, tool)
            rec.update(ok=not problems, problems=problems, existing=True,
                       sha256=sha256_file(final) if not problems else None)  # fmt: skip
            return rec
        tmp = final.with_name(final.name + ".tmp")
        run(
            [
                tool["ffmpeg"],
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(NEW_GEN / entry["source_path"]),
                *REWRITE,
                str(tmp),
            ]
        )
        problems = clip_problems(tmp, entry, tool)
        if problems:
            tmp.unlink(missing_ok=True)
            rec.update(ok=False, problems=problems)
            return rec
        rec["sha256"] = sha256_file(tmp)
        os.replace(tmp, final)
        rec.update(ok=True, problems=[], time=now(), host=socket.gethostname())
    except Exception as exc:  # noqa: BLE001
        rec.update(ok=False, problems=[f"error: {exc}"])
    return rec


def clip_problems(path: Path, entry: dict[str, Any], tool: dict[str, str]) -> list[str]:
    """Everything a released clip must satisfy, checked against its planned source."""
    problems = []
    fp = fingerprint(path, tool["ffmpeg"])
    if fp != entry["fingerprint"]:
        problems.append(f"decoded content differs from source: {fp} vs {entry['fingerprint']}")
    problems += format_problems(path, tool["ffprobe"], entry["row"]["length"])
    data = path.read_bytes()
    problems += [f"version text {t!r} left" for t in L.version_text(data)]
    return problems


def check_released(entry: dict[str, Any], tool: dict[str, str], fixed: bool = False) -> dict[str, Any]:
    """Verify worker: re-check one released clip from scratch (as fixed, if fixed)."""
    path = RELEASE / entry["release_path"]
    try:
        if fixed:
            problems = fixed_problems(path, entry, tool, decode_sound(NEW_GEN / entry["source_path"], tool["ffmpeg"]))
        else:
            problems = clip_problems(path, entry, tool)
        return {"release_path": entry["release_path"], "sha256": sha256_file(path), "problems": problems}
    except Exception as exc:  # noqa: BLE001
        return {"release_path": entry["release_path"], "sha256": None, "problems": [f"error: {exc}"]}


def run_bytes(cmd: list[str], data: bytes | None = None) -> bytes:
    """Run a command with binary input and output; raise with stderr on failure."""
    proc = subprocess.run(cmd, input=data, capture_output=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.decode(errors='replace').strip()[-400:]}"
        )
    return proc.stdout


def decode_sound(path: Path, ffmpeg: str) -> np.ndarray:
    """Return a clip's sound as 16-bit samples (16 kHz mono PCM, so exact)."""
    return np.frombuffer(
        run_bytes([ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-"]), "<i2"
    )


def moov_has_pasp(data: bytes) -> bool:
    """Return whether the mov header (not the media data) holds a pasp box."""
    i = 0
    while i + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[i : i + 8])
        if kind == b"moov":
            return b"pasp" in data[i : i + size]
        if size < 8:
            return False
        i += size
    return False


def fixed_problems(path: Path, entry: dict[str, Any], tool: dict[str, str], source: np.ndarray) -> list[str]:
    """Everything a fixed clip must satisfy, checked against its planned source.

    The source picture exactly, the source sound with the treated start exactly, the
    standard sound packets, a 1:1 aspect ratio in both headers, no version text.
    """
    length = entry["row"]["length"]
    problems = format_problems(path, tool["ffprobe"], length)
    fp = fingerprint(path, tool["ffmpeg"])
    if (fp["video"], fp["frames"]) != (entry["fingerprint"]["video"], entry["fingerprint"]["frames"]):
        problems.append("decoded picture differs from source")
    if not np.array_equal(decode_sound(path, tool["ffmpeg"]), L.treat_start(source)):
        problems.append("sound is not the source sound with the treated start")
    info = json.loads(run([tool["ffprobe"], "-v", "error", "-show_entries", "stream=sample_aspect_ratio:packet=stream_index,size",
                           "-of", "json", str(path)]))  # fmt: skip
    if info["streams"][0].get("sample_aspect_ratio") != "1:1":
        problems.append(f"aspect ratio {info['streams'][0].get('sample_aspect_ratio')}")
    if [int(p["size"]) for p in info["packets"] if p["stream_index"] == 1] != L.audio_packet_sizes(length):
        problems.append("sound packets differ from the standard layout")
    data = path.read_bytes()
    if not moov_has_pasp(data):
        problems.append("no pixel-aspect-ratio box in the mov header")
    problems += [f"version text {t!r} left" for t in L.version_text(data)]
    return problems


def remux(clip: Path, sound: np.ndarray, out: Path, ffmpeg: str) -> None:
    """Write ``out``: the clip's video copied with VIDEO_FIX, and ``sound`` as PCM.

    The sound is framed in the terminal mux's 1024-sample packets, so the file's sound
    packets stay exactly as they were.
    """
    run_bytes([ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "s16le", "-ar", str(L.AUDIO_RATE), "-ac", "1", "-i", "pipe:0",
               "-i", str(clip), "-map", "1:v:0", "-map", "0:a:0", "-c:v", "copy", *VIDEO_FIX,
               "-af", f"asetnsamples=n={L.AUDIO_PACKET_SAMPLES}:p=0", "-c:a", "pcm_s16le",
               "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
               "-f", "mov", str(out)], data=sound.astype("<i2").tobytes())  # fmt: skip


def fix_one(task: tuple[dict[str, Any], str], tool: dict[str, str]) -> dict[str, Any]:
    """Fix worker: rewrite one released clip in place, check it, return its record.

    Starts only from the copied clip (sha256 = copy journal). A file already replaced by
    an earlier run that stopped before journaling is accepted if it passes every check.
    """
    entry, copied_sha = task
    path = RELEASE / entry["release_path"]
    rec: dict[str, Any] = {"release_path": entry["release_path"], "sha256_copied": copied_sha}
    try:
        source = decode_sound(NEW_GEN / entry["source_path"], tool["ffmpeg"])
        if sha256_file(path) == copied_sha:
            tmp = path.with_name(path.name + ".tmp")
            remux(path, L.treat_start(source), tmp, tool["ffmpeg"])
            problems = fixed_problems(tmp, entry, tool, source)
            if problems:
                tmp.unlink(missing_ok=True)
                rec.update(ok=False, problems=problems)
                return rec
            rec["sha256"] = sha256_file(tmp)
            os.replace(tmp, path)
        else:
            problems = fixed_problems(path, entry, tool, source)
            if problems:
                rec.update(ok=False, problems=["neither the copied clip nor a fixed one", *problems])
                return rec
            rec.update(sha256=sha256_file(path), existing=True)
        rec.update(ok=True, problems=[], time=now(), host=socket.gethostname())
    except Exception as exc:  # noqa: BLE001
        rec.update(ok=False, problems=[f"error: {exc}"])
    return rec


# --------------------------------------------------------------------------- #
# plan: inputs
# --------------------------------------------------------------------------- #
def load_manifest(family: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    """(pair_id, label, release arm) -> manifest row; controls are skipped."""
    rows = {}
    for r in read_jsonl(S.new_gen_manifest_path(f"bulk_v1_{family}")):
        arm = L.MANIFEST_ARM.get(r["arm"])
        if arm:
            rows[(r["pair_id"], r["class_name"], arm)] = r
    return rows


Decision = tuple[dict[str, Any], Path]


def decision_folders(family: str) -> list[Path]:
    """Every batch's decision folder of a build, by name; a folder renamed with a suffix was set aside."""
    return sorted(p for p in REPORTS.glob(f"bulk_v1_{family}_selection_*") if p.is_dir() and "." not in p.name)


def load_decisions(family: str) -> tuple[dict[str, Decision], dict[str, Decision]]:
    """Return (current, pooled) gate decisions: pair -> (row, gates folder scored from).

    current = each batch's own decision, else the first batch's joint decision (pooled).
    """

    def read(folder: Path) -> dict[str, Decision]:
        summary = json.loads((folder / "selection_summary.json").read_text())
        gates = REPORTS / Path(summary["inputs"]["gates_dir"]).name
        return {r["pair_id"]: (r, gates) for r in read_jsonl(folder / "selection_pairs.jsonl")}

    pooled_dir = REPORTS / f"bulk_v1_{family}_selection"
    pooled = read(pooled_dir) if pooled_dir.is_dir() else {}
    current = dict(pooled)
    for folder in decision_folders(family):
        current.update(read(folder))
    return current, pooled


def kept_pairs(family: str, manifest: dict[tuple[str, str, str], dict[str, Any]], limits: dict[str, Any],
               errors: list[str]) -> dict[str, dict[str, Any]]:  # fmt: skip
    """Return pair -> where its clips are and the decision that kept it.

    Kept = every pair its gate decision accepts whose clips are all there: shipped in the final
    folder, or set aside whole in _withdrawn/<pair>/ by a step other than the gates (in the first
    batches a matching step set aside pairs that had passed their own checks; the release puts them
    back). A pair whose only failure is its missing clips counts as accepted when the first batch's
    joint decision accepted it. Every kept pair is re-checked against the frozen rule
    (``limits`` = S.GATES), independently of the gate's own flag.
    """
    current, pooled = load_decisions(family)
    final = NEW_GEN / f"final_bulk_v1_{family}"
    shipped = {n[: -len("__RVRA.mov")] for n in os.listdir(final / "RVRA") if n.endswith("__RVRA.mov")}
    clips: dict[str, list[str]] = collections.defaultdict(list)  # pair -> the sample ids of its clips
    for (pair, _label, _arm), row in manifest.items():
        clips[pair].append(row["sample_id"])
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(shipped):
        row, gates = current.get(p, ({}, Path()))
        if not row.get("accepted"):
            errors.append(f"{family}: shipped pair {p} has no accepting gate decision")
            continue
        out[p] = {"origin": "final", "decision": row, "gates": gates}
    for p, (row, gates) in sorted(current.items()):
        if p in shipped:
            continue
        accepting: Decision | None = (row, gates) if row["accepted"] else None
        if (
            accepting is None
            and not set(row["failing"]) - {"complete_group"}
            and pooled.get(p, ({}, Path()))[0].get("accepted")
        ):
            accepting = pooled[p]
        if accepting is None:
            continue
        aside = final / "_withdrawn" / p
        if not clips.get(p) or not all((aside / f"{s}.mov").exists() for s in clips[p]):
            if row["accepted"]:
                errors.append(f"{family}: {p} passed its gate but its clips are neither shipped nor set aside whole")
            continue
        out[p] = {"origin": "withdrawn", "decision": accepting[0], "gates": accepting[1]}
    for p, k in out.items():
        measured = {c["criterion"]: c.get("measured") for c in k["decision"]["checks"]}
        fails = L.recheck_gate(measured, family, limits)
        if fails:
            errors.append(f"{family}: {p} fails the frozen rule on re-check: {fails}")
    return out


def source_path(family: str, kept: dict[str, Any], pair: str, sample_id: str, label: str) -> str:
    """Source clip path relative to new_generated/."""
    if kept["origin"] == "final":
        return f"final_bulk_v1_{family}/{label}/{sample_id}.mov"
    return f"final_bulk_v1_{family}/_withdrawn/{pair}/{sample_id}.mov"


def voice_scores(gates: Path, conversion_key: str, cache: dict[Path, dict[str, dict[str, Any]]]) -> dict[str, float]:
    """Return voice_target / voice_source / voice_vocoded from the gate's own report."""
    if gates not in cache:
        cache[gates] = {r["job_id"]: r for r in read_jsonl(gates / "sv_report.jsonl") if "target_cosine" in r}
    main = cache[gates][conversion_key]
    ctrl = cache[gates].get(f"{conversion_key}__ctrl_real")
    return {
        "voice_target": main["target_cosine"],
        "voice_source": main["source_cosine"],
        "voice_vocoded": ctrl["source_cosine"] if ctrl else None,
    }


def voice_reference(rvfa: dict[str, Any], errors: list[str]) -> str | None:
    """Return the one target recording Seed-VC used, from the conversion's own record.

    Checked against the manifest's ``voice_ref_clip_id``; the manifest's 40-clip
    ``voice_ref_clip_ids`` is only the candidate pool and is never released.
    """
    record = S.new_gen_conversion_dir() / f"{rvfa['conversion_key']}.json"
    try:
        used = json.loads(record.read_text())["target_ref_clip_id"]
    except (FileNotFoundError, KeyError) as exc:
        errors.append(f"{rvfa['conversion_key']}: no conversion record of the reference used: {exc}")
        return None
    if used != rvfa["voice_ref_clip_id"]:
        errors.append(f"{rvfa['conversion_key']}: conversion used {used}, manifest says {rvfa['voice_ref_clip_id']}")
    if not used.startswith(f"{rvfa['pair_source_identity_id']}/"):
        errors.append(f"{rvfa['conversion_key']}: reference {used} is not a clip of the target")
    return L.clip_id(used)


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def step_plan(args: argparse.Namespace) -> int:  # noqa: PLR0912, PLR0915
    """Decide the release files, fingerprint them, check the rules, write the plan."""
    started = now()
    tool = tools()
    out_dir = BUILD / "plan"
    errors: list[str] = []
    manifests = {f: load_manifest(f) for f in L.FAMILIES}
    kept = {f: kept_pairs(f, manifests[f], S.GATES, errors) for f in L.FAMILIES}
    for f, k in kept.items():
        log(
            f"{f}: kept {len(k)} pairs ({sum(v['origin'] == 'withdrawn' for v in k.values())} restored from _withdrawn)"
        )

    # every clip of every kept pair, in every family that kept it
    candidates: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for f, pairs in kept.items():
        length = L.FAMILIES[f][1]
        for p, k in pairs.items():
            for label, arm in L.REAL_SIDE + L.FAKE_SIDE:
                row = manifests[f].get((p, label, arm))
                if row is None:
                    errors.append(f"{f}: {p} has no manifest row for {label} {arm}")
                    continue
                rel = source_path(f, k, p, row["sample_id"], label)
                if not (NEW_GEN / rel).exists():
                    errors.append(f"{f}: missing source {rel}")
                    continue
                candidates[(f, p, label, arm)] = {"family": f, "pair": p, "label": label, "arm": arm, "length": length,
                                                  "source_path": rel, "manifest": row}  # fmt: skip
    log(f"fingerprinting {len(candidates)} source clips with {args.workers} workers")
    tasks = [("|".join(key), c["source_path"], c["length"]) for key, c in candidates.items()]
    settings: collections.Counter[str] = collections.Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(partial(describe_source, ffmpeg=tool["ffmpeg"]), tasks, chunksize=32), 1):
            found = candidates[tuple(res["key"].split("|"))]
            found.update(fingerprint=res.get("fingerprint"), sha256=res.get("sha256"))
            errors += [f"{found['source_path']}: {x}" for x in res["problems"]]
            if res.get("settings"):
                settings[res["settings"]] += 1
            if i % 20000 == 0:
                log(f"  {i}/{len(tasks)} fingerprinted")
    if len(settings) != 1:
        errors.append(f"encode settings differ across source clips: {len(settings)} variants")

    # design rules: which clips must be identical
    violations: list[dict[str, Any]] = []
    canonical: dict[tuple[str, str], str] = {}
    excluded: set[tuple[str, str]] = set()

    def same(a: tuple[str, str, str, str], b: tuple[str, str, str, str], part: str) -> bool:
        fa, fb = candidates.get(a, {}).get("fingerprint"), candidates.get(b, {}).get("fingerprint")
        return bool(fa and fb and fa[part] == fb[part])

    for f, pairs in kept.items():
        for p in pairs:

            def key(label: str, arm: str, fam: str = f, pair: str = p) -> tuple[str, str, str, str]:
                return (fam, pair, label, arm)

            rules = [
                ("FVRA main / FVRA vocoded / FVFA share one video",
                 same(key("FVRA", "main"), key("FVRA", "vocoded"), "video") and same(key("FVRA", "main"), key("FVFA", "main"), "video")),
                ("RVRA main / RVRA vocoded / RVFA share one video",
                 same(key("RVRA", "main"), key("RVRA", "vocoded"), "video") and same(key("RVRA", "main"), key("RVFA", "main"), "video")),
                ("RVRA main and FVRA main share the natural voice", same(key("RVRA", "main"), key("FVRA", "main"), "audio")),
                ("RVRA vocoded and FVRA vocoded share the vocoded voice", same(key("RVRA", "vocoded"), key("FVRA", "vocoded"), "audio")),
                ("RVFA and FVFA share the fake voice", same(key("RVFA", "main"), key("FVFA", "main"), "audio")),
            ]  # fmt: skip
            for name, ok in rules:
                if not ok:
                    violations.append({"family": f, "pair": p, "rule": name})
                    excluded.add((f, p))
    for length, order in L.REAL_SOURCE_ORDER.items():
        pair_ids = sorted(set().union(*(kept[f].keys() for f in order)))
        for p in pair_ids:
            holders = [f for f in order if p in kept[f] and (f, p) not in excluded]
            if not holders:
                continue
            groups: dict[tuple[Any, ...], list[str]] = collections.defaultdict(list)
            for f in holders:
                sig = tuple((candidates[(f, p, lb, a)]["fingerprint"]["video"], candidates[(f, p, lb, a)]["fingerprint"]["audio"])
                            for lb, a in L.REAL_SIDE)  # fmt: skip
                groups[sig].append(f)
            best = max(groups.values(), key=lambda fs: (len(fs), -order.index(fs[0])))
            canonical[(length, p)] = best[0]
            for f in holders:
                if f not in best:
                    violations.append({"family": f, "pair": p, "rule": "real-side clips identical across generators",
                                       "kept_copy_from": best[0]})  # fmt: skip
                    excluded.add((f, p))
    if violations and not args.exclude_inconsistent:
        log(
            f"{len(violations)} design-rule violations; status BLOCKED (rerun with --exclude-inconsistent to drop them)"
        )
    if args.exclude_inconsistent:
        for f, p in excluded:
            kept[f].pop(p, None)
        canonical = {k: v for k, v in canonical.items() if k[1] in kept[v]}

    # release entries
    entries: list[dict[str, Any]] = []
    pairs_out: list[dict[str, Any]] = []
    people: dict[str, str] = {}
    voice_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    for length, order in L.REAL_SOURCE_ORDER.items():
        for p in sorted({q for f in order for q in kept[f]}):
            holders = [f for f in order if p in kept[f]]
            if not holders:
                continue
            c = canonical.get((length, p), holders[0])
            rvfa = manifests[c][(p, "RVFA", "main")]
            src_clip = L.clip_id(rvfa["driver_clip_id"])
            # The pair's own ids, never per-row fields: RVRA vocoded rows store the
            # source person in pair_source_identity_id.
            source_id, target_id = rvfa["driver_identity_id"], rvfa["pair_source_identity_id"]
            if not p.endswith(f"__x__{target_id}") or not p.startswith(f"{source_id}_"):
                errors.append(f"{p}: source/target {source_id}/{target_id} do not match the pair id")
            pid = L.release_pair_id(rvfa["driver_clip_id"], target_id)
            split = rvfa["split"]
            for person in (source_id, target_id):
                if people.setdefault(person, split) != split:
                    errors.append(f"{person} is in two splits")
            sides = [(c, lb, a) for lb, a in L.REAL_SIDE] + [(f, lb, a) for f in holders for lb, a in L.FAKE_SIDE]
            for f, label, arm in sides:
                cand = candidates[(f, p, label, arm)]
                face, voice = L.expected_face_voice(label, source_id, target_id)
                m = cand["manifest"]
                if (m["face_identity_id"], m["voice_identity_id"]) != (face, voice):
                    errors.append(
                        f"{f}: {m['sample_id']} face/voice {m['face_identity_id']}/{m['voice_identity_id']} != {face}/{voice}"
                    )
                name = L.release_name(label, f, arm, src_clip, target_id)
                entries.append({
                    "release_path": f"{L.release_folder(label, f, arm)}/{name}",
                    "source_path": cand["source_path"], "origin": kept[f][p]["origin"], "family": f,
                    "legacy_sample_id": m["sample_id"], "source_sha256": cand["sha256"],
                    "fingerprint": cand["fingerprint"],
                    "row": {"file_name": name, "label": label, "video_generator": L.video_generator(label, f),
                            "audio_generator": L.audio_generator(label, arm), "length": length, "arm": arm,
                            "split": split, "face_id": face, "voice_id": voice, "gender": m["gender_driver"], "pair_id": pid},
                })  # fmt: skip
            try:
                scores: dict[str, Any] = {
                    "voice": voice_scores(kept[c][p]["gates"], rvfa["conversion_key"], voice_cache)
                }
            except (KeyError, FileNotFoundError) as exc:
                errors.append(f"{c}: {p} voice scores not found in {kept[c][p]['gates'].name}: {exc}")
                scores = {"voice": None}
            canon_voice = {x["criterion"]: x.get("measured") for x in kept[c][p]["decision"]["checks"]}.get(
                "speaker"
            ) or {}
            if scores["voice"] and scores["voice"]["voice_target"] != canon_voice.get("converted_min"):
                errors.append(f"{c}: {p} voice_target does not match the gate's own value")
            for f in holders:
                meas = {x["criterion"]: x.get("measured") for x in kept[f][p]["decision"]["checks"]}
                face_m, pic = meas["arcface"], meas["picture"]["FVRA"]
                scores[L.FAMILIES[f][0]] = {"face_target": face_m["target"], "face_source": face_m["leak"],
                                            "face_baseline": face_m["floor"], "flicker": pic["flicker"],
                                            "melt": pic["melt_ratio"], "sharp": pic["sharp_ratio"]}  # fmt: skip
            # Seed-VC used ONE target recording: the one its own record names.
            voice_ref = voice_reference(rvfa, errors)
            pairs_out.append({
                "pair_id": pid, "length": length, "split": split, "gender": rvfa["gender_driver"],
                "source_id": source_id, "target_id": target_id, "source_clip": src_clip,
                "target_clips": {"face": L.clip_id(rvfa["face_ref_clip_id"]), "voice": voice_ref},
                "kept_by": [g for g in L.GENERATOR_ORDER if g in {L.FAMILIES[f][0] for f in holders}],
                "scores": scores,
            })  # fmt: skip
    names = collections.Counter(e["release_path"] for e in entries)
    errors += [f"two files would get the name {n}" for n, k in names.items() if k > 1]

    # write the plan
    folders: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for e in entries:
        folders[e["release_path"].rsplit("/", 1)[0]][e["row"]["split"]] += 1
    status = "READY" if not errors and (not violations or args.exclude_inconsistent) else "BLOCKED"
    write_jsonl(out_dir / "files.jsonl", entries)
    write_jsonl(out_dir / "pairs.jsonl", pairs_out)
    write_json(out_dir / "splits.json", dict(sorted(people.items())))
    write_jsonl(out_dir / "violations.jsonl", violations)
    write_text_atomic(out_dir / "errors.txt", "".join(e + "\n" for e in errors))
    write_json(out_dir / "encode_settings.json", json.loads(next(iter(settings))) if settings else {})
    counts = {
        "files": len(entries), "pairs": {lg: sum(1 for x in pairs_out if x["length"] == lg) for lg in L.LENGTH_FRAMES},
        "people": len(people), "folders": {k: dict(v) for k, v in sorted(folders.items())},
        "restored_pairs": {f: sum(1 for v in kept[f].values() if v["origin"] == "withdrawn") for f in L.FAMILIES},
        "violations": dict(collections.Counter(v["rule"] for v in violations)),
        "excluded_pair_generators": len(excluded) if args.exclude_inconsistent else 0,
    }  # fmt: skip
    record = run_record("plan", started, {"status": status, "tools": tool, "counts": counts, "errors": len(errors),
                                          "exclude_inconsistent": args.exclude_inconsistent,
                                          "inputs": {"thresholds_sha256": gates_sha256(),
                                                     "code_commit": code_commit(),
                                                     "manifests_sha256": {f: sha256_file(S.new_gen_manifest_path(f"bulk_v1_{f}")) for f in L.FAMILIES}}})  # fmt: skip
    write_json(out_dir / "run.json", record)
    write_text_atomic(out_dir / "report.md", plan_report(record, violations, errors))
    log(
        f"plan {status}: {len(entries)} files, {counts['pairs']} pairs, {len(errors)} errors, {len(violations)} violations"
    )
    return 0 if status == "READY" else 1


def plan_report(record: dict[str, Any], violations: list[dict[str, Any]], errors: list[str]) -> str:
    """Human-readable plan report."""
    c = record["counts"]
    lines = [f"# MOZAIC plan: {record['status']}", "", f"Made {record['finished']} on {record['host']} "
             f"(job {record['slurm_job']}), {record['tools']['version']}.", "",
             f"- files to release: **{c['files']}**", f"- pairs: {c['pairs']}", f"- people: {c['people']}",
             f"- pairs restored from _withdrawn: {c['restored_pairs']}", f"- errors: {record['errors']}",
             f"- design-rule violations: {sum(c['violations'].values())} {c['violations']}",
             f"- pair-generator entries excluded: {c['excluded_pair_generators']}", "",
             "| folder | train | val | test | total |", "|---|---|---|---|---|"]  # fmt: skip
    for folder, s in c["folders"].items():
        lines.append(f"| {folder} | {s.get('train', 0)} | {s.get('val', 0)} | {s.get('test', 0)} | {sum(s.values())} |")
    by_family = collections.Counter((v["family"], v["rule"]) for v in violations)
    if by_family:
        lines += ["", "## Design-rule violations", "", "| family | rule | pairs |", "|---|---|---|"]
        lines += [f"| {f} | {r} | {n} |" for (f, r), n in sorted(by_family.items())]
    if errors:
        lines += ["", "## Errors (first 50)", ""] + [f"- {e}" for e in errors[:50]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# copy
# --------------------------------------------------------------------------- #
def dropped_pairs() -> set[str]:
    """Release pair ids removed after the build (_mozaic_build/drop/pairs.json)."""
    return set(json.loads(DROP.read_text())["pairs"]) if DROP.exists() else set()


def load_ready_plan() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the plan's files (no dropped pairs) and record; refuse if not READY."""
    record = json.loads((BUILD / "plan" / "run.json").read_text())
    if record["status"] != "READY":
        raise SystemExit("refusing: the plan is not READY (see _mozaic_build/plan/report.md)")
    dropped = dropped_pairs()
    return [e for e in read_jsonl(BUILD / "plan" / "files.jsonl") if e["row"]["pair_id"] not in dropped], record


def journal() -> dict[str, dict[str, Any]]:
    """release_path -> journal record of every clip copied and checked so far."""
    path = BUILD / "copy" / "journal.jsonl"
    return {r["release_path"]: r for r in read_jsonl(path)} if path.exists() else {}


def step_copy(args: argparse.Namespace) -> int:
    """Rewrite every planned clip into mozaic/, check it, journal it."""
    started = now()
    entries, plan_record = load_ready_plan()
    tool = tools()
    if tool["ffmpeg"] != plan_record["tools"]["ffmpeg"]:
        raise SystemExit(f"refusing: ffmpeg {tool['ffmpeg']} is not the plan's {plan_record['tools']['ffmpeg']}")
    done = journal()
    todo = [e for e in entries if e["release_path"] not in done or not (RELEASE / e["release_path"]).exists()]
    log(f"{len(entries)} planned, {len(entries) - len(todo)} already copied, {len(todo)} to do")
    if not args.apply:
        log("dry run: pass --apply to copy")
        return 0
    jpath = BUILD / "copy" / "journal.jsonl"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with ProcessPoolExecutor(args.workers) as pool, jpath.open("a") as jf:
        for i, rec in enumerate(pool.map(partial(copy_one, tool=tool), todo, chunksize=16), 1):
            if rec["ok"]:
                jf.write(json.dumps(rec) + "\n")
                jf.flush()
                os.fsync(jf.fileno())
            else:
                failures.append(rec)
            if i % 10000 == 0:
                log(f"  {i}/{len(todo)} done, {len(failures)} failed")
    write_jsonl(BUILD / "copy" / "failures.jsonl", failures)
    done = journal()
    folders: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in entries:
        folders[e["release_path"].rsplit("/", 1)[0]].append(e)
    complete = 0
    for folder, items in sorted(folders.items()):
        if all(e["release_path"] in done and (RELEASE / e["release_path"]).exists() for e in items):
            listing = "".join(
                f"{e['release_path']} {done[e['release_path']]['sha256']}\n"
                for e in sorted(items, key=lambda x: x["release_path"])
            )
            write_json(BUILD / "copy" / "DONE" / f"{folder.replace('/', '__')}.json",
                       {"folder": folder, "files": len(items), "listing_sha256": hashlib.sha256(listing.encode()).hexdigest(), "at": now()})  # fmt: skip
            complete += 1
    write_json(BUILD / "copy" / "run.json", run_record("copy", started, {"tools": tool, "planned": len(entries),
                                                                         "journaled": len(done), "failures": len(failures),
                                                                         "folders_done": f"{complete}/{len(folders)}"}))  # fmt: skip
    log(
        f"copy finished: {len(done)}/{len(entries)} journaled, {len(failures)} failures, {complete}/{len(folders)} folders DONE"
    )
    return 0 if not failures and complete == len(folders) else 1


# --------------------------------------------------------------------------- #
# fix
# --------------------------------------------------------------------------- #
def fix_journal() -> dict[str, dict[str, Any]]:
    """release_path -> journal record of every clip fixed and checked so far."""
    path = BUILD / "fix" / "journal.jsonl"
    return {r["release_path"]: r for r in read_jsonl(path)} if path.exists() else {}


def released_sha256() -> dict[str, str]:
    """release_path -> sha256 of each clip as released now (fixed, else copied)."""
    shas = {p: r["sha256"] for p, r in journal().items()}
    shas.update({p: r["sha256"] for p, r in fix_journal().items()})
    return shas


def step_fix(args: argparse.Namespace) -> int:
    """Fix every copied clip in place (aspect tag, sound start), check, journal."""
    started = now()
    entries, plan_record = load_ready_plan()
    tool = tools()
    if tool["ffmpeg"] != plan_record["tools"]["ffmpeg"]:
        raise SystemExit(f"refusing: ffmpeg {tool['ffmpeg']} is not the plan's {plan_record['tools']['ffmpeg']}")
    copied, fixed = journal(), fix_journal()
    if any(e["release_path"] not in copied for e in entries):
        raise SystemExit("refusing: copy is not complete")
    todo = [e for e in entries if e["release_path"] not in fixed]
    log(f"{len(entries)} planned, {len(entries) - len(todo)} already fixed, {len(todo)} to do")
    if not args.apply:
        log("dry run: pass --apply to fix")
        return 0
    jpath = BUILD / "fix" / "journal.jsonl"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    tasks = [(e, copied[e["release_path"]]["sha256"]) for e in todo]
    with ProcessPoolExecutor(args.workers) as pool, jpath.open("a") as jf:
        for i, rec in enumerate(pool.map(partial(fix_one, tool=tool), tasks, chunksize=16), 1):
            if rec["ok"]:
                jf.write(json.dumps(rec) + "\n")
                jf.flush()
                os.fsync(jf.fileno())
            else:
                failures.append(rec)
            if i % 10000 == 0:
                log(f"  {i}/{len(todo)} done, {len(failures)} failed")
    write_jsonl(BUILD / "fix" / "failures.jsonl", failures)
    fixed = fix_journal()
    write_json(BUILD / "fix" / "run.json", run_record("fix", started, {"tools": tool, "planned": len(entries),
                                                                       "journaled": len(fixed), "failures": len(failures),
                                                                       "video_fix": VIDEO_FIX,
                                                                       "sound_start_ms": [L.START_MUTE_MS, L.START_FADE_MS]}))  # fmt: skip
    log(f"fix finished: {len(fixed)}/{len(entries)} journaled, {len(failures)} failures")
    return 0 if not failures and len(fixed) == len(entries) else 1


# --------------------------------------------------------------------------- #
# voice: the pairs' voice scores, re-measured on the released sound
# --------------------------------------------------------------------------- #
_VOICE: dict[str, Any] = {}  # the speaker model, loaded once per worker process


def _voice_init() -> None:
    """Load the gate's speaker code and model once per worker (gates/voice_check.py)."""
    import torch  # noqa: PLC0415
    from speechbrain.inference.speaker import SpeakerRecognition  # noqa: PLC0415

    torch.set_num_threads(1)
    from mozaic.data_generation.gates import voice_check as sv  # noqa: PLC0415

    _VOICE.update(sv=sv, model=SpeakerRecognition.from_hparams(source=sv.MODEL_SOURCE))


def voice_one(task: dict[str, Any]) -> dict[str, Any]:
    """Voice worker: a pair's three voice scores, exactly as the gate computes them.

    voice_target / voice_source: the fake voice (RVFA) against the target's reference
    recording and the source's own recording; voice_vocoded: the vocoded real voice
    against the source's recording. Control pairs are also scored on the gated sound.
    """
    sv, model = _VOICE["sv"], _VOICE["model"]
    rec: dict[str, Any] = {k: task[k] for k in ("pair_id", "length", "reference", "source_recording")}
    try:
        fake = sv.run_pair(model, task["rvfa"], task["reference"], task["source_recording"])
        vocoded = sv.run_pair(model, task["vocoded"], task["reference"], task["source_recording"])
        rec.update(zip(VOICE_KEYS, (fake["target_cosine"], fake["source_cosine"], vocoded["source_cosine"])))
        if task["control"]:
            fake = sv.run_pair(model, task["rvfa_gated"], task["reference"], task["source_recording"])
            vocoded = sv.run_pair(model, task["vocoded_gated"], task["reference"], task["source_recording"])
            rec["control"] = dict(
                zip(VOICE_KEYS, (fake["target_cosine"], fake["source_cosine"], vocoded["source_cosine"]))
            )
        rec.update(ok=True, problems=[], host=socket.gethostname())
    except Exception as exc:  # noqa: BLE001
        rec.update(ok=False, problems=[f"error: {exc}"])
    return rec


def voice_journal() -> dict[str, dict[str, Any]]:
    """pair_id -> voice re-score record of every pair scored so far."""
    path = BUILD / "voice" / "journal.jsonl"
    return {r["pair_id"]: r for r in read_jsonl(path)} if path.exists() else {}


def voice_task(
    pair: dict[str, Any], by_path: dict[str, dict[str, Any]], rows: dict[tuple[str, str], dict[str, Any]], control: bool
) -> dict[str, Any]:
    """Clips and the gate's two reference recordings for one pair."""
    fam = next(f for f, (_, lg) in L.FAMILIES.items() if lg == pair["length"])  # real-side folders are shared
    rvfa, voc = (by_path[f"{L.release_folder(lb, fam, arm)}/{L.release_name(lb, fam, arm, pair['source_clip'], pair['target_id'])}"]
                 for lb, arm in (("RVFA", "main"), ("RVRA", "vocoded")))  # fmt: skip
    row = rows[(rvfa["family"], rvfa["legacy_sample_id"])]
    used = json.loads((S.new_gen_conversion_dir() / f"{row['conversion_key']}.json").read_text())["target_ref_clip_id"]
    clip = pair["target_clips"]["voice"]
    if used != f"{clip[:7]}/{clip[8:-6]}/{clip[-5:]}":
        raise SystemExit(f"{pair['pair_id']}: conversion used {used}, pairs.jsonl says {clip}")
    return {
        "pair_id": pair["pair_id"], "length": pair["length"], "control": control,
        "rvfa": str(RELEASE / rvfa["release_path"]), "vocoded": str(RELEASE / voc["release_path"]),
        "rvfa_gated": str(NEW_GEN / rvfa["source_path"]), "vocoded_gated": str(NEW_GEN / voc["source_path"]),
        "reference": str(S.VOXCELEB2_MP4_ROOT / f"{used}.mp4"), "source_recording": str(S.crop_audio_path(row["crop_key"])),
    }  # fmt: skip


def voice_summary(pairs: list[dict[str, Any]], voice: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Control check, change against the gate's numbers, pairs below the voice floor."""
    gate = {p["pair_id"]: p["scores"]["voice"] for p in pairs}
    floor = {lg: S.GATES["A" if lg == "77f" else "B"]["sv_floor"] for lg in L.LENGTH_FRAMES}
    controls = [r for r in voice.values() if "control" in r]
    worst = (
        {k: max(abs(r["control"][k] - gate[r["pair_id"]][k]) for r in controls) for k in VOICE_KEYS} if controls else {}
    )
    diffs = {k: [r[k] - gate[p][k] for p, r in voice.items() if gate[p].get(k) is not None] for k in VOICE_KEYS}
    change = {k: {"mean": float(np.mean(v)), "max_abs": float(np.max(np.abs(v)))} for k, v in diffs.items() if v}
    below = {
        k: sorted(p for p, r in voice.items() if r[k] < floor[r["length"]]) for k in ("voice_target", "voice_vocoded")
    }
    ok = (
        len(voice) == len(pairs)
        and len(controls) == VOICE_CONTROL
        and all(v <= VOICE_CONTROL_TOL for v in worst.values())
    )
    return {"status": "PASS" if ok else "FAIL", "pairs": len(pairs), "scored": len(voice), "control_pairs": len(controls),
            "control_tolerance": VOICE_CONTROL_TOL, "control_worst_diff": worst, "change_vs_gate": change, "floor": floor,
            "below_floor": {k: len(v) for k, v in below.items()}, "below_floor_pairs": below}  # fmt: skip


def step_voice(args: argparse.Namespace) -> int:
    """Re-score every pair's voice on the released sound (gate's code); journal it."""
    started = now()
    entries, _ = load_ready_plan()
    tool = tools()
    fixed = fix_journal()
    if any(e["release_path"] not in fixed for e in entries):
        raise SystemExit("refusing: the fix step is not complete, so the released sound is not final")
    by_path = {e["release_path"]: e for e in entries}
    rows = {(f, r["sample_id"]): r for f in L.FAMILIES for r in read_jsonl(S.new_gen_manifest_path(f"bulk_v1_{f}"))}
    dropped = dropped_pairs()
    pairs = sorted((p for p in read_jsonl(BUILD / "plan" / "pairs.jsonl") if p["pair_id"] not in dropped),
                   key=lambda r: r["pair_id"])  # fmt: skip
    control = {pairs[i]["pair_id"] for i in np.random.default_rng(42).choice(len(pairs), VOICE_CONTROL, replace=False)}
    done = voice_journal()
    tasks = [voice_task(p, by_path, rows, p["pair_id"] in control) for p in pairs if p["pair_id"] not in done]
    log(f"{len(pairs)} pairs, {len(done)} already scored, {len(tasks)} to do ({len(control)} also on the gated sound)")
    jpath = BUILD / "voice" / "journal.jsonl"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with ProcessPoolExecutor(args.workers, initializer=_voice_init) as pool, jpath.open("a") as jf:
        for i, rec in enumerate(pool.map(voice_one, tasks, chunksize=4), 1):
            if rec["ok"]:
                jf.write(json.dumps(rec) + "\n")
                jf.flush()
                os.fsync(jf.fileno())
            else:
                failures.append(rec)
            if i % 1000 == 0:
                log(f"  {i}/{len(tasks)} scored, {len(failures)} failed")
    write_jsonl(BUILD / "voice" / "failures.jsonl", failures)
    summary = voice_summary(pairs, voice_journal())
    if failures:
        summary["status"] = "FAIL"
    write_json(
        BUILD / "voice" / "run.json",
        run_record("voice", started, {**summary, "failures": len(failures), "tools": tool}),
    )
    log(f"voice {summary['status']}: {summary['scored']}/{summary['pairs']} scored, control worst {summary['control_worst_diff']}, "
        f"below floor {summary['below_floor']}")  # fmt: skip
    return 0 if summary["status"] == "PASS" else 1


def with_voice(pairs: list[dict[str, Any]], voice: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the pairs with their voice scores replaced by the re-scores."""
    missing = [p["pair_id"] for p in pairs if p["pair_id"] not in voice]
    if missing:
        raise SystemExit(f"refusing: no voice re-score for {len(missing)} pairs, e.g. {missing[:3]}")
    return [{**p, "scores": {**p["scores"], "voice": {k: voice[p["pair_id"]][k] for k in VOICE_KEYS}}} for p in pairs]


# --------------------------------------------------------------------------- #
# drop
# --------------------------------------------------------------------------- #
def step_drop(args: argparse.Namespace) -> int:
    """Move the clips of dropped pairs out of mozaic/ into _mozaic_build/drop/clips/."""
    started = now()
    dropped = dropped_pairs()
    if not dropped:
        raise SystemExit(f"refusing: no pairs listed in {DROP}")
    todo = [e for e in read_jsonl(BUILD / "plan" / "files.jsonl") if e["row"]["pair_id"] in dropped]
    here = [e for e in todo if (RELEASE / e["release_path"]).exists()]
    log(f"{len(dropped)} pairs dropped: {len(todo)} planned clips, {len(here)} still in mozaic/")
    if not args.apply:
        log("dry run: pass --apply to move them")
        return 0
    for e in here:
        dst = BUILD / "drop" / "clips" / e["release_path"]
        if dst.exists():
            raise SystemExit(f"refusing: {dst} exists")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(RELEASE / e["release_path"], dst)
    left = [e["release_path"] for e in todo if (RELEASE / e["release_path"]).exists()]
    kept = sum((BUILD / "drop" / "clips" / e["release_path"]).exists() for e in todo)
    write_json(BUILD / "drop" / "run.json", run_record("drop", started, {"pairs": len(dropped), "clips": len(todo),
                                                                         "moved_now": len(here), "kept_in_drop": kept,
                                                                         "left_in_release": left}))  # fmt: skip
    log(f"drop finished: {kept}/{len(todo)} clips in _mozaic_build/drop/clips, {len(left)} left in mozaic/")
    return 0 if not left and kept == len(todo) else 1


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #
def step_manifests(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Write the manifests into mozaic/, and SOURCES.csv into the build."""
    started = now()
    entries, plan_record = load_ready_plan()
    done = journal()
    folders = {e["release_path"].rsplit("/", 1)[0] for e in entries}
    markers = {p.stem for p in (BUILD / "copy" / "DONE").glob("*.json")}
    missing = sorted(f for f in folders if f.replace("/", "__") not in markers)
    if missing or any(e["release_path"] not in done for e in entries):
        raise SystemExit(f"refusing: copy is not complete ({len(missing)} folders without DONE)")
    fixed = fix_journal()
    if fixed and any(e["release_path"] not in fixed for e in entries):
        raise SystemExit(f"refusing: fix is not complete ({len(fixed)}/{len(entries)})")
    shas = released_sha256()
    rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in entries:
        rows[e["release_path"].rsplit("/", 1)[0]].append({**e["row"], "sha256": shas[e["release_path"]]})
    dropped = dropped_pairs()
    pairs = sorted((p for p in read_jsonl(BUILD / "plan" / "pairs.jsonl") if p["pair_id"] not in dropped),
                   key=lambda r: (r["length"], r["pair_id"]))  # fmt: skip
    if (BUILD / "voice" / "run.json").exists():  # voice scores re-measured on the released sound
        if json.loads((BUILD / "voice" / "run.json").read_text())["status"] != "PASS":
            raise SystemExit("refusing: the voice re-score did not pass (see _mozaic_build/voice/run.json)")
        pairs = with_voice(pairs, voice_journal())
    write_jsonl(RELEASE / "pairs.jsonl", pairs)
    by_pair = {p["pair_id"]: p for p in pairs}
    index = sorted((L.clip_index_row(folder, r, by_pair[r["pair_id"]]) for folder, items in rows.items() for r in items),
                   key=lambda r: r["file_name"])  # fmt: skip
    write_jsonl(RELEASE / "metadata.jsonl", index)
    write_parquet(RELEASE / "metadata_table.parquet", index)
    splits = json.loads((BUILD / "plan" / "splits.json").read_text())
    write_json(RELEASE / "splits.json", splits)
    summary = build_summary(rows, pairs, splits, fixed=bool(fixed))
    write_json(RELEASE / "summary.json", summary)
    moved = retire_old_manifests(folders)
    write_text_atomic(RELEASE / SUMS_FILE, sums_text({r["file_name"]: r["sha256"] for r in index}))
    with (BUILD / "SOURCES.csv.tmp").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["release_path", "source_path", "origin", "family", "legacy_sample_id",
                    "source_sha256", "video_md5", "audio_md5", "release_sha256"])  # fmt: skip
        for e in sorted(entries, key=lambda x: x["release_path"]):
            w.writerow([e["release_path"], e["source_path"], e["origin"], e["family"],
                        e["legacy_sample_id"], e["source_sha256"], e["fingerprint"]["video"], e["fingerprint"]["audio"],
                        shas[e["release_path"]]])  # fmt: skip
    os.replace(BUILD / "SOURCES.csv.tmp", BUILD / "SOURCES.csv")
    write_json(BUILD / "manifests_run.json", run_record("manifests", started, {"plan_finished": plan_record["finished"],
                                                                               "folders": len(rows), "pairs": len(pairs),
                                                                               "old_manifests_moved": moved}))  # fmt: skip
    log(
        f"manifests written: {len(index)} rows from {len(rows)} folders, {len(pairs)} pairs; "
        f"{moved} old manifest files moved to {OLD_MANIFESTS_HISTORY}"
    )
    return 0


def metadata_arrow_schema() -> Any:
    """Return the Parquet schema of metadata_table.parquet, from L.METADATA_SCHEMA."""
    import pyarrow as pa  # noqa: PLC0415 (the voice step runs in a venv without pyarrow)

    kinds = {"string": pa.string(), "bool": pa.bool_(), "float": pa.float64(), "float or empty": pa.float64(),
             "list of strings": pa.list_(pa.string())}  # fmt: skip
    return pa.schema([pa.field(name, kinds[kind], nullable=kind == "float or empty")
                      for name, kind, _ in L.METADATA_SCHEMA])  # fmt: skip


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the metadata rows as a typed Parquet table, atomically."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    tmp = path.with_name(path.name + ".tmp")
    table = pa.Table.from_pylist(rows, schema=metadata_arrow_schema())
    pq.write_table(table, tmp, compression="zstd")  # type: ignore[no-untyped-call]  # pyarrow: no hints
    os.replace(tmp, path)


def sums_text(clip_sha: dict[str, str]) -> str:
    """Return SHA256SUMS: every file in mozaic/ but itself, for ``sha256sum -c``.

    Clips take their checksum from the rows (verify re-hashes them on disk); the
    other top-level files are hashed here, after they are written. The dataset card
    (README.md, DATASHEET.md) is listed when it is there.
    """
    top = [f for f in DATA_FILES if f != SUMS_FILE] + [f for f in DOC_FILES if (RELEASE / f).exists()]
    sums = {**clip_sha, **{f: sha256_file(RELEASE / f) for f in top}}
    return "".join(f"{sha}  {path}\n" for path, sha in sorted(sums.items()))


def retire_old_manifests(folders: set[str]) -> int:
    """Move old manifest files from mozaic/ into the build history; return the count."""
    moved = 0
    for rel in [f"{f}/metadata.jsonl" for f in sorted(folders)] + list(OLD_MANIFESTS):
        if not (RELEASE / rel).exists():
            continue
        dst = OLD_MANIFESTS_HISTORY / rel
        if dst.exists():
            raise SystemExit(f"refusing: {dst} already exists")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(RELEASE / rel, dst)
        moved += 1
    return moved


def build_summary(
    rows: dict[str, list[dict[str, Any]]], pairs: list[dict[str, Any]], splits: dict[str, str], fixed: bool = False
) -> dict[str, Any]:
    """Return summary.json: everything dataset-wide, and every count."""
    encode = json.loads((BUILD / "plan" / "encode_settings.json").read_text())
    versions: dict[str, set[str]] = collections.defaultdict(set)
    for f in L.FAMILIES:
        for r in read_jsonl(S.new_gen_manifest_path(f"bulk_v1_{f}")):
            video, _, voice = r["generator_version"].partition("+")
            versions[L.FAMILIES[f][0]].add(video)
            versions["seed_vc"].add(voice)
    by_folder ={folder: dict(collections.Counter(r["split"] for r in items)) for folder, items in sorted(rows.items())}
    all_rows = [r for items in rows.values() for r in items]
    return {
        "name": "MOZAIC", "version": "1.0", "schema_version": "1.0",
        "release_date": "TO FILL", "cite": "TO FILL", "licence": "CC BY-NC 4.0", "access": "TO FILL", "removal_requests": "TO FILL",
        "labels": {"RVRA": "real face, real voice", "RVFA": "real face, fake voice",
                   "FVRA": "fake face, real voice", "FVFA": "fake face, fake voice"},
        "arms": {"main": "the clip as made", "vocoded": "the same clip with its voice passed through BigVGAN; the label does not change"},
        "source": {"dataset": "VoxCeleb2 (dev set)", "people": len(splits),
                   "cite": "Chung, Nagrani, Zisserman. VoxCeleb2: Deep Speaker Recognition. Interspeech 2018"},
        "video": {"container": encode["container"], "codec": f"h264 ({encode['video_codec']}), crf {encode['crf']}, preset {encode['preset']}, {encode['pixel_format']}",
                  "size": f"{encode['resolution']}x{encode['resolution']}", "fps": encode["fps"],
                  "lengths": {"77f": "77 frames (3.08 s)", "153f": "153 frames (6.12 s)"}},
        "audio": {"codec": encode["audio_codec"], "sample_rate": encode["audio_sample_rate"], "channels": 1},
        "applied_to_every_clip": {"brightness": encode["luma"], "colour": encode["chroma"], "loudness": encode["loudness"],
                                  "silence_trimmed": encode["silence_trim"],
                                  **({"sound_start": f"first {L.START_MUTE_MS} ms silent, then a {L.START_FADE_MS} ms fade-in",
                                      "pixel_aspect_ratio": "1:1"} if fixed else {})},
        "generators": {**{g: sorted(v) for g, v in sorted(versions.items())}, "bigvgan": BIGVGAN, "draw_seed": 42},
        "counts": {"clips": len(all_rows), "pairs": {lg: sum(1 for p in pairs if p["length"] == lg) for lg in L.LENGTH_FRAMES},
                   "by_label": dict(collections.Counter(r["label"] for r in all_rows)),
                   "by_split": dict(collections.Counter(r["split"] for r in all_rows)), "by_folder": by_folder},
        "people_per_split": dict(collections.Counter(splits.values())),
    }  # fmt: skip


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def expected_files(planned: set[str], on_disk: set[str]) -> set[str]:
    """Files mozaic/ must hold: the planned clips, the data files, and the dataset card if it is there."""
    return planned | set(DATA_FILES) | (set(DOC_FILES) & on_disk)


def step_verify(args: argparse.Namespace) -> int:  # noqa: PLR0912, PLR0915
    """Re-check the whole release against the plan, from the files on disk."""
    started = now()
    entries, plan_record = load_ready_plan()
    tool = tools()
    done = released_sha256()
    fixed = fix_journal()
    fails: list[str] = []
    planned = {e["release_path"]: e for e in entries}
    if fixed and set(planned) - set(fixed):
        fails.append(f"fix covers {len(set(fixed) & set(planned))} of {len(planned)} planned clips")
    on_disk = {str(p.relative_to(RELEASE)) for p in RELEASE.rglob("*") if p.is_file()}
    expected = expected_files(set(planned), on_disk)
    fails += [f"unexpected file {x}" for x in sorted(on_disk - expected)]
    fails += [f"missing file {x}" for x in sorted(expected - on_disk)]
    pairs = read_jsonl(RELEASE / "pairs.jsonl")
    pair_of = {p["pair_id"]: p for p in pairs}
    index = read_jsonl(RELEASE / "metadata.jsonl") if (RELEASE / "metadata.jsonl").exists() else []
    rows: dict[str, dict[str, Any]] = {}
    for r in index:
        if r["file_name"] in rows:
            fails.append(f"two rows for {r['file_name']}")
        rows[r["file_name"]] = r
    fails += [f"row sha256 != journal for {p}" for p in planned if rows.get(p, {}).get("sha256") != done.get(p)]
    fails += [f"metadata.jsonl row differs from the plan and pairs.jsonl for {p}" for p, e in planned.items()
              if e["row"]["pair_id"] not in pair_of or rows.get(p) != L.clip_index_row(
                  p.rsplit("/", 1)[0], {**e["row"], "sha256": done.get(p)}, pair_of[e["row"]["pair_id"]])]  # fmt: skip
    if (RELEASE / "metadata_table.parquet").exists():
        import pyarrow.parquet as pq  # noqa: PLC0415

        table = pq.read_table(RELEASE / "metadata_table.parquet")  # type: ignore[no-untyped-call]
        if not table.schema.equals(metadata_arrow_schema()):
            fails.append("metadata_table.parquet: column names or types differ from the schema")
        if table.to_pylist() != index:
            fails.append("metadata_table.parquet: rows differ from metadata.jsonl")
    log(f"re-checking {len(entries)} released clips with {args.workers} workers" + (" (as fixed)" if fixed else ""))
    disk_sha: dict[str, str] = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for res in pool.map(partial(check_released, tool=tool, fixed=bool(fixed)), entries, chunksize=16):
            if res["problems"]:
                fails.append(f"{res['release_path']}: {res['problems']}")
            if res["sha256"] != rows.get(res["release_path"], {}).get("sha256"):
                fails.append(f"{res['release_path']}: sha256 on disk != metadata.jsonl")
            if res["sha256"]:
                disk_sha[res["release_path"]] = res["sha256"]
    splits = json.loads((RELEASE / "splits.json").read_text())
    pair_ids = collections.Counter(p["pair_id"] for p in pairs)
    if (BUILD / "voice" / "run.json").exists():
        voice = voice_journal()
        fails += [f"pair {p['pair_id']}: voice scores differ from the voice re-score" for p in pairs
                  if p["scores"]["voice"] != {k: voice.get(p["pair_id"], {}).get(k) for k in VOICE_KEYS}]  # fmt: skip
    fails += [f"pair {p} listed twice" for p, n in pair_ids.items() if n > 1]
    by_pair: dict[str, set[tuple[str, str, str]]] = collections.defaultdict(set)
    for r in rows.values():
        by_pair[r["pair_id"]].add((r["label"], r["video_generator"], r["arm"]))
        if r["pair_id"] not in pair_ids:
            fails.append(f"{r['file_name']}: pair not in pairs.jsonl")
    for p in pairs:
        want = {(lb, "real", a) for lb, a in L.REAL_SIDE} | {(lb, g, a) for g in p["kept_by"] for lb, a in L.FAKE_SIDE}
        if by_pair[p["pair_id"]] != want:
            fails.append(f"pair {p['pair_id']}: clips {sorted(by_pair[p['pair_id']] ^ want)} differ from kept_by")
        if splits.get(p["source_id"]) != p["split"] or splits.get(p["target_id"]) != p["split"]:
            fails.append(f"pair {p['pair_id']}: split differs from splits.json")
        clips = p.get("target_clips", {})
        if not (isinstance(clips.get("face"), str) and isinstance(clips.get("voice"), str)):
            fails.append(f"pair {p['pair_id']}: target_clips must name one face clip and one voice clip")
        missing_scores = [g for g in ["voice", *p["kept_by"]] if g not in p["scores"]]
        if missing_scores:
            fails.append(f"pair {p['pair_id']}: no scores for {missing_scores}")
    for r in rows.values():
        if splits.get(r["face_id"]) != r["split"] or splits.get(r["voice_id"]) != r["split"]:
            fails.append(f"{r['file_name']}: split differs from splits.json")
    summary = json.loads((RELEASE / "summary.json").read_text()) if (RELEASE / "summary.json").exists() else {}
    if summary.get("counts", {}).get("clips") != len(rows) or summary["counts"]["pairs"] != {
        lg: sum(1 for p in pairs if p["length"] == lg) for lg in L.LENGTH_FRAMES
    }:
        fails.append("summary.json counts differ from the rows")
    four_way = {
        "plan": len(planned),
        "journal": len(set(done) & set(planned)),
        "on_disk": len(on_disk & set(planned)),
        "rows": len(rows),
    }
    if len(set(four_way.values())) != 1:
        fails.append(f"plan / journal / disk / rows disagree: {four_way}")
    actual = {x: disk_sha.get(x) or sha256_file(RELEASE / x) for x in sorted(on_disk & expected)}
    listed: dict[str, str] = {}
    if (RELEASE / SUMS_FILE).exists():
        for line in (RELEASE / SUMS_FILE).read_text().splitlines():
            sha, _, path = line.partition("  ")
            listed[path] = sha
    if listed != {x: sha for x, sha in actual.items() if x != SUMS_FILE}:
        fails.append(f"{SUMS_FILE} differs from the files on disk")
    status = "PASS" if not fails else "FAIL"
    sums = "".join(f"{sha}  {x}\n" for x, sha in actual.items())
    write_text_atomic(BUILD / "verify" / "RELEASE_SHA256SUMS", sums)
    record = run_record("verify", started, {"status": status, "tools": tool, "four_way": four_way, "failures": len(fails),
                                            "plan_finished": plan_record["finished"], "checked_as_fixed": bool(fixed)})  # fmt: skip
    write_json(BUILD / "verify" / "run.json", record)
    write_text_atomic(BUILD / "verify" / "report.md", f"# MOZAIC verify: {status}\n\n- four-way count: {four_way}\n- failures: {len(fails)}\n\n"
                      + "".join(f"- {x}\n" for x in fails[:200]))  # fmt: skip
    log(f"verify {status}: {four_way}, {len(fails)} failures")
    return 0 if status == "PASS" else 1


def main() -> int:
    """Command line."""
    os.umask(0o002)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["plan", "copy", "fix", "voice", "drop", "manifests", "verify"])
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument("--apply", action="store_true", help="copy, fix, drop: actually change mozaic/")
    p.add_argument("--exclude-inconsistent", action="store_true",
                   help="plan: drop pair-generator entries whose clips break a design rule (otherwise BLOCKED)")  # fmt: skip
    args = p.parse_args()
    steps = {"plan": step_plan, "copy": step_copy, "fix": step_fix, "voice": step_voice, "drop": step_drop,
             "manifests": step_manifests, "verify": step_verify}  # fmt: skip
    return steps[args.step](args)


if __name__ == "__main__":
    raise SystemExit(main())
