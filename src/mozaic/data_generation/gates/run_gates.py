"""The gate stage of one batch: score every build, keep or drop each pair, withdraw, record.

A batch is a list of pairs per clip length (stratum A: 77 frames, B: 153 frames), made by
several builds (one per video family and length). Run it after the mux of every build of
the batch has finished. It never makes new pairs or renders. In order:

  1. score     for each build: the face check (face_check, in the face environment), the
               voice check's list (make_voice_check_list), the lip-sync check's list of the
               batch's first 20 pairs (make_lip_sync_list), the picture check (picture_check),
               the voice check (voice_check, in the voice environment) and the lip-sync check
               (lip_sync_check, in the SyncNet environment); a build whose reports are all
               there is skipped
  2. select    for each build: keep_or_drop applies the keep/drop rule to the batch's pairs
  3. withdraw  for each build: the rejected pairs' clips move to <final>/_withdrawn/<pair>/
               (final_clips/withdraw_incomplete_pairs.py)
  4. ledger    one line per dropped pair: <work-dir>/<batch>.drops.jsonl

A build's clips are in <new_generated>/final_<build>/, its reports in <new_generated>/_reports/:
<build>_pairs<N>_<hash>_gates/ (the scores) and <build>_selection_<batch>/ (the decisions).

    python -m mozaic.data_generation.gates.run_gates --batch batch_02 \\
        --pairs A=<pairs>.A.txt --pairs B=<pairs>.B.txt \\
        --build bulk_v1_ff77:A --build bulk_v1_ff153:B --build bulk_v1_s2v77:A \\
        --build bulk_v1_wan77:A --build bulk_v1_s2v153:B \\
        --face-python <face env>/bin/python --voice-python <voice env>/bin/python \\
        --syncnet-python <SyncNet env>/bin/python
    ... --dry-run         # print every step; run and write nothing
    ... --force-gates     # score and select again although the reports exist (adds drops, never restores clips)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S

STRATA = ("A", "B")
GATES = "mozaic.data_generation.gates"
WITHDRAW = "mozaic.data_generation.final_clips.withdraw_incomplete_pairs"
SRC = Path(__file__).resolve().parents[3]  # src/, put on every step's path
# Reports a scored build has, besides identity_gate.json (see gates_complete).
GATE_REPORTS = ("picture_report.jsonl", "sv_report.jsonl", "sync_report.jsonl")
LIP_SYNC = SRC / "mozaic" / "data_generation" / "gates" / "lip_sync_check.py"  # runs as a script in its own env
# One thread per math library in every step, as in the release run.
THREAD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="The gate stage of one batch: score, select, withdraw, drop ledger.")
    p.add_argument("--batch", required=True, help="name of the batch; names its selection folders and ledger")
    p.add_argument("--pairs", action="append", required=True, metavar="STRATUM=FILE",
                   help="the batch's pair list of one stratum (A or B), one pair id per line; once per stratum")
    p.add_argument("--build", action="append", required=True, metavar="BUILD_ID:STRATUM",
                   help="a build of the batch and its stratum; once per build, in the order to run them")
    p.add_argument("--face-python", default=os.environ.get("FACE_PYTHON"),
                   help="python of the environment with insightface (default: $FACE_PYTHON)")
    p.add_argument("--voice-python", default=os.environ.get("VOICE_PYTHON"),
                   help="python of the environment with speechbrain (default: $VOICE_PYTHON)")
    p.add_argument("--syncnet-python", default=os.environ.get("SYNCNET_PYTHON"),
                   help="python of the SyncNet environment (default: $SYNCNET_PYTHON); SyncNet's code in "
                        "$SYNCNET_UPSTREAM")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="ledger, summary, done marker and withdraw lists "
                        "(default: <new_generated>/_reports/gate_stage)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force-gates", action="store_true",
                   help="score and select again even if the done marker or reports exist; "
                        "appends new drops, never restores media")
    return p.parse_args()


# --------------------------------------------------------------------------- small helpers (tested)
def parse_builds(specs: list[str]) -> list[tuple[str, str]]:
    """'bulk_v1_ff77:A' -> ('bulk_v1_ff77', 'A'), in the given order."""
    builds = []
    for spec in specs:
        build, _, stratum = spec.rpartition(":")
        if not build or stratum not in STRATA:
            raise SystemExit(f"--build {spec!r}: want BUILD_ID:A or BUILD_ID:B")
        builds.append((build, stratum))
    return builds


def parse_pair_lists(specs: list[str]) -> dict[str, Path]:
    """'A=lists/batch_02.A.txt' -> {'A': Path(...)}."""
    lists = {}
    for spec in specs:
        stratum, _, path = spec.partition("=")
        if stratum not in STRATA or not path:
            raise SystemExit(f"--pairs {spec!r}: want A=FILE or B=FILE")
        lists[stratum] = Path(path)
    return lists


def read_ids(path: Path) -> set[str]:
    """The pair ids of a list, one per line."""
    return {x.strip() for x in path.read_text().splitlines() if x.strip()}


def final_dir_of(build_id: str) -> Path:
    """Where a build's clips ship: final_<build>/ beside the other builds' folders."""
    return S.NEW_GEN_ROOT / f"final_{build_id}"


def gate_dir_of(reports: Path, build: str, pair_ids_file: Path) -> Path:
    """The build's gate folder for this pair list, named by the list's content hash."""
    suffix = S.batch_suffix_for({x.strip() for x in pair_ids_file.read_text().split()})
    return reports / f"{build}{suffix}_gates"


def gates_complete(gate_dir: Path, identity_report: Path) -> bool:
    """All the gate reports exist (the identity report in the gate folder or beside it)."""
    ident = (gate_dir / "identity_gate.json").exists() or identity_report.exists()
    return ident and all((gate_dir / r).exists() for r in GATE_REPORTS)


def selection_complete(out_dir: Path) -> bool:
    """Return whether keep_or_drop wrote a clean report (no unexplained errors)."""
    summary = out_dir / "selection_summary.json"
    rejected = out_dir / "pair_ids_rejected.txt"
    if not summary.exists() or not rejected.exists():
        return False
    try:
        data = json.loads(summary.read_text())
    except json.JSONDecodeError:
        return False
    return int(data.get("n_errors", 1)) == 0


def parse_rejected_txt(text: str) -> list[tuple[str, list[str]]]:
    """First column is pair_id; optional tab-separated failing criteria."""
    out: list[tuple[str, list[str]]] = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        pid, _, rest = ln.partition("\t")
        pid = pid.strip()
        if not pid:
            continue
        crit = [c for c in rest.strip().split(",") if c]
        out.append((pid, crit))
    return out


def incomplete_mux_ids(final_dir: Path, pair_ids: set[str]) -> set[str]:
    """Pairs of the list the mux already moved aside because a class was missing."""
    withdrawn = final_dir / "_withdrawn"
    if not withdrawn.is_dir():
        return set()
    found: set[str] = set()
    for pair_id in pair_ids:
        reason = withdrawn / pair_id / "reason.json"
        if not reason.exists():
            continue
        try:
            data = json.loads(reason.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("reason") == "incomplete_group":
            found.add(pair_id)
    return found


def drop_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Identity of one ledger line: pair, build, why it left."""
    return (str(row.get("pair_id", "")), str(row.get("build_id", "")), str(row.get("source", "")))


def load_drops(path: Path) -> list[dict[str, Any]]:
    """The ledger's rows, or none when it does not exist yet."""
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def merge_drops(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append-only: keep history, skip duplicate (pair_id, build_id, source)."""
    seen = {drop_key(r) for r in existing}
    out = list(existing)
    for row in new:
        k = drop_key(row)
        if k in seen or not k[0]:
            continue
        seen.add(k)
        out.append(row)
    return out


def write_drops(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the whole ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def ledger_rows(
    build: str, stratum: str, batch: str, rejected: list[tuple[str, list[str]]], muxed: set[str],
    now: str, final_dir: Path, sel_dir: Path,
) -> list[dict[str, Any]]:
    """One ledger row per dropped pair of a build: the rule's rejects, then mux drops the rule did not list."""

    def row(pid: str, source: str, criteria: list[str]) -> dict[str, Any]:
        return {
            "pair_id": pid, "build_id": build, "stratum": stratum, "batch": batch, "source": source,
            "criteria": criteria, "withdrawn_at": now, "final_dir": str(final_dir),
            "withdrawn_dir": str(final_dir / "_withdrawn" / pid), "selection_dir": str(sel_dir),
        }

    rows = [row(pid, "incomplete_mux" if pid in muxed else "rule", crit) for pid, crit in rejected]
    listed = {pid for pid, _ in rejected}
    rows += [row(pid, "incomplete_mux", []) for pid in sorted(muxed) if pid not in listed]
    return rows


# --------------------------------------------------------------------------- running the steps
def log(msg: str) -> None:
    """Print one timestamped line."""
    print(f"[gate {dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def step_env() -> dict[str, str]:
    """Environment of every step: this repo's code on the path, one thread per math library."""
    path = os.pathsep.join(x for x in (str(SRC), os.environ.get("PYTHONPATH", "")) if x)
    return {**os.environ, **THREAD_ENV, "PYTHONPATH": path}


def run(argv: list[str], dry_run: bool) -> int:
    """Run one step to completion (its output goes straight to the log); return its exit code."""
    log("$ " + " ".join(argv))
    if dry_run:
        return 0
    return subprocess.run(argv, env=step_env(), check=False).returncode


def run_or_stop(argv: list[str], dry_run: bool) -> None:
    """Run one step; stop the stage if it fails."""
    rc = run(argv, dry_run)
    if rc != 0:
        raise SystemExit(f"step failed (rc={rc}): {argv[:3]}")


def check_scoring_inputs(final_dir: Path, pairs_file: Path) -> None:
    """Refuse to score an empty or missing final folder or pair list."""
    if not final_dir.is_dir():
        raise SystemExit(f"no such final folder: {final_dir}")
    if not any(p.suffix in S.SHIPPED_MEDIA_EXTS for p in final_dir.rglob("*")):
        raise SystemExit(f"no shipped clips under {final_dir}")
    if not pairs_file.exists() or pairs_file.stat().st_size == 0:
        raise SystemExit(f"pair list missing or empty: {pairs_file}")


def score(
    build: str, pairs_file: Path, gate_dir: Path, face_python: str, voice_python: str, syncnet_python: str,
    dry_run: bool,
) -> list[str]:
    """Run the scorers of one build in order; return the steps that failed.

    A failed step does not stop the others, so one problem never costs the other reports.
    """
    final_dir = final_dir_of(build)
    if not dry_run:
        check_scoring_inputs(final_dir, pairs_file)
        gate_dir.mkdir(parents=True, exist_ok=True)
    common = ["--build-id", build, "--final-dir", str(final_dir)]
    pairs = ["--pair-ids-file", str(pairs_file)]
    steps = [
        ("face_check", [face_python, "-m", f"{GATES}.face_check", *common,
                           "--out", str(gate_dir / "identity_gate.json"), *pairs]),
        ("make_voice_check_list", [sys.executable, "-m", f"{GATES}.make_voice_check_list", *common,
                             "--out-dir", str(gate_dir), *pairs]),
        ("make_lip_sync_list", [sys.executable, "-m", f"{GATES}.make_lip_sync_list", *common,
                                "--out-dir", str(gate_dir), *pairs]),
        ("picture_check", [sys.executable, "-m", f"{GATES}.picture_check", *common,
                             "--out", str(gate_dir / "picture_report.jsonl"), *pairs]),
    ]
    failed = [name for name, argv in steps if run(argv, dry_run) != 0]
    sv_pairs = gate_dir / "sv_pairs.jsonl"
    if dry_run or (sv_pairs.exists() and sv_pairs.stat().st_size > 0):
        voice = [voice_python, "-m", f"{GATES}.voice_check", "--pairs", str(sv_pairs),
                 "--out", str(gate_dir / "sv_report.jsonl")]
        if run(voice, dry_run) != 0:
            failed.append("voice_check")
    else:
        log(f"{build}: voice_check skipped, no voice pairs")
    sync_pairs = gate_dir / "sync_pairs.jsonl"
    if dry_run or (sync_pairs.exists() and sync_pairs.stat().st_size > 0):
        lip = [syncnet_python, str(LIP_SYNC), "--pairs", str(sync_pairs), "--out", str(gate_dir / "sync_report.jsonl")]
        if run(lip, dry_run) != 0:
            failed.append("lip_sync_check")
    else:
        log(f"{build}: lip_sync_check skipped, no clips to score")
    return failed


# --------------------------------------------------------------------------- the stage
def score_builds(args: argparse.Namespace, builds: list[tuple[str, str]], lists: dict[str, Path]) -> None:
    """Step 1: score every build whose reports are missing; stop if some are still missing."""
    reports = S.new_gen_report_dir()

    def complete(build: str, stratum: str) -> bool:
        return gates_complete(gate_dir_of(reports, build, lists[stratum]), reports / f"{build}_identity_gate.json")

    todo = [(b, s) for b, s in builds if args.force_gates or not complete(b, s)]
    if not todo:
        log("every build is scored already")
        return
    if not args.dry_run and not (args.face_python and args.voice_python and args.syncnet_python):
        raise SystemExit("scoring needs --face-python, --voice-python and --syncnet-python "
                         "(or FACE_PYTHON, VOICE_PYTHON and SYNCNET_PYTHON)")
    face_python = args.face_python or "<face python>"  # placeholders show only in a dry run
    voice_python = args.voice_python or "<voice python>"
    syncnet_python = args.syncnet_python or "<syncnet python>"
    failed: dict[str, list[str]] = {}
    for build, stratum in todo:
        gate_dir = gate_dir_of(reports, build, lists[stratum])
        failed[build] = score(build, lists[stratum], gate_dir, face_python, voice_python, syncnet_python,
                              args.dry_run)
    if args.dry_run:
        return
    still = [b for b, s in builds if not complete(b, s)]
    if still:
        raise SystemExit("gate reports still missing for "
                         + ", ".join(f"{b} (failed steps: {failed.get(b) or 'none'})" for b in still))


def select_builds(args: argparse.Namespace, builds: list[tuple[str, str]], lists: dict[str, Path]) -> None:
    """Step 2: apply the rule to every build not already selected cleanly."""
    reports = S.new_gen_report_dir()
    for build, stratum in builds:
        out_dir = reports / f"{build}_selection_{args.batch}"
        if selection_complete(out_dir) and not args.force_gates:
            log(f"{build}: selection already complete at {out_dir}")
            continue
        run_or_stop([sys.executable, "-m", f"{GATES}.keep_or_drop", "--build-id", build,
                     "--final-dir", str(final_dir_of(build)), "--stratum", stratum,
                     "--gates-dir", str(gate_dir_of(reports, build, lists[stratum])),
                     "--out-dir", str(out_dir), "--pair-ids-file", str(lists[stratum])], args.dry_run)


def withdraw_and_record(
    args: argparse.Namespace, builds: list[tuple[str, str]], lists: dict[str, Path], work: Path
) -> None:
    """Steps 3 and 4: move every rejected pair aside, then write the drop ledger and the done marker."""
    reports = S.new_gen_report_dir()
    now = f"{dt.datetime.now():%Y-%m-%d %H:%M}"
    new_rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for build, stratum in builds:
        final_dir = final_dir_of(build)
        sel_dir = reports / f"{build}_selection_{args.batch}"
        rejected = parse_rejected_txt((sel_dir / "pair_ids_rejected.txt").read_text() if not args.dry_run else "")
        rejected_ids = [pid for pid, _ in rejected]
        muxed = incomplete_mux_ids(final_dir, read_ids(lists[stratum])) if not args.dry_run else set()
        wpath = work / "withdraw" / f"{args.batch}.{build}.withdraw.txt"
        if not args.dry_run:
            wpath.parent.mkdir(parents=True, exist_ok=True)
            wpath.write_text("".join(x + "\n" for x in rejected_ids))
        if rejected_ids:
            run_or_stop([sys.executable, "-m", WITHDRAW, "--build-id", build, "--final-dir", str(final_dir),
                         "--withdraw-pairs", str(wpath), "--reason", f"selection rule, {args.batch}, {dt.date.today()}",
                         "--apply"], args.dry_run)
        new_rows += ledger_rows(build, stratum, args.batch, rejected, muxed, now, final_dir, sel_dir)
        counts[build] = {"rejected": len(rejected_ids), "incomplete_mux": len(muxed)}
        log(f"{build}: {len(rejected_ids)} rule reject(s), {len(muxed)} incomplete mux")

    if args.dry_run:
        return
    drops_path = work / f"{args.batch}.drops.jsonl"
    merged = merge_drops(load_drops(drops_path), new_rows)
    write_drops(drops_path, merged)
    summary = {"batch": args.batch, "gated_at": now, "drops": str(drops_path), "counts": counts,
               "n_drop_rows": len(merged)}
    (work / f"{args.batch}.gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (work / f"{args.batch}.GATE_DONE").write_text(f"{now} gated {args.batch} drops {len(merged)}\n")


def main() -> int:
    """Run the gate stage of one batch."""
    args = parse_args()
    os.umask(0o002)  # files group-writable, like the rest of the data folder
    builds = parse_builds(args.build)
    lists = parse_pair_lists(args.pairs)
    work = args.work_dir or (S.new_gen_report_dir() / "gate_stage")
    done_marker = work / f"{args.batch}.GATE_DONE"
    if done_marker.exists() and not args.force_gates:
        log(f"already gated: {done_marker}")
        return 0
    for build, stratum in builds:
        if stratum not in lists or not lists[stratum].exists():
            raise SystemExit(f"{build}: no pair list for stratum {stratum} (--pairs {stratum}=FILE)")
    if not args.dry_run:
        work.mkdir(parents=True, exist_ok=True)

    log("1/4 score")
    score_builds(args, builds, lists)
    log("2/4 select")
    select_builds(args, builds, lists)
    log("3/4 withdraw, 4/4 drop ledger")
    withdraw_and_record(args, builds, lists, work)
    log(f"{args.batch}: GATE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
