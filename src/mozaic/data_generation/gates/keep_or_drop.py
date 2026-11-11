"""Keep or drop whole pairs of one build by whether generation succeeded.

Applies the fixed rule (limits in configs/run.yaml, gates:) to every pair of one build,
reading only finished reports. It never edits a manifest and never decides what gets
muxed: selection happens after render, mux and scoring of every candidate.

Criteria, in order; any failure rejects the ENTIRE four-class pair:

    1  complete group                    every main-arm sample shipped
    2  relative ArcFace (cross render)   leak - floor <= cap, target >= floor,
                                         target > leak, target >= abs_min
    3  speaker cosine, both arms         converted >= sv_floor; vocoded-real vs
                                         its own natural clip >= sv_floor
    4  picture quality                   vs the SAME pair's real clip: flicker
                                         <= real p99, face melting <= 2.0x,
                                         face sharpness >= 0.5x
    5  refusal                           a <render>.refused.json or .failed.json
                                         = rejected, reason "refused"

A missing input is an error only when nothing explains it: a refusal or a withdrawal
by the mux explains a missing render. A criterion with no score (no face found, a
voice not scored) is reported as not measured: it never rejects, and the run is
marked PROVISIONAL. What this never reads: a detector score or any other benchmark result.

    python -m mozaic.data_generation.gates.keep_or_drop --build-id <build> --final-dir <final> \\
        --stratum A --gates-dir <gates> --pair-ids-file <pairs.txt> --out-dir <selection>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

from mozaic.data_generation import settings as S

GENERATED_VIDEO_CLASSES = ("FVRA", "FVFA")
FAKE_CLASSES = ("RVFA", "FVRA", "FVFA")
CRITERIA = ("complete_group", "arcface", "speaker", "syncnet", "picture", "refused")


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Apply the fixed pair-selection rule to one build.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--final-dir", type=Path, required=True, help="the shipped clips this build was scored on")
    p.add_argument("--stratum", required=True, help="A (77 frames) or B (153 frames): which stratum's limits apply")
    p.add_argument("--identity-report", type=Path, default=None, help="default: <gates-dir>/identity_gate.json")
    p.add_argument("--gates-dir", type=Path, default=None, help="default: _reports/<build>_gates")
    p.add_argument("--pair-ids-file", type=Path, default=None, help="restrict to these pairs (one batch)")
    p.add_argument("--out-dir", type=Path, default=None, help="default: _reports/<build>_selection")
    return p.parse_args()


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    """Rows of a JSONL file, or an empty list when it is absent."""
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path: Path | None) -> dict[str, Any] | None:
    """Read a JSON object, or None when the file is absent."""
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def stratum_thresholds(stratum: str) -> dict[str, Any]:
    """One stratum's limits from the settings, the picture rule nested under "picture"."""
    return {**S.GATES[stratum], "picture": S.GATES["picture"]}


# --------------------------------------------------------------------------- the rule, one pair at a time
def evaluate_pair(
    pair_rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    shipped: dict[str, bool],
    identity: dict[str, dict[str, Any]],
    sv: dict[str, dict[str, Any]],
    refused: dict[str, Path],
    waived: frozenset[str] = frozenset(),
    not_applied: frozenset[str] = frozenset(),
    picture: dict[str, dict[str, Any]] | None = None,
    withdrawn: frozenset[str] = frozenset(),
    sync: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Every criterion's measured value, threshold and verdict for one pair.

    waived names sub-checks the rule skips for this pair's video family (arcface_excess
    for FaceFusion, picture_melting for S2V). A waived value is still measured and
    written; it just cannot reject.

    not_applied names limits the rule declines to enforce (e.g. sv_floor): such a
    criterion reads not measured, never a rejection, and the gap shows in the report.
    It differs from a waiver, which says a check does not apply to a family.
    """
    pair_id = pair_rows[0]["pair_id"]
    main = [r for r in pair_rows if r["arm"] == "main"]
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    not_measured: list[str] = []

    def add(criterion: str, ok: bool | None, measured: Any, threshold: Any, note: str = "") -> None:
        checks.append({"criterion": criterion, "ok": ok, "measured": measured, "threshold": threshold, "note": note})

    # refusal first: it explains missing renders, so it must be known before the group check
    cross = [r["render_key"] for r in main if r["class_name"] == "FVRA" and r.get("render_key")]
    settled = [k for k in cross + [r.get("render_key") for r in pair_rows] if k in refused]
    render_key = settled[0] if settled else None
    is_refused = render_key is not None
    if is_refused:
        sidecar = refused[str(render_key)]
        note = (
            "generator failed on the input (documented)"
            if sidecar.name.endswith(".failed.json")
            else "generator declined the input"
        )
        add("refused", False, sidecar.name, None, note)

    # 1 complete group
    missing = [r["sample_id"] for r in main if not shipped.get(r["sample_id"])]
    # A pair the mux dropped by design (picture or loudness tolerance) was moved whole
    # to <final>/_withdrawn/<pair>/: absent on purpose, rejected here, not an error.
    is_withdrawn = pair_id in withdrawn
    if missing and not is_refused and not is_withdrawn:
        errors.append(f"missing shipped sample(s) with no refusal: {missing[:3]}")
    add(
        "complete_group",
        not missing,
        {"missing": missing},
        "withdrawn by the terminal stage (mux drop)" if is_withdrawn else "all main-arm samples shipped",
    )

    # 2 relative ArcFace on the cross render
    rec = identity.get(pair_id)
    if rec is None:
        if not is_refused and not is_withdrawn:
            not_measured.append("arcface")
        add("arcface", None, None, None, "no cross render scored")
    else:
        excess = rec["leak"] - rec["floor"]
        fails = []
        waived_note = ""
        if excess > thresholds["arcface_excess_cap"]:
            if "arcface_excess" in waived:
                waived_note = f"excess {excess:.3f} over cap, waived for this family"
            else:
                fails.append("excess")
        if rec["target"] < thresholds["arcface_target_floor"]:
            fails.append("target_floor")
        if not rec["target"] > rec["leak"]:  # not "<=": a NaN leak must fail here
            fails.append("target_le_leak")
        if rec["target"] < thresholds["arcface_target_abs_min"]:
            fails.append("target_abs_min")
        add(
            "arcface",
            not fails,
            {"target": rec["target"], "leak": rec["leak"], "floor": rec["floor"], "excess": excess},
            {
                "excess_cap": thresholds["arcface_excess_cap"],
                "target_floor": thresholds["arcface_target_floor"],
                "target_abs_min": thresholds["arcface_target_abs_min"],
            },
            "; ".join(x for x in (", ".join(fails), waived_note) if x),
        )

    # 3 speaker cosine, converted arm and vocoded-real arm
    conv_keys = sorted({r["conversion_key"] for r in main if r.get("conversion_key")})
    conv = [sv[k] for k in conv_keys if k in sv]
    ctrl = [sv[f"{k}__ctrl_real"] for k in conv_keys if f"{k}__ctrl_real" in sv]
    has_vocoded = any(r["audio_variant"] == S.AUDIO_VARIANT_VOCODED for r in pair_rows)
    if conv_keys and "sv_floor" in not_applied:
        not_measured.append("speaker")
        add("speaker", None, None, thresholds["sv_floor"], "floor not applied for this stratum yet")
    elif conv_keys and not conv:
        not_measured.append("speaker")
        add("speaker", None, None, thresholds["sv_floor"], "conversion not scored")
    elif conv_keys:
        conv_min = min(r["target_cosine"] for r in conv)
        voc_min = min((r["source_cosine"] for r in ctrl), default=None)
        ok = conv_min >= thresholds["sv_floor"] and (voc_min is None or voc_min >= thresholds["sv_floor"])
        note = (
            "" if ok else ("converted below floor" if conv_min < thresholds["sv_floor"] else "vocoded-real below floor")
        )
        if has_vocoded and voc_min is None:
            note = (note + "; " if note else "") + "vocoded arm not scored"
        add("speaker", ok, {"converted_min": conv_min, "vocoded_vs_natural_min": voc_min}, thresholds["sv_floor"], note)

    # 4 lip sync (SyncNet), on the batch's sampled pairs only: a floor on every scored main-arm clip, and a
    # margin under the pair's own real clip for each fake one. An unsampled pair is not measured.
    sync = sync or {}
    sync_limits = {"floor": thresholds["sync_real_floor"], "margin": thresholds["sync_fake_margin"]}
    scored = {r["sample_id"]: sync[r["sample_id"]] for r in main if r["sample_id"] in sync}
    rvra = next((r for r in main if r["class_name"] == "RVRA"), None)
    if "sync_real_floor" in not_applied:
        not_measured.append("syncnet")
        add("syncnet", None, None, sync_limits, "floor not applied for this stratum yet")
    elif not scored:
        not_measured.append("syncnet")
        add("syncnet", None, None, sync_limits, "not scored")
    else:
        fails = []
        ref = scored.get(rvra["sample_id"]) if rvra else None
        ref_conf = ref.get("confidence") if ref else None
        if ref is not None and ref_conf is None:
            # the real clip itself cannot be tracked: the check cannot be read for this pair
            add("syncnet", None, {"rvra": ref.get("error")}, None, "real clip untrackable: criterion N/A")
        else:
            for row in scored.values():
                conf = row.get("confidence")
                cls = row.get("class_name")
                if conf is None:
                    if cls in FAKE_CLASSES:
                        fails.append(f"{cls}: no face track")
                    continue
                if conf < thresholds["sync_real_floor"]:
                    fails.append(f"{cls}: {conf:.2f} < floor")
                if cls in FAKE_CLASSES and ref_conf is not None and ref_conf - conf > thresholds["sync_fake_margin"]:
                    fails.append(f"{cls}: {ref_conf - conf:.2f} under real > margin")
            add("syncnet", not fails, {str(row.get("class_name")): row.get("confidence") for row in scored.values()},
                sync_limits, "; ".join(fails))  # fmt: skip

    # 5 picture quality: each generated clip against the pair's own real clip
    pic_rule: Any = thresholds.get("picture")  # a nested rule, not a float like the other limits
    if pic_rule is None or picture is None:
        add("picture", None, None, None, "not measured" if pic_rule else "no picture rule for this stratum")
        if pic_rule:
            not_measured.append("picture")
    else:
        real_row = next((r for r in main if r["class_name"] == "RVRA"), None)
        real = picture.get(real_row["sample_id"]) if real_row else None
        # Only the classes whose VIDEO is generated: RVFA carries the real footage,
        # so its picture is the real clip's and cannot be a generation failure.
        gen = {
            r["class_name"]: picture[r["sample_id"]]
            for r in main
            if r["class_name"] in GENERATED_VIDEO_CLASSES and r["sample_id"] in picture
        }
        if real is None or not gen:
            not_measured.append("picture")
            add("picture", None, None, None, "real or generated clip not measured")
        else:
            fails = []
            waived_notes: list[str] = []
            stratum = "B" if S.row_crop_frames(main[0]) > 100 else "A"
            flick_max = pic_rule["flicker_max"][stratum]
            measured: dict[str, Any] = {"real_wobble": real["wobble_face"], "real_sharp": real["sharp_face_lf"]}
            for cls, rec in gen.items():
                melt = rec["wobble_face"] / max(real["wobble_face"], 1e-6)
                sharp = rec["sharp_face_lf"] / max(real["sharp_face_lf"], 1e-6)
                measured[cls] = {"flicker": rec["jit_whole"], "melt_ratio": melt, "sharp_ratio": sharp}
                if rec["jit_whole"] > flick_max:
                    fails.append(f"{cls}: flicker {rec['jit_whole']:.2f} > {flick_max}")
                if melt > pic_rule["melt_ratio_max"]:
                    if "picture_melting" in waived:
                        waived_notes.append(f"{cls}: melting x{melt:.2f} over cap, waived for this family")
                    else:
                        fails.append(f"{cls}: melting x{melt:.2f} > {pic_rule['melt_ratio_max']}")
                if sharp < pic_rule["sharp_ratio_min"]:
                    fails.append(f"{cls}: sharpness x{sharp:.2f} < {pic_rule['sharp_ratio_min']}")
            add(
                "picture",
                not fails,
                measured,
                {
                    "flicker_max": flick_max,
                    "melt_ratio_max": pic_rule["melt_ratio_max"],
                    "sharp_ratio_min": pic_rule["sharp_ratio_min"],
                },
                "; ".join([*fails, *waived_notes]),
            )

    failing = [c["criterion"] for c in checks if c["ok"] is False]
    return {
        "pair_id": pair_id,
        "accepted": not failing and not errors,
        "failing": failing,
        "not_measured": not_measured,
        "errors": errors,
        "checks": checks,
    }


# --------------------------------------------------------------------------- loading
def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Every report the rule reads, keyed for per-pair lookup."""
    rows = read_jsonl(S.new_gen_manifest_path(args.build_id))
    if args.pair_ids_file:
        keep = {x.strip() for x in args.pair_ids_file.read_text().splitlines() if x.strip()}
        rows = [r for r in rows if r["pair_id"] in keep]
    gates = args.gates_dir or (S.new_gen_report_dir() / f"{args.build_id}_gates")
    final_dir = args.final_dir.resolve()

    shipped = {
        r["sample_id"]: S.find_final_sample(r["class_name"], r["sample_id"], final_dir) is not None for r in rows
    }
    ident_path = args.identity_report or (gates / "identity_gate.json")
    if not ident_path.exists():  # older runs wrote it beside the gates folder
        ident_path = S.new_gen_report_dir() / f"{args.build_id}_identity_gate.json"
    ident = read_json(ident_path)
    identity = {}
    if ident:
        for r in ident["pairs"]:
            if not r["self_drive"] and not math.isnan(r["target"]):
                identity[r["pair_id"]] = r
    sv = {r["job_id"]: r for r in read_jsonl(gates / "sv_report.jsonl")}
    sync = {r["job_id"]: r for r in read_jsonl(gates / "sync_report.jsonl")}
    picture_rows = read_jsonl(gates / "picture_report.jsonl")
    picture = {r["sample_id"]: r for r in picture_rows} if picture_rows else None
    refused: dict[str, Path] = {}
    for r in rows:
        key = r.get("render_key")
        if key and S.render_refusal_path(key).exists():
            refused[key] = S.render_refusal_path(key)
        elif key and S.render_failure_path(key).exists():
            # The generator failed twice on this input and documented it: the
            # pair is rejected like a refused one, not held as an error.
            refused[key] = S.render_failure_path(key)
    failed = sorted(
        {r["render_key"] for r in rows if r.get("render_key") and S.render_failure_path(r["render_key"]).exists()}
    )
    withdrawn_dir = final_dir / "_withdrawn"
    withdrawn = (
        frozenset(p.name for p in withdrawn_dir.iterdir() if p.is_dir()) if withdrawn_dir.is_dir() else frozenset()
    )
    return {
        "rows": rows, "final_dir": final_dir, "shipped": shipped, "identity": identity,
        "identity_path": args.identity_report, "sv": sv, "sync": sync, "refused": refused, "failed": failed,
        "gates": gates, "withdrawn": withdrawn, "picture": picture,
    }


def render_markdown(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    """Per-pair table: id, criterion, measured value, threshold, action."""
    lines = [f"# Pair selection — {summary['build_id']} ({summary['status']})", ""]
    lines.append(
        f"final_dir: `{summary['final_dir']}`  stratum: {summary['stratum']}  "
        f"thresholds: `{summary['thresholds_file']}`  date: {summary['date']}"
    )
    lines.append("")
    lines.append(
        f"**{summary['n_accepted']} accepted, {summary['n_rejected']} rejected of {summary['n_pairs']} pairs** "
        f"({summary['rejection_rate']:.1%}); errors {summary['n_errors']}; not measured {summary['n_not_measured']}"
    )
    lines.append("")
    lines.append("| criterion | rejected | not measured |")
    lines.append("|---|---|---|")
    for c in CRITERIA:
        lines.append(
            f"| {c} | {summary['rejected_by_criterion'].get(c, 0)} | {summary['not_measured_by_criterion'].get(c, 0)} |"
        )
    lines.append("")
    lines.append("## Rejected pairs")
    lines.append("")
    lines.append("| pair | criterion | measured | threshold | note |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        for c in r["checks"]:
            if c["ok"] is False:
                lines.append(
                    f"| {r['pair_id']} | {c['criterion']} | {json.dumps(c['measured'], default=float)[:80]} "
                    f"| {json.dumps(c['threshold'], default=float)[:60]} | {c['note']} |"
                )
    if summary["errors"]:
        lines.append("")
        lines.append("## Errors (unexplained, must be resolved before this report counts)")
        lines.append("")
        for pair_id, errs in summary["errors"].items():
            lines.append(f"- {pair_id}: {'; '.join(errs)}")
    return "\n".join(lines) + "\n"


def summarize(
    args: argparse.Namespace, thresholds: dict[str, Any], inputs: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Counts by criterion, the run's status and what was read."""
    accepted = [r for r in results if r["accepted"]]
    rejected = [r for r in results if not r["accepted"]]
    errors = {r["pair_id"]: r["errors"] for r in results if r["errors"]}
    by_crit: dict[str, int] = {}
    nm_crit: dict[str, int] = {}
    for r in results:
        for c in r["failing"]:
            by_crit[c] = by_crit.get(c, 0) + 1
        for c in r["not_measured"]:
            nm_crit[c] = nm_crit.get(c, 0) + 1
    status = "PROVISIONAL" if nm_crit else "FINAL"
    if errors:
        status += " WITH ERRORS"
    return {
        "build_id": args.build_id, "stratum": args.stratum, "final_dir": str(inputs["final_dir"]),
        "thresholds_file": str(S.CONFIGS / "run.yaml"), "thresholds": thresholds,
        "date": dt.date.today().isoformat(), "status": status,
        "n_pairs": len(results), "n_accepted": len(accepted), "n_rejected": len(rejected),
        "rejection_rate": len(rejected) / len(results),
        "rejected_by_criterion": by_crit, "not_measured_by_criterion": nm_crit,
        "n_errors": len(errors), "errors": errors, "n_not_measured": sum(nm_crit.values()),
        "n_refused": sum(1 for r in results if "refused" in r["failing"]),
        "n_failed_renders": len(inputs["failed"]),
        "inputs": {"identity": len(inputs["identity"]), "sv_rows": len(inputs["sv"]), "sync_rows": len(inputs["sync"]),
                   "picture_rows": len(inputs["picture"] or {}), "gates_dir": str(inputs["gates"])},
    }


def write_outputs(out_dir: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """The kept and dropped lists, the per-pair records, the summary and the readable report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pair_ids_accepted.txt").write_text("".join(r["pair_id"] + "\n" for r in results if r["accepted"]))
    (out_dir / "pair_ids_rejected.txt").write_text(
        "".join(f"{r['pair_id']}\t{','.join(r['failing']) or 'error'}\n" for r in results if not r["accepted"])
    )
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (out_dir / "selection_pairs.jsonl").write_text("".join(json.dumps(r, default=float) + "\n" for r in results))
    (out_dir / "selection_report.md").write_text(render_markdown(summary, results))


def main() -> int:
    """Apply the rule, write the lists and the report, exit 1 on unexplained errors."""
    args = parse_args()
    thresholds = stratum_thresholds(args.stratum)
    not_applied = frozenset(thresholds.get("not_applied", ()))
    if not_applied:
        print(f"[select] stratum {args.stratum}: thresholds NOT applied yet: {sorted(not_applied)}")
    exceptions = S.GATES.get("family_exceptions", {})
    inputs = load_inputs(args)
    rows = inputs["rows"]
    if not rows:
        print("[select] no manifest rows selected", file=sys.stderr)
        return 2
    if inputs["failed"]:
        print(
            f"[select] {len(inputs['failed'])} render(s) carry a .failed.json — resolve or re-render: "
            f"{inputs['failed'][:3]}",
            file=sys.stderr,
        )

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    results = []
    for _pid, prs in sorted(by_pair.items()):
        family = next((r["video_method"] for r in prs if r.get("needs_render")), None)
        waived = frozenset(exceptions.get(family, []))
        results.append(
            evaluate_pair(prs, thresholds, inputs["shipped"], inputs["identity"], inputs["sv"], inputs["refused"],
                          waived, not_applied, inputs["picture"], inputs["withdrawn"], inputs["sync"])
        )

    summary = summarize(args, thresholds, inputs, results)
    out_dir = args.out_dir or (S.new_gen_report_dir() / f"{args.build_id}_selection")
    write_outputs(out_dir, summary, results)

    print(f"[select] {args.build_id} [{summary['status']}]: {summary['n_accepted']} accepted, "
          f"{summary['n_rejected']} rejected of {summary['n_pairs']} ({summary['rejection_rate']:.1%}); "
          f"by criterion {summary['rejected_by_criterion']}; not measured {summary['not_measured_by_criterion']}; "
          f"errors {summary['n_errors']}")
    print(f"[select] -> {out_dir}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
