"""Tests for the keep/drop gates: the rule on made-up reports, the face gate's pair filter,
and the gate stage's helpers and step order.

Made-up rows only: no clip is read and no model is loaded. What is pinned: a corrupted
pair is rejected on the criterion that measured it; a documented refusal or a mux
withdrawal is a rejection without an error; an unexplained missing render is an error;
an unscored criterion is "not measured", never a pass or a rejection; the limits are
the release's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mozaic.data_generation.gates import face_check as I  # noqa: E741
from mozaic.data_generation.gates import run_gates as G
from mozaic.data_generation.gates import keep_or_drop as SP

PAIR = "id01_abc_00001__x__id02"
RULE = {
    "arcface_excess_cap": 0.10,
    "arcface_target_floor": 0.15,
    "arcface_target_abs_min": 0.30,
    "sv_floor": 0.25,
    "sync_real_floor": 3.4,
    "sync_fake_margin": 3.8,
}


def _rows() -> list[dict[str, Any]]:
    base = {"pair_id": PAIR, "arm": "main", "audio_variant": "natural_real", "conversion_key": None, "render_key": None}
    return [
        {**base, "sample_id": f"{PAIR}__RVRA", "class_name": "RVRA"},
        {**base, "sample_id": f"{PAIR}__RVFA", "class_name": "RVFA", "conversion_key": "conv__x",
         "audio_variant": "converted_fake"},
        {**base, "sample_id": f"{PAIR}__FVRA", "class_name": "FVRA", "render_key": "cross__x"},
        {**base, "sample_id": f"{PAIR}__FVFA", "class_name": "FVFA", "render_key": "cross__x",
         "conversion_key": "conv__x", "audio_variant": "converted_fake"},
    ]


def _good() -> dict[str, Any]:
    rows = _rows()
    return {
        "pair_rows": rows,
        "thresholds": RULE,
        "shipped": {r["sample_id"]: True for r in rows},
        "identity": {PAIR: {"target": 0.55, "leak": 0.03, "floor": 0.02}},
        "sv": {
            "conv__x": {"target_cosine": 0.5, "source_cosine": 0.1},
            "conv__x__ctrl_real": {"target_cosine": 0.1, "source_cosine": 0.99},
        },
        "refused": {},
        "sync": {f"{PAIR}__{c}": {"class_name": c, "confidence": v}
                 for c, v in (("RVRA", 8.0), ("RVFA", 7.9), ("FVRA", 7.6), ("FVFA", 8.0))},
    }


# --------------------------------------------------------------------------- the rule
def test_clean_pair_is_accepted() -> None:
    r = SP.evaluate_pair(**_good())
    assert r["accepted"] and not r["failing"] and not r["errors"] and not r["not_measured"]
    assert [c["criterion"] for c in r["checks"]] == ["complete_group", "arcface", "speaker", "syncnet", "picture"]


def test_each_corruption_rejects_on_its_own_criterion() -> None:
    g = _good()
    g["identity"] = {PAIR: {"target": 0.55, "leak": 0.20, "floor": 0.02}}
    assert SP.evaluate_pair(**g)["failing"] == ["arcface"]
    g = _good()
    g["identity"] = {PAIR: {"target": 0.28, "leak": 0.03, "floor": 0.02}}
    r = SP.evaluate_pair(**g)
    assert r["failing"] == ["arcface"] and "target_abs_min" in r["checks"][1]["note"]
    g = _good()
    g["sv"]["conv__x"]["target_cosine"] = 0.1
    assert SP.evaluate_pair(**g)["failing"] == ["speaker"]
    g = _good()
    g["sv"]["conv__x__ctrl_real"]["source_cosine"] = 0.1
    assert SP.evaluate_pair(**g)["failing"] == ["speaker"]


def test_a_nan_leak_fails_target_above_leak() -> None:
    """No face found in the real clip: leak and floor are NaN, and the face check fails."""
    g = _good()
    g["identity"] = {PAIR: {"target": 0.55, "leak": float("nan"), "floor": float("nan")}}
    r = SP.evaluate_pair(**g)
    assert r["failing"] == ["arcface"] and r["checks"][1]["note"] == "target_le_leak"


def test_documented_refusal_is_a_rejection_not_an_error() -> None:
    g = _good()
    g["refused"] = {"cross__x": Path("/renders/cross__x.refused.json")}
    g["shipped"][f"{PAIR}__FVRA"] = False
    g["shipped"][f"{PAIR}__FVFA"] = False
    g["identity"] = {}
    r = SP.evaluate_pair(**g)
    assert not r["accepted"] and "refused" in r["failing"] and r["errors"] == [] and r["not_measured"] == []


def test_failed_self_drive_render_explains_the_missing_pair() -> None:
    """The self-drive render failed, the mux dropped the pair, the cross render was fine."""
    g = _good()
    g["pair_rows"] = [
        *g["pair_rows"],
        {**g["pair_rows"][2], "sample_id": f"{PAIR}__FVRA__self_drive", "arm": "self_drive", "render_key": "self__x"},
    ]
    g["refused"] = {"self__x": Path("/renders/self__x.failed.json")}
    g["shipped"] = dict.fromkeys(g["shipped"], False)
    r = SP.evaluate_pair(**g)
    assert "refused" in r["failing"] and r["errors"] == []
    assert r["checks"][0]["measured"] == "self__x.failed.json"


def test_withdrawn_pair_is_a_rejection_not_an_error() -> None:
    """The mux dropped it and the terminal stage moved it aside: absent on purpose."""
    g = _good()
    g["shipped"] = dict.fromkeys(g["shipped"], False)
    g["withdrawn"] = frozenset({g["pair_rows"][0]["pair_id"]})
    r = SP.evaluate_pair(**g)
    assert "complete_group" in r["failing"] and not r["errors"]
    assert "arcface" not in r["not_measured"]


def test_unexplained_missing_render_is_an_error() -> None:
    g = _good()
    g["shipped"][f"{PAIR}__FVRA"] = False
    r = SP.evaluate_pair(**g)
    assert r["errors"] and not r["accepted"]


def test_unscored_voice_is_not_measured_never_a_rejection() -> None:
    g = _good()
    g["sv"] = {}
    r = SP.evaluate_pair(**g)
    assert r["accepted"] and r["not_measured"] == ["speaker"] and "speaker" not in r["failing"]


def test_family_waiver_reports_the_excess_but_does_not_reject() -> None:
    g = _good()
    g["identity"] = {PAIR: {"target": 0.55, "leak": 0.20, "floor": 0.02}}
    r = SP.evaluate_pair(**g, waived=frozenset({"arcface_excess"}))
    assert r["accepted"] and "waived" in r["checks"][1]["note"]
    g["identity"] = {PAIR: {"target": 0.10, "leak": 0.20, "floor": 0.02}}
    r = SP.evaluate_pair(**g, waived=frozenset({"arcface_excess"}))
    assert r["failing"] == ["arcface"] and "target_le_leak" in r["checks"][1]["note"]


def test_a_threshold_that_is_not_applied_cannot_reject() -> None:
    """A limit named in not_applied is measured and written, never a rejection."""
    below = _good()
    below["sv"]["conv__x"]["target_cosine"] = 0.01
    assert "speaker" in SP.evaluate_pair(**below)["failing"]  # enforced, it rejects
    relaxed = SP.evaluate_pair(**below, not_applied=frozenset({"sv_floor"}))
    assert relaxed["failing"] == [] and relaxed["accepted"] is True
    assert relaxed["not_measured"] == ["speaker"]
    notes = {c["criterion"]: c["note"] for c in relaxed["checks"]}
    assert "not applied" in notes["speaker"]


def test_not_applied_does_not_touch_the_other_criteria() -> None:
    """It must silence exactly the limits named, and nothing else."""
    g = _good()
    g["identity"] = {PAIR: {"target": 0.55, "leak": 0.20, "floor": 0.02}}
    r = SP.evaluate_pair(**g, not_applied=frozenset({"sv_floor"}))
    assert r["failing"] == ["arcface"]


PICTURE_RULE = {
    **RULE,
    "picture": {"flicker_max": {"A": 1.0, "B": 1.0}, "melt_ratio_max": 2.0, "sharp_ratio_min": 0.5},
}


def _picture(**gen: dict[str, float]) -> dict[str, dict[str, float]]:
    """Build a picture report: the real clip plus every other clip, by sample id."""
    real = {"jit_whole": 0.2, "wobble_face": 3.0, "sharp_face_lf": 100.0}
    out = {f"{PAIR}__RVRA": real}
    for cls in ("RVFA", "FVRA", "FVFA"):
        out[f"{PAIR}__{cls}"] = {**real, **gen.get(cls, {})}
    return out


def test_picture_criterion_compares_each_generated_clip_with_the_pairs_own_real_clip() -> None:
    """Flicker, melting and softness reject the whole pair."""
    g = {**_good(), "thresholds": PICTURE_RULE, "picture": _picture()}
    r = SP.evaluate_pair(**g)
    assert r["accepted"] and "picture" not in r["not_measured"]
    g["picture"] = _picture(FVRA={"wobble_face": 6.3})  # 2.1x the real face's change: melting
    r = SP.evaluate_pair(**g)
    note = next(c for c in r["checks"] if c["criterion"] == "picture")["note"]
    assert r["failing"] == ["picture"] and "melting x2.10" in note
    g["picture"] = _picture(FVFA={"sharp_face_lf": 40.0})  # 0.4x the real sharpness: soft
    assert SP.evaluate_pair(**g)["failing"] == ["picture"]
    g["picture"] = _picture(FVRA={"jit_whole": 1.4})  # pulsing brighter and darker
    assert SP.evaluate_pair(**g)["failing"] == ["picture"]
    g["picture"] = _picture(RVFA={"wobble_face": 6.3})  # the real-video class is not judged on the render
    assert SP.evaluate_pair(**g)["accepted"]


def test_picture_criterion_is_not_measured_without_a_report_and_absent_without_a_rule() -> None:
    g = {**_good(), "thresholds": PICTURE_RULE, "picture": None}
    r = SP.evaluate_pair(**g)
    assert r["accepted"] and r["not_measured"] == ["picture"]
    g = {**_good(), "picture": _picture(FVRA={"wobble_face": 9.0})}  # RULE has no picture rule: nothing to apply
    r = SP.evaluate_pair(**g)
    assert r["accepted"] and not r["not_measured"]


def test_melting_is_waived_for_a_family_that_invents_its_own_motion() -> None:
    """S2V: audio-driven motion is not melting; written, never a rejection."""
    g = {
        **_good(),
        "thresholds": PICTURE_RULE,
        "picture": _picture(FVRA={"wobble_face": 9.0}),
        "waived": frozenset({"picture_melting"}),
    }
    r = SP.evaluate_pair(**g)
    assert r["accepted"]
    assert "waived for this family" in next(c for c in r["checks"] if c["criterion"] == "picture")["note"]


def test_limits_are_the_release_ones() -> None:
    """The per-stratum limits the rule reads, exactly as the release run used them."""
    picture = {"flicker_max": {"A": 0.993, "B": 0.843}, "melt_ratio_max": 2.0, "sharp_ratio_min": 0.5}
    face = {"arcface_excess_cap": 0.0977, "arcface_target_floor": 0.1442, "arcface_target_abs_min": 0.3}
    sync_a = {"sync_real_floor": 3.411, "sync_fake_margin": 3.761}
    sync_b = {"sync_real_floor": 4.1096, "sync_fake_margin": 3.8225}
    assert SP.stratum_thresholds("A") == {**face, "sv_floor": 0.2496, **sync_a, "picture": picture}
    assert SP.stratum_thresholds("B") == {**face, "sv_floor": 0.3048, **sync_b, "picture": picture}
    assert list(SP.stratum_thresholds("A")) == [*face, "sv_floor", *sync_a, "picture"]
    assert SP.S.GATES["family_exceptions"] == {"facefusion": ["arcface_excess"], "s2v": ["picture_melting"]}
    assert SP.CRITERIA == ("complete_group", "arcface", "speaker", "syncnet", "picture", "refused")


# --------------------------------------------------------------------------- the face gate
def _render_row(pair: str, cls: str = "FVRA", key: str | None = None) -> dict[str, object]:
    return {"pair_id": pair, "class_name": cls, "needs_render": cls == "FVRA", "render_key": key or f"cross__{pair}"}


def test_pair_list_restricts_the_renders_scored(tmp_path: Path) -> None:
    rows = [_render_row("p1"), _render_row("p2"), _render_row("p3"), _render_row("p1", "RVRA", "cross__p1")]
    assert set(I.cross_renders(rows, None)) == {"cross__p1", "cross__p2", "cross__p3"}
    lst = tmp_path / "pairs.txt"
    lst.write_text("p2\n\np3\n")
    assert set(I.cross_renders(rows, I.read_pair_ids(lst))) == {"cross__p2", "cross__p3"}
    assert I.read_pair_ids(None) is None


def test_insightface_root_honors_flag_then_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("INSIGHTFACE_ROOT", raising=False)
    assert I.insightface_root() == Path.home() / ".insightface"
    monkeypatch.setenv("INSIGHTFACE_ROOT", str(tmp_path / "env"))
    assert I.insightface_root() == tmp_path / "env"
    assert I.insightface_root(tmp_path / "explicit") == tmp_path / "explicit"


# --------------------------------------------------------------------------- the gate stage
def test_rejected_txt_takes_the_first_column_and_criteria() -> None:
    text = "p1\tspeaker,arcface\np2\t\n\np3\tpicture\n"
    assert G.parse_rejected_txt(text) == [("p1", ["speaker", "arcface"]), ("p2", []), ("p3", ["picture"])]


def test_drop_ledger_is_append_only_and_keyed() -> None:
    a = {"pair_id": "p1", "build_id": "bulk_v1_ff77", "source": "rule", "criteria": ["arcface"]}
    b = {"pair_id": "p1", "build_id": "bulk_v1_ff77", "source": "rule", "criteria": ["ignored"]}
    c = {"pair_id": "p1", "build_id": "bulk_v1_s2v77", "source": "rule"}
    d = {"pair_id": "p1", "build_id": "bulk_v1_ff77", "source": "incomplete_mux"}
    merged = G.merge_drops([a], [b, c, d, a])
    assert [G.drop_key(r) for r in merged] == [
        ("p1", "bulk_v1_ff77", "rule"),
        ("p1", "bulk_v1_s2v77", "rule"),
        ("p1", "bulk_v1_ff77", "incomplete_mux"),
    ]
    assert merged[0]["criteria"] == ["arcface"], "history is not rewritten"


def test_drops_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "batch_02.drops.jsonl"
    rows = [{"pair_id": "p1", "build_id": "bulk_v1_ff77", "source": "rule"}]
    G.write_drops(path, rows)
    assert G.merge_drops(G.load_drops(path), rows) == rows


def test_selection_complete_requires_zero_errors(tmp_path: Path) -> None:
    d = tmp_path / "sel"
    d.mkdir()
    assert not G.selection_complete(d)
    (d / "pair_ids_rejected.txt").write_text("p1\tspeaker\n")
    (d / "selection_summary.json").write_text('{"n_errors": 1}')
    assert not G.selection_complete(d)
    (d / "selection_summary.json").write_text('{"n_errors": 0}')
    assert G.selection_complete(d)


def test_gate_dirs_are_named_by_the_pair_list(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("p1\np2\n")
    b.write_text("p9\n")
    reports = tmp_path / "_reports"
    da = G.gate_dir_of(reports, "bulk_v1_ff77", a)
    assert da != G.gate_dir_of(reports, "bulk_v1_ff77", b)
    assert da.name.startswith("bulk_v1_ff77_pairs2_") and da.name.endswith("_gates")


def test_gates_complete_needs_every_report(tmp_path: Path) -> None:
    gates = tmp_path / "g"
    gates.mkdir()
    beside = tmp_path / "bulk_v1_ff77_identity_gate.json"
    for name in ("picture_report.jsonl", "sv_report.jsonl", "sync_report.jsonl"):
        (gates / name).write_text("")
    assert not G.gates_complete(gates, beside)
    beside.write_text("{}")  # an identity report beside the gate folder counts
    assert G.gates_complete(gates, beside)
    (gates / "sv_report.jsonl").unlink()
    assert not G.gates_complete(gates, beside)


def test_ledger_rows_name_the_rule_and_the_mux(tmp_path: Path) -> None:
    final = tmp_path / "final_bulk_v1_ff77"
    for pid, reason in (("p1", "incomplete_group"), ("p3", "incomplete_group"), ("p4", "selection")):
        (final / "_withdrawn" / pid).mkdir(parents=True)
        (final / "_withdrawn" / pid / "reason.json").write_text(f'{{"reason": "{reason}"}}')
    muxed = G.incomplete_mux_ids(final, {"p1", "p2", "p3", "p4"})
    assert muxed == {"p1", "p3"}
    rows = G.ledger_rows("bulk_v1_ff77", "A", "batch_02", [("p1", ["complete_group"]), ("p2", ["speaker"])],
                         muxed, "now", final, tmp_path / "sel")
    assert [(r["pair_id"], r["source"], r["criteria"]) for r in rows] == [
        ("p1", "incomplete_mux", ["complete_group"]), ("p2", "rule", ["speaker"]), ("p3", "incomplete_mux", [])]
    assert list(rows[0]) == ["pair_id", "build_id", "stratum", "batch", "source", "criteria", "withdrawn_at",
                             "final_dir", "withdrawn_dir", "selection_dir"]
    assert rows[1]["withdrawn_dir"] == str(final / "_withdrawn" / "p2")


def test_build_and_pair_list_flags() -> None:
    assert G.parse_builds(["bulk_v1_ff77:A", "bulk_v1_ff153:B"]) == [("bulk_v1_ff77", "A"), ("bulk_v1_ff153", "B")]
    assert G.parse_pair_lists(["A=a.txt", "B=b.txt"]) == {"A": Path("a.txt"), "B": Path("b.txt")}
    with pytest.raises(SystemExit):
        G.parse_builds(["bulk_v1_ff77"])
    with pytest.raises(SystemExit):
        G.parse_pair_lists(["C=c.txt"])


def test_dry_run_prints_the_steps_in_order_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lst = tmp_path / "batch_02.A.txt"
    lst.write_text("p1\np2\n")
    work = tmp_path / "work"
    monkeypatch.setattr(G.os, "umask", lambda mask: 0)  # keep the test process's umask
    monkeypatch.delenv("FACE_PYTHON", raising=False)
    monkeypatch.delenv("VOICE_PYTHON", raising=False)
    monkeypatch.delenv("SYNCNET_PYTHON", raising=False)
    monkeypatch.setattr("sys.argv", ["run_gates", "--batch", "batch_02", "--pairs", f"A={lst}",
                                     "--build", "bulk_v1_ff77:A", "--work-dir", str(work), "--dry-run"])
    assert G.main() == 0
    out = capsys.readouterr().out.splitlines()
    steps = [ln.split(" -m ", 1)[1].split()[0] for ln in out if " -m " in ln]
    gates = "mozaic.data_generation.gates."
    assert steps == [gates + "face_check", gates + "make_voice_check_list", gates + "make_lip_sync_list",
                     gates + "picture_check", gates + "voice_check", gates + "keep_or_drop"]
    assert any("<syncnet python>" in ln and "lip_sync_check.py" in ln for ln in out)  # the lip-sync check, as a script
    assert not work.exists()


def test_gate_stage_never_makes_new_pairs_or_renders() -> None:
    src = Path(G.__file__).read_text()
    for word in ("make_reserve", "draw_pairs", "--through", "refill", "sbatch", "squeue"):
        assert word not in src


# --------------------------------------------------------------------------- lip sync
def _sync(**conf: float | None) -> dict[str, dict[str, Any]]:
    base = {"RVRA": 8.0, "RVFA": 7.9, "FVRA": 7.6, "FVFA": 8.0, **conf}
    return {f"{PAIR}__{c}": {"class_name": c, "confidence": v} for c, v in base.items()}


def test_lip_sync_floor_on_every_scored_clip_and_margin_under_the_pairs_real_clip() -> None:
    g = _good()
    assert SP.evaluate_pair(**{**g, "sync": _sync(FVRA=3.0)})["failing"] == ["syncnet"]  # below the floor
    assert SP.evaluate_pair(**{**g, "sync": _sync(RVRA=3.0, RVFA=3.5, FVRA=3.5, FVFA=3.5)})["failing"] == ["syncnet"]
    r = SP.evaluate_pair(**{**g, "sync": _sync(RVRA=9.0, FVFA=5.0)})  # 4.0 under its own real clip > 3.8
    assert r["failing"] == ["syncnet"] and "under real > margin" in r["checks"][3]["note"]
    assert SP.evaluate_pair(**{**g, "sync": _sync(RVRA=9.0, FVFA=5.3)})["accepted"]  # 3.7 under: kept


def test_lip_sync_is_a_sample_an_unsampled_pair_is_not_measured_never_dropped() -> None:
    r = SP.evaluate_pair(**{**_good(), "sync": {}})
    assert r["accepted"] and r["not_measured"] == ["syncnet"]


def test_lip_sync_real_clip_untrackable_is_not_applicable_a_fake_one_fails() -> None:
    g = _good()
    na = SP.evaluate_pair(**{**g, "sync": _sync(RVRA=None)})
    assert na["accepted"] and "syncnet" not in na["not_measured"] and na["checks"][3]["ok"] is None
    assert SP.evaluate_pair(**{**g, "sync": _sync(FVRA=None)})["failing"] == ["syncnet"]  # no face track


def test_lip_sync_list_takes_the_batchs_first_pairs_every_shipped_clip(tmp_path: Path) -> None:
    from mozaic.data_generation.gates import make_lip_sync_list as LS

    rows = []
    for pair in ("p3", "p1", "p2"):
        for cls in ("RVRA", "FVRA"):
            sid = f"{pair}__{cls}"
            rows.append({"pair_id": pair, "class_name": cls, "sample_id": sid, "audio_variant": "a", "video_driver": "v"})
            clip = LS.shipped(cls, sid, tmp_path)
            if sid != "p2__FVRA":  # a clip that did not ship is left out
                clip.parent.mkdir(parents=True, exist_ok=True)
                clip.write_bytes(b"")
    out = LS.build_sync_pairs(rows, 2, tmp_path)
    assert [(r["pair_id"], r["class_name"]) for r in out] == [("p1", "FVRA"), ("p1", "RVRA"), ("p2", "RVRA")]
    assert out[0]["job_id"] == "p1__FVRA" and out[0]["clip"] == str(LS.shipped("FVRA", "p1__FVRA", tmp_path))
