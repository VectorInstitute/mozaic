"""Tests for the MOZAIC release build: names, ids, labels, fingerprints, gate, manifests."""

from __future__ import annotations

import argparse
import copy
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mozaic.data_generation.release import build_mozaic as B  # noqa: N812
from mozaic.data_generation.release import mozaic_layout as L  # noqa: N812


def test_clip_id_keeps_every_part() -> None:
    """VoxCeleb2 path -> underscore id; a video id starting with _ gives a double _."""
    assert L.clip_id("id00058/6PiUqI-q-1w/00025") == "id00058_6PiUqI-q-1w_00025"
    cid = L.clip_id("id00417/_WTIf5_EgYQ/00303")
    assert cid == "id00417__WTIf5_EgYQ_00303"
    assert (cid[:7], cid[8:-6], cid[-5:]) == ("id00417", "_WTIf5_EgYQ", "00303")


def test_release_pair_id_has_no_length_or_x() -> None:
    """Same rule for both lengths; the old ``__f153`` and ``__x__`` parts are gone."""
    assert L.release_pair_id("id00885/40mpZRU47T4/00001", "id04597") == "id00885_40mpZRU47T4_00001__id04597"


@pytest.mark.parametrize(
    ("label", "family", "arm", "folder", "name"),
    [
        ("RVRA", "ff77", "main", "videos/RVRA/real/77f/main", "RVRA__real__77f__main__c__id1.mov"),
        ("RVRA", "s2v153", "vocoded", "videos/RVRA/real/153f/vocoded", "RVRA__real__153f__vocoded__c__id1.mov"),
        ("RVFA", "wan77", "main", "videos/RVFA/seed_vc/77f/main", "RVFA__seed_vc__77f__main__c__id1.mov"),
        ("FVRA", "s2v77", "vocoded", "videos/FVRA/s2v/77f/vocoded", "FVRA__s2v__77f__vocoded__c__id1.mov"),
        ("FVFA", "wan77", "main", "videos/FVFA/animate/77f/main", "FVFA__animate__77f__main__c__id1.mov"),
        ("FVFA", "ff153", "main", "videos/FVFA/facefusion/153f/main", "FVFA__facefusion__153f__main__c__id1.mov"),
    ],
)
def test_folder_and_name(label: str, family: str, arm: str, folder: str, name: str) -> None:
    """Folder = label/generator/length/arm; name = six parts joined by ``__``."""
    assert L.release_folder(label, family, arm) == folder
    assert L.release_name(label, family, arm, "c", "id1") == name


def test_face_voice_and_generators_per_label() -> None:
    """Whose face, whose voice, and which generators each label has."""
    assert L.expected_face_voice("RVRA", "s", "t") == ("s", "s")
    assert L.expected_face_voice("RVFA", "s", "t") == ("s", "t")
    assert L.expected_face_voice("FVRA", "s", "t") == ("t", "s")
    assert L.expected_face_voice("FVFA", "s", "t") == ("t", "t")
    assert [L.video_generator(lb, "s2v77") for lb in L.LABELS] == ["real", "real", "s2v", "s2v"]
    assert [L.audio_generator(lb, "main") for lb in L.LABELS] == ["none", "seed_vc", "none", "seed_vc"]
    assert L.audio_generator("RVRA", "vocoded") == L.audio_generator("FVRA", "vocoded") == "bigvgan"


def test_audio_samples_per_length() -> None:
    """16 kHz at 25 fps: 640 samples per frame."""
    assert L.expected_audio_samples("77f") == 49280
    assert L.expected_audio_samples("153f") == 97920


FRAMEMD5 = """#format: frame checksums
#stream#, dts,        pts, duration,     size, hash
0,          0,          0,        1,    37632, aaa
1,          0,          0,     1024,     2048, bbb
0,          1,          1,        1,    37632, ccc
1,       1024,       1024,     1024,     2048, ddd
"""


def test_parse_framemd5_counts_and_ignores_header() -> None:
    """Frames, audio samples, and a fingerprint that ignores header lines."""
    fp = L.parse_framemd5(FRAMEMD5)
    assert fp["frames"] == 2
    assert fp["audio_samples"] == 2048
    assert L.parse_framemd5("#another header\n" + FRAMEMD5) == fp
    assert L.parse_framemd5(FRAMEMD5.replace("ccc", "eee"))["video"] != fp["video"]
    assert L.parse_framemd5(FRAMEMD5.replace("ccc", "eee"))["audio"] == fp["audio"]


# The keep/drop limits, shaped as settings.GATES (both lengths the same here, except flicker).
STRATUM: dict[str, Any] = {
    "arcface_excess_cap": 0.0977, "arcface_target_floor": 0.1442, "arcface_target_abs_min": 0.3, "sv_floor": 0.2496,
}  # fmt: skip
GATES: dict[str, Any] = {
    "A": STRATUM, "B": STRATUM,
    "picture": {"flicker_max": {"A": 0.993, "B": 0.843}, "melt_ratio_max": 2.0, "sharp_ratio_min": 0.5},
    "family_exceptions": {"facefusion": ["arcface_excess"], "s2v": ["picture_melting"]},
}  # fmt: skip
GOOD: dict[str, Any] = {
    "arcface": {"target": 0.7, "leak": 0.05, "floor": 0.04},
    "speaker": {"converted_min": 0.5, "vocoded_vs_natural_min": 0.9},
    "picture": {c: {"flicker": 0.2, "melt_ratio": 1.0, "sharp_ratio": 0.9} for c in ("FVRA", "FVFA")},
}


def test_recheck_passes_a_good_pair() -> None:
    """A pair inside every limit passes for every family."""
    for family in L.FAMILIES:
        assert L.recheck_gate(GOOD, family, GATES) == []


def test_recheck_catches_each_failure() -> None:
    """Each limit, on its own, rejects."""
    cases = [
        (("arcface", "target"), 0.25, "face_target below minimum"),
        (("speaker", "converted_min"), 0.2, "voice_target below floor"),
        (("speaker", "vocoded_vs_natural_min"), 0.2, "voice_vocoded below floor"),
    ]
    for (crit, field), value, expected in cases:
        m = copy.deepcopy(GOOD)
        m[crit][field] = value
        assert expected in L.recheck_gate(m, "wan77", GATES)
    m = copy.deepcopy(GOOD)
    m["picture"]["FVFA"]["sharp_ratio"] = 0.4
    assert L.recheck_gate(m, "ff153", GATES) == ["FVFA sharp"]
    m = copy.deepcopy(GOOD)
    m["picture"]["FVRA"]["flicker"] = 0.9  # under the 77f limit, over the 153f one
    assert L.recheck_gate(m, "ff77", GATES) == []
    assert L.recheck_gate(m, "ff153", GATES) == ["FVRA flicker"]


def test_recheck_applies_the_two_waivers_only_where_frozen() -> None:
    """face_source excess is waived for FaceFusion only; melt for S2V only."""
    m = copy.deepcopy(GOOD)
    m["arcface"]["leak"] = 0.3  # 0.26 above the floor
    assert L.recheck_gate(m, "ff77", GATES) == []
    assert "face_source above cap" in L.recheck_gate(m, "s2v77", GATES)
    m = copy.deepcopy(GOOD)
    m["picture"]["FVRA"]["melt_ratio"] = 3.0
    assert L.recheck_gate(m, "s2v153", GATES) == []
    assert L.recheck_gate(m, "wan77", GATES) == ["FVRA melt"]


def test_version_text_finds_real_labels_only() -> None:
    """Real encoder labels are caught; the same 4 letters by chance in video are not."""
    assert L.version_text(b"\x00\x01encoder\x00Lavf58.76.100\x00") == ["Lavf58.76"]
    assert L.version_text(b"..x264 - core 164 r3191 4613ac3..Lavc libx264..") == ["Lavc libx264", "x264 - core"]
    assert L.version_text(b"\xe3\xeeLavc\x97 /\xe8\x8f" + b"V{Lavc\xc9F" + b"\xf1\xdaLavf\xc0\xef") == []


def test_treat_start_silences_then_fades() -> None:
    """First 50 ms exactly zero, next 10 ms a rising linear ramp, the rest untouched."""
    x = np.full(L.expected_audio_samples("77f"), 1000, dtype=np.int16)
    y = L.treat_start(x)
    assert y.dtype == np.int16
    assert (y[:800] == 0).all()
    assert (y[800], y[959]) == (round(1000 / 161), round(1000 * 160 / 161))
    assert (np.diff(y[800:960].astype(int)) >= 0).all()
    assert (y[960:] == 1000).all()
    assert (x == 1000).all()  # the input is not changed


def test_audio_packet_sizes_follow_the_terminal_framing() -> None:
    """1024-sample packets of 16-bit mono PCM, the last one short."""
    assert L.audio_packet_sizes("77f") == [2048] * 48 + [256]
    assert L.audio_packet_sizes("153f") == [2048] * 95 + [1280]


def test_moov_has_pasp_looks_only_in_the_mov_header() -> None:
    """A pasp box inside moov counts; the same four bytes in the media data do not."""

    def box(kind: bytes, payload: bytes = b"") -> bytes:
        return struct.pack(">I4s", 8 + len(payload), kind) + payload

    header_box = box(b"ftyp", b"qt  ") + box(b"mdat", b"media") + box(b"moov", box(b"mvhd") + box(b"pasp", bytes(8)))
    media_only = box(b"ftyp", b"qt  ") + box(b"mdat", b"pasp" * 4) + box(b"moov", box(b"mvhd"))
    assert B.moov_has_pasp(header_box)
    assert not B.moov_has_pasp(media_only)


def test_with_voice_replaces_only_the_voice_scores() -> None:
    """The voice part is replaced; face scores, key order and the input are kept."""
    pairs: list[dict[str, Any]] = [{"pair_id": "a", "length": "77f",
              "scores": {"voice": {"voice_target": 0.5, "voice_source": 0.1, "voice_vocoded": 0.9},
                         "facefusion": {"face_target": 0.8}}}]  # fmt: skip
    voice = {"a": {"pair_id": "a", "voice_target": 0.51, "voice_source": 0.11, "voice_vocoded": 0.89, "ok": True}}
    out = B.with_voice(pairs, voice)
    assert out[0]["scores"] == {
        "voice": {"voice_target": 0.51, "voice_source": 0.11, "voice_vocoded": 0.89},
        "facefusion": {"face_target": 0.8},
    }
    assert list(out[0]["scores"]) == ["voice", "facefusion"]
    assert pairs[0]["scores"]["voice"]["voice_target"] == 0.5
    with pytest.raises(SystemExit):
        B.with_voice(pairs, {})


def test_clip_index_row_takes_the_right_scores() -> None:
    """Fake faces get their generator's face scores, real faces none; voice for both."""
    pair = {"pair_id": "c__id2", "source_id": "id1", "target_id": "id2", "source_clip": "c",
            "target_clips": {"face": "f", "voice": "v"}, "kept_by": ["facefusion", "s2v"],
            "scores": {"voice": {"voice_target": 0.5, "voice_source": 0.1, "voice_vocoded": 0.9},
                       "facefusion": dict.fromkeys(L.FACE_SCORES, 0.8), "s2v": dict.fromkeys(L.FACE_SCORES, 0.4)}}  # fmt: skip
    base = {"length": "77f", "arm": "main", "split": "train", "gender": "m", "pair_id": "c__id2", "sha256": "x"}
    fake = L.clip_index_row("videos/FVFA/s2v/77f/main", {**base, "file_name": "a.mov", "label": "FVFA",
                            "video_generator": "s2v", "audio_generator": "seed_vc", "face_id": "id2", "voice_id": "id2"}, pair)  # fmt: skip
    real = L.clip_index_row("videos/RVFA/seed_vc/77f/main", {**base, "file_name": "b.mov", "label": "RVFA",
                            "video_generator": "real", "audio_generator": "seed_vc", "face_id": "id1", "voice_id": "id2"}, pair)  # fmt: skip
    assert fake["file_name"] == "videos/FVFA/s2v/77f/main/a.mov" and fake["face_target"] == 0.4
    assert (fake["face_real"], fake["voice_real"]) == (False, False)
    assert (real["face_real"], real["voice_real"]) == (True, False)
    assert all(real[k] is None for k in L.FACE_SCORES)
    assert fake["voice_target"] == real["voice_target"] == 0.5
    assert list(fake) == list(real) == [name for name, _, _ in L.METADATA_SCHEMA]


def _metadata_rows() -> list[dict[str, Any]]:
    """Return a fake-face row and a real-face row (no face scores), as manifests."""
    pair = {"pair_id": "c__id2", "source_id": "id1", "target_id": "id2", "source_clip": "c",
            "target_clips": {"face": "f", "voice": "v"}, "kept_by": ["s2v"],
            "scores": {"voice": dict.fromkeys(L.VOICE_SCORES, 0.5), "s2v": dict.fromkeys(L.FACE_SCORES, 0.25)}}  # fmt: skip
    base = {"length": "77f", "arm": "main", "split": "train", "gender": "f", "pair_id": "c__id2", "sha256": "ab"}
    return [
        L.clip_index_row("videos/FVFA/s2v/77f/main", {**base, "file_name": "a.mov", "label": "FVFA",
                         "video_generator": "s2v", "audio_generator": "seed_vc", "face_id": "id2", "voice_id": "id2"}, pair),
        L.clip_index_row("videos/RVRA/real/77f/main", {**base, "file_name": "b.mov", "label": "RVRA",
                         "video_generator": "real", "audio_generator": "none", "face_id": "id1", "voice_id": "id1"}, pair),
    ]  # fmt: skip


def test_parquet_table_holds_exactly_the_metadata_rows(tmp_path: Path) -> None:
    """Typed columns, empty face scores for real faces, and the same rows back."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    rows = _metadata_rows()
    B.write_parquet(tmp_path / "metadata_table.parquet", rows)
    table = pq.read_table(tmp_path / "metadata_table.parquet")  # type: ignore[no-untyped-call]
    assert table.schema.equals(B.metadata_arrow_schema())
    assert table.to_pylist() == rows
    assert table.schema.field("face_target").nullable and not table.schema.field("file_name").nullable
    with pytest.raises(Exception, match="null"):
        B.write_parquet(tmp_path / "bad.parquet", [{**rows[0], "label": None}])


def test_sha256sums_lists_every_file_but_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``sha256sum -c`` format, sorted, clips from the rows, top files (and the dataset card) hashed."""
    monkeypatch.setattr(B, "RELEASE", tmp_path)
    for name in (*B.DATA_FILES, *B.DOC_FILES):
        (tmp_path / name).write_text(name)
    text = B.sums_text({"videos/a.mov": "1" * 64})
    lines = text.splitlines()
    paths = [line.split("  ", 1)[1] for line in lines]
    assert paths == sorted(paths) and B.SUMS_FILE not in paths
    assert set(paths) == {"videos/a.mov", *B.DATA_FILES, *B.DOC_FILES} - {B.SUMS_FILE}
    assert f"{B.sha256_file(tmp_path / 'README.md')}  README.md" in lines
    assert all(len(line.split("  ", 1)[0]) == 64 for line in lines)


def test_the_dataset_card_is_optional_the_data_files_are_not(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No README.md/DATASHEET.md: SHA256SUMS leaves them out and verify neither wants nor flags them."""
    monkeypatch.setattr(B, "RELEASE", tmp_path)
    for name in B.DATA_FILES:
        (tmp_path / name).write_text(name)
    paths = {line.split("  ", 1)[1] for line in B.sums_text({"videos/a.mov": "1" * 64}).splitlines()}
    assert paths == {"videos/a.mov", *B.DATA_FILES} - {B.SUMS_FILE}
    planned = {"videos/a.mov"}
    data = planned | set(B.DATA_FILES)
    assert B.expected_files(planned, data) == data  # absent: not missing
    assert B.expected_files(planned, data | {"README.md"}) == data | {"README.md"}  # present: not unexpected
    assert B.expected_files(planned, set()) == data  # the data files are always required
    (tmp_path / "pairs.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        B.sums_text({})


def test_old_manifests_are_moved_not_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-folder metadata.jsonl, clips.jsonl and metadata.json go to the history."""
    release, history = tmp_path / "mozaic", tmp_path / "history"
    monkeypatch.setattr(B, "RELEASE", release)
    monkeypatch.setattr(B, "OLD_MANIFESTS_HISTORY", history)
    for rel in ("videos/RVRA/real/77f/main/metadata.jsonl", "clips.jsonl", "metadata.json"):
        (release / rel).parent.mkdir(parents=True, exist_ok=True)
        (release / rel).write_text(rel)
    (release / "videos/RVRA/real/77f/main/a.mov").write_text("clip")
    assert B.retire_old_manifests({"videos/RVRA/real/77f/main"}) == 3
    assert (history / "clips.jsonl").read_text() == "clips.jsonl"
    assert sorted(p.name for p in release.rglob("*") if p.is_file()) == ["a.mov"]
    assert B.retire_old_manifests({"videos/RVRA/real/77f/main"}) == 0


def test_decision_folders_skip_the_set_aside_ones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every batch's decision folder is read, in name order; a folder renamed with a suffix is not."""
    for name in ("bulk_v1_ff77_selection", "bulk_v1_ff77_selection_b02", "bulk_v1_ff77_selection_b01",
                 "bulk_v1_ff77_selection_b02.failed_20260930", "bulk_v1_ff153_selection_b01"):
        (tmp_path / name).mkdir()
    (tmp_path / "bulk_v1_ff77_selection_b03.txt").write_text("")
    monkeypatch.setattr(B, "REPORTS", tmp_path)
    assert [p.name for p in B.decision_folders("ff77")] == ["bulk_v1_ff77_selection_b01", "bulk_v1_ff77_selection_b02"]


def _decision(folder: Path, rows: list[dict[str, Any]]) -> None:
    folder.mkdir(parents=True)
    (folder / "selection_summary.json").write_text(json.dumps({"inputs": {"gates_dir": f"/x/{folder.name}_gates"}}))
    (folder / "selection_pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_kept_pairs_are_accepted_and_complete_wherever_they_sit(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """Kept: shipped + accepted, or set aside whole + accepted (own decision, or the first batch's
    joint one when only the missing clips failed). A rejected pair, or an accepted one with clips
    missing, is not kept; the second is an error."""
    reports, final = tmp_path / "_reports", tmp_path / "final_bulk_v1_ff77"
    monkeypatch.setattr(B, "REPORTS", reports)
    monkeypatch.setattr(B, "NEW_GEN", tmp_path)
    monkeypatch.setattr(B.L, "recheck_gate", lambda measured, family, limits: [])

    def row(pair: str, accepted: bool, failing: list[str]) -> dict[str, Any]:
        return {"pair_id": pair, "accepted": accepted, "failing": failing, "checks": []}

    _decision(reports / "bulk_v1_ff77_selection", [row("D", True, [])])
    _decision(reports / "bulk_v1_ff77_selection_b01", [row("A", True, []), row("B", True, []), row("C", True, []),
                                                       row("D", False, ["complete_group"]),
                                                       row("E", False, ["arcface_target"])])  # fmt: skip
    manifest = {(p, label, "main"): {"sample_id": f"{p}__{label}"} for p in "ABCDE" for label in ("RVRA", "FVRA")}
    (final / "RVRA").mkdir(parents=True)
    (final / "RVRA" / "A__RVRA.mov").write_bytes(b"")
    for p, labels in (("B", ("RVRA", "FVRA")), ("C", ("RVRA",)), ("D", ("RVRA", "FVRA")), ("E", ("RVRA", "FVRA"))):
        for label in labels:
            (final / "_withdrawn" / p).mkdir(parents=True, exist_ok=True)
            (final / "_withdrawn" / p / f"{p}__{label}.mov").write_bytes(b"")
    errors: list[str] = []
    kept = B.kept_pairs("ff77", manifest, {}, errors)
    assert {p: k["origin"] for p, k in kept.items()} == {"A": "final", "B": "withdrawn", "D": "withdrawn"}
    assert kept["D"]["decision"]["accepted"] is True  # the joint decision kept it
    assert len(errors) == 1 and "C" in errors[0]


def test_code_commit_is_read_never_made_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The commit comes from CODE_COMMIT; without that file it is None."""
    monkeypatch.setattr(B, "CODE_COMMIT", tmp_path / "CODE_COMMIT")
    assert B.code_commit() is None
    (tmp_path / "CODE_COMMIT").write_text("0123456789abcdef0123456789abcdef01234567\n")
    assert B.code_commit() == "0123456789abcdef0123456789abcdef01234567"


# summary.json's keys, in order: the same as the Hub copy (no limits block, no code block).
SUMMARY_KEYS = ["name", "version", "schema_version", "release_date", "cite", "licence", "access", "removal_requests",
                "labels", "arms", "source", "video", "audio", "applied_to_every_clip", "generators", "counts",
                "people_per_split"]  # fmt: skip


def _made_up_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make a finished copy step for one made-up 77f pair kept by FaceFusion; return mozaic/."""
    release, build = tmp_path / "mozaic", tmp_path / "_mozaic_build"
    for name, value in (("RELEASE", release), ("BUILD", build), ("DROP", build / "drop" / "pairs.json"),
                        ("OLD_MANIFESTS_HISTORY", build / "history")):
        monkeypatch.setattr(B, name, value)  # fmt: skip
    monkeypatch.setattr(B.S, "new_gen_manifest_path", lambda build_id: tmp_path / f"{build_id}.jsonl")
    for family in L.FAMILIES:
        version = f"{L.FAMILIES[family][0]}-v1+seedvc-v1"
        (tmp_path / f"bulk_v1_{family}.jsonl").write_text(json.dumps({"generator_version": version}) + "\n")
    pid, source, target = "id1_v_00001__id2", "id1_v_00001", "id2"
    entries, journal = [], []
    for label, arm in L.REAL_SIDE + L.FAKE_SIDE:
        name = L.release_name(label, "ff77", arm, source, target)
        face, voice = L.expected_face_voice(label, "id1", "id2")
        rel = f"{L.release_folder(label, 'ff77', arm)}/{name}"
        entries.append({"release_path": rel, "source_path": f"final_bulk_v1_ff77/{label}/{name}", "origin": "final",
                        "family": "ff77", "legacy_sample_id": name[:-4],
                        "source_sha256": "0" * 64, "fingerprint": {"video": "v", "audio": "a", "frames": 77,
                                                                   "audio_samples": 49280},
                        "row": {"file_name": name, "label": label, "video_generator": L.video_generator(label, "ff77"),
                                "audio_generator": L.audio_generator(label, arm), "length": "77f", "arm": arm,
                                "split": "test", "face_id": face, "voice_id": voice, "gender": "f", "pair_id": pid}})  # fmt: skip
        (release / rel).parent.mkdir(parents=True, exist_ok=True)
        (release / rel).write_bytes(label.encode())
        journal.append({"release_path": rel, "sha256": B.sha256_file(release / rel), "ok": True})
    pair = {"pair_id": pid, "length": "77f", "split": "test", "gender": "f", "source_id": "id1", "target_id": "id2",
            "source_clip": source, "target_clips": {"face": "id2_w_00002", "voice": "id2_w_00003"},
            "kept_by": ["facefusion"], "scores": {"voice": dict.fromkeys(L.VOICE_SCORES, 0.5),
                                                  "facefusion": dict.fromkeys(L.FACE_SCORES, 0.25)}}  # fmt: skip
    B.write_jsonl(build / "plan" / "files.jsonl", entries)
    B.write_jsonl(build / "plan" / "pairs.jsonl", [pair])
    B.write_json(build / "plan" / "splits.json", {"id1": "test", "id2": "test"})
    B.write_json(build / "plan" / "encode_settings.json", {"container": "mov", "video_codec": "libx264", "crf": 18,
                 "preset": "medium", "pixel_format": "yuv420p", "resolution": 224, "fps": 25,
                 "audio_codec": "pcm_s16le", "audio_sample_rate": 16000, "luma": {}, "chroma": {}, "loudness": {},
                 "silence_trim": False})  # fmt: skip
    B.write_json(build / "plan" / "run.json", {"status": "READY", "finished": "x"})
    B.write_jsonl(build / "copy" / "journal.jsonl", journal)
    for folder in {e["release_path"].rsplit("/", 1)[0] for e in entries}:
        B.write_json(build / "copy" / "DONE" / f"{folder.replace('/', '__')}.json", {"folder": folder})
    return release


def test_manifests_writes_the_data_files_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No README.md or DATASHEET.md is written; summary.json has the Hub copy's keys, in its order."""
    release = _made_up_build(tmp_path, monkeypatch)
    assert B.step_manifests(argparse.Namespace()) == 0
    assert sorted(p.name for p in release.iterdir() if p.is_file()) == sorted(B.DATA_FILES)
    summary = json.loads((release / "summary.json").read_text())
    assert list(summary) == SUMMARY_KEYS
    assert summary["counts"]["clips"] == 6 and summary["counts"]["pairs"] == {"77f": 1, "153f": 0}
    sums = dict(line.split("  ", 1)[::-1] for line in (release / B.SUMS_FILE).read_text().splitlines())
    assert set(sums) == {r["file_name"] for r in B.read_jsonl(release / "metadata.jsonl")} | (set(B.DATA_FILES) - {B.SUMS_FILE})


def test_manifests_lists_a_dataset_card_put_in_by_hand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A README.md already in mozaic/ is kept as it is and listed in SHA256SUMS."""
    release = _made_up_build(tmp_path, monkeypatch)
    (release / "README.md").write_text("# MOZAIC\n")
    assert B.step_manifests(argparse.Namespace()) == 0
    assert (release / "README.md").read_text() == "# MOZAIC\n" and not (release / "DATASHEET.md").exists()
    assert f"{B.sha256_file(release / 'README.md')}  README.md" in (release / B.SUMS_FILE).read_text().splitlines()
