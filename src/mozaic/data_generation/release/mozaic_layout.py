"""MOZAIC release rules: folders, file names, ids, row fields and the gate re-check.

Every naming and labelling rule of the release lives here, so the builder and the
tests read one definition. Nothing in this module touches the filesystem.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np


LENGTH_FRAMES = {"77f": 77, "153f": 153}
AUDIO_RATE = 16000
AUDIO_PACKET_SAMPLES = 1024  # the terminal mux frames the PCM sound in 1024-sample packets
FPS = 25
# Applied to every clip after the first build: converted voices fade in over their
# first ~50 ms and real ones do not, so every clip's sound now starts the same way.
START_MUTE_MS = 50
START_FADE_MS = 10

# bulk_v1 family -> (video generator, length)
FAMILIES: dict[str, tuple[str, str]] = {
    "ff77": ("facefusion", "77f"),
    "s2v77": ("s2v", "77f"),
    "wan77": ("animate", "77f"),
    "ff153": ("facefusion", "153f"),
    "s2v153": ("s2v", "153f"),
}
LABELS = ("RVRA", "RVFA", "FVRA", "FVFA")
GENERATOR_ORDER = ("facefusion", "s2v", "animate")
# Order in which a real-side clip (RVRA, RVFA) is taken when several families have it.
REAL_SOURCE_ORDER = {"77f": ("ff77", "s2v77", "wan77"), "153f": ("ff153", "s2v153")}

# The six clips a kept pair contributes; controls (self_drive, same_speaker) dropped.
REAL_SIDE = (("RVRA", "main"), ("RVRA", "vocoded"), ("RVFA", "main"))
FAKE_SIDE = (("FVRA", "main"), ("FVRA", "vocoded"), ("FVFA", "main"))
MANIFEST_ARM = {"main": "main", "vocoded_real": "vocoded", "vocoded_fake": "vocoded"}

# Text that reveals which encoder build wrote a file; none may stay in a clip. Specific
# on purpose: a bare 4-byte "Lavc" also occurs by chance inside compressed video.
VERSION_PATTERNS = (re.compile(rb"Lav[cf][0-9]+\.[0-9]+"), re.compile(rb"Lavc libx264"), re.compile(rb"x264 - core"))


def version_text(data: bytes) -> list[str]:
    """Return every encoder version string found in a file's bytes."""
    return sorted({m.group().decode("latin-1") for p in VERSION_PATTERNS for m in p.finditer(data)})


def video_generator(label: str, family: str) -> str:
    """``real`` for real faces, otherwise the family's video generator."""
    return "real" if label in ("RVRA", "RVFA") else FAMILIES[family][0]


def audio_generator(label: str, arm: str) -> str:
    """Return ``bigvgan`` if vocoded, ``seed_vc`` for fake voices, else ``none``."""
    if arm == "vocoded":
        return "bigvgan"
    return "seed_vc" if label in ("RVFA", "FVFA") else "none"


def generator_folder(label: str, family: str) -> str:
    """Return the 2nd folder level: real (RVRA), seed_vc (RVFA), else the generator."""
    return {"RVRA": "real", "RVFA": "seed_vc"}.get(label, FAMILIES[family][0])


def clip_id(voxceleb_path: str) -> str:
    """``id00058/6PiUqI-q-1w/00025`` -> ``id00058_6PiUqI-q-1w_00025``.

    Readable back by position: person = first 7 characters, segment = last 5.
    """
    person, video, segment = voxceleb_path.split("/")
    return f"{person}_{video}_{segment}"


def release_pair_id(source_clip_path: str, target_id: str) -> str:
    """Release pair id: ``<source_clip>__<target_id>`` (no length, no ``__x__``)."""
    return f"{clip_id(source_clip_path)}__{target_id}"


def release_folder(label: str, family: str, arm: str) -> str:
    """``videos/<label>/<generator>/<length>/<arm>``."""
    return f"videos/{label}/{generator_folder(label, family)}/{FAMILIES[family][1]}/{arm}"


def release_name(label: str, family: str, arm: str, source_clip: str, target_id: str) -> str:
    """``<label>__<generator>__<length>__<arm>__<source_clip>__<target_id>.mov``."""
    return f"{label}__{generator_folder(label, family)}__{FAMILIES[family][1]}__{arm}__{source_clip}__{target_id}.mov"


def expected_face_voice(label: str, source_id: str, target_id: str) -> tuple[str, str]:
    """Whose face and whose voice a clip of this label shows."""
    face = target_id if label in ("FVRA", "FVFA") else source_id
    voice = target_id if label in ("RVFA", "FVFA") else source_id
    return face, voice


def expected_audio_samples(length: str) -> int:
    """Audio samples in a clip of this length (16 kHz, 25 fps)."""
    return LENGTH_FRAMES[length] * AUDIO_RATE // FPS


CLIP_FIELDS = ("file_name", "label", "video_generator", "audio_generator", "length", "arm", "split", "face_id",
               "voice_id", "gender")  # fmt: skip
VOICE_SCORES = ("voice_target", "voice_source", "voice_vocoded")
FACE_SCORES = ("face_target", "face_source", "face_baseline", "flicker", "melt", "sharp")
# Every metadata.jsonl field, in order: (name, type, meaning). The types of
# metadata_table.parquet are made from this list.
METADATA_SCHEMA: tuple[tuple[str, str, str], ...] = (
    ("file_name", "string", "path of the video inside the dataset folder"),
    ("label", "string", "RVRA, RVFA, FVRA or FVFA"),
    ("video_generator", "string", "real, facefusion, s2v or animate"),
    ("audio_generator", "string", "none, seed_vc or bigvgan"),
    ("length", "string", "77f (3.08 s) or 153f (6.12 s)"),
    ("arm", "string", "main or vocoded"),
    ("split", "string", "train, val or test"),
    ("face_id", "string", "whose face is shown (VoxCeleb2 person id)"),
    ("voice_id", "string", "whose voice is heard (VoxCeleb2 person id)"),
    ("gender", "string", "m or f"),
    ("face_real", "bool", "the face is real"),
    ("voice_real", "bool", "the voice is real"),
    ("pair_id", "string", "the pair the video belongs to"),
    ("source_id", "string", "the source person: who is in the original clip"),
    ("target_id", "string", "the target person: who the fake turns them into"),
    ("source_clip", "string", "the VoxCeleb2 clip the video was made from"),
    ("target_face_clip", "string", "the target's clip the face picture came from"),
    ("target_voice_clip", "string", "the target's recording Seed-VC used as its voice reference"),
    ("kept_by", "list of strings", "the generators that kept the pair"),
    ("voice_target", "float", "converted voice vs the target's reference recording (ECAPA cosine)"),
    ("voice_source", "float", "converted voice vs the source's real voice (ECAPA cosine)"),
    ("voice_vocoded", "float", "vocoded real voice vs the real voice (ECAPA cosine)"),
    ("face_target", "float or empty", "fake face vs the target's face picture (ArcFace cosine)"),
    ("face_source", "float or empty", "fake face vs the source's real face (ArcFace cosine)"),
    ("face_baseline", "float or empty", "source's real face vs the target's face picture (ArcFace cosine)"),
    ("flicker", "float or empty", "brightness jumping between frames"),
    ("melt", "float or empty", "face wobble, relative to the real clip"),
    ("sharp", "float or empty", "face sharpness, relative to the real clip"),
    ("sha256", "string", "checksum of the video file"),
)


def clip_index_row(folder: str, row: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    """One metadata.jsonl row: the clip's own fields, its pair's, and its scores.

    ``file_name`` is the path from the dataset folder, as Hugging Face's folder loader
    reads it. Face and picture scores belong to the generator that made the face, so
    real-face clips (RVRA, RVFA) leave them empty; voice scores apply to every clip.
    """
    face = pair["scores"].get(row["video_generator"], {})
    return {
        "file_name": f"{folder}/{row['file_name']}",
        **{k: row[k] for k in CLIP_FIELDS if k != "file_name"},
        "face_real": row["label"] in ("RVRA", "RVFA"),
        "voice_real": row["label"] in ("RVRA", "FVRA"),
        "pair_id": row["pair_id"],
        "source_id": pair["source_id"],
        "target_id": pair["target_id"],
        "source_clip": pair["source_clip"],
        "target_face_clip": pair["target_clips"]["face"],
        "target_voice_clip": pair["target_clips"]["voice"],
        "kept_by": pair["kept_by"],
        **{k: pair["scores"]["voice"].get(k) for k in VOICE_SCORES},
        **{k: face.get(k) for k in FACE_SCORES},
        "sha256": row["sha256"],
    }


def audio_packet_sizes(length: str) -> list[int]:
    """Byte size of every sound packet in a clip of this length (16-bit mono PCM)."""
    full, rest = divmod(expected_audio_samples(length), AUDIO_PACKET_SAMPLES)
    return [2 * AUDIO_PACKET_SAMPLES] * full + ([2 * rest] if rest else [])


def treat_start(samples: np.ndarray) -> np.ndarray:
    """Silence the first 50 ms of 16 kHz samples, then fade the next 10 ms in linearly.

    The same for every clip, so the start of the sound carries no label. Fixed ramp and
    round-half-to-even, so the verify step recomputes it exactly from the source.
    """
    mute = AUDIO_RATE * START_MUTE_MS // 1000
    fade = AUDIO_RATE * START_FADE_MS // 1000
    out = np.array(samples, dtype=np.int16)
    out[:mute] = 0
    ramp = np.arange(1, fade + 1, dtype=np.float64) / (fade + 1)
    out[mute : mute + fade] = np.round(out[mute : mute + fade] * ramp).astype(np.int16)
    return out


def parse_framemd5(text: str) -> dict[str, Any]:
    """Fingerprint of a clip from ``ffmpeg -map 0:v:0 -map 0:a:0 -f framemd5``.

    The video and audio hashes cover every decoded frame or packet together with its
    timestamps, so two clips with equal fingerprints show the same picture and sound at
    the same times, whatever their container bytes.
    """
    video: list[str] = []
    audio: list[str] = []
    audio_bytes = 0
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split(",")]
        if parts[0] == "0":
            video.append(",".join(parts))
        else:
            audio.append(",".join(parts))
            audio_bytes += int(parts[4])
    return {
        "video": hashlib.md5("\n".join(video).encode()).hexdigest(),
        "audio": hashlib.md5("\n".join(audio).encode()).hexdigest(),
        "frames": len(video),
        "audio_samples": audio_bytes // 2,  # pcm_s16le mono
    }


def recheck_gate(measured: dict[str, Any], family: str, gates: dict[str, Any]) -> list[str]:
    """Re-apply the frozen pass rule to one pair's gate values; return the failed tests.

    Independent of the gate's own ``accepted`` flag. ``gates`` holds the keep/drop
    limits as settings.GATES does: ``A`` (77f) and ``B`` (153f) per length, ``picture``,
    and ``family_exceptions`` (generator -> the checks it skips).
    """
    length = FAMILIES[family][1]
    stratum = "A" if length == "77f" else "B"
    t = gates[stratum]
    generator = FAMILIES[family][0]
    waived = set(gates.get("family_exceptions", {}).get(generator, []))
    fails: list[str] = []
    face = measured.get("arcface")
    if not face:
        fails.append("face not measured")
    else:
        if face["target"] < max(t["arcface_target_abs_min"], t["arcface_target_floor"]):
            fails.append("face_target below minimum")
        if not face["target"] > face["leak"]:
            fails.append("face_target not above face_source")
        if "arcface_excess" not in waived and face["leak"] - face["floor"] > t["arcface_excess_cap"]:
            fails.append("face_source above cap")
    voice = measured.get("speaker")
    if not voice:
        fails.append("voice not measured")
    else:
        if voice["converted_min"] < t["sv_floor"]:
            fails.append("voice_target below floor")
        if voice.get("vocoded_vs_natural_min") is not None and voice["vocoded_vs_natural_min"] < t["sv_floor"]:
            fails.append("voice_vocoded below floor")
    picture = measured.get("picture")
    if not picture:
        fails.append("picture not measured")
    else:
        p = gates["picture"]
        for label in ("FVRA", "FVFA"):
            m = picture.get(label)
            if not m:
                fails.append(f"picture {label} not measured")
                continue
            if m["flicker"] > p["flicker_max"][stratum]:
                fails.append(f"{label} flicker")
            if "picture_melting" not in waived and m["melt_ratio"] > p["melt_ratio_max"]:
                fails.append(f"{label} melt")
            if m["sharp_ratio"] < p["sharp_ratio_min"]:
                fails.append(f"{label} sharp")
    return fails
