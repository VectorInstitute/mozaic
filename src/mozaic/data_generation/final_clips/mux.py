"""The one mux every final clip goes through, real and fake alike.

A video stream and an audio stream go in, one .mov comes out at one fixed spec. Nothing here
branches on the class, so the mux itself cannot become a real/fake cue.

- Length: the video must already carry exactly target_frames frames (checked, never capped:
  -frames:v on a muxed output cuts the audio too). The audio is conformed to target_frames x 640
  samples, and its last 160 samples are silence on every clip.
- Picture: resized through 624 px to 224 px, then a gamma on Y (mean luma 72) and a gain on U/V
  (chroma 17), both solved from the clip's own pixels and refined twice against the encoded output.
- Loudness: loudnorm is used only as a meter; one constant gain to -23 LUFS, capped so the true
  peak stays under -2 dBTP. No silence trim, no compression.
- Checks: every output is measured again. A clip that misses the picture target is dropped
  (TerminalDropError: recorded, its pair is withdrawn next); any other broken check is a failure.
  Neither leaves a clip on disk.

    python -m mozaic.data_generation.final_clips.mux --video v.mp4 --audio a.wav --out out.mov
    python -m mozaic.data_generation.final_clips.mux --jobs jobs.jsonl
    python -m mozaic.data_generation.final_clips.mux --null-test --video v.mp4 --audio a.wav
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mozaic.data_generation import settings as S

TIMEOUT_S = 600
ONSET_SEARCH_MS = 50  # cross-correlation window; a linear gain must land at exactly 0
# How far input audio may be from the contract length before it is refused rather than conformed:
# ~13x what Seed-VC drifts (1.5 ms), far below one video frame (40 ms).
AUDIO_LENGTH_TOLERANCE_S = 0.020
# Share of the asked-for chroma gain that must take before a missed target counts as the solver's error.
CHROMA_EFFICIENCY_MIN = 0.90
RENDER_DIR_NAME = "renders"  # generated videos live in new_generated/renders/


class TerminalError(RuntimeError):
    """A clip failed the mux's contract. No output is written."""


class TerminalDropError(TerminalError):
    """A clip the mux cannot bring onto the target: dropped by design, not a failure of the step.

    It is recorded beside the job list, and its half-shipped pair is withdrawn by the next step.
    """


# --------------------------------------------------------------------------- probes (each isolates its stream)
def _run(cmd: list[str], timeout: int = TIMEOUT_S) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)


def decoded_video_frames(path: Path) -> int:
    """Frames that decode from the video stream only.

    -map 0:v:0 -an because framemd5 on a muxed file lists every stream. Decoded, not counted
    packets: a broken timestamp chain can keep every packet and still lose frames.
    """
    out = _run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-f", "framemd5", "-"])
    return sum(1 for line in out.stdout.decode(errors="replace").splitlines() if line and not line.startswith("#"))


def video_frame_rate(path: Path) -> str | None:
    """The video stream's r_frame_rate as its raw 'num/den' string."""
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path),
    ])  # fmt: skip
    return out.stdout.decode().strip() or None


def video_dimensions(path: Path) -> tuple[int, int] | None:
    """(width, height) of the video stream."""
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
    ])  # fmt: skip
    try:
        w, h = out.stdout.decode().strip().split(",")
        return int(w), int(h)
    except ValueError:
        return None


def container_audio_samples(path: Path) -> int | None:
    """The audio stream's declared length in samples (duration_ts, which respects the edit list)."""
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=duration_ts", "-of", "csv=p=0", str(path),
    ])  # fmt: skip
    try:
        return int(out.stdout.decode().strip())
    except ValueError:
        return None


def decoded_audio_pcm(path: Path, sample_rate: int = S.AUDIO_SAMPLE_RATE_HZ) -> np.ndarray:
    """Decode the audio stream to mono float32 at sample_rate."""
    out = _run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-vn", "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-",
    ])  # fmt: skip
    return np.frombuffer(out.stdout, dtype=np.float32)


def decoded_audio_samples(path: Path) -> int:
    """How many samples a plain PCM decode gives (AAC would round up to whole frames)."""
    return int(decoded_audio_pcm(path).size)


def measure_loudness(path: Path) -> dict[str, str]:
    """EBU R128 analysis: loudnorm's own JSON block. Only the measurement is used, never its gain."""
    af = f"loudnorm=I={S.TERMINAL_LOUDNESS_I}:TP={S.TERMINAL_TRUE_PEAK_CEILING}:print_format=json"
    out = _run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af", af, "-f", "null", "-"])
    text = out.stderr.decode(errors="replace")
    try:
        block = text[text.rindex("{") : text.rindex("}") + 1]
        return json.loads(block)
    except (ValueError, json.JSONDecodeError) as exc:
        msg = f"loudnorm analysis produced no JSON for {path.name}: {text[-300:]}"
        raise TerminalError(msg) from exc


# --------------------------------------------------------------------------- loudness and audio length
def compute_gain_db(measured: dict[str, str]) -> dict[str, Any]:
    """The one loudness rule: the gain that reaches the target, capped by the true-peak ceiling.

    loudnorm's own linear mode falls back to a dynamic, content-chosen mode on some clips; a capped
    constant gain is the same operation for every clip. A clip the cap binds lands short of the
    target, and the shortfall is recorded.
    """
    input_i = float(measured["input_i"])
    input_tp = float(measured["input_tp"])
    requested = S.TERMINAL_LOUDNESS_I - input_i
    headroom = S.TERMINAL_TRUE_PEAK_CEILING - input_tp
    applied = min(requested, headroom, S.TERMINAL_MAX_GAIN_DB)
    return {
        "input_i": input_i,
        "input_tp": input_tp,
        "input_lra": float(measured["input_lra"]),
        "requested_gain_db": round(requested, 4),
        "peak_headroom_db": round(headroom, 4),
        "applied_gain_db": round(applied, 4),
        "peak_capped": applied < requested - 1e-9,
        "loudness_shortfall_db": round(requested - applied, 4),
    }


def audio_filter(gain: dict[str, Any], target_frames: int) -> str:
    """The whole audio chain: gain, resample, conform to the exact length. The same for every class.

    The resampler is swresample's default (the ffmpeg 4.4 build used has no soxr).
    """
    end_sample = target_frames * S.SAMPLES_PER_FRAME
    trimmed = end_sample - S.TERMINAL_AUDIO_TAIL_PAD_SAMPLES
    return (
        f"volume={gain['applied_gain_db']:.4f}dB,"
        f"aresample={S.AUDIO_SAMPLE_RATE_HZ},"
        # Pad, cut short of the length, pad back: the trailing silence is then the same on every clip,
        # not the amount a voice happened to fall short by.
        f"apad,"
        f"atrim=end_sample={trimmed},"
        f"apad=whole_len={end_sample},"
        f"asetpts=PTS-STARTPTS,"
        # Last, so every clip gets one packet layout; p=0 because padding would add samples.
        f"asetnsamples=n={S.TERMINAL_AUDIO_FRAME_SAMPLES}:p=0"
    )


# --------------------------------------------------------------------------- brightness and colour
def scale_chain() -> str:
    """The resampling every clip shares (through 624 px, then 224 px), up to the pixel format.

    A strict prefix of the encode's filter graph, so the corrections are solved on exactly the
    pixels they are applied to.
    """
    return (
        f"scale={S.TERMINAL_RESAMPLE_VIA}:{S.TERMINAL_RESAMPLE_VIA}:flags={S.TERMINAL_SCALE_FLAGS},"
        f"scale={S.TARGET_RESOLUTION}:{S.TARGET_RESOLUTION}:flags={S.TERMINAL_SCALE_FLAGS},"
        f"format={S.TERMINAL_PIXEL_FORMAT}"
    )


def video_filter(gamma: float | None, saturation: float | None = None) -> str:
    """The whole video chain: resampling, then the clip's luma gamma and chroma gain."""
    steps = [scale_chain()]
    if gamma is not None:
        # The comma is escaped so ffmpeg reads it as pow()'s separator, not the end of the filter.
        steps.append(rf"lutyuv=y=255*pow(val/255\,{gamma:.6f})")
    if saturation is not None:
        steps.append(f"eq=saturation={saturation:.6f}")  # chroma only: luma stays bit-identical
    return ",".join(steps)


def measure_video(path: Path, filters: str = "") -> dict[str, Any]:
    """The luma histogram and the mean chroma magnitude, in one decode of the raw planes.

    Raw planes, no colour conversion, so the meter is in the units lutyuv and eq write. Chroma
    magnitude is the mean of sqrt((U-128)^2 + (V-128)^2), the distance from grey.
    """
    chain = ",".join(x for x in (filters, f"format={S.TERMINAL_PIXEL_FORMAT}") if x)
    out = _run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-an", "-vf", chain, "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", S.TERMINAL_PIXEL_FORMAT, "-",
    ])  # fmt: skip
    if out.returncode != 0 or not out.stdout:
        msg = f"could not read pixels from {path.name}: {out.stderr.decode(errors='replace')[-300:]}"
        raise TerminalError(msg)
    n = S.TARGET_RESOLUTION
    luma_size, chroma_size = n * n, n * n // 4
    frame_size = luma_size + 2 * chroma_size
    data = np.frombuffer(out.stdout, dtype=np.uint8)
    frames = data[: len(data) // frame_size * frame_size].reshape(-1, frame_size)
    if not len(frames):
        msg = f"{path.name} decoded no complete frames"
        raise TerminalError(msg)
    u = frames[:, luma_size : luma_size + chroma_size].astype(np.float32) - 128.0
    v = frames[:, luma_size + chroma_size :].astype(np.float32) - 128.0
    return {
        "luma_hist": np.bincount(frames[:, :luma_size].reshape(-1), minlength=256).astype(np.float64),
        "chroma": float(np.mean(np.sqrt(u * u + v * v))),
    }


def _mean_after_gamma(hist: np.ndarray, gamma: float) -> float:
    """Mean luma of this histogram after the power law, rounded to 8 bits as lutyuv writes it."""
    levels = np.arange(256, dtype=np.float64) / 255.0
    mapped = np.clip(np.round(255.0 * levels**gamma), 0.0, 255.0)
    return float((hist * mapped).sum() / hist.sum())


def solve_luma_gamma(hist: np.ndarray, target: float = S.TERMINAL_LUMA_TARGET) -> dict[str, Any]:
    """The exponent that puts this clip's mean luma on the target, by bisection (60 halvings).

    A gamma maps 0-255 onto itself, so it never clips. At the edge of the range the clip is not
    refused: it lands short of the target and the shortfall is recorded.
    """
    measured = _mean_after_gamma(hist, 1.0)
    low, high = S.TERMINAL_LUMA_GAMMA_MIN, S.TERMINAL_LUMA_GAMMA_MAX
    if _mean_after_gamma(hist, low) < target:
        gamma, capped = low, True
    elif _mean_after_gamma(hist, high) > target:
        gamma, capped = high, True
    else:
        capped = False
        for _ in range(60):
            gamma = 0.5 * (low + high)
            if _mean_after_gamma(hist, gamma) > target:
                low = gamma
            else:
                high = gamma
        gamma = 0.5 * (low + high)
    predicted = _mean_after_gamma(hist, gamma)
    return {
        "measured_luma": round(measured, 4),
        "target_luma": target,
        "applied_gamma": round(float(gamma), 6),
        "predicted_luma": round(predicted, 4),
        "gamma_capped": capped,
        "luma_shortfall": round(target - predicted, 4),
    }


def solve_chroma_gain(measured: float, target: float = S.TERMINAL_CHROMA_TARGET) -> dict[str, Any]:
    """The chroma gain that puts this clip on the target: target / measured (eq is linear), capped."""
    if measured <= 0:
        return {"measured_chroma": 0.0, "target_chroma": target, "applied_gain": 1.0,
                "gain_capped": True, "chroma_shortfall": target}  # fmt: skip
    raw = target / measured
    gain = min(max(raw, S.TERMINAL_CHROMA_GAIN_MIN), S.TERMINAL_CHROMA_GAIN_MAX)
    return {
        "measured_chroma": round(measured, 4),
        "target_chroma": target,
        "applied_gain": round(float(gain), 6),
        "gain_capped": gain != raw,
        "chroma_shortfall": round(target - measured * gain, 4),
    }


def solve_corrections(video_in: Path) -> dict[str, Any]:
    """Both picture corrections, from one measurement of the resampled input."""
    measured = measure_video(video_in, scale_chain())
    return {
        "luma": solve_luma_gamma(measured["luma_hist"]),
        "chroma": solve_chroma_gain(measured["chroma"]),
        "input_hist": measured["luma_hist"],
    }


def refine_corrections(corrections: dict[str, Any], achieved: dict[str, float]) -> dict[str, Any]:
    """Correct the corrections against what the lossy encoder really produced.

    Luma is solved again against a target moved by the miss; chroma, being linear, is rescaled.
    """
    luma, chroma = dict(corrections["luma"]), dict(corrections["chroma"])
    luma_target = luma["target_luma"] + (luma["target_luma"] - achieved["luma"])
    refined_luma = solve_luma_gamma(corrections["input_hist"], luma_target)
    luma["applied_gamma"] = refined_luma["applied_gamma"]
    luma["refined_against"] = round(achieved["luma"], 4)
    if achieved["chroma"] > 0 and not chroma["gain_capped"]:
        scaled = chroma["applied_gain"] * chroma["target_chroma"] / achieved["chroma"]
        chroma["applied_gain"] = round(
            float(min(max(scaled, S.TERMINAL_CHROMA_GAIN_MIN), S.TERMINAL_CHROMA_GAIN_MAX)), 6
        )
        chroma["refined_against"] = round(achieved["chroma"], 4)
    return {"luma": luma, "chroma": chroma, "input_hist": corrections["input_hist"]}


def achieved_picture(path: Path) -> dict[str, float]:
    """Mean luma and chroma of a finished clip."""
    measured = measure_video(path)
    return {"luma": _mean_after_gamma(measured["luma_hist"], 1.0), "chroma": measured["chroma"]}


def onset_lag_ms(reference: np.ndarray, actual: np.ndarray, sample_rate: int = S.AUDIO_SAMPLE_RATE_HZ) -> float:
    """Lag of `actual` behind `reference` in ms, by peak cross-correlation; a gain must give 0.0."""
    lag = int(round(ONSET_SEARCH_MS * sample_rate / 1000))
    n = min(reference.size, actual.size)
    if n <= 2 * lag:
        return 0.0
    ref = reference[:n].astype(np.float64)
    act = actual[:n].astype(np.float64)
    ref -= ref.mean()
    act -= act.mean()
    if ref.std() == 0 or act.std() == 0:
        return 0.0
    correlation = np.correlate(act, ref[lag:-lag], mode="valid")
    return float((int(np.argmax(correlation)) - lag) * 1000.0 / sample_rate)


def file_md5(path: Path) -> str:
    """Hash a whole file, for the null test (our own outputs, not a security check)."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_md5(path: Path, video: bool) -> str:
    """Hash one stream's decoded content, so metadata cannot hide a difference."""
    if video:
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-f", "framemd5", "-"]
        out = _run(cmd).stdout
    else:
        out = decoded_audio_pcm(path).tobytes()
    return hashlib.md5(out).hexdigest()


# --------------------------------------------------------------------------- the chain
@dataclass
class TerminalResult:
    """What one clip's trip through the mux produced and checked."""

    ok: bool
    out_path: Path
    target_frames: int
    checks: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    dropped: bool = False  # True for a TerminalDropError: recorded, not counted as a failure


def audio_sample_rate(path: Path) -> int | None:
    """The audio stream's native sample rate."""
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(path),
    ])  # fmt: skip
    try:
        return int(out.stdout.decode().strip())
    except ValueError:
        return None


def _assert_inputs(video_in: Path, audio_in: Path, target_frames: int) -> dict[str, Any]:
    """Check what goes in: the video length exactly, the audio length within AUDIO_LENGTH_TOLERANCE_S.

    A generator cannot be asked for an exact sample count (Seed-VC comes out ~1.5 ms short), so the
    audio gets a tolerance and is conformed to the exact length in the encode, for every class.
    """
    in_frames = decoded_video_frames(video_in)
    if in_frames != target_frames:
        msg = f"input video decodes {in_frames} frames, expected {target_frames}"
        # A short render is that generator's fault on one clip: the pair is dropped.
        # A short crop is our cropper breaking its contract: a failure that stops the step.
        if video_in.parent.name == RENDER_DIR_NAME:
            raise TerminalDropError(msg + " (the generator dropped frames)")
        raise TerminalError(msg)

    in_rate = audio_sample_rate(audio_in)
    in_samples = container_audio_samples(audio_in)
    if in_rate is None or in_samples is None:
        msg = f"cannot read the audio stream of {audio_in.name}"
        raise TerminalError(msg)
    target_s = target_frames / S.TARGET_FPS
    in_s = in_samples / in_rate
    drift_ms = (in_s - target_s) * 1000.0
    if abs(in_s - target_s) > AUDIO_LENGTH_TOLERANCE_S:
        msg = (
            f"input audio is {in_s:.6f}s, expected {target_s:.6f}s "
            f"({drift_ms:+.1f} ms, over the {AUDIO_LENGTH_TOLERANCE_S * 1000:.0f} ms tolerance)"
        )
        raise TerminalError(msg)
    return {
        "input_video_frames": in_frames,
        "input_audio_samples": in_samples,
        "input_audio_rate": in_rate,
        "input_audio_drift_ms": round(drift_ms, 4),
    }


def verify_audio_conform(audio_in: Path, gain: dict[str, Any], target_frames: int) -> dict[str, Any]:
    """Check on this clip that the audio chain lands on exactly the contract length.

    Run into a plain wav, so no codec framing can round a difference away. Also records how many
    samples the conform added (0 for a real crop, a few for a generated voice).
    """
    expected = target_frames * S.SAMPLES_PER_FRAME
    native = container_audio_samples(audio_in)
    rate = audio_sample_rate(audio_in) or S.AUDIO_SAMPLE_RATE_HZ
    with tempfile.TemporaryDirectory() as tmpdir:
        probe = Path(tmpdir) / "conformed.wav"
        rc = _run([
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(audio_in),
            "-af", audio_filter(gain, target_frames),
            "-ac", "1", "-f", "wav", str(probe),
        ])  # fmt: skip
        if rc.returncode != 0:
            msg = f"audio conform probe failed: {rc.stderr.decode(errors='replace')[-300:]}"
            raise TerminalError(msg)
        after = container_audio_samples(probe)
    if after != expected:
        msg = f"audio conform produced {after} samples, expected exactly {expected}"
        raise TerminalError(msg)
    resampled = round((native or 0) * S.AUDIO_SAMPLE_RATE_HZ / rate)
    return {
        "conform_samples_out": after,
        "conform_delta_samples": expected - resampled,
        "conform_delta_ms": round((expected - resampled) * 1000.0 / S.AUDIO_SAMPLE_RATE_HZ, 4),
    }


def _encode(
    video_in: Path,
    audio_in: Path,
    tmp_out: Path,
    gain: dict[str, Any],
    target_frames: int,
    gamma: float | None,
    saturation: float | None = None,
) -> None:
    """Run the single mux: one command for every class, with the clip's own gain, gamma and saturation."""
    audio_flags = ["-c:a", S.TERMINAL_AUDIO_CODEC]
    if S.TERMINAL_AUDIO_BITRATE:  # PCM takes no bitrate
        audio_flags += ["-b:a", S.TERMINAL_AUDIO_BITRATE]
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-fflags", "+bitexact",
        "-i", str(video_in),
        "-i", str(audio_in),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", video_filter(gamma, saturation),
        "-c:v", S.TERMINAL_VIDEO_CODEC, "-crf", str(S.TERMINAL_CRF), "-preset", S.TERMINAL_PRESET,
        "-af", audio_filter(gain, target_frames),
        "-ar", str(S.AUDIO_SAMPLE_RATE_HZ), "-ac", "1",
        *audio_flags,
        "-flags:v", "+bitexact", "-flags:a", "+bitexact",
        "-movflags", "+faststart",
        "-f", S.TERMINAL_CONTAINER_FORMAT, str(tmp_out),
    ]  # fmt: skip
    result = _run(cmd)
    if result.returncode != 0:
        msg = f"terminal encode failed rc={result.returncode}: {result.stderr.decode(errors='replace')[-400:]}"
        raise TerminalError(msg)


def _assert_picture(checks: dict[str, Any], corrections: dict[str, Any]) -> None:
    """Drop a clip whose brightness or colour missed the target although its correction could reach it.

    A correction at the edge of its range lands short by design (recorded). For colour, a source
    with almost no colour (below the source floor) or a gain that did not take (eq clips at the
    chroma limits) is out of reach too, and is recorded as clipped.
    """
    if not corrections["luma"]["gamma_capped"]:
        drift = abs(checks["luma"]["achieved_luma"] - corrections["luma"]["target_luma"])
        if drift > S.TERMINAL_LUMA_TOLERANCE:
            msg = (
                f"output mean luma {checks['luma']['achieved_luma']:.2f} misses the "
                f"{corrections['luma']['target_luma']:.1f} target by {drift:.2f} "
                f"(tolerance {S.TERMINAL_LUMA_TOLERANCE})"
            )
            raise TerminalDropError(msg)
    if not corrections["chroma"]["gain_capped"]:
        chroma_checks = checks["chroma"]
        drift = abs(chroma_checks["achieved_chroma"] - corrections["chroma"]["target_chroma"])
        measured = corrections["chroma"]["measured_chroma"]
        asked = corrections["chroma"]["applied_gain"]
        moved = (chroma_checks["achieved_chroma"] / measured) if measured > 0 else 0.0
        efficiency = (moved / asked) if asked else 0.0
        monochrome = measured < S.TERMINAL_CHROMA_SOURCE_FLOOR
        unreachable = monochrome or efficiency < CHROMA_EFFICIENCY_MIN
        chroma_checks["gain_efficiency"] = round(efficiency, 4)
        chroma_checks["monochrome_source"] = bool(monochrome)
        chroma_checks["chroma_clipped"] = bool(drift > S.TERMINAL_CHROMA_TOLERANCE and unreachable)
        if drift > S.TERMINAL_CHROMA_TOLERANCE and not unreachable:
            msg = (
                f"output chroma {chroma_checks['achieved_chroma']:.2f} misses the "
                f"{corrections['chroma']['target_chroma']:.1f} target by {drift:.2f} "
                f"(tolerance {S.TERMINAL_CHROMA_TOLERANCE}) with the gain taking at "
                f"{efficiency:.2f} — the solve is wrong, not the clip"
            )
            raise TerminalDropError(msg)


def _assert_output(
    out: Path, audio_in: Path, target_frames: int, corrections: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Check what came out against the contract; returns the measurements."""
    expected_samples = target_frames * S.SAMPLES_PER_FRAME
    # A framed codec (AAC) decodes to whole coding frames; PCM decodes to the exact count.
    expected_decoded = expected_samples
    if S.TERMINAL_AUDIO_CODEC in S.FRAMED_AUDIO_CODECS:
        expected_decoded = -(-expected_samples // S.AAC_FRAME_SAMPLES) * S.AAC_FRAME_SAMPLES

    frames = decoded_video_frames(out)
    fps = video_frame_rate(out)
    dims = video_dimensions(out)
    samples = container_audio_samples(out)
    decoded = decoded_audio_samples(out)
    achieved = measure_loudness(out)
    lag = onset_lag_ms(decoded_audio_pcm(audio_in), decoded_audio_pcm(out))

    checks: dict[str, Any] = {
        "output_video_frames": frames,
        "output_fps": fps,
        "output_dimensions": list(dims) if dims else None,
        "output_audio_samples": samples,
        "output_audio_decoded_samples": decoded,
        "expected_audio_samples": expected_samples,
        "expected_decoded_samples": expected_decoded,
        "achieved_loudness_i": float(achieved["input_i"]),
        "achieved_true_peak": float(achieved["input_tp"]),
        "onset_lag_ms": round(lag, 4),
        "duration_s": round(target_frames / S.TARGET_FPS, 6),
    }
    if corrections is not None:
        landed = achieved_picture(out)
        checks["luma"] = {**corrections["luma"], "achieved_luma": round(landed["luma"], 4)}
        checks["chroma"] = {**corrections["chroma"], "achieved_chroma": round(landed["chroma"], 4)}
        checks["luma"].pop("input_hist", None)

    if frames != target_frames:
        msg = f"output decodes {frames} frames, expected {target_frames}"
        raise TerminalError(msg)
    if fps != f"{S.TARGET_FPS}/1":
        msg = f"output fps is {fps}, expected {S.TARGET_FPS}/1"
        raise TerminalError(msg)
    if dims != (S.TARGET_RESOLUTION, S.TARGET_RESOLUTION):
        msg = f"output is {dims}, expected {S.TARGET_RESOLUTION}x{S.TARGET_RESOLUTION}"
        raise TerminalError(msg)
    if samples != expected_samples:
        msg = f"output audio declares {samples} samples, expected {expected_samples}"
        raise TerminalError(msg)
    if decoded != expected_decoded:
        msg = f"output audio decodes {decoded} samples, expected {expected_decoded} (AAC frame padding)"
        raise TerminalError(msg)
    if corrections is not None:
        _assert_picture(checks, corrections)
    # The gain cap keeps this true; 0.5 dB covers the meter's own resolution.
    if float(achieved["input_tp"]) > S.TERMINAL_TRUE_PEAK_CEILING + 0.5:
        # A voice the capped gain could not bring under the ceiling: the clip is dropped, its pair withdrawn.
        msg = f"output true peak {achieved['input_tp']} dBTP exceeds the {S.TERMINAL_TRUE_PEAK_CEILING} ceiling"
        raise TerminalDropError(msg)
    if lag != 0.0:
        msg = f"audio onset moved {lag:.3f} ms against its video (a linear gain cannot shift audio)"
        raise TerminalError(msg)
    return checks


def _readable(path: Path) -> bool:
    """True when ffprobe can open the container and finds a video stream."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def process(
    video_in: Path,
    audio_in: Path,
    out_path: Path,
    target_frames: int,
    overwrite: bool = False,
    audit: bool = True,
) -> TerminalResult:
    """Run one clip through the mux.

    The input must carry exactly target_frames frames; the output is checked to still carry them,
    with target_frames x 640 audio samples. On any failed check no clip is written.
    """
    if out_path.exists() and not overwrite:
        if _readable(out_path):
            return TerminalResult(ok=True, out_path=out_path, target_frames=target_frames, checks={"skipped": "exists"})
        # A clip that exists but cannot be opened was cut off mid-write: make it again.
        print(f"[terminal] redo {out_path.name}: exists but cannot be read", file=sys.stderr)
        out_path.unlink(missing_ok=True)
    if target_frames <= 0:
        return TerminalResult(
            False, out_path, target_frames, error=f"target_frames must be positive, got {target_frames}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Each writer has its own temp file (host + pid), renamed into place, so two muxes over one
    # job list never write the same file.
    tmp_out = out_path.with_name(f"{out_path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    checks: dict[str, Any] = {}
    try:
        checks.update(_assert_inputs(video_in, audio_in, target_frames))
        gain = compute_gain_db(measure_loudness(audio_in))
        checks["gain"] = gain
        corrections = solve_corrections(video_in)
        if audit:
            checks.update(verify_audio_conform(audio_in, gain, target_frames))
        _encode(
            video_in, audio_in, tmp_out, gain, target_frames,
            corrections["luma"]["applied_gamma"], corrections["chroma"]["applied_gain"],
        )  # fmt: skip
        # Measure what the encoder produced and correct against it. The same number of passes for
        # every clip, so the work done never depends on the clip's content.
        for _ in range(S.TERMINAL_REFINE_PASSES):
            corrections = refine_corrections(corrections, achieved_picture(tmp_out))
            _encode(
                video_in, audio_in, tmp_out, gain, target_frames,
                corrections["luma"]["applied_gamma"], corrections["chroma"]["applied_gain"],
            )  # fmt: skip
        checks["refine_passes"] = S.TERMINAL_REFINE_PASSES
        checks.update(_assert_output(tmp_out, audio_in, target_frames, corrections))
    except TerminalError as exc:
        tmp_out.unlink(missing_ok=True)
        return TerminalResult(
            False, out_path, target_frames, checks=checks, error=str(exc), dropped=isinstance(exc, TerminalDropError)
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        tmp_out.unlink(missing_ok=True)
        return TerminalResult(False, out_path, target_frames, checks=checks, error=f"{type(exc).__name__}: {exc}")

    tmp_out.replace(out_path)
    return TerminalResult(True, out_path, target_frames, checks=checks)


def write_provenance(out_path: Path, result: TerminalResult, job: dict[str, Any] | None = None) -> None:
    """Write the clip's record beside it: the spec it was made under and its own checks."""
    record = {
        "sample_id": (job or {}).get("sample_id"),
        "class_name": (job or {}).get("class_name"),
        "target_frames": result.target_frames,
        "fps": S.TARGET_FPS,
        "resolution": S.TARGET_RESOLUTION,
        "video_codec": S.TERMINAL_VIDEO_CODEC,
        "crf": S.TERMINAL_CRF,
        "preset": S.TERMINAL_PRESET,
        "pixel_format": S.TERMINAL_PIXEL_FORMAT,
        "scale_flags": S.TERMINAL_SCALE_FLAGS,
        "resample_via": S.TERMINAL_RESAMPLE_VIA,
        "container": S.TERMINAL_CONTAINER_FORMAT,
        "audio_codec": S.TERMINAL_AUDIO_CODEC,
        "audio_bitrate": S.TERMINAL_AUDIO_BITRATE,
        "audio_sample_rate": S.AUDIO_SAMPLE_RATE_HZ,
        "luma": {
            "target": S.TERMINAL_LUMA_TARGET,
            "tolerance": S.TERMINAL_LUMA_TOLERANCE,
            "mode": "per-clip gamma on Y, solved from the clip's own histogram",
        },
        "chroma": {
            "target": S.TERMINAL_CHROMA_TARGET,
            "tolerance": S.TERMINAL_CHROMA_TOLERANCE,
            "mode": "per-clip gain on U/V, solved from the clip's own saturation",
        },
        "refine_passes_configured": S.TERMINAL_REFINE_PASSES,
        # No noise is added to any clip; the two keys stay so every record has the same fields.
        "noise_strength": None,
        "noise_seed": None,
        "loudness": {
            "target_i": S.TERMINAL_LOUDNESS_I,
            "true_peak_ceiling": S.TERMINAL_TRUE_PEAK_CEILING,
            "mode": "peak-capped constant gain",
        },
        "silence_trim": False,
        "checks": result.checks,
        # the family that made the clip, carried from its pair-list row (null-test runs have no job)
        "generator_version": (job or {}).get("generator_version", S.GENERATOR_VERSION),
    }
    out_path.with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- null test
def null_test(video_in: Path, audio_in: Path, target_frames: int, workdir: Path) -> bool:
    """Show the mux cannot see the label, and that the comparison can fail.

    1. The same inputs, made twice to two output paths, must match to the byte and to the
       decoded frame and sample.
    2. A slightly altered input must be reported as different, or step 1 proves nothing.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    left = workdir / f"null_left{S.TERMINAL_CONTAINER_EXT}"
    right = workdir / f"null_right{S.TERMINAL_CONTAINER_EXT}"

    print("[null] half 1: identical inputs, two class labels, two output paths")
    r_left = process(video_in, audio_in, left, target_frames, overwrite=True)
    r_right = process(video_in, audio_in, right, target_frames, overwrite=True)
    if not (r_left.ok and r_right.ok):
        print(f"[null] FAIL — chain errored: {r_left.error or r_right.error}", file=sys.stderr)
        return False

    same_bytes = file_md5(left) == file_md5(right)
    same_video = stream_md5(left, video=True) == stream_md5(right, video=True)
    same_audio = stream_md5(left, video=False) == stream_md5(right, video=False)
    print(f"[null]   bytes identical         : {same_bytes}")
    print(f"[null]   decoded video identical : {same_video}")
    print(f"[null]   decoded audio identical : {same_audio}")
    if not (same_bytes and same_video and same_audio):
        print("[null] FAIL — the chain is not label-independent", file=sys.stderr)
        return False

    print("[null] half 2: positive control — an altered input MUST be reported different")
    altered_src = workdir / "altered_audio.wav"
    # A small attenuation: far too large for a bit-exact comparison to miss.
    rc = _run([
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(audio_in),
        "-af", "volume=0.97", "-ar", str(S.AUDIO_SAMPLE_RATE_HZ), "-ac", "1", "-f", "wav", str(altered_src),
    ])  # fmt: skip
    if rc.returncode != 0:
        print("[null] FAIL — could not build the positive control", file=sys.stderr)
        return False
    control = workdir / f"null_control{S.TERMINAL_CONTAINER_EXT}"
    r_control = process(video_in, altered_src, control, target_frames, overwrite=True)
    if not r_control.ok:
        print(f"[null] FAIL — control clip errored: {r_control.error}", file=sys.stderr)
        return False
    detected = file_md5(left) != file_md5(control) and stream_md5(left, video=False) != stream_md5(control, video=False)
    print(f"[null]   altered input detected  : {detected}")
    if not detected:
        print("[null] FAIL — the comparison cannot see a real difference; half 1 proves nothing", file=sys.stderr)
        return False

    print("[null] PASS — label-independent, and the test can fail")
    return True


# --------------------------------------------------------------------------- command line
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="The one mux for all four classes.")
    p.add_argument("--video", type=Path, help="video-only input (crop or render)")
    p.add_argument("--audio", type=Path, help="audio input (crop wav, conversion, or vocoded)")
    p.add_argument("--out", type=Path, help="output clip (.mov)")
    p.add_argument("--jobs", type=Path, help="JSONL: {video, audio, out, target_frames, sample_id, class_name}")
    p.add_argument("--out-dir", type=Path, help="root for --jobs outputs whose 'out' is relative")
    p.add_argument("--frames", type=int, default=S.CROP_FRAMES,
                   help="target frame count for --video/--audio (default: clips.crop_frames; --jobs gives it per clip)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-audit", action="store_true", help="skip the per-clip audio-length probe")
    p.add_argument("--refine-passes", type=int, default=None,
                   help="passes against the encoded output; overrides mux.refine_passes for this run only")
    p.add_argument("--null-test", action="store_true", help="run the null test and exit")
    p.add_argument("--workdir", type=Path, default=None, help="scratch dir for --null-test")
    return p.parse_args()


def run_null_test(args: argparse.Namespace) -> int:
    if not (args.video and args.audio):
        print("--null-test needs --video and --audio", file=sys.stderr)
        return 2
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="terminal_null_"))
    try:
        return 0 if null_test(args.video, args.audio, args.frames, workdir) else 1
    finally:
        if args.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)


def run_jobs(args: argparse.Namespace) -> int:
    """Mux every job of a list; drops go to <jobs>_dropped.jsonl, and only failures fail the step."""
    failures = 0
    dropped: list[dict[str, Any]] = []
    total = 0
    with args.jobs.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            job = json.loads(line)
            total += 1
            out = Path(job["out"])
            if not out.is_absolute() and args.out_dir:
                out = args.out_dir / out
            result = process(
                Path(job["video"]), Path(job["audio"]), out,
                int(job["target_frames"]), args.overwrite, not args.no_audit,
            )  # fmt: skip
            if result.ok:
                if result.checks.get("skipped") and out.with_suffix(".json").exists():
                    # A finished clip keeps its record; it is not rewritten as {"skipped": "exists"}.
                    print(f"[terminal] skip {job.get('sample_id', out.name)} (exists, sidecar kept)")
                    continue
                write_provenance(out, result, job)
                print(f"[terminal] ok   {job.get('sample_id', out.name)}")
            elif result.dropped:
                dropped.append(
                    {
                        "sample_id": job.get("sample_id"),
                        "class_name": job.get("class_name"),
                        "pair_id": job.get("pair_id"),
                        "out": str(out),
                        "reason": result.error,
                    }
                )
                print(f"[terminal] DROP {job.get('sample_id', out.name)}: {result.error}")
            else:
                failures += 1
                print(f"[terminal] FAIL {job.get('sample_id', out.name)}: {result.error}", file=sys.stderr)
    dropped_path = args.jobs.with_name(args.jobs.stem + "_dropped.jsonl")
    with dropped_path.open("w") as fh:
        for row in dropped:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        f"[terminal] {total - failures - len(dropped)}/{total} ok, {len(dropped)} dropped by design "
        f"(-> {dropped_path.name}), {failures} FAILED"
    )
    return 1 if failures else 0


def main() -> int:
    """Run one clip, a job list, or the null test."""
    args = parse_args()
    if args.refine_passes is not None:
        S.TERMINAL_REFINE_PASSES = args.refine_passes
    if args.null_test:
        return run_null_test(args)
    if args.jobs:
        return run_jobs(args)

    if not (args.video and args.audio and args.out):
        print("need --video, --audio and --out (or --jobs, or --null-test)", file=sys.stderr)
        return 2
    result = process(args.video, args.audio, args.out, args.frames, args.overwrite, not args.no_audit)
    if not result.ok:
        print(f"[terminal] FAIL: {result.error}", file=sys.stderr)
        return 1
    if not (result.checks.get("skipped") and args.out.with_suffix(".json").exists()):
        write_provenance(args.out, result)
    print(json.dumps(result.checks, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
