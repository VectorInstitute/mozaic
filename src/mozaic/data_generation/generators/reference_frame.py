"""Pick the face photo a video generator starts from: the best frame of a person's own VoxCeleb2 clip.

All three video generators build every output frame from one still of the person, so that still fixes the
face, clothes, framing and background of the whole render. A still that catches motion blur, a blink or a
grimace is copied into every frame, and no later check can see it. So instead of a fixed time, the job makers
score candidate frames on what ruins a still and keep the best:

- sharpness: variance of the Laplacian over the face (talking heads move; blur is the common defect);
- a frontal face: OpenCV's Haar frontal-face detector;
- both eyes found: the Haar eye detector, which also rejects most blinks (a closed eye usually fails it).

The scores multiply, so a frame that fails one cannot win on another. Haar needs only OpenCV and runs on a
CPU (about a second per clip); VoxCeleb2 frames are already tight face crops, so it only has to confirm the
face. A clip with no scored frame falls back to S.FACE_REF_AT_S, so it still renders.

    python -m mozaic.data_generation.generators.reference_frame --clip id04652/AZTGcx8hmIo/00085
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from mozaic.data_generation import settings as S

MIN_FACE_FRACTION = 0.05  # the face must cover at least this much of the frame


def load_cascades() -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    """The frontal-face and eye Haar cascades that ship with OpenCV."""
    base = cv2.data.haarcascades  # type: ignore[attr-defined]
    return (
        cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml"),
        cv2.CascadeClassifier(base + "haarcascade_eye.xml"),
    )


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: the usual cheap blur measure."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_frames(
    video: Path, candidates: int, cascades: tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]
) -> list[tuple[float, float, dict[str, float]]]:
    """Score `candidates` frames spread over the clip. Returns (time, score, parts)."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or S.TARGET_FPS)
    if total <= 0 or fps <= 0:
        cap.release()
        return []

    # Skip the first and last 10%: a clip's edges often catch a cut or a fade.
    lo, hi = int(total * 0.1), int(total * 0.9)
    picks = np.linspace(lo, max(lo + 1, hi - 1), num=min(candidates, max(1, hi - lo)), dtype=int)

    face_cc, eye_cc = cascades
    out: list[tuple[float, float, dict[str, float]]] = []
    for frame_idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = face_cc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(boxes) == 0:
            continue
        fx, fy, fw, fh = max(boxes, key=lambda b: int(b[2]) * int(b[3]))
        face_frac = (fw * fh) / float(w * h)
        if face_frac < MIN_FACE_FRACTION:
            continue

        face_gray = gray[fy : fy + fh, fx : fx + fw]
        if face_gray.size == 0:
            continue
        blur = sharpness(face_gray)
        # Two eyes found: roughly frontal and probably not mid-blink.
        eyes = eye_cc.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6)
        eye_term = 1.0 if len(eyes) >= 2 else 0.25
        out.append(
            (
                float(frame_idx) / fps,
                blur * eye_term,
                {"sharpness": blur, "eyes_found": float(len(eyes)), "eye_term": eye_term, "face_fraction": face_frac},
            )
        )
    cap.release()
    return out


def best_reference_time(video: Path, candidates: int = S.FACE_REF_CANDIDATES) -> tuple[float, dict[str, float]]:
    """(time in s, score parts) of the best frame in `video`; S.FACE_REF_AT_S and {} when none scores."""
    scored = score_frames(video, candidates, load_cascades())
    if not scored:
        return S.FACE_REF_AT_S, {}
    t, _, parts = max(scored, key=lambda r: r[1])
    return round(t, 3), parts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick the best face photo frame in a clip.")
    p.add_argument("--clip", required=True, help="clip id like id04652/AZTGcx8hmIo/00085, or a path")
    p.add_argument("--candidates", type=int, default=S.FACE_REF_CANDIDATES)
    return p.parse_args()


def main() -> int:
    """Print the chosen time and its score parts."""
    args = parse_args()
    video = (
        Path(args.clip)
        if "/" in args.clip and args.clip.endswith(".mp4")
        else S.VOXCELEB2_MP4_ROOT / f"{args.clip}.mp4"
    )
    if not video.exists():
        print(f"no such clip: {video}", file=sys.stderr)
        return 1
    t, parts = best_reference_time(video, args.candidates)
    print(f"best reference frame: t={t}s")
    for k, v in parts.items():
        print(f"  {k:14}= {v:.4f}")
    print(f"  (fixed default would have been t={S.FACE_REF_AT_S}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
