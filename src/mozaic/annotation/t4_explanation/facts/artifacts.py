"""The closed list of artifact categories an evidence item can name, per part."""

from __future__ import annotations

CATEGORIES = {
    "video": [
        "texture jittering",
        "unnatural blur",
        "color or lighting inconsistency",
        "camera motion inconsistency",
        "blending seam",
        "person identity inconsistency",
        "shape or structure distortion",
        "unnatural human movement",
    ],
    "audio": [
        "metallic or buzzy timbre",
        "unnatural pitch or loudness",
        "smeared breaths or consonants",
    ],
}
