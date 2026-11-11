"""Face check: is the generated face the target person, and how much of the driver is left?

ArcFace embeddings (insightface pack buffalo_l, recognition model w600k_r50), the usual
identity score in face-swap papers. A cosine has no absolute scale (it moves with pose,
light and the two faces), so every pair is measured against both ends of its own scale:

    ceiling   the real clip against itself (two halves of its frames): the same face
    floor     the real driver against the target's photo: two different real people
    target    the generated face against the target's photo: wants to be near the ceiling
    leak      the generated face against the real driver: wants to be near the floor

The swap worked when target > leak. The target's photo is refs/<render_key>.png, the
exact still the generator was given, so the check scores against what the generator saw.

Self-drive renders (the driver re-made as themselves) are a control: there target and
leak should both be high, so they are reported apart from the cross renders.

Needs insightface and onnxruntime; run it in an environment that has both (the
FaceFusion one does):

    <face env>/bin/python -m mozaic.data_generation.gates.face_check --build-id <build> \\
        --final-dir <final> --pair-ids-file <pairs.txt> --out <gates>/identity_gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from mozaic.data_generation import settings as S

EMBED_FRAMES = 8  # frames per clip; their mean embedding, so one bad detection matters little
MODEL_PACK = "buffalo_l"  # its recognition model is ArcFace w600k_r50


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="ArcFace identity gate for generated video.")
    p.add_argument("--build-id", required=True)
    p.add_argument("--frames", type=int, default=EMBED_FRAMES)
    p.add_argument("--limit", type=int, default=None, help="first N pairs")
    p.add_argument(
        "--pair-ids-file",
        type=Path,
        default=None,
        help="score only these pairs (one id per line), e.g. one batch instead of the whole build",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="cap onnxruntime's intra-op threads (default: onnxruntime's own choice, the machine's core count)",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--min-target",
        type=float,
        default=S.IDENTITY_TARGET_MIN,
        help="ArcFace cosine to the target's photo below which a render is flagged",
    )
    p.add_argument(
        "--final-dir",
        type=Path,
        default=None,
        help="root holding the shipped classes (default: <new_generated>/final); point it at a re-mux to score that",
    )
    p.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="InsightFace model folder (default: $INSIGHTFACE_ROOT, then ~/.insightface)",
    )
    return p.parse_args()


def insightface_root(model_root: Path | None = None) -> Path:
    """Return the InsightFace model folder.

    InsightFace ignores INSIGHTFACE_HOME. The only override that works is
    FaceAnalysis(root=...), fed by --model-root or $INSIGHTFACE_ROOT.
    """
    if model_root is not None:
        return model_root.expanduser()
    return Path(os.environ.get("INSIGHTFACE_ROOT", "~/.insightface")).expanduser()


def build_app(det_size: int = 320, threads: int | None = None, model_root: Path | None = None) -> Any:
    """Load insightface (face detection and recognition only) on the CPU.

    threads caps onnxruntime's thread pool. Left alone, onnxruntime starts one thread
    per core of the machine, far more than a job given a few CPUs can run. insightface
    passes only the providers to its sessions, so the cap goes in by wrapping its
    session class. None keeps onnxruntime's default, as in the release run.
    """
    from insightface.app import FaceAnalysis

    if threads:
        import onnxruntime
        from insightface.model_zoo import model_zoo

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        original_init = model_zoo.PickableInferenceSession.__init__

        def capped_init(self: Any, model_path: str, **kwargs: Any) -> None:
            original_init(self, model_path, sess_options=opts, **kwargs)

        model_zoo.PickableInferenceSession.__init__ = capped_init

    root = insightface_root(model_root)
    print(f"[identity] model root: {root}")
    app = FaceAnalysis(name=MODEL_PACK, root=str(root), allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def frame_embeddings(app: Any, path: Path, n: int) -> np.ndarray:
    """L2-normalised ArcFace embeddings for up to `n` frames of a clip or image."""
    import cv2

    images: list[np.ndarray] = []
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)
    else:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or S.CROP_FRAMES
        for idx in np.linspace(2, max(2, total - 3), n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                images.append(frame)
        cap.release()

    embeds: list[np.ndarray] = []
    for img in images:
        faces = app.get(img)
        if not faces:
            continue
        # Largest detection: these are single-speaker face crops, so a second
        # detection is a false positive on background texture.
        face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        vec = np.asarray(face.normed_embedding, dtype=np.float64)
        embeds.append(vec / (np.linalg.norm(vec) or 1.0))
    return np.stack(embeds) if embeds else np.zeros((0, 512))


def mean_embedding(embeds: np.ndarray) -> np.ndarray | None:
    """Average a set of unit embeddings back onto the unit sphere."""
    if len(embeds) == 0:
        return None
    m = embeds.mean(axis=0)
    norm = float(np.linalg.norm(m))
    return m / norm if norm else None


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine similarity, or NaN when either side had no usable detection."""
    if a is None or b is None:
        return float("nan")
    return float(np.dot(a, b))


def split_half_ceiling(embeds: np.ndarray) -> float:
    """Cosine between two halves of one clip's frames: the ceiling for this face."""
    if len(embeds) < 2:
        return float("nan")
    half = len(embeds) // 2
    return cosine(mean_embedding(embeds[:half]), mean_embedding(embeds[half:]))


def read_pair_ids(path: Path | None) -> set[str] | None:
    """Return the pair ids listed in *path*, or None for every pair."""
    if path is None:
        return None
    return {x.strip() for x in path.read_text().splitlines() if x.strip()}


def cross_renders(rows: list[dict[str, Any]], keep: set[str] | None) -> dict[str, dict[str, Any]]:
    """One row per cross render, restricted to *keep* when a pair list is given.

    With a list, the gate scores one batch, not every render of the build, so its run
    time stays the batch's.
    """
    renders: dict[str, dict[str, Any]] = {}
    for r in rows:
        if keep is not None and r.get("pair_id") not in keep:
            continue
        if r.get("needs_render") and r["class_name"] == "FVRA":
            renders.setdefault(r["render_key"], r)
    return renders


def main() -> int:
    """Score every cross render against its target's photo and its driver."""
    args = parse_args()
    rows = [
        json.loads(line) for line in S.new_gen_manifest_path(args.build_id).read_text().splitlines() if line.strip()
    ]
    # The main-arm RVRA is the driver's untouched clip; the vocoded twin shares
    # its pixels, so main-arm is the unambiguous choice.
    real_by_pair = {r["pair_id"]: r for r in rows if r["class_name"] == "RVRA" and r["arm"] == "main"}
    # One entry per render, not per pair: keyed by pair, a pair's self-drive render
    # would replace its cross render, which would then never be scored.
    renders = cross_renders(rows, read_pair_ids(args.pair_ids_file))

    final_dir = S.resolve_final_dir(args.final_dir)
    app = build_app(threads=args.threads, model_root=args.model_root)
    print(
        f"[identity] insightface {MODEL_PACK} ready; {len(renders)} renders across "
        f"{len({r['pair_id'] for r in renders.values()})} pairs"
    )
    print(f"[identity] scoring {final_dir}")

    results: list[dict[str, Any]] = []
    keys = sorted(renders)[: args.limit] if args.limit else sorted(renders)
    for n, render_key in enumerate(keys, 1):
        r = renders[render_key]
        pair_id = r["pair_id"]
        fake_row = r
        real_row = real_by_pair.get(pair_id)
        ref_png = S.ref_image_path(r["render_key"])
        if fake_row is None or real_row is None or not ref_png.exists():
            continue
        fake_clip = S.find_final_sample("FVRA", fake_row["sample_id"], final_dir)
        real_clip = S.find_final_sample("RVRA", real_row["sample_id"], final_dir)
        if not (fake_clip and real_clip):
            continue

        fake_e = frame_embeddings(app, fake_clip, args.frames)
        real_e = frame_embeddings(app, real_clip, args.frames)
        ref_e = frame_embeddings(app, ref_png, 1)
        fake_m, real_m, ref_m = mean_embedding(fake_e), mean_embedding(real_e), mean_embedding(ref_e)

        rec = {
            "pair_id": pair_id,
            "render_key": r["render_key"],
            "self_drive": r["video_driver"] == S.VIDEO_DRIVER_SELF,
            "frames_detected": {"fake": int(len(fake_e)), "real": int(len(real_e)), "ref": int(len(ref_e))},
            "ceiling": split_half_ceiling(real_e),
            "floor": cosine(real_m, ref_m),
            "target": cosine(fake_m, ref_m),
            "leak": cosine(fake_m, real_m),
        }
        rec["swapped"] = bool(rec["target"] > rec["leak"]) if not np.isnan(rec["target"]) else None
        results.append(rec)
        if n % 20 == 0 or n == len(keys):
            print(f"[identity]   {n}/{len(keys)}", flush=True)

    ok = [r for r in results if not np.isnan(r["target"]) and not np.isnan(r["leak"])]
    cross = [r for r in ok if not r["self_drive"]]
    selfd = [r for r in ok if r["self_drive"]]

    def col(rs: list[dict[str, Any]], k: str) -> str:
        vals = [r[k] for r in rs if not np.isnan(r[k])]
        return f"{statistics.mean(vals):.3f}" if vals else "  n/a"

    print(f"\n=== ArcFace identity (CSIM), {len(ok)}/{len(results)} pairs with usable detections ===")
    print(f"{'arm':14s} {'n':>3s} {'ceiling':>8s} {'floor':>7s} {'target':>7s} {'leak':>7s} {'swapped':>9s}")
    for label, rs in (("cross (swap)", cross), ("self-drive", selfd)):
        if not rs:
            continue
        sw = sum(1 for r in rs if r["swapped"])
        print(
            f"{label:14s} {len(rs):3d} {col(rs, 'ceiling'):>8s} {col(rs, 'floor'):>7s} "
            f"{col(rs, 'target'):>7s} {col(rs, 'leak'):>7s} {sw:6d}/{len(rs)}"
        )
    if cross:
        tgt = [r["target"] for r in cross if not np.isnan(r["target"])]
        flo = [r["floor"] for r in cross if not np.isnan(r["floor"])]
        cei = [r["ceiling"] for r in cross if not np.isnan(r["ceiling"])]
        span = statistics.mean(cei) - statistics.mean(flo)
        print(
            f"\n  target sits {(statistics.mean(tgt) - statistics.mean(flo)) / span * 100:.0f}% "
            "of the way from FLOOR to CEILING"
        )
        print("  (100% = indistinguishable from the donor; 0% = no more like the donor than a stranger)")

    # The gate. A healthy mean can sit beside renders that do not look like the
    # target at all, so the ones below the limit are counted and written out as a list.
    below = sorted(
        (r for r in cross if not np.isnan(r["target"]) and r["target"] < args.min_target),
        key=lambda r: r["target"],
    )
    print(f"\n=== identity-quality gate: target >= {args.min_target:.2f} ===")
    if cross:
        print(f"  {len(below)}/{len(cross)} cross renders below the threshold ({len(below) / len(cross) * 100:.1f}%)")
    for r in below[:10]:
        print(f"    {r['target']:.3f}  {r['pair_id']}")
    if len(below) > 10:
        print(f"    ... and {len(below) - 10} more")
    print(f"  VERDICT: {'PASS — every cross render resembles its donor' if not below else 'FLAG — see the list above'}")

    out = args.out or (S.new_gen_report_dir() / f"{args.build_id}_identity_gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "build_id": args.build_id,
        "final_dir": str(final_dir),
        "n_pairs": len(results),
        "min_target": args.min_target,
        "pairs": results,
    }
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nwrote {out}")
    if below:
        excluded = out.with_name(f"{args.build_id}_identity_excluded.txt")
        excluded.write_text("".join(f"{r['pair_id']}\n" for r in below))
        print(f"wrote {excluded}")
        print("  feed it to --pair-ids-file to re-render or to ship without them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
