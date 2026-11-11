"""Settings and file names for MOZAIC generation.

Values come from configs/*.yaml, the two folders from configs/paths.yaml. Every step imports
this module (as S) and reads the same names, so a value lives in one place only.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

CONFIGS = Path(__file__).resolve().parent / "configs"


def _load(name: str) -> dict[str, Any]:
    with (CONFIGS / name).open() as f:
        return yaml.safe_load(f)


def _paths() -> dict[str, Any]:
    """The two folders, from configs/paths.yaml or the file named by MOZAIC_GENERATION_PATHS."""
    path = Path(os.environ.get("MOZAIC_GENERATION_PATHS") or CONFIGS / "paths.yaml")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: copy configs/paths.example.yaml to configs/paths.yaml")
    with path.open() as f:
        return yaml.safe_load(f)


RUN = _load("run.yaml")
FACEFUSION = _load("facefusion.yaml")
S2V = _load("s2v.yaml")
ANIMATE = _load("animate.yaml")
SEEDVC = _load("seed_vc.yaml")
_P = _paths()

# --------------------------------------------------------------------------- folders
VOXCELEB2_ROOT = Path(_P["voxceleb2"]).expanduser()
VOXCELEB2_MP4_ROOT = VOXCELEB2_ROOT / "dev" / "mp4"  # <id>/<video>/<clip>.mp4
VOXCELEB2_META_CSV = VOXCELEB2_ROOT / "meta" / "vox2_meta.csv"
DATA_ROOT = Path(_P["data"]).expanduser()
GEN_ROOT = DATA_ROOT / "voxceleb2"  # clip list, identity split, extracted voice audio
AUDIO_ROOT = GEN_ROOT / "audio"
NEW_GEN_ROOT = DATA_ROOT / "new_generated"  # everything a build makes
RELEASE_ROOT = DATA_ROOT / "mozaic"
BUILD_ROOT = DATA_ROOT / "_mozaic_build"

# --------------------------------------------------------------------------- clip list + split
TIMEOUT_FFPROBE = RUN["inventory"]["ffprobe_timeout_s"]
INVENTORY_CONCURRENCY = RUN["inventory"]["concurrency"]
MIN_CLIP_DURATION_S = RUN["inventory"]["min_clip_duration_s"]
DROP_REASON_CORRUPT = "corrupt"
DROP_REASON_TOO_SHORT = "too_short"
DROP_REASON_NO_AUDIO = "no_audio"
SPLIT_SEED = RUN["seed"]
SPLIT_RATIO = tuple(RUN["split"]["ratio"])
STRATIFY_KEY = RUN["split"]["stratify_key"]
AUDIO_SAMPLE_RATE_HZ = RUN["audio"]["sample_rate_hz"]
AUDIO_CHANNELS = RUN["audio"]["channels"]
AUDIO_EXTRACT_CONCURRENCY = RUN["audio"]["extract_concurrency"]

# --------------------------------------------------------------------------- pairs
RING_SIZE = RUN["pairs"]["ring_size"]
RING_GENDERS = tuple(RUN["pairs"]["ring_genders"])
FOLDERS_PER_CELEB = RUN["pairs"]["folders_per_celeb"]
CLIPS_PER_FOLDER = RUN["pairs"]["clips_per_folder"]
MIN_DRIVER_CLIP_MARGIN_S = RUN["pairs"]["min_driver_clip_margin_s"]
SAME_SPEAKER_ARM_FRAC = RUN["pairs"]["same_speaker_arm_frac"]
SELF_DRIVE_CONTROL_FRAC = RUN["pairs"]["self_drive_control_frac"]
VOCODED_REAL_FRAC = RUN["pairs"]["vocoded_real_frac"]

# --------------------------------------------------------------------------- clips
TARGET_FPS = RUN["clips"]["fps"]
TARGET_RESOLUTION = RUN["clips"]["size"]
CROP_FRAMES = RUN["clips"]["crop_frames"]
CROP_START_S = RUN["clips"]["crop_start_s"]
CROP_CONCURRENCY = RUN["clips"]["crop_concurrency"]
CROP_DURATION_S = CROP_FRAMES / TARGET_FPS
MIN_DRIVER_CLIP_DURATION_S = CROP_START_S + CROP_DURATION_S + MIN_DRIVER_CLIP_MARGIN_S
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE_HZ // TARGET_FPS  # 640: a whole frame is a whole number of samples
CROP_AUDIO_SAMPLES = CROP_FRAMES * SAMPLES_PER_FRAME

# --------------------------------------------------------------------------- face photo for the generators
FACE_REF_AT_S = RUN["reference_frame"]["at_s"]
FACE_REF_CANDIDATES = RUN["reference_frame"]["candidates"]

# --------------------------------------------------------------------------- mux
_MUX = RUN["mux"]
TERMINAL_VIDEO_CODEC = _MUX["video_codec"]
TERMINAL_CRF = _MUX["crf"]
TERMINAL_PRESET = _MUX["preset"]
TERMINAL_PIXEL_FORMAT = _MUX["pixel_format"]
TERMINAL_SCALE_FLAGS = _MUX["scale_flags"]
TERMINAL_RESAMPLE_VIA = _MUX["resample_via"]
TERMINAL_LUMA_TARGET = _MUX["luma"]["target"]
TERMINAL_LUMA_TOLERANCE = _MUX["luma"]["tolerance"]
TERMINAL_LUMA_GAMMA_MIN = _MUX["luma"]["gamma_min"]
TERMINAL_LUMA_GAMMA_MAX = _MUX["luma"]["gamma_max"]
TERMINAL_CHROMA_TARGET = _MUX["chroma"]["target"]
TERMINAL_CHROMA_TOLERANCE = _MUX["chroma"]["tolerance"]
TERMINAL_CHROMA_SOURCE_FLOOR = _MUX["chroma"]["source_floor"]
TERMINAL_CHROMA_GAIN_MIN = _MUX["chroma"]["gain_min"]
TERMINAL_CHROMA_GAIN_MAX = _MUX["chroma"]["gain_max"]
TERMINAL_REFINE_PASSES = _MUX["refine_passes"]
TERMINAL_LOUDNESS_I = _MUX["loudness"]["integrated_lufs"]
TERMINAL_TRUE_PEAK_CEILING = _MUX["loudness"]["true_peak_ceiling_db"]
TERMINAL_MAX_GAIN_DB = _MUX["loudness"]["max_gain_db"]
TERMINAL_AUDIO_CODEC = _MUX["audio"]["codec"]
TERMINAL_AUDIO_BITRATE = _MUX["audio"]["bitrate"]
TERMINAL_AUDIO_FRAME_SAMPLES = _MUX["audio"]["frame_samples"]
TERMINAL_AUDIO_TAIL_PAD_SAMPLES = _MUX["audio"]["tail_pad_samples"]
TERMINAL_CONTAINER_FORMAT = _MUX["container"]["format"]
TERMINAL_CONTAINER_EXT = _MUX["container"]["ext"]
SHIPPED_MEDIA_EXTS = (".mov", ".mp4")  # readers accept both; the mux writes TERMINAL_CONTAINER_EXT
AAC_FRAME_SAMPLES = 1024
FRAMED_AUDIO_CODECS = ("aac",)  # codecs whose decode runs on to a whole coding frame

# --------------------------------------------------------------------------- keep/drop
GATES = RUN["gates"]
IDENTITY_TARGET_MIN = GATES["A"]["arcface_target_abs_min"]

# --------------------------------------------------------------------------- generators
FACEFUSION_SWAPPER_MODEL = FACEFUSION["swapper_model"]
FACEFUSION_DETECTOR_MODEL = FACEFUSION["detector_model"]
FACEFUSION_DETECTOR_SCORE = FACEFUSION["detector_score"]
FACEFUSION_FACE_ENHANCER = FACEFUSION["face_enhancer"]
FACEFUSION_PIXEL_BOOST = FACEFUSION["pixel_boost"]
FACEFUSION_OUTPUT_VIDEO_QUALITY = FACEFUSION["output_video_quality"]
FACEFUSION_SEC_PER_RENDER = FACEFUSION["sec_per_render"]
S2V_SIZE = S2V["size"]
S2V_INFER_FRAMES = S2V["infer_frames"]
S2V_SAMPLE_STEPS = S2V["sample_steps"]
S2V_GUIDE_SCALE = S2V["guide_scale"]
S2V_START_FROM_REF = S2V["start_from_ref"]
S2V_PROMPT = S2V["prompt"]
S2V_NATIVE_FPS = S2V["native_fps"]
S2V_SECONDS_PER_CLIP = S2V_INFER_FRAMES / S2V_NATIVE_FPS
S2V_RETIME_METHOD = S2V["retime_method"]
S2V_MIN_PER_CHUNK = S2V["min_per_chunk"]
ANIMATE_PROMPT = ANIMATE["prompt"]
ANIMATE_RESOLUTION_AREA = ANIMATE["resolution_area"]
ANIMATE_SAMPLE_STEPS = ANIMATE["sample_steps"]
ANIMATE_SAMPLE_SHIFT = ANIMATE["sample_shift"]
ANIMATE_SAMPLE_GUIDE_SCALE = ANIMATE["guide_scale"]
ANIMATE_FRAME_NUM = ANIMATE["frame_num"]
ANIMATE_REFERT_NUM = ANIMATE["refert_num"]
SEEDVC_DIFFUSION_STEPS = SEEDVC["diffusion_steps"]
SEEDVC_INFERENCE_CFG_RATE = SEEDVC["inference_cfg_rate"]
SEEDVC_LENGTH_ADJUST = SEEDVC["length_adjust"]

# --------------------------------------------------------------------------- names used in every file
CLASS_NAMES = ("RVRA", "RVFA", "FVRA", "FVFA")
LABEL_REAL = "real"
LABEL_FAKE = "fake"
VIDEO_METHOD_NONE = "none"
VIDEO_METHOD_ANIMATE = "animate"
VIDEO_METHOD_FACEFUSION = "facefusion"
VIDEO_METHOD_S2V = "s2v"
VIDEO_METHODS = (VIDEO_METHOD_ANIMATE, VIDEO_METHOD_FACEFUSION, VIDEO_METHOD_S2V)
VIDEO_DRIVER_NONE = "none"
VIDEO_DRIVER_CROSS = "cross"  # the other person's face put into the driver clip
VIDEO_DRIVER_SELF = "self"  # the driver re-made as themselves (a control)
AUDIO_METHOD_NONE = "none"
AUDIO_METHOD_SEEDVC = "seedvc"
AUDIO_VARIANT_NATURAL = "natural_real"
AUDIO_VARIANT_VOCODED = "vocoded_real"
AUDIO_VARIANT_CONVERTED = "converted_fake"
TEST_SUBSET_NATURAL = "natural"
TEST_SUBSET_VOCODED = "vocoded"
TEST_SUBSET_BOTH = "both"

_GENERATORS = {VIDEO_METHOD_ANIMATE: ANIMATE, VIDEO_METHOD_FACEFUSION: FACEFUSION, VIDEO_METHOD_S2V: S2V}
GENERATOR_VERSION = ANIMATE["generator_version"]
GENERATOR_VERSION_BY_VIDEO_METHOD = {m: g["generator_version"] for m, g in _GENERATORS.items()}
RENDER_KEY_PREFIXES = {
    m: {VIDEO_DRIVER_CROSS: g["render_key_prefixes"]["cross"], VIDEO_DRIVER_SELF: g["render_key_prefixes"]["self"]}
    for m, g in _GENERATORS.items()
}


def generator_version(video_method: str = VIDEO_METHOD_ANIMATE) -> str:
    """Version string of a video family (with the voice settings)."""
    return GENERATOR_VERSION_BY_VIDEO_METHOD.get(video_method, GENERATOR_VERSION)


def sample_id_for(pair_id: str, class_name: str, video_method: str = VIDEO_METHOD_ANIMATE, arm_suffix: str = "") -> str:
    """Name of a shipped clip. Fake-video clips of FaceFusion and S2V carry the family."""
    generated = class_name in (CLASS_NAMES[2], CLASS_NAMES[3])
    method = f"__{video_method}" if generated and video_method != VIDEO_METHOD_ANIMATE else ""
    return f"{pair_id}__{class_name}{method}{arm_suffix}"


def render_key_cross(crop_key: str, source_identity: str, video_method: str = VIDEO_METHOD_ANIMATE) -> str:
    """Name of the render that FVRA and FVFA of one pair both use."""
    prefix = RENDER_KEY_PREFIXES[video_method][VIDEO_DRIVER_CROSS]
    return f"{prefix}__{crop_key}__face_{source_identity}"


def render_key_self(crop_key: str, video_method: str = VIDEO_METHOD_ANIMATE) -> str:
    """Name of the control render (the driver as themselves)."""
    prefix = RENDER_KEY_PREFIXES[video_method][VIDEO_DRIVER_SELF]
    return f"{prefix}__{crop_key}"


# --------------------------------------------------------------------------- clip list + split files
def log_dir() -> Path:
    return GEN_ROOT / "_logs"


def report_dir() -> Path:
    return GEN_ROOT / "_reports"


def inventory_shard_log_path(shard_index: int, num_shards: int) -> Path:
    return log_dir() / f"inventory_shard{shard_index}_of_{num_shards}.jsonl"


def inventory_shard_done_path(shard_index: int, num_shards: int) -> Path:
    return log_dir() / f"inventory_shard{shard_index}_of_{num_shards}.done"


def inventory_parquet_path() -> Path:
    return report_dir() / "inventory.parquet"


def split_map_path() -> Path:
    return GEN_ROOT / "split_map.json"


def audio_shard_log_path(shard_index: int, num_shards: int) -> Path:
    return log_dir() / f"audio_shard{shard_index}_of_{num_shards}.jsonl"


def audio_shard_done_path(shard_index: int, num_shards: int) -> Path:
    return log_dir() / f"audio_shard{shard_index}_of_{num_shards}.done"


def audio_wav_path(rel_path: str) -> Path:
    """The extracted wav of a clip-list rel_path (<id>/<video>/<clip>.mp4)."""
    return AUDIO_ROOT / Path(rel_path).with_suffix(".wav")


def ensure_dirs() -> None:
    log_dir().mkdir(parents=True, exist_ok=True)
    report_dir().mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- build files
def new_gen_manifest_dir(build_id: str) -> Path:
    return NEW_GEN_ROOT / "manifests" / build_id


def new_gen_manifest_path(build_id: str) -> Path:
    return new_gen_manifest_dir(build_id) / "manifest.jsonl"


def new_gen_manifest_summary_path(build_id: str) -> Path:
    return new_gen_manifest_dir(build_id) / "manifest_summary.json"


def new_gen_validate_report_path(build_id: str) -> Path:
    return new_gen_manifest_dir(build_id) / "validate_report.json"


def new_gen_crop_dir() -> Path:
    return NEW_GEN_ROOT / "crops"


def crop_video_path(crop_key: str) -> Path:
    return new_gen_crop_dir() / f"{crop_key}.mp4"


def crop_audio_path(crop_key: str) -> Path:
    return new_gen_crop_dir() / f"{crop_key}.wav"


def new_gen_log_dir() -> Path:
    return NEW_GEN_ROOT / "_logs"


def crop_log_path(build_id: str) -> Path:
    return new_gen_log_dir() / f"crop_{build_id}.jsonl"


def crop_done_path(build_id: str) -> Path:
    return new_gen_log_dir() / f"crop_{build_id}.done"


def new_gen_render_dir() -> Path:
    return NEW_GEN_ROOT / "renders"


def new_gen_ref_dir() -> Path:
    return NEW_GEN_ROOT / "refs"


def render_path(render_key: str) -> Path:
    return new_gen_render_dir() / f"{render_key}.mp4"


def ref_image_path(render_key: str) -> Path:
    """The face photo a render starts from."""
    return new_gen_ref_dir() / f"{render_key}.png"


def new_gen_conversion_dir() -> Path:
    return NEW_GEN_ROOT / "conversions"


def new_gen_vocoded_dir() -> Path:
    return NEW_GEN_ROOT / "vocoded"


def new_gen_report_dir() -> Path:
    return NEW_GEN_ROOT / "_reports"


def new_gen_final_dir(class_name: str, root: Path | None = None) -> Path:
    """Folder a class's final clips land in; root is the build's final folder."""
    return (root or (NEW_GEN_ROOT / "final")) / class_name


def final_sample_path(class_name: str, sample_id: str, root: Path | None = None) -> Path:
    """Where the mux writes a final clip."""
    return new_gen_final_dir(class_name, root) / f"{sample_id}{TERMINAL_CONTAINER_EXT}"


def find_final_sample(class_name: str, sample_id: str, root: Path | None = None) -> Path | None:
    """Where a reader finds a final clip, in any accepted container."""
    directory = new_gen_final_dir(class_name, root)
    for ext in SHIPPED_MEDIA_EXTS:
        candidate = directory / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def resolve_final_dir(root: Path | None = None) -> Path:
    """The final folder a step really used, as an absolute path (written into its reports)."""
    return (root or (NEW_GEN_ROOT / "final")).resolve()


def ensure_new_gen_dirs() -> None:
    for d in (
        new_gen_crop_dir(),
        new_gen_render_dir(),
        new_gen_ref_dir(),
        new_gen_conversion_dir(),
        new_gen_vocoded_dir(),
        new_gen_report_dir(),
        new_gen_log_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- dropped renders and voices
def render_refusal_path(render_key: str) -> Path:
    """Written when a generator refused its input (a documented drop)."""
    return new_gen_render_dir() / f"{render_key}.refused.json"


def render_failure_path(render_key: str) -> Path:
    """Written when a generator failed on its input twice (a documented drop)."""
    return new_gen_render_dir() / f"{render_key}.failed.json"


def conversion_failure_path(conversion_key: str) -> Path:
    """Written when Seed-VC failed on its input twice (a documented drop)."""
    return new_gen_conversion_dir() / f"{conversion_key}.failed.json"


def refused_pair_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    """Pairs that lost a render or a voice; such a pair ships in no class."""
    return {
        r["pair_id"]
        for r in rows
        if r.get("pair_id")
        and (
            (
                r.get("render_key")
                and (render_refusal_path(r["render_key"]).exists() or render_failure_path(r["render_key"]).exists())
            )
            or (r.get("conversion_key") and conversion_failure_path(r["conversion_key"]).exists())
        )
    }


# --------------------------------------------------------------------------- lengths and batches
def crop_key_for(driver_clip_id: str, crop_frames: int = CROP_FRAMES) -> str:
    """Crop name: the clip id, plus the length when it is not the default."""
    legacy = driver_clip_id.replace("/", "_")
    return legacy if crop_frames == CROP_FRAMES else f"{legacy}__f{crop_frames}"


def crop_duration_s(crop_frames: int = CROP_FRAMES) -> float:
    return crop_frames / TARGET_FPS


def crop_audio_samples(crop_frames: int = CROP_FRAMES) -> int:
    return crop_frames * SAMPLES_PER_FRAME


def min_driver_clip_duration_s(crop_frames: int = CROP_FRAMES) -> float:
    return CROP_START_S + crop_duration_s(crop_frames) + MIN_DRIVER_CLIP_MARGIN_S


def row_crop_frames(row: dict[str, Any]) -> int:
    """Crop length of a pair-list row (rows without the field use the default)."""
    return int(row.get("crop_frames") or CROP_FRAMES)


def batch_suffix_for(pair_ids: Iterable[str] | None) -> str:
    """'_pairs<N>_<sha8>' for a run on a list of pairs, '' for a whole build.

    Job makers and the runner both use it, so a batch's job lists are always found.
    """
    if pair_ids is None:
        return ""
    ids = sorted(set(pair_ids))
    digest = sha256("\n".join(ids).encode()).hexdigest()[:8]
    return f"_pairs{len(ids)}_{digest}"
