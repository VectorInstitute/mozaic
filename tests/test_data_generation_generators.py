"""Tests for the generator runners' environment: the GPU libraries their child processes get, and offline use."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mozaic.data_generation.generators.animate import run as WA  # noqa: N812
from mozaic.data_generation.generators.facefusion import run as FF  # noqa: N812


def _lib(root: Path, rel: str) -> str:
    path = root / rel
    path.mkdir(parents=True)
    return str(path)


def test_facefusion_gets_the_venv_cuda_libraries_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The venv's nvidia-*-cu12 lib folders go before what LD_LIBRARY_PATH already had."""
    site = "lib/python3.12/site-packages/nvidia"
    cublas, cudnn = _lib(tmp_path, f"{site}/cublas/lib"), _lib(tmp_path, f"{site}/cudnn/lib")
    (tmp_path / site / "README").write_text("")  # not a library folder
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/already/there")
    assert FF.gpu_env()["LD_LIBRARY_PATH"] == f"{cublas}:{cudnn}:/already/there"
    monkeypatch.delenv("LD_LIBRARY_PATH")
    assert FF.gpu_env()["LD_LIBRARY_PATH"] == f"{cublas}:{cudnn}"


def test_facefusion_stops_without_its_models(tmp_path: Path) -> None:
    """No swapper model: a clear stop before any job, never a silent CPU run."""
    with pytest.raises(SystemExit, match="not found"):
        FF.require_gpu(tmp_path / "inswapper_128_fp16.onnx", {})


def test_animate_gets_cuda_torch_and_cuda11_libraries_and_stays_offline(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Order: the venv's nvidia libs, torch's own, the CUDA 11 ones, then the old value; offline unless set."""
    site = "lib/python3.12/site-packages"
    nv, torch = _lib(tmp_path, f"{site}/nvidia/cuda_runtime/lib"), _lib(tmp_path, f"{site}/torch/lib")
    cuda11 = _lib(tmp_path / "cu11", "nvidia/cudnn/lib")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setenv("WA_ONNX_CUDA11_LIBS", str(tmp_path / "cu11"))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/already/there")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    WA.model_env.cache_clear()
    env = WA.model_env()
    WA.model_env.cache_clear()
    assert env["LD_LIBRARY_PATH"] == f"{nv}:{torch}:{cuda11}:/already/there"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "0"  # a value already set is kept
