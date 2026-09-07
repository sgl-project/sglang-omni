# SPDX-License-Identifier: Apache-2.0
"""Backend-selection contracts for MiniMax Music 3 Apple stages."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.minimax_music3 import stages
from sglang_omni.utils import device as device_utils


def _select_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stages, "_use_mlx_backend", lambda: True)
    monkeypatch.setattr(
        device_utils, "resolve_concrete_device", lambda *_: torch.device("mps:0")
    )


def _select_torch_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stages, "_use_mlx_backend", lambda: False)
    monkeypatch.setattr(
        device_utils, "resolve_concrete_device", lambda *_: torch.device("mps:0")
    )


def test_create_ar_executor_selects_native_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_mlx(monkeypatch)
    observed = {}

    class Scheduler:
        def __init__(self, model_path, *, revision):
            observed.update(model_path=model_path, revision=revision)

    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.minimax_music3.mlx.ar_scheduler",
        SimpleNamespace(MiniMaxMusic3MlxARScheduler=Scheduler),
    )

    scheduler = stages.create_ar_executor(
        "mlx-community/MiniMax-Music3-mxfp8",
        mlx_model_revision="revision-a",
    )

    assert isinstance(scheduler, Scheduler)
    assert observed == {
        "model_path": "mlx-community/MiniMax-Music3-mxfp8",
        "revision": "revision-a",
    }


def test_create_acoustic_executor_selects_native_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_mlx(monkeypatch)
    observed = {}

    class Decoder:
        dtype = "bfloat16"
        dit_steps = 4
        dit_cfg_scale = 1.25

        def __init__(self, model_path, *, revision, dit_steps, dit_cfg_scale):
            self.dit_steps = dit_steps
            self.dit_cfg_scale = dit_cfg_scale
            observed.update(
                model_path=model_path,
                revision=revision,
                dit_steps=dit_steps,
                dit_cfg_scale=dit_cfg_scale,
            )

    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.minimax_music3.mlx.acoustic",
        SimpleNamespace(MiniMaxMusic3MlxAcousticDecoder=Decoder),
    )

    scheduler = stages.create_dit_dav_executor(
        "mlx-community/MiniMax-Music3-mxfp8",
        dit_steps=4,
        dit_cfg_scale=1.25,
        mlx_model_revision="revision-b",
    )

    assert scheduler._decoder.__class__ is Decoder
    assert observed == {
        "model_path": "mlx-community/MiniMax-Music3-mxfp8",
        "revision": "revision-b",
        "dit_steps": 4,
        "dit_cfg_scale": 1.25,
    }


@pytest.mark.parametrize("option", ["cache_dit", "breakable_cuda_graph"])
def test_mlx_acoustic_rejects_cuda_only_options(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
) -> None:
    _select_mlx(monkeypatch)

    with pytest.raises(ValueError, match="unavailable with MLX"):
        stages.create_dit_dav_executor(
            "mlx-community/MiniMax-Music3-mxfp8",
            **{option: True},
        )


def test_create_ar_executor_selects_torch_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_torch_mps(monkeypatch)
    observed = {}

    class Scheduler:
        def __init__(self, model_path, *, revision):
            observed.update(model_path=model_path, revision=revision)

    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.minimax_music3.torch_mps",
        SimpleNamespace(MiniMaxMusic3TorchMpsARScheduler=Scheduler),
    )

    scheduler = stages.create_ar_executor(
        "MiniMaxAI/MiniMax-Music3",
        torch_model_revision="revision-c",
    )

    assert isinstance(scheduler, Scheduler)
    assert observed == {
        "model_path": "MiniMaxAI/MiniMax-Music3",
        "revision": "revision-c",
    }


def test_create_acoustic_executor_selects_torch_mps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _select_torch_mps(monkeypatch)
    observed = {}

    class Decoder:
        dtype = "bfloat16"
        dit_steps = 4
        dit_cfg_scale = 1.25

        def __init__(self, model_path, **kwargs):
            observed.update(model_path=model_path, **kwargs)

    monkeypatch.setattr(stages, "MiniMaxMusic3AcousticDecoder", Decoder)
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.minimax_music3.torch_mps",
        SimpleNamespace(resolve_torch_mps_directory=lambda *_: tmp_path),
    )

    scheduler = stages.create_dit_dav_executor(
        "MiniMaxAI/MiniMax-Music3",
        dit_steps=4,
        dit_cfg_scale=1.25,
        torch_model_revision="revision-d",
    )

    assert scheduler._decoder.__class__ is Decoder
    assert observed == {
        "model_path": str(tmp_path),
        "device": "mps:0",
        "dtype": "bfloat16",
        "dit_steps": 4,
        "dit_cfg_scale": 1.25,
        "attention_backend": "torch_sdpa",
        "cache_dit": False,
        "compile_acoustic": False,
        "breakable_cuda_graph": False,
    }


@pytest.mark.parametrize("option", ["cache_dit", "breakable_cuda_graph"])
def test_torch_mps_acoustic_rejects_cuda_only_options(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
) -> None:
    _select_torch_mps(monkeypatch)

    with pytest.raises(ValueError, match="unavailable with Torch MPS"):
        stages.create_dit_dav_executor(
            "MiniMaxAI/MiniMax-Music3",
            **{option: True},
        )
