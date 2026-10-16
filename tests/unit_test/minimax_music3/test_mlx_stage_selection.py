# SPDX-License-Identifier: Apache-2.0
"""Backend-selection contracts for MiniMax Music 3 MLX stages."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from sglang_omni.models.minimax_music3 import stages


def select_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stages, "use_mlx_backend", lambda: True)
    monkeypatch.setattr(stages.current_platform, "is_mps", lambda: True)


def test_create_ar_executor_selects_native_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_mlx(monkeypatch)
    observed = {}

    class Scheduler:
        def __init__(self, model_path, *, revision, serial_offload):
            observed.update(
                model_path=model_path,
                revision=revision,
                serial_offload=serial_offload,
            )

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
        "serial_offload": False,
    }


def test_create_ar_executor_forwards_serial_offload_to_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_mlx(monkeypatch)
    observed = {}

    class Scheduler:
        def __init__(self, model_path, *, revision, serial_offload):
            observed.update(
                model_path=model_path,
                revision=revision,
                serial_offload=serial_offload,
            )

    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.minimax_music3.mlx.ar_scheduler",
        SimpleNamespace(MiniMaxMusic3MlxARScheduler=Scheduler),
    )

    stages.create_ar_executor("artifact", enable_serial_offload=True)

    assert observed["serial_offload"] is True


def test_create_acoustic_executor_selects_native_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_mlx(monkeypatch)
    observed = {}

    class Decoder:
        dtype = "bfloat16"
        dit_steps = 4
        dit_cfg_scale = 1.25

        def __init__(
            self, model_path, *, revision, dit_steps, dit_cfg_scale, serial_offload
        ):
            self.dit_steps = dit_steps
            self.dit_cfg_scale = dit_cfg_scale
            observed.update(
                model_path=model_path,
                revision=revision,
                dit_steps=dit_steps,
                dit_cfg_scale=dit_cfg_scale,
                serial_offload=serial_offload,
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

    assert scheduler.decoder.__class__ is Decoder
    assert observed == {
        "model_path": "mlx-community/MiniMax-Music3-mxfp8",
        "revision": "revision-b",
        "dit_steps": 4,
        "dit_cfg_scale": 1.25,
        "serial_offload": False,
    }


def test_create_acoustic_executor_forwards_serial_offload_to_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_mlx(monkeypatch)
    observed = {}

    class Decoder:
        dtype = "bfloat16"
        dit_steps = 4
        dit_cfg_scale = 1.25

        def __init__(self, model_path, **kwargs):
            del model_path
            observed.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.minimax_music3.mlx.acoustic",
        SimpleNamespace(MiniMaxMusic3MlxAcousticDecoder=Decoder),
    )

    stages.create_dit_dav_executor("artifact", enable_serial_offload=True)

    assert observed["serial_offload"] is True


@pytest.mark.parametrize("option", ["cache_dit", "breakable_cuda_graph"])
def test_mlx_acoustic_rejects_cuda_only_options(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
) -> None:
    select_mlx(monkeypatch)

    with pytest.raises(ValueError, match="unavailable with MLX"):
        stages.create_dit_dav_executor(
            "mlx-community/MiniMax-Music3-mxfp8",
            **{option: True},
        )
