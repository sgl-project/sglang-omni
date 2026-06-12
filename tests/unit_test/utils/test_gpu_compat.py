# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import sglang_omni.utils.gpu_compat as gpu_compat


def test_get_gpu_compat_env_defaults_respects_existing_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(gpu_compat._FLASHINFER_USE_CUDA_NORM, "0")
    monkeypatch.setattr(
        gpu_compat,
        "visible_gpus_need_flashinfer_cuda_norm",
        lambda: True,
    )

    assert gpu_compat.get_gpu_compat_env_defaults() == {}


def test_get_gpu_compat_env_defaults_for_blackwell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(gpu_compat._FLASHINFER_USE_CUDA_NORM, raising=False)
    monkeypatch.setattr(
        gpu_compat,
        "visible_gpus_need_flashinfer_cuda_norm",
        lambda: True,
    )

    assert gpu_compat.get_gpu_compat_env_defaults() == {
        gpu_compat._FLASHINFER_USE_CUDA_NORM: "1",
    }


def test_visible_gpus_need_flashinfer_cuda_norm_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_compat, "_visible_gpu_ids", lambda: [0])
    monkeypatch.setattr(
        gpu_compat,
        "get_gpu_device_info",
        lambda _gpu_id: SimpleNamespace(name="NVIDIA B200"),
    )
    monkeypatch.setattr(gpu_compat, "_get_compute_capability", lambda _gpu_id: None)

    assert gpu_compat.visible_gpus_need_flashinfer_cuda_norm() is True


def test_visible_gpus_need_flashinfer_cuda_norm_by_compute_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_compat, "_visible_gpu_ids", lambda: [0])
    monkeypatch.setattr(
        gpu_compat,
        "get_gpu_device_info",
        lambda _gpu_id: SimpleNamespace(name="NVIDIA H200"),
    )
    monkeypatch.setattr(gpu_compat, "_get_compute_capability", lambda _gpu_id: (10, 0))

    assert gpu_compat.visible_gpus_need_flashinfer_cuda_norm() is True


def test_visible_gpus_do_not_need_flashinfer_cuda_norm_on_hopper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_compat, "_visible_gpu_ids", lambda: [0])
    monkeypatch.setattr(
        gpu_compat,
        "get_gpu_device_info",
        lambda _gpu_id: SimpleNamespace(name="NVIDIA H200"),
    )
    monkeypatch.setattr(gpu_compat, "_get_compute_capability", lambda _gpu_id: (9, 0))

    assert gpu_compat.visible_gpus_need_flashinfer_cuda_norm() is False


def test_apply_gpu_compat_env_defaults_sets_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(gpu_compat._FLASHINFER_USE_CUDA_NORM, raising=False)
    monkeypatch.setattr(
        gpu_compat,
        "get_gpu_compat_env_defaults",
        lambda _env=None: {gpu_compat._FLASHINFER_USE_CUDA_NORM: "1"},
    )

    applied = gpu_compat.apply_gpu_compat_env_defaults()

    assert applied == {gpu_compat._FLASHINFER_USE_CUDA_NORM: "1"}
    assert os.environ[gpu_compat._FLASHINFER_USE_CUDA_NORM] == "1"
