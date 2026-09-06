# SPDX-License-Identifier: Apache-2.0
"""Architecture dispatch for Omni's MLX worker."""

from __future__ import annotations

import pytest

from sglang_omni.model_runner import mlx_model_worker


def test_qwen3_asr_resolves_to_its_runner_factory() -> None:
    pytest.importorskip("mlx.core")
    from sglang_omni.models.qwen3_asr.mlx.runner import make_qwen3_asr_mlx_runner_class

    factory = mlx_model_worker.resolve_mlx_runner_factory(
        "Qwen3ASRForConditionalGeneration"
    )
    assert factory is make_qwen3_asr_mlx_runner_class


def test_moss_tts_local_resolves_to_its_runner_factory() -> None:
    pytest.importorskip("mlx.core")
    from sglang_omni.models.moss_tts_local.mlx.runner import (
        make_moss_tts_local_mlx_runner_class,
    )

    factory = mlx_model_worker.resolve_mlx_runner_factory("MossTTSLocalSGLangModel")
    assert factory is make_moss_tts_local_mlx_runner_class


def test_unregistered_architecture_lists_the_supported_ones() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        mlx_model_worker.resolve_mlx_runner_factory("SomeOtherArch")

    message = str(excinfo.value)
    assert "SomeOtherArch" in message
    assert "Qwen3ASRForConditionalGeneration" in message


def test_missing_architecture_is_rejected() -> None:
    with pytest.raises(NotImplementedError):
        mlx_model_worker.resolve_mlx_runner_factory(None)


def test_registration_adds_an_architecture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mlx_model_worker,
        "_MLX_RUNNER_FACTORIES",
        dict(mlx_model_worker._MLX_RUNNER_FACTORIES),
    )
    mlx_model_worker.register_mlx_runner_factory(
        "FakeArch", f"{__name__}:_fake_runner_factory"
    )

    assert (
        mlx_model_worker.resolve_mlx_runner_factory("FakeArch") is _fake_runner_factory
    )


def _fake_runner_factory() -> type:
    return object
