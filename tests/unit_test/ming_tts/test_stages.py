# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from typing import Any

import pytest


def test_ming_tts_audio_decode_factory_exposes_only_supported_contract() -> None:
    from sglang_omni.models.ming_tts.config import (
        MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
        MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
    )
    from sglang_omni.models.ming_tts.stages import create_audio_decode_executor

    parameters = inspect.signature(create_audio_decode_executor).parameters

    assert "decode_mode" not in parameters
    assert (
        parameters["initial_chunk_patches"].default
        == MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
    )
    assert (
        parameters["steady_chunk_patches"].default
        == MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES
    )
    assert parameters["max_batch_size"].default == 1
    assert parameters["max_batch_wait_ms"].default == 0


def test_ming_tts_legacy_engine_factory_alias_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.ming_tts import stages

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sentinel = object()

    def fake_factory(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(stages, "create_sglang_tts_engine_executor", fake_factory)

    result = stages.create_tts_engine_executor("model", gpu_id=2)

    assert result is sentinel
    assert calls == [(("model",), {"gpu_id": 2})]


@pytest.mark.parametrize(
    ("factory_args", "error"),
    [
        ({"max_batch_size": 2}, "max_batch_size=1 only"),
        ({"max_batch_wait_ms": 1}, "max_batch_wait_ms=0 only"),
    ],
)
def test_ming_tts_audio_decode_factory_rejects_batch_config_before_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    factory_args: dict[str, int],
    error: str,
) -> None:
    from sglang_omni.models.ming_tts import stages

    def fail_if_called(model_path: str) -> str:
        raise AssertionError(f"unexpected checkpoint resolution for {model_path}")

    monkeypatch.setattr(stages, "_resolve_checkpoint", fail_if_called)

    with pytest.raises(ValueError, match=error):
        stages.create_audio_decode_executor("unused", **factory_args)


@pytest.mark.parametrize("field", ["initial_chunk_patches", "steady_chunk_patches"])
@pytest.mark.parametrize("value", [True, 1.5, "2", 0, -1])
def test_audio_decode_factory_rejects_invalid_cadence_before_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    from sglang_omni.models.ming_tts import stages

    def fail_if_called(model_path: str) -> str:
        raise AssertionError(f"unexpected checkpoint resolution for {model_path}")

    monkeypatch.setattr(stages, "_resolve_checkpoint", fail_if_called)

    with pytest.raises(
        ValueError,
        match=f"{field} must be a positive integer",
    ):
        stages.create_audio_decode_executor(
            "unused",
            **{field: value},
        )
