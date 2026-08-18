# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from sglang_omni.models.ming_tts.engine_builder import MingTtsEngineBuilder
from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner


def _adjust_overrides(key: str, value: Any) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        key: value,
    }
    MingTtsEngineBuilder().adjust_overrides(overrides)
    return overrides


def test_ming_tts_abort_callback_resets_runner_state() -> None:
    runner = object.__new__(MingTTSModelRunner)
    runner._request_states = {"req-ming-tts": object()}
    builder = object.__new__(MingTtsEngineBuilder)
    builder._model_runner = runner

    abort_callback = builder.make_abort_callback()
    abort_callback("req-ming-tts")
    abort_callback("req-ming-tts")

    assert runner._request_states == {}


@pytest.mark.parametrize(
    "key",
    ["disable_overlap_schedule", "disable_radix_cache"],
)
@pytest.mark.parametrize(
    "value",
    [True, 1, "1", "true", "True", " yes ", "on"],
)
def test_ming_tts_accepts_affirmative_unsupported_feature_flags(
    key: str, value: Any
) -> None:
    overrides = _adjust_overrides(key, value)

    assert overrides[key] is True


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("disable_overlap_schedule", "does not currently support SGLang overlap"),
        ("disable_radix_cache", "requires disable_radix_cache=true"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [False, 0, "false", "no", "", None, "maybe"],
)
def test_ming_tts_rejects_enabled_unsupported_feature_flags(
    key: str, message: str, value: Any
) -> None:
    with pytest.raises(ValueError, match=message):
        _adjust_overrides(key, value)


@pytest.mark.parametrize("value", [False, 0, "false", "no", "", None])
def test_ming_tts_accepts_disabled_torch_compile(value: Any) -> None:
    overrides = _adjust_overrides("enable_torch_compile", value)

    assert overrides["enable_torch_compile"] is value


@pytest.mark.parametrize("value", [True, 1, "1", "true", " yes ", "on"])
def test_ming_tts_rejects_enabled_torch_compile(value: Any) -> None:
    with pytest.raises(ValueError, match="torch.compile is not currently supported"):
        _adjust_overrides("enable_torch_compile", value)
