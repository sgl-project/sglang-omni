# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from sglang_omni.models.ming_tts import CAPABILITIES
from sglang_omni.models.ming_tts.engine_builder import MingTtsEngineBuilder
from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)


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


def test_ming_tts_declares_breakable_prefill_cuda_graph_support() -> None:
    assert CAPABILITIES.supports_breakable_prefill_cuda_graph is True
    assert (
        MingTtsEngineBuilder.supports_breakable_prefill_cuda_graph
        is CAPABILITIES.supports_breakable_prefill_cuda_graph
    )


def test_ming_tts_prefill_graph_policy_requires_explicit_opt_in() -> None:
    builder = object.__new__(MingTtsEngineBuilder)
    builder.context_length = 8192
    defaults = builder.generation_defaults(dtype="bfloat16")
    max_running_requests = defaults.pop("max_running_requests")

    disabled = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        **defaults,
    )
    assert disabled["disable_cuda_graph"] is True
    assert "cuda_graph_backend_prefill" not in disabled
    assert "cuda_graph_bs_prefill" not in disabled

    selected = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides={
            "disable_cuda_graph": False,
            "cuda_graph_backend_prefill": "breakable",
            "cuda_graph_bs_prefill": [128, 256],
        },
        **defaults,
    )
    assert selected["disable_cuda_graph"] is False
    assert selected["cuda_graph_backend_prefill"] == "breakable"
    assert selected["cuda_graph_bs_prefill"] == [128, 256]
    assert selected["cuda_graph_max_bs_prefill"] == 256


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
