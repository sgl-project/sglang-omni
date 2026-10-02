# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sglang_omni.models.ming_tts import engine_builder
from sglang_omni.models.ming_tts.engine_builder import (
    MingTtsEngineBuilder,
    ming_tts_uses_mlx,
)
from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner


@pytest.fixture(
    params=[True, False],
    ids=["mlx", "torch_mps"],
)
def mlx_backend(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> bool:
    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(engine_builder, "ming_tts_uses_mlx", lambda: request.param)
    monkeypatch.setattr(current_platform, "is_mps", lambda: True)
    return request.param


@pytest.mark.parametrize(
    "selected,apple,expected",
    [(False, False, False), (True, True, True), (True, False, None), (False, True, False)],
)
def test_backend_selection(
    monkeypatch: pytest.MonkeyPatch, selected: bool, apple: bool, expected: bool | None
) -> None:
    from sglang.srt.hardware_backend.mlx import runtime

    from sglang_omni import platforms

    monkeypatch.setattr(runtime, "use_mlx", lambda: selected)
    monkeypatch.setattr(platforms, "current_platform", SimpleNamespace(is_mps=lambda: apple))
    if expected is None:
        with pytest.raises(ValueError):
            ming_tts_uses_mlx()
    else:
        assert ming_tts_uses_mlx() is expected


def test_builder_defaults(mlx_backend: bool) -> None:
    builder = MingTtsEngineBuilder()
    builder.context_length = 2048
    defaults = builder.generation_defaults(dtype="bfloat16")
    builder.adjust_overrides(defaults)
    assert defaults["max_running_requests"] == 1
    assert defaults["max_total_tokens"] == 2048
    assert defaults["attention_backend"] == "torch_native"
    assert defaults["chunked_prefill_size"] == 0
    if mlx_backend:
        assert builder.get_model_buffer_bs(None) is None
    else:
        model = SimpleNamespace(decode_input_embedding=SimpleNamespace(num_embeddings=1))
        assert builder.get_model_buffer_bs(model) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_running_requests", 2),
        ("disable_cuda_graph", False),
        ("attention_backend", "triton"),
        ("max_total_tokens", 10),
        ("max_prefill_tokens", 10),
        ("chunked_prefill_size", 128),
        ("prefill_attention_backend", "triton"),
        ("decode_attention_backend", "triton"),
        ("speculative_algorithm", "EAGLE"),
    ],
)
@pytest.mark.usefixtures("mlx_backend")
def test_builder_rejects_unsupported_execution(key: str, value: Any) -> None:
    builder = MingTtsEngineBuilder()
    builder.context_length = 2048
    overrides = builder.generation_defaults(dtype="bfloat16")
    overrides[key] = value
    with pytest.raises(ValueError):
        builder.adjust_overrides(overrides)


@pytest.mark.usefixtures("mlx_backend")
def test_builder_rejects_tp() -> None:
    builder = MingTtsEngineBuilder(tp_size=2, nccl_port=12345)
    builder.context_length = 2048
    with pytest.raises(ValueError, match="TP=1"):
        builder.adjust_overrides(builder.generation_defaults(dtype="bfloat16"))


def test_builder_quantization(mlx_backend: bool) -> None:
    builder = MingTtsEngineBuilder()
    builder.context_length = 64
    overrides = builder.generation_defaults(dtype="bfloat16")
    overrides["quantization"] = "mlx_q4"
    if mlx_backend:
        builder.adjust_overrides(overrides)
    else:
        with pytest.raises(ValueError, match="does not support quantization"):
            builder.adjust_overrides(overrides)


def adjust_overrides(key: str, value: Any) -> dict[str, Any]:
    builder = MingTtsEngineBuilder()
    overrides: dict[str, Any] = {
        **builder.generation_defaults(dtype="bfloat16"),
        key: value,
    }
    builder.adjust_overrides(overrides)
    return overrides


def test_ming_tts_abort_callback_resets_runner_state() -> None:
    runner = object.__new__(MingTTSModelRunner)
    runner.request_states = {"req-ming-tts": object()}
    builder = object.__new__(MingTtsEngineBuilder)
    builder.model_runner = runner

    abort_callback = builder.make_abort_callback()
    abort_callback("req-ming-tts")
    abort_callback("req-ming-tts")

    assert runner.request_states == {}


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
    overrides = adjust_overrides(key, value)

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
        adjust_overrides(key, value)


@pytest.mark.parametrize("value", [False, 0, "false", "no", "", None])
def test_ming_tts_accepts_disabled_torch_compile(value: Any) -> None:
    overrides = adjust_overrides("enable_torch_compile", value)

    assert overrides["enable_torch_compile"] is value


@pytest.mark.parametrize("value", [True, 1, "1", "true", " yes ", "on"])
def test_ming_tts_rejects_enabled_torch_compile(value: Any) -> None:
    with pytest.raises(ValueError, match="torch.compile is not currently supported"):
        adjust_overrides("enable_torch_compile", value)
