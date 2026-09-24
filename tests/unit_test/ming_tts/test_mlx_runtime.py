# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sglang_omni.models.ming_tts.apple_runtime import MingTtsMlxEngineBuilder, ming_tts_uses_mlx


@pytest.mark.parametrize("selected,apple,expected", [
    (False, False, False), (True, True, True), (True, False, None), (False, True, None),
])
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


def test_mlx_builder_defaults() -> None:
    builder = MingTtsMlxEngineBuilder()
    builder.context_length = 2048
    defaults = builder.generation_defaults(dtype="bfloat16")
    builder.adjust_overrides(defaults)
    assert defaults["max_running_requests"] == 1
    assert defaults["max_total_tokens"] == 2048
    assert defaults["attention_backend"] == "torch_native"
    assert defaults["chunked_prefill_size"] == 0
    assert builder.get_model_buffer_bs(None) is None


@pytest.mark.parametrize("key,value", [
    ("max_running_requests", 2), ("disable_cuda_graph", False),
    ("attention_backend", "triton"), ("max_total_tokens", 10),
    ("max_prefill_tokens", 10), ("disable_overlap_schedule", False),
    ("disable_radix_cache", False), ("chunked_prefill_size", 128),
    ("prefill_attention_backend", "triton"),
    ("decode_attention_backend", "triton"), ("speculative_algorithm", "EAGLE"),
])
def test_mlx_builder_rejects_unsupported_execution(key: str, value: Any) -> None:
    builder = MingTtsMlxEngineBuilder()
    builder.context_length = 2048
    overrides = builder.generation_defaults(dtype="bfloat16")
    overrides[key] = value
    with pytest.raises(ValueError):
        builder.adjust_overrides(overrides)


def test_mlx_builder_rejects_tp() -> None:
    builder = MingTtsMlxEngineBuilder(tp_size=2, nccl_port=12345)
    builder.context_length = 2048
    with pytest.raises(ValueError, match="TP=1"):
        builder.adjust_overrides(builder.generation_defaults(dtype="bfloat16"))


def test_engine_stage_dispatches_to_mlx_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.ming_tts import apple_runtime, stages

    monkeypatch.setattr(apple_runtime, "ming_tts_uses_mlx", lambda: True)
    calls = []

    def build(self: Any, model_path: str, **kwargs: Any) -> str:
        calls.append((model_path, self.requested_context_length, kwargs))
        return "scheduler"

    monkeypatch.setattr(MingTtsMlxEngineBuilder, "build", build)
    assert stages.create_sglang_tts_engine_executor("local-model", context_length=2048) == "scheduler"
    assert calls[0][:2] == ("local-model", 2048)


def test_audio_stage_dispatches_without_importing_torch_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.ming_tts import apple_runtime, stages
    from sglang_omni.models.ming_tts.mlx import stages as mlx_stages

    monkeypatch.setattr(apple_runtime, "ming_tts_uses_mlx", lambda: True)
    calls = []

    def create(model_path: str, **kwargs: Any) -> str:
        calls.append((model_path, kwargs))
        return "audio-scheduler"

    monkeypatch.setattr(mlx_stages, "create_mlx_audio_decode_executor", create)
    assert stages.create_audio_decode_executor("local-model") == "audio-scheduler"
    assert calls[0][1]["initial_chunk_patches"] == 2
    with pytest.raises(ValueError, match="streaming_cuda_graph=false"):
        stages.create_audio_decode_executor("local-model", streaming_cuda_graph=True)
