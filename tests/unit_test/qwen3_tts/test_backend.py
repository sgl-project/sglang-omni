# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS execution-backend boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.qwen3_tts.backend import Qwen3TTSBackend, get_qwen3_tts_backend


def test_non_mlx_execution_keeps_the_canonical_torch_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.utils import tensor_bridge

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)

    # CPU, CUDA, and a future MPS path are devices within this backend.
    assert get_qwen3_tts_backend() is Qwen3TTSBackend.TORCH


def test_mlx_opt_in_selects_the_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang.srt.utils import tensor_bridge

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: True)

    assert get_qwen3_tts_backend() is Qwen3TTSBackend.MLX


def test_mlx_defaults_disable_torch_only_optimizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.utils import tensor_bridge

    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: True)

    defaults = Qwen3TtsEngineBuilder().generation_defaults(dtype="bfloat16")

    assert defaults["disable_cuda_graph"] is True
    assert defaults["disable_radix_cache"] is True
    assert defaults["chunked_prefill_size"] == -1
    assert defaults["enable_torch_compile"] is False


def test_torch_mps_uses_eager_native_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang.srt.utils import tensor_bridge

    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)
    builder = Qwen3TtsEngineBuilder()
    builder.device = "mps"

    defaults = builder.generation_defaults(dtype="float16")

    assert defaults == {
        "max_running_requests": 1,
        "max_queued_requests": 1,
        "dtype": "float16",
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "enable_torch_compile": False,
        "context_length": 2048,
        "max_total_tokens": 2048,
        "max_prefill_tokens": 2048,
        "chunked_prefill_size": -1,
        "attention_backend": "torch_native",
        "sampling_backend": "pytorch",
        "trust_remote_code": True,
    }

    builder.adjust_overrides(defaults)
    assert builder.context_length == 2048
    assert "context_length" not in defaults


def test_torch_mps_rejects_more_than_one_running_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.utils import tensor_bridge

    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)
    builder = Qwen3TtsEngineBuilder()
    builder.device = "mps"

    with pytest.raises(ValueError, match="max_running_requests=1"):
        builder.validate_before_infrastructure(SimpleNamespace(max_running_requests=2))
