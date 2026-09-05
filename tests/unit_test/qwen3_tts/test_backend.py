# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS execution-backend boundaries."""

from __future__ import annotations

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
