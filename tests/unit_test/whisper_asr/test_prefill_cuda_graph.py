# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.model_runner.whisper_prefill_cuda_graph_runner import (
    WhisperPrefillCudaGraphRunner,
)


@pytest.mark.parametrize("batch_size", [1])
def test_whisper_prefill_capture_populates_encoder_metadata(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    runner = object.__new__(WhisperPrefillCudaGraphRunner)
    capture_batch = SimpleNamespace(
        batch_size=batch_size,
        encoder_lens=None,
        encoder_lens_cpu=None,
        encoder_cached=None,
        encoder_out_cache_loc=None,
    )
    monkeypatch.setattr(
        WhisperPrefillCudaGraphRunner.__mro__[1],
        "capture_prepare",
        lambda self, num_tokens: (capture_batch, object()),
    )
    runner.device = "cpu"

    batch, _ = runner.capture_prepare(4)

    assert batch.encoder_lens.tolist() == [1] * batch_size
    assert batch.encoder_lens_cpu == [1] * batch_size
    assert batch.encoder_cached == [True] * batch_size
    assert batch.encoder_out_cache_loc is None


@pytest.mark.parametrize("forward_mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_whisper_prefill_replay_preserves_encoder_metadata(
    monkeypatch: pytest.MonkeyPatch,
    forward_mode: ForwardMode,
) -> None:
    runner = object.__new__(WhisperPrefillCudaGraphRunner)
    static_batch = SimpleNamespace(
        encoder_lens_cpu=None,
        encoder_cached=None,
        encoder_out_cache_loc=None,
    )
    live_batch = SimpleNamespace(
        forward_mode=forward_mode,
        encoder_lens_cpu=[7, 5],
        encoder_cached=[False, True],
        encoder_out_cache_loc=torch.tensor([11, 12, 13]),
    )
    monkeypatch.setattr(
        WhisperPrefillCudaGraphRunner.__mro__[1],
        "load_batch",
        lambda self, forward_batch, **kwargs: static_batch,
    )

    replay_batch = runner.load_batch(live_batch)

    assert replay_batch.encoder_lens_cpu == live_batch.encoder_lens_cpu
    assert replay_batch.encoder_cached == live_batch.encoder_cached
    assert replay_batch.encoder_out_cache_loc is live_batch.encoder_out_cache_loc


@pytest.mark.parametrize(
    ("architecture", "backend", "expected"),
    [
        (
            "WhisperForConditionalGeneration",
            "breakable",
            WhisperPrefillCudaGraphRunner,
        ),
        ("WhisperForConditionalGeneration", "disabled", None),
        ("Qwen3ASRForConditionalGeneration", "breakable", None),
    ],
)
def test_model_runner_selects_whisper_prefill_adapter_only_when_needed(
    architecture: str,
    backend: str,
    expected: type[WhisperPrefillCudaGraphRunner] | None,
) -> None:
    runner = object.__new__(SGLModelRunner)
    runner._model_arch_override = architecture
    runner.server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend=backend))
    )

    assert runner._prefill_cuda_graph_runner_cls() is expected


@pytest.mark.parametrize(
    ("architecture", "backend", "expects_override"),
    [
        ("WhisperForConditionalGeneration", "breakable", True),
        ("WhisperForConditionalGeneration", "disabled", False),
    ],
)
def test_model_runner_scopes_whisper_prefill_adapter_override(
    architecture: str,
    backend: str,
    expects_override: bool,
) -> None:
    from sglang.srt.model_executor.model_runner_components import cuda_graph_setup

    runner = object.__new__(SGLModelRunner)
    runner._model_arch_override = architecture
    runner.server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend=backend))
    )
    original = cuda_graph_setup.PrefillCudaGraphRunner

    with runner._prefill_cuda_graph_runner_override():
        expected = WhisperPrefillCudaGraphRunner if expects_override else original
        assert cuda_graph_setup.PrefillCudaGraphRunner is expected

    assert cuda_graph_setup.PrefillCudaGraphRunner is original
