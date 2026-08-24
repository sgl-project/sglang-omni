# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from transformers import WhisperConfig

from sglang_omni.models.arkasr.audio_lengths import arkasr_num_audio_tokens
from sglang_omni.models.arkasr.audio_tower import ArkAudioMLPAdapter
from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig
from sglang_omni.models.arkasr.encoder_cuda_graph import (
    ArkasrEncoderCudaGraphRunner,
    _bucket_batch,
    _bucket_frames,
)
from sglang_omni.models.arkasr.encoder_service import ArkasrPreLMEncoderService
from sglang_omni.models.arkasr.sglang_model import ArkasrForConditionalGeneration

_MEL_BINS = 8
_HIDDEN_SIZE = 4
_MERGE_FACTOR = 4
_STARTUP_GRAPH_KEYS = {
    (batch, frames) for batch in (1, 2, 4, 8) for frames in (512, 768, 1024)
}


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [
        (1, 1),
        (2, 2),
        (3, 4),
        (5, 8),
        (9, None),
    ],
)
def test_bucket_batch_rounds_up_to_configured_bucket(
    batch_size: int,
    expected: int | None,
) -> None:
    assert _bucket_batch(batch_size) == expected


@pytest.mark.parametrize(
    ("mel_frames", "expected"),
    [
        (1, 512),
        (512, 512),
        (513, 768),
        (769, 1024),
        (1025, None),
    ],
)
def test_bucket_frames_selects_captured_bucket(
    mel_frames: int, expected: int | None
) -> None:
    assert _bucket_frames(mel_frames) == expected


def test_run_falls_back_for_non_precaptured_bucket_without_capture() -> None:
    runner = object.__new__(ArkasrEncoderCudaGraphRunner)
    runner._graphs = {}

    result = runner.run(torch.zeros(1, _MEL_BINS, 1100), [1100])

    assert result is None


def _precapture_runner(
    *,
    enough_free_vram: bool = True,
) -> ArkasrEncoderCudaGraphRunner:
    runner = object.__new__(ArkasrEncoderCudaGraphRunner)
    runner._mel_bins = _MEL_BINS
    runner._device = torch.device("cpu")
    runner._min_free_bytes = 3 * 1024**3
    runner._graphs = {}
    runner._lock = threading.Lock()
    runner._enough_free_vram = MagicMock(
        return_value=(enough_free_vram, 4 * 1024**3 if enough_free_vram else 0)
    )
    runner._capture = MagicMock(side_effect=lambda batch, frames: (batch, frames))
    return runner


def test_capture_startup_buckets_captures_fixed_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _precapture_runner()
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    runner.capture_startup_buckets()

    assert set(runner._graphs) == _STARTUP_GRAPH_KEYS
    assert {
        (call.args[0], call.args[1]) for call in runner._capture.call_args_list
    } == _STARTUP_GRAPH_KEYS


def test_capture_startup_buckets_stops_on_low_vram() -> None:
    runner = _precapture_runner(enough_free_vram=False)

    runner.capture_startup_buckets()

    assert runner._graphs == {}
    runner._capture.assert_not_called()
    runner._enough_free_vram.assert_called_once()


def test_replay_failure_removes_graph_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = MagicMock()
    graph.replay.side_effect = RuntimeError("replay failed")
    runner = object.__new__(ArkasrEncoderCudaGraphRunner)
    runner._device = torch.device("cpu")
    runner._lock = threading.Lock()
    runner._done_event = MagicMock()
    runner._event_recorded = False
    runner._graphs = {
        (1, 512): SimpleNamespace(
            graph=graph,
            features=torch.zeros(1, _MEL_BINS, 512),
            mask=torch.zeros(1, 512, dtype=torch.bool),
            frame_index=torch.arange(512).unsqueeze(0),
            output=torch.zeros(1, 128, _HIDDEN_SIZE),
        )
    }
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())

    features = torch.zeros(1, _MEL_BINS, 17)
    assert runner.run(features, [17]) is None
    assert runner.run(features, [17]) is None
    assert runner._graphs == {}
    graph.replay.assert_called_once()


class _RecordingAudioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.merge_factor = _MERGE_FACTOR
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...] | None]] = []

    @property
    def dtype(self) -> torch.dtype:
        return self.param.dtype

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls.append(
            (
                tuple(features.shape),
                None if attention_mask is None else tuple(attention_mask.shape),
            )
        )
        batch_size = features.shape[0]
        token_count = arkasr_num_audio_tokens(
            features.shape[-1],
            self.merge_factor,
        )
        rows = [
            torch.full(
                (token_count, _HIDDEN_SIZE),
                float(batch_index + 10),
                dtype=features.dtype,
                device=features.device,
            )
            for batch_index in range(batch_size)
        ]
        return torch.stack(rows, dim=0)


def _model_with(
    audio_encoder: nn.Module | None = None,
    graph_runner: object | None = None,
) -> ArkasrForConditionalGeneration:
    model = object.__new__(ArkasrForConditionalGeneration)
    nn.Module.__init__(model)
    model.audio_encoder = audio_encoder or _RecordingAudioEncoder()
    model.encoder_max_batch_size = model.DEFAULT_ENCODER_MAX_BATCH_SIZE
    if graph_runner is not None:
        model.encoder_cuda_graph_runner = graph_runner
    return model


def _item(num_frames: int, *, hash_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        hash=hash_id,
        feature=torch.randn(1, _MEL_BINS, num_frames),
        feature_attention_mask=torch.ones(1, num_frames, dtype=torch.long),
        num_audio_tokens=arkasr_num_audio_tokens(num_frames, _MERGE_FACTOR),
    )


def test_get_audio_feature_routes_through_graph_runner() -> None:
    observed: dict[str, object] = {}

    class _Runner:
        def run(self, features: torch.Tensor, lengths: list[int]) -> torch.Tensor:
            observed["features_shape"] = tuple(features.shape)
            observed["lengths"] = list(lengths)
            token_count = arkasr_num_audio_tokens(features.shape[-1], _MERGE_FACTOR)
            return torch.stack(
                [
                    torch.full(
                        (token_count, _HIDDEN_SIZE),
                        float(batch_index + 1),
                        dtype=features.dtype,
                        device=features.device,
                    )
                    for batch_index in range(features.shape[0])
                ],
                dim=0,
            )

    model = _model_with(graph_runner=_Runner())
    lengths = [17, 9]

    output = model.get_audio_feature(
        [_item(length, hash_id=index + 1) for index, length in enumerate(lengths)]
    )

    assert observed == {
        "features_shape": (2, _MEL_BINS, 17),
        "lengths": lengths,
    }
    assert model.audio_encoder.calls == []
    expected_rows = sum(
        arkasr_num_audio_tokens(length, _MERGE_FACTOR) for length in lengths
    )
    assert output.shape == (expected_rows, _HIDDEN_SIZE)
    first_rows = arkasr_num_audio_tokens(lengths[0], _MERGE_FACTOR)
    assert torch.equal(output[:first_rows], torch.ones(first_rows, _HIDDEN_SIZE))
    assert torch.equal(
        output[first_rows:],
        torch.full(
            (arkasr_num_audio_tokens(lengths[1], _MERGE_FACTOR), _HIDDEN_SIZE),
            2.0,
        ),
    )


def test_get_audio_feature_falls_back_to_eager_when_graph_runner_declines() -> None:
    class _DecliningRunner:
        def run(self, features: torch.Tensor, lengths: list[int]) -> None:
            del features, lengths
            return None

    model = _model_with(graph_runner=_DecliningRunner())
    lengths = [17, 9]

    output = model.get_audio_feature(
        [_item(length, hash_id=index + 1) for index, length in enumerate(lengths)]
    )

    assert model.audio_encoder.calls == [((2, _MEL_BINS, 17), (2, 17))]
    assert output.shape == (
        sum(arkasr_num_audio_tokens(length, _MERGE_FACTOR) for length in lengths),
        _HIDDEN_SIZE,
    )


def test_pre_lm_service_splits_graph_backed_flat_embeddings() -> None:
    class _Runner:
        def run(self, features: torch.Tensor, lengths: list[int]) -> torch.Tensor:
            del lengths
            token_count = arkasr_num_audio_tokens(features.shape[-1], _MERGE_FACTOR)
            return torch.stack(
                [
                    torch.full(
                        (token_count, _HIDDEN_SIZE),
                        float(batch_index + 3),
                        dtype=features.dtype,
                        device=features.device,
                    )
                    for batch_index in range(features.shape[0])
                ],
                dim=0,
            )

    lengths = [17, 9]
    model = _model_with(graph_runner=_Runner())
    items = [_item(length, hash_id=index + 1) for index, length in enumerate(lengths)]
    embedding = model.get_audio_feature(items)

    service = object.__new__(ArkasrPreLMEncoderService)
    service._hidden_size = _HIDDEN_SIZE
    service._dtype = torch.float32

    parts = service.split_embeddings(items, embedding)

    assert [tuple(part.shape) for part in parts] == [
        (arkasr_num_audio_tokens(length, _MERGE_FACTOR), _HIDDEN_SIZE)
        for length in lengths
    ]
    assert torch.equal(parts[0], torch.full_like(parts[0], 3.0))
    assert torch.equal(parts[1], torch.full_like(parts[1], 4.0))
    embedding.zero_()
    assert torch.equal(parts[0], torch.full_like(parts[0], 3.0))
    assert torch.equal(parts[1], torch.full_like(parts[1], 4.0))


def _tiny_config() -> ArkasrConfig:
    whisper = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=_MEL_BINS,
        max_source_positions=512,
    )
    return ArkasrConfig(
        whisper_config=whisper,
        merge_factor=_MERGE_FACTOR,
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=256,
        audio_token_id=151663,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_graph_matches_eager_embeddings_on_mixed_mel_lengths() -> None:
    torch.manual_seed(123)
    model = _model_with(ArkAudioMLPAdapter(_tiny_config()).eval().cuda())
    lengths = [9, 18, 25]
    items = [_item(length, hash_id=index + 1) for index, length in enumerate(lengths)]

    with torch.inference_mode():
        eager = model.get_audio_feature(items)
        model.encoder_cuda_graph_runner = ArkasrEncoderCudaGraphRunner(
            model.audio_encoder,
            min_free_gb=0.0,
            warmup_iters=1,
        )
        model.encoder_cuda_graph_runner.capture_startup_buckets()
        graph = model.get_audio_feature(items)
        torch.cuda.synchronize()

    assert torch.allclose(graph, eager, atol=1e-4, rtol=1e-4)
    assert model.encoder_cuda_graph_runner._graphs
