# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch

from sglang_omni.models.whisper_asr.encoder_cuda_graph import (
    WhisperEncoderCudaGraphRunner,
    _CapturedGraph,
)


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls: list[tuple[int, ...]] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(features.shape))
        return features.sum(dim=1, keepdim=False) * self.weight


class _Graph:
    def __init__(
        self,
        encoder: _Encoder,
        static_features: torch.Tensor,
        static_output: torch.Tensor,
    ) -> None:
        self._encoder = encoder
        self._static_features = static_features
        self._static_output = static_output
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self._static_output.copy_(self._encoder(self._static_features))


def test_run_uses_smallest_bucket_and_zeroes_padding(caplog) -> None:
    encoder = _Encoder()
    runner = WhisperEncoderCudaGraphRunner(
        encoder,
        num_mel_bins=2,
        input_feature_len=5,
    )
    static_features = torch.full((4, 2, 5), 99.0)
    static_output = torch.empty((4, 5))
    graph = _Graph(encoder, static_features, static_output)
    runner._graphs[4] = _CapturedGraph(
        graph=graph,
        input_features=static_features,
        output=static_output,
    )

    features = torch.arange(30, dtype=torch.float32).reshape(3, 2, 5)
    with caplog.at_level(
        logging.INFO,
        logger="sglang_omni.models.whisper_asr.encoder_cuda_graph",
    ):
        output = runner.run(features)
        runner.run(features)

    assert graph.replays == 2
    assert runner.captured_buckets == (4,)
    assert torch.equal(output, features.sum(dim=1))
    assert torch.count_nonzero(static_features[3]) == 0
    assert output.shape == (3, 5)
    assert output.dtype == features.dtype
    assert output.device == features.device
    assert output.data_ptr() != static_output.data_ptr()
    output.zero_()
    assert torch.count_nonzero(static_output) > 0
    replay_logs = [
        record for record in caplog.records if record.message.startswith("Replaying")
    ]
    assert len(replay_logs) == 1
    assert "batch=4 request_batch=3" in replay_logs[0].message


def test_run_falls_back_for_uncaptured_or_wrong_feature_shape() -> None:
    encoder = _Encoder()
    runner = WhisperEncoderCudaGraphRunner(
        encoder,
        num_mel_bins=2,
        input_feature_len=5,
    )
    features = torch.ones((2, 2, 6))

    output = runner.run(features)

    assert encoder.calls == [(2, 2, 6)]
    assert torch.equal(output, features.sum(dim=1))

    wrong_rank = torch.ones((2, 5))
    output = runner.run(wrong_rank)

    assert encoder.calls[-1] == (2, 5)
    assert torch.equal(output, wrong_rank.sum(dim=1))


def test_capture_is_noop_for_cpu_encoder() -> None:
    runner = WhisperEncoderCudaGraphRunner(
        _Encoder(),
        num_mel_bins=2,
        input_feature_len=5,
    )

    runner.capture([1, 2])

    assert runner.captured_buckets == ()
