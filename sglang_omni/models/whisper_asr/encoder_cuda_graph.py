# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA-graph runner for the fixed-shape Whisper encoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class _CapturedGraph:
    graph: torch.cuda.CUDAGraph
    input_features: torch.Tensor
    output: torch.Tensor


class WhisperEncoderCudaGraphRunner:
    """Capture the stateless encoder at selected request-batch sizes.

    Whisper preprocessing pads every request to one fixed mel-frame length, so
    only the leading request batch dimension varies. Inputs between buckets are
    zero-padded and the output is trimmed back to the real request count.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_mel_bins: int,
        input_feature_len: int,
        min_free_gb: float = 3.0,
        warmup_iters: int = 3,
    ) -> None:
        self._encoder = encoder
        self._num_mel_bins = int(num_mel_bins)
        self._input_feature_len = int(input_feature_len)
        parameter = next(encoder.parameters())
        self._device = parameter.device
        self._dtype = parameter.dtype
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._warmup_iters = max(int(warmup_iters), 1)
        self._graphs: dict[int, _CapturedGraph] = {}
        self._logged_replay_buckets: set[int] = set()

    @property
    def captured_buckets(self) -> tuple[int, ...]:
        """Return the captured request-batch buckets in ascending order."""
        return tuple(sorted(self._graphs))

    def _enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    def _warmup(self, static_features: torch.Tensor) -> None:
        stream = torch.cuda.Stream(device=self._device)
        stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._encoder(static_features)
        torch.cuda.current_stream(self._device).wait_stream(stream)
        torch.cuda.synchronize(self._device)

    def _capture_bucket(self, batch_size: int) -> None:
        static_features = torch.zeros(
            batch_size,
            self._num_mel_bins,
            self._input_feature_len,
            device=self._device,
            dtype=self._dtype,
        )
        self._warmup(static_features)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            static_output = self._encoder(static_features)
        self._graphs[batch_size] = _CapturedGraph(
            graph=graph,
            input_features=static_features,
            output=static_output,
        )
        logger.info(
            "Captured Whisper encoder CUDA graph batch=%d output_shape=%s",
            batch_size,
            tuple(static_output.shape),
        )

    @torch.no_grad()
    def capture(self, batch_buckets: list[int] | tuple[int, ...]) -> None:
        """Capture valid buckets, skipping buckets that fail or lack headroom."""
        normalized = {int(value) for value in batch_buckets}
        buckets = sorted(value for value in normalized if value >= 1)
        if not buckets or self._device.type != "cuda":
            return
        with torch.cuda.device(self._device):
            for batch_size in reversed(buckets):
                if batch_size in self._graphs:
                    continue
                enough, free = self._enough_free_vram()
                if not enough:
                    logger.warning(
                        "Whisper encoder CUDA graph skipped batch=%d: free VRAM "
                        "%.1f GB is below %.1f GB headroom",
                        batch_size,
                        free / (1024**3),
                        self._min_free_bytes / (1024**3),
                    )
                    continue
                try:
                    self._capture_bucket(batch_size)
                except Exception as exc:
                    logger.warning(
                        "Whisper encoder CUDA graph capture failed for batch=%d: %s; "
                        "using eager execution for that bucket",
                        batch_size,
                        exc,
                    )
                    self._graphs.pop(batch_size, None)

    @torch.no_grad()
    def run(self, input_features: torch.Tensor) -> torch.Tensor:
        """Replay the smallest fitting bucket, or run the encoder eagerly."""
        if input_features.ndim != 3:
            return self._encoder(input_features)
        batch_size = int(input_features.shape[0])
        bucket = min(
            (size for size in self._graphs if size >= batch_size), default=None
        )
        if (
            bucket is None
            or input_features.shape[1] != self._num_mel_bins
            or input_features.shape[2] != self._input_feature_len
        ):
            return self._encoder(input_features)
        captured = self._graphs[bucket]
        captured.input_features[:batch_size].copy_(input_features)
        if batch_size < bucket:
            captured.input_features[batch_size:].zero_()
        if bucket not in self._logged_replay_buckets:
            logger.info(
                "Replaying Whisper encoder CUDA graph batch=%d request_batch=%d",
                bucket,
                batch_size,
            )
            self._logged_replay_buckets.add(bucket)
        captured.graph.replay()
        return captured.output[:batch_size].clone()


__all__ = ["WhisperEncoderCudaGraphRunner"]
