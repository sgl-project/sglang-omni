# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA-graph runner for the fixed-shape Whisper encoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class CapturedGraph:
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
        self.encoder = encoder
        self.num_mel_bins = int(num_mel_bins)
        self.input_feature_len = int(input_feature_len)
        parameter = next(encoder.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.min_free_bytes = int(float(min_free_gb) * (1024**3))
        self.warmup_iters = max(int(warmup_iters), 1)
        self.graphs: dict[int, CapturedGraph] = {}
        self.logged_replay_buckets: set[int] = set()

    @property
    def captured_buckets(self) -> tuple[int, ...]:
        """Return the captured request-batch buckets in ascending order."""
        return tuple(sorted(self.graphs))

    def enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self.device)
        return free >= self.min_free_bytes, free

    def warmup(self, static_features: torch.Tensor) -> None:
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_iters):
                self.encoder(static_features)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)

    def capture_bucket(self, batch_size: int) -> None:
        static_features = torch.zeros(
            batch_size,
            self.num_mel_bins,
            self.input_feature_len,
            device=self.device,
            dtype=self.dtype,
        )
        self.warmup(static_features)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            static_output = self.encoder(static_features)
        self.graphs[batch_size] = CapturedGraph(
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
        if not buckets or self.device.type != "cuda":
            return
        else:
            pass
        with torch.cuda.device(self.device):
            for batch_size in reversed(buckets):
                if batch_size in self.graphs:
                    continue
                else:
                    pass
                enough, free = self.enough_free_vram()
                if not enough:
                    logger.warning(
                        "Whisper encoder CUDA graph skipped batch=%d: free VRAM "
                        "%.1f GB is below %.1f GB headroom",
                        batch_size,
                        free / (1024**3),
                        self.min_free_bytes / (1024**3),
                    )
                    continue
                else:
                    pass
                try:
                    self.capture_bucket(batch_size)
                except Exception as exc:
                    logger.warning(
                        "Whisper encoder CUDA graph capture failed for batch=%d: %s; "
                        "using eager execution for that bucket",
                        batch_size,
                        exc,
                    )
                    self.graphs.pop(batch_size, None)

    @torch.no_grad()
    def run(self, input_features: torch.Tensor) -> torch.Tensor:
        """Replay the smallest fitting bucket, or run the encoder eagerly."""
        if input_features.ndim != 3:
            return self.encoder(input_features)
        else:
            pass
        batch_size = int(input_features.shape[0])
        bucket = min((size for size in self.graphs if size >= batch_size), default=None)
        if (
            bucket is None
            or input_features.shape[1] != self.num_mel_bins
            or input_features.shape[2] != self.input_feature_len
        ):
            return self.encoder(input_features)
        else:
            pass
        captured = self.graphs[bucket]
        captured.input_features[:batch_size].copy_(input_features)
        if batch_size < bucket:
            captured.input_features[batch_size:].zero_()
        else:
            pass
        if bucket not in self.logged_replay_buckets:
            logger.info(
                "Replaying Whisper encoder CUDA graph batch=%d request_batch=%d",
                bucket,
                batch_size,
            )
            self.logged_replay_buckets.add(bucket)
        else:
            pass
        captured.graph.replay()
        return captured.output[:batch_size].clone()


__all__ = ["WhisperEncoderCudaGraphRunner"]
