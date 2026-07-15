# SPDX-License-Identifier: Apache-2.0
"""Per-chunk-count CUDA graph for the MOSS-TD Whisper encoder.

The Whisper encoder is a fixed-shape, stateless pure function: input mel
[num_chunks, num_mel_bins, input_feature_len] -> [num_chunks, encoder_len, d_model].

Only the first dim (chunk count) varies, so we bucket over chunk count and pad
up to the nearest captured bucket on replay.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class WhisperEncoderCudaGraphRunner:
    def __init__(
        self,
        encoder,
        num_mel_bins: int,
        input_feature_len: int,
        min_free_gb: float = 3.0,
        warmup_iters: int = 3,
    ) -> None:
        self._encoder = encoder
        self._num_mel_bins = int(num_mel_bins)
        self._input_feature_len = int(input_feature_len)
        self._device = next(encoder.parameters()).device
        self._dtype = next(encoder.parameters()).dtype
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._warmup_iters = int(warmup_iters)
        self._graphs: dict[int, tuple] = {}
        self._pool = None
        self._forward_batch = None

    def _enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    def _warmup(self, static_feat, static_pos, forward_batch) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._encoder(static_feat, static_pos, forward_batch)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    def _capture_bucket(self, c: int, encoder_len: int, forward_batch) -> None:
        static_feat = torch.zeros(
            c,
            self._num_mel_bins,
            self._input_feature_len,
            device=self._device,
            dtype=self._dtype,
        )
        static_pos = torch.arange(encoder_len, device=self._device, dtype=torch.long)
        self._warmup(static_feat, static_pos, forward_batch)

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=self._pool, capture_error_mode="thread_local"
        ):
            static_out = self._encoder(static_feat, static_pos, forward_batch)
        self._graphs[c] = (graph, static_feat, static_pos, static_out)
        logger.info(
            "Captured MOSS-TD encoder CUDA graph chunks=%d -> out %s (%d cached)",
            c,
            tuple(static_out.shape),
            len(self._graphs),
        )

    @torch.no_grad()
    def capture(self, chunk_buckets, forward_batch=None) -> None:
        """Capture one graph per chunk-count bucket, once, at warmup."""
        self._forward_batch = forward_batch
        encoder_len = (self._input_feature_len - 1) // 2 + 1

        with torch.cuda.device(self._device):
            for c in sorted(
                {int(x) for x in chunk_buckets if int(x) >= 1}, reverse=True
            ):
                if c in self._graphs:
                    continue
                enough, free = self._enough_free_vram()
                if not enough:
                    logger.warning(
                        "MOSS-TD encoder CUDA graph: free VRAM %.1fGB < %.1fGB "
                        "headroom; skipping chunks=%d",
                        free / 1024**3,
                        self._min_free_bytes / 1024**3,
                        c,
                    )
                    continue
                try:
                    self._capture_bucket(c, encoder_len, forward_batch)
                except Exception as exc:
                    logger.warning(
                        "MOSS-TD encoder CUDA graph capture failed for chunks=%d: "
                        "%s; will use a larger captured graph or eager",
                        c,
                        exc,
                    )
                    self._graphs.pop(c, None)

    @torch.no_grad()
    def run(self, input_features, encoder_position_ids, forward_batch):
        """Replay the graph for [n, num_mel_bins, input_feature_len] features,
        padding up to the nearest captured bucket. Falls back to eager if no
        bucket fits or the input_feature_len differs from capture."""
        n = input_features.shape[0]
        chunk_bucket = min((c for c in self._graphs if c >= n), default=None)
        if chunk_bucket is None or input_features.shape[-1] != self._input_feature_len:
            return self._encoder(input_features, encoder_position_ids, forward_batch)
        graph, static_feat, _static_pos, static_out = self._graphs[chunk_bucket]
        static_feat[:n].copy_(input_features)
        if n < chunk_bucket:
            static_feat[n:].zero_()
        graph.replay()
        return static_out[:n].clone()
