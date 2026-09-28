# SPDX-License-Identifier: Apache-2.0
"""Per-chunk-count device graph for the MOSS-TD Whisper encoder.

The Whisper encoder is a fixed-shape, stateless pure function: input mel
[num_chunks, num_mel_bins, input_feature_len] -> [num_chunks, encoder_len, d_model].

Only the first dim (chunk count) varies, so we bucket over chunk count and pad
up to the nearest captured bucket on replay.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.utils.common import get_available_gpu_memory

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend

logger = logging.getLogger(__name__)


class WhisperEncoderCudaGraphRunner:
    def __init__(
        self,
        encoder,
        num_mel_bins: int,
        input_feature_len: int,
        graph_backend: DeviceGraphBackend,
        min_free_gb: float = 3.0,
        warmup_iters: int = 3,
    ) -> None:
        self.encoder = encoder
        self.graph_backend = graph_backend
        self.num_mel_bins = int(num_mel_bins)
        self.input_feature_len = int(input_feature_len)
        self.device = next(encoder.parameters()).device
        self.dtype = next(encoder.parameters()).dtype
        self.device_module = torch.get_device_module(self.device)
        self.min_free_bytes = int(float(min_free_gb) * (1024**3))
        self.warmup_iters = int(warmup_iters)
        self.graphs: dict[int, tuple] = {}
        self.pool = None
        self.capture_stream = self.device_module.Stream(device=self.device)
        self.forward_batch = None

    def enough_free_vram(self) -> tuple[bool, int]:
        free_gib = get_available_gpu_memory(self.device.type, self.device.index)
        free = int(free_gib * (1 << 30))
        return free >= self.min_free_bytes, free

    def warmup(self, static_feat, static_pos, forward_batch) -> None:
        stream = self.capture_stream
        stream.wait_stream(self.device_module.current_stream(self.device))
        with self.device_module.stream(stream):
            for _ in range(self.warmup_iters):
                self.encoder(static_feat, static_pos, forward_batch)
        self.device_module.current_stream(self.device).wait_stream(stream)
        self.device_module.synchronize(self.device)

    def capture_bucket(self, c: int, encoder_len: int, forward_batch) -> None:
        static_feat = torch.zeros(
            c,
            self.num_mel_bins,
            self.input_feature_len,
            device=self.device,
            dtype=self.dtype,
        )
        static_pos = torch.arange(encoder_len, device=self.device, dtype=torch.long)
        with current_platform.graph_capture_attention():
            self.warmup(static_feat, static_pos, forward_batch)
            if self.pool is None:
                self.pool = self.device_module.graph_pool_handle()
            else:
                pass
            with self.graph_backend.capture(
                pool=self.pool,
                stream=self.capture_stream,
                thread_local_errors=True,
            ) as graph:
                static_out = self.encoder(static_feat, static_pos, forward_batch)
        self.graphs[c] = (graph, static_feat, static_pos, static_out)
        logger.info(
            f"Captured MOSS-TD encoder graph chunks={c} -> "
            f"out {tuple(static_out.shape)} ({len(self.graphs)} cached)"
        )

    @torch.no_grad()
    def capture(self, chunk_buckets, forward_batch=None) -> None:
        """Capture one graph per chunk-count bucket, once, at warmup."""
        self.forward_batch = forward_batch
        encoder_len = (self.input_feature_len - 1) // 2 + 1

        with self.device_module.device(self.device):
            for c in sorted(
                {int(x) for x in chunk_buckets if int(x) >= 1}, reverse=True
            ):
                if c in self.graphs:
                    continue
                else:
                    pass
                enough, free = self.enough_free_vram()
                if not enough:
                    logger.warning(
                        f"MOSS-TD encoder graph: free VRAM {free / 1024**3:.1f}GB "
                        f"< {self.min_free_bytes / 1024**3:.1f}GB headroom; "
                        f"skipping chunks={c}"
                    )
                    continue
                else:
                    pass
                try:
                    self.capture_bucket(c, encoder_len, forward_batch)
                except Exception as exc:
                    logger.warning(
                        f"MOSS-TD encoder graph capture failed for chunks={c}: "
                        f"{exc}; will use a larger captured graph or eager"
                    )
                    self.graphs.pop(c, None)

    @torch.no_grad()
    def run(self, input_features, encoder_position_ids, forward_batch):
        """Replay the graph for [n, num_mel_bins, input_feature_len] features,
        padding up to the nearest captured bucket. Falls back to eager if no
        bucket fits or the input_feature_len differs from capture."""
        n = input_features.shape[0]
        chunk_bucket = min((c for c in self.graphs if c >= n), default=None)
        if chunk_bucket is None or input_features.shape[-1] != self.input_feature_len:
            return self.encoder(input_features, encoder_position_ids, forward_batch)
        else:
            pass
        graph, static_feat, _static_pos, static_out = self.graphs[chunk_bucket]
        static_feat[:n].copy_(input_features)
        if n < chunk_bucket:
            static_feat[n:].zero_()
        else:
            pass
        graph.replay()
        return static_out[:n].clone()
