# SPDX-License-Identifier: Apache-2.0
"""Exact compute-only CUDA graphs for MOSS-TTS-Realtime local decode."""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import torch

logger = logging.getLogger(__name__)


class MossTTSRealtimeLocalCudaGraphReplayError(RuntimeError):
    """A captured local-compute graph could not be replayed safely."""


class _CapturedLocalComputeGraph(NamedTuple):
    graph: torch.cuda.CUDAGraph
    static_input: torch.Tensor
    static_logits: torch.Tensor


class MossTTSRealtimeLocalCudaGraphRunner:
    """Graph local transformer + LM head while exact sampling stays eager."""

    def __init__(
        self,
        local_transformer: Any,
        *,
        batch_sizes: list[int],
        max_batch_size: int,
        warmup_iters: int = 3,
        min_free_gb: float = 0.25,
    ) -> None:
        normalized = sorted({int(batch_size) for batch_size in batch_sizes})
        if not normalized or any(batch_size < 1 for batch_size in normalized):
            raise ValueError("local CUDA graph batch sizes must be positive")
        self.local_transformer = local_transformer
        self.local_model = local_transformer.model
        self.device = local_transformer.device
        self.dtype = local_transformer.dtype
        self.batch_sizes = normalized
        self.max_batch_size = int(max_batch_size)
        self.warmup_iters = int(warmup_iters)
        self.min_free_bytes = int(float(min_free_gb) * (1024**3))
        if self.max_batch_size < max(self.batch_sizes):
            raise ValueError("local CUDA graph batch size exceeds max batch size")
        if self.warmup_iters < 1:
            raise ValueError("local CUDA graph warmup_iters must be positive")
        if self.min_free_bytes < 0:
            raise ValueError("local CUDA graph min_free_gb must be nonnegative")

        self._graphs: dict[tuple[int, int], _CapturedLocalComputeGraph] = {}
        self._graph_pools: dict[int, Any] = {}
        self._disabled = False
        self._replay_total = 0
        self._fallback_total = 0
        self._failure_total = 0

    def _has_free_memory(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self.device)
        return free >= self.min_free_bytes, int(free)

    @torch.no_grad()
    def _capture_batch_size(self, batch_size: int) -> None:
        static_inputs = [
            torch.zeros(
                batch_size,
                self.local_transformer.config.hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.local_transformer.config.rvq)
        ]
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(self.warmup_iters):
                for codebook in range(self.local_transformer.config.rvq):
                    hidden = self.local_model.step(
                        static_inputs[codebook],
                        codebook,
                    )
                    self.local_transformer.local_lm_heads[codebook](hidden)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        graph_pool = torch.cuda.graph_pool_handle()
        self._graph_pools[batch_size] = graph_pool
        for codebook in range(self.local_transformer.config.rvq):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                pool=graph_pool,
                capture_error_mode="thread_local",
            ):
                hidden = self.local_model.step(
                    static_inputs[codebook],
                    codebook,
                )
                static_logits = self.local_transformer.local_lm_heads[codebook](hidden)
            self._graphs[(batch_size, codebook)] = _CapturedLocalComputeGraph(
                graph=graph,
                static_input=static_inputs[codebook],
                static_logits=static_logits,
            )

    @torch.no_grad()
    def warmup(self) -> list[int]:
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return []
        with torch.cuda.device(self.device):
            enough, free = self._has_free_memory()
            if not enough:
                logger.warning(
                    "MOSS-TTS-Realtime local CUDA graph skipped: free VRAM "
                    "%.2f GiB < %.2f GiB headroom",
                    free / 1024**3,
                    self.min_free_bytes / 1024**3,
                )
                return []
            self.local_model._ensure_kv_cache(
                self.max_batch_size,
                device=self.device,
                dtype=self.dtype,
            )
            self.local_model.freeze_kv_cache()
            captured: list[int] = []
            for batch_size in self.batch_sizes:
                try:
                    self._capture_batch_size(batch_size)
                except Exception as exc:
                    self._failure_total += 1
                    for codebook in range(self.local_transformer.config.rvq):
                        self._graphs.pop((batch_size, codebook), None)
                    self._graph_pools.pop(batch_size, None)
                    logger.warning(
                        "MOSS-TTS-Realtime local CUDA graph capture failed for "
                        "batch=%d: %s; using eager fallback",
                        batch_size,
                        exc,
                    )
                    continue
                captured.append(batch_size)
            logger.info(
                "MOSS-TTS-Realtime local compute CUDA graphs captured for bs=%s",
                captured,
            )
            return captured

    def supports(self, batch_size: int) -> bool:
        return not self._disabled and all(
            (int(batch_size), codebook) in self._graphs
            for codebook in range(self.local_transformer.config.rvq)
        )

    def record_fallback(self) -> None:
        self._fallback_total += 1

    @torch.no_grad()
    def compute(self, hidden_states: torch.Tensor, codebook: int) -> torch.Tensor:
        batch_size = int(hidden_states.shape[0])
        entry = self._graphs.get((batch_size, int(codebook)))
        if self._disabled or entry is None:
            raise MossTTSRealtimeLocalCudaGraphReplayError(
                "local CUDA graph is unavailable for the requested shape"
            )
        try:
            entry.static_input.copy_(hidden_states)
            entry.graph.replay()
        except Exception as exc:
            self._disabled = True
            self._failure_total += 1
            raise MossTTSRealtimeLocalCudaGraphReplayError(
                "local CUDA graph replay failed; future frames will use eager"
            ) from exc
        self._replay_total += 1
        return entry.static_logits

    def resource_snapshot(self) -> dict[str, int]:
        captured = sorted(
            {batch_size for batch_size, codebook in self._graphs if codebook == 0}
        )
        return {
            "local_cuda_graph_captured_batch_count": len(captured),
            "local_cuda_graph_max_batch_size": max(captured, default=0),
            "local_cuda_graph_replay_total": self._replay_total,
            "local_cuda_graph_fallback_total": self._fallback_total,
            "local_cuda_graph_failure_total": self._failure_total,
            "local_cuda_graph_disabled": int(self._disabled),
        }
