# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs for the MOSS-TTS-Realtime streaming codec decoder."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, NamedTuple

import torch

logger = logging.getLogger(__name__)


class _CapturedVocoderGraph(NamedTuple):
    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor
    static_lengths: torch.Tensor
    static_audio: torch.Tensor
    static_audio_lengths: torch.Tensor


class MossTTSRealtimeVocoderCudaGraphRunner:
    """Capture one fixed-slot codec graph for each requested frame count."""

    def __init__(
        self,
        codec: Any,
        state_adapter: Any,
        *,
        batch_size: int,
        n_vq: int,
        max_frames: int,
        warmup_iters: int = 3,
        min_free_gb: float = 3.0,
    ) -> None:
        self._codec = codec
        self._state_adapter = state_adapter
        self._batch_size = int(batch_size)
        self._n_vq = int(n_vq)
        self._device = next(codec.parameters()).device
        self._max_frames = int(max_frames)
        self._warmup_iters = int(warmup_iters)
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._graphs: dict[int, _CapturedVocoderGraph] = {}
        self._pool = None
        self._sealed = False

    def _enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    @torch.no_grad()
    def _reset_state(self) -> None:
        self._state_adapter.reset_slots(
            list(range(self._batch_size)),
            batch_size=self._batch_size,
        )

    @torch.no_grad()
    def _capture_frame_count(self, frame_count: int) -> None:
        static_codes = torch.zeros(
            self._n_vq,
            self._batch_size,
            frame_count,
            dtype=torch.long,
            device=self._device,
        )
        static_lengths = torch.full(
            (self._batch_size,),
            frame_count,
            dtype=torch.long,
            device=self._device,
        )
        exec_mask = torch.ones(
            self._batch_size,
            dtype=torch.bool,
            device=self._device,
        )
        self._state_adapter.set_exec_mask(exec_mask)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._codec._decode_frame(static_codes, static_lengths)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self._reset_state()
        self._state_adapter.set_exec_mask(exec_mask)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=self._pool,
            capture_error_mode="thread_local",
        ):
            result = self._codec._decode_frame(static_codes, static_lengths)
            static_audio = result.audio
            static_audio_lengths = result.audio_lengths
        self._graphs[frame_count] = _CapturedVocoderGraph(
            graph=graph,
            static_codes=static_codes,
            static_lengths=static_lengths,
            static_audio=static_audio,
            static_audio_lengths=static_audio_lengths,
        )
        logger.info(
            "Captured MOSS-TTS-Realtime vocoder CUDA graph T=%d (B=%d)",
            frame_count,
            self._batch_size,
        )

    @torch.no_grad()
    def warmup(self, frames: Iterable[int]) -> None:
        if self._sealed:
            logger.warning(
                "MOSS-TTS-Realtime vocoder CUDA graph warmup called after seal"
            )
            return
        with torch.cuda.device(self._device):
            for frame_count in sorted(
                dict.fromkeys(int(frame) for frame in frames),
                reverse=True,
            ):
                if not 1 <= frame_count <= self._max_frames:
                    logger.warning(
                        "Skipping MOSS-TTS-Realtime vocoder CUDA graph T=%d; "
                        "supported range is [1, %d]",
                        frame_count,
                        self._max_frames,
                    )
                    continue
                enough, free = self._enough_free_vram()
                if not enough:
                    logger.warning(
                        "MOSS-TTS-Realtime vocoder CUDA graph has %.1f GiB free "
                        "VRAM below the %.1f GiB headroom; remaining shapes use eager",
                        free / 1024**3,
                        self._min_free_bytes / 1024**3,
                    )
                    break
                try:
                    self._capture_frame_count(frame_count)
                except Exception as exc:
                    self._graphs.pop(frame_count, None)
                    logger.warning(
                        "MOSS-TTS-Realtime vocoder CUDA graph capture failed for "
                        "T=%d: %s; this shape uses eager",
                        frame_count,
                        exc,
                    )
        self._sealed = True
        logger.info(
            "MOSS-TTS-Realtime vocoder CUDA graphs sealed: T=%s",
            self.captured_frames(),
        )

    def captured_frames(self) -> list[int]:
        return sorted(self._graphs)

    @torch.no_grad()
    def decode_step(
        self,
        codes: torch.Tensor,
        exec_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not codes.is_cuda:
            return None
        n_vq, batch_size, frame_count = codes.shape
        if n_vq != self._n_vq or batch_size != self._batch_size:
            return None
        captured = self._graphs.get(int(frame_count))
        if captured is None:
            return None
        self._state_adapter.set_exec_mask(exec_mask)
        captured.static_codes.copy_(codes)
        captured.graph.replay()
        return captured.static_audio, captured.static_audio_lengths
