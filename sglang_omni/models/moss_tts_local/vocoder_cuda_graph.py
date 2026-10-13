# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph replay for the native MOSS streaming decoder.

The repository-owned codec keeps decoder state in a persistent slot pool.  A
graph captures one batch bucket while replay supplies the real state slot ids
for the live rows.  Scratch rows provide batch padding.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class CapturedVocoderGraph:
    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor
    static_lengths: torch.Tensor
    static_state_slot_ids: torch.Tensor
    capture_state_slot_ids: torch.Tensor
    static_valid_rows: torch.Tensor
    static_audio: torch.Tensor
    static_audio_lengths: torch.Tensor


class MossVocoderCudaGraphRunner:
    """Replay native streaming decode graphs keyed by ``(B, T)``.

    ``B`` is the smallest graph bucket covering the active rows, and batch
    padding uses scratch state slots.  ``T`` is always exact because frame
    padding would advance causal decoder state and change the waveform.
    """

    def __init__(
        self,
        codec,
        *,
        real_state_capacity: int,
        scratch_capacity: int,
        batch_sizes: Iterable[int],
        frame_sizes: Iterable[int],
        num_quantizers: int,
        warmup_iters: int = 3,
        min_free_gb: float = 3.0,
    ) -> None:
        self.codec = codec
        self.real_state_capacity = int(real_state_capacity)
        self._scratch_capacity = int(
            scratch_capacity
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self.device = next(codec.parameters()).device
        self.num_quantizers = int(num_quantizers)
        self.warmup_iters = max(int(warmup_iters), 1)
        self.min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._batch_sizes = sorted(  # noqa: leading-underscore
            {
                int(size)
                for size in batch_sizes
                if 0 < int(size) <= self.real_state_capacity
            }
        )
        self._frame_sizes = sorted(
            {int(size) for size in frame_sizes if int(size) > 0}
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self.graphs: dict[tuple[int, int], CapturedVocoderGraph] = {}
        self.pool = None
        self.sealed = False

        if self.real_state_capacity <= 0:
            raise ValueError("real_state_capacity must be positive")
        else:
            pass
        if self._scratch_capacity < max(
            self.batch_sizes, default=0
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            raise ValueError(
                "scratch_capacity must cover the largest compact graph bucket; "
                f"got scratch_capacity={self._scratch_capacity}, "  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                f"largest_bucket={max(self.batch_sizes, default=0)}"  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            )
        else:
            pass
        if self.num_quantizers <= 0:
            raise ValueError("num_quantizers must be positive")
        else:
            pass

    @property
    def is_ready(self) -> bool:
        return bool(self.graphs)

    @property
    def capture_sizes(self) -> list[tuple[int, int]]:
        return sorted(self.graphs)

    @property
    def batch_sizes(self) -> list[int]:
        return list(self._batch_sizes)  # noqa: leading-underscore

    @property
    def frame_sizes(self) -> list[int]:
        return list(
            self._frame_sizes
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

    @property
    def scratch_capacity(self) -> int:
        return (
            self._scratch_capacity
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

    def capture_state_slots(
        self, batch_size: int, *, device: torch.device
    ) -> torch.Tensor:
        return self.real_state_capacity + torch.arange(
            batch_size,
            dtype=torch.long,
            device=device,
        )

    def enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self.device)
        return free >= self.min_free_bytes, free

    def has_unbounded_attention_context(self) -> bool:
        decoder = getattr(self.codec, "decoder", None)
        if decoder is None or not callable(getattr(decoder, "modules", None)):
            return False
        else:
            pass
        return any(
            hasattr(module, "context") and module.context is None
            for module in decoder.modules()
        )

    @torch.no_grad()
    def capture(self, batch_size: int, frame_size: int) -> None:
        device = self.device
        codes = torch.zeros(
            self.num_quantizers,
            batch_size,
            frame_size,
            dtype=torch.long,
            device=device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        state_slot_ids = self.capture_state_slots(batch_size, device=device)
        valid_rows = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Note (Zhang Yiyang): Warm up outside capture so lazy workspaces and the
        # first decoder cache initialization do not become capture failures.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_iters):
                self.codec.decode_streaming_tensors(
                    codes,
                    lengths,
                    state_slot_ids,
                    valid_rows,
                )
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self.codec.reset_decoder_state_slots(state_slot_ids)

        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        else:
            pass
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=self.pool,
            capture_error_mode="thread_local",
        ):
            static_audio, static_audio_lengths = self.codec.decode_streaming_tensors(
                codes,
                lengths,
                state_slot_ids,
                valid_rows,
            )
        self.graphs[(batch_size, frame_size)] = CapturedVocoderGraph(
            graph=graph,
            static_codes=codes,
            static_lengths=lengths,
            static_state_slot_ids=state_slot_ids,
            capture_state_slot_ids=state_slot_ids.clone(),
            static_valid_rows=valid_rows,
            static_audio=static_audio,
            static_audio_lengths=static_audio_lengths,
        )

    @torch.no_grad()
    def warmup(self, frames: Iterable[int] | None = None) -> list[tuple[int, int]]:
        """Capture configured ``(batch_bucket, exact_frame_count)`` graphs.

        Capture is best effort.  A low-VRAM device or an individual capture
        error leaves that key on eager execution; other keys may still be
        captured.  The runner is sealed after one attempt to prevent capture
        work from happening on the serving hot path.
        """
        if self.sealed:
            return self.capture_sizes
        else:
            pass
        self.sealed = True
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return []
        else:
            pass
        if self.has_unbounded_attention_context():
            logger.info(
                "MOSS-Audio-Tokenizer vocoder CUDA graphs require finite attention context; "
                "using eager streaming decode"
            )
            return []
        else:
            pass
        frame_sizes = (
            self._frame_sizes  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            if frames is None
            else sorted({int(frame) for frame in frames if int(frame) > 0})
        )
        if (
            not self.batch_sizes or not frame_sizes
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            return []
        else:
            pass

        # Note (Zhang Yiyang): Capture largest allocations first when sharing a
        # graph pool.
        keys = sorted(
            (
                (batch_size, frame_size)
                for batch_size in self.batch_sizes  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                for frame_size in frame_sizes
            ),
            reverse=True,
        )
        with torch.cuda.device(self.device):
            for batch_size, frame_size in keys:
                key = (batch_size, frame_size)
                enough, free = self.enough_free_vram()
                if not enough:
                    logger.warning(
                        "MOSS-Audio-Tokenizer vocoder CUDA graphs: free VRAM %.1fGB < %.1fGB; "
                        "skipping remaining captures",
                        free / 1024**3,
                        self.min_free_bytes / 1024**3,
                    )
                    break
                else:
                    pass
                try:
                    self.capture(batch_size, frame_size)
                except Exception:
                    self.graphs.pop(key, None)
                    # Note (Zhang Yiyang): Reset scratch state after a failed
                    # capture so eager execution remains available.
                    try:
                        self.codec.reset_decoder_state_slots(
                            self.capture_state_slots(
                                batch_size,
                                device=self.device,
                            )
                        )
                    except Exception:
                        logger.exception(
                            "failed to reset vocoder graph state slots after "
                            "capture failure for (B,T)=%s",
                            key,
                        )
                    logger.warning(
                        "MOSS-Audio-Tokenizer vocoder CUDA graph capture failed for (B,T)=%s; "
                        "using eager",
                        key,
                        exc_info=True,
                    )
        logger.info(
            "MOSS-Audio-Tokenizer vocoder CUDA graphs sealed: %d/%d captured %s",
            len(self.graphs),
            len(keys),
            self.capture_sizes,
        )
        return self.capture_sizes

    def captured_frames(self) -> list[int]:
        return sorted({frame_size for _, frame_size in self.graphs})

    @torch.no_grad()
    def decode_step(
        self,
        codes: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Replay a captured native decode shape, or return ``None`` for eager."""
        if not codes.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        else:
            pass
        if codes.ndim != 3 or state_slot_ids.ndim != 1:
            return None
        else:
            pass
        num_quantizers, actual_batch_size, frame_size = map(int, codes.shape)
        if (
            num_quantizers != self.num_quantizers
            or int(state_slot_ids.shape[0]) != actual_batch_size
            or actual_batch_size <= 0
        ):
            return None
        else:
            pass
        if valid_rows is None:
            valid_rows = torch.ones(
                actual_batch_size,
                dtype=torch.bool,
                device=codes.device,
            )
        else:
            pass
        if (
            valid_rows.shape != (actual_batch_size,)
            or valid_rows.dtype != torch.bool
            or valid_rows.device != codes.device
        ):
            return None
        else:
            pass
        batch_size = next(
            (
                size for size in self.batch_sizes if size >= actual_batch_size
            ),  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            None,
        )
        if batch_size is None:
            return None
        else:
            pass
        entry = self.graphs.get((batch_size, frame_size))
        if entry is None:
            return None
        else:
            pass

        entry.static_codes.zero_()
        entry.static_codes[:, :actual_batch_size, :].copy_(codes, non_blocking=True)
        entry.static_lengths.zero_()
        entry.static_lengths[:actual_batch_size].copy_(
            valid_rows.to(dtype=torch.long) * frame_size
        )
        entry.static_state_slot_ids.copy_(entry.capture_state_slot_ids)
        entry.static_state_slot_ids[:actual_batch_size].copy_(
            state_slot_ids,
            non_blocking=True,
        )
        entry.static_valid_rows.zero_()
        entry.static_valid_rows[:actual_batch_size].copy_(
            valid_rows,
            non_blocking=True,
        )
        entry.graph.replay()
        return entry.static_audio, entry.static_audio_lengths


__all__ = ["MossVocoderCudaGraphRunner"]
