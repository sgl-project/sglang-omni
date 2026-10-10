# SPDX-License-Identifier: Apache-2.0
"""Streaming codec runner: GPU-resident stream histories and CUDA graphs."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterator, Sequence

import torch

from sglang_omni.models.easymagpie_tts.codec import EasyMagpieCodec
from sglang_omni.scheduling.streaming_vocoder import vocoder_decode_stream_priority

logger = logging.getLogger(__name__)

GRAPH_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)


class StreamingCodecRunner:
    """Decode stream chunks against codec histories kept in fixed GPU slots.

    Every stream owns one slot row in each causal layer's history. A step
    gathers the participants' rows, runs the codec and scatters the new
    histories back, so a CUDA graph replays it from just the codes and slot
    ids. Padding rows read and write a scratch slot that no stream owns. On
    CUDA every step runs on a high-priority stream, ahead of talker decode
    kernels queued on the default stream.
    """

    def __init__(self, codec: EasyMagpieCodec, *, max_streams: int) -> None:
        if max_streams < 1:
            raise ValueError("EasyMagpie vocoder max_streams must be positive")
        else:
            pass
        self.codec = codec
        self.device = codec.dequantizer.levels.device
        self.max_streams = int(max_streams)
        self.scratch = self.max_streams
        self.free = list(range(self.max_streams - 1, -1, -1))
        with torch.inference_mode():
            self.histories = codec.empty_stream_state(self.max_streams + 1)
        if self.device.type == "cuda":
            self.stream = torch.cuda.Stream(
                self.device, priority=vocoder_decode_stream_priority(torch.cuda)
            )
        else:
            self.stream = None
        # (frames, batch) -> (graph, static codes, static slots/fresh, audio)
        self.graphs: dict[
            tuple[int, int],
            tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def acquire(self) -> int:
        if not self.free:
            raise RuntimeError(
                f"EasyMagpie vocoder has no free codec slot: all "
                f"{self.max_streams} streaming slots are in use"
            )
        else:
            return self.free.pop()

    def release(self, slot: int) -> None:
        self.free.append(slot)

    @contextlib.contextmanager
    def on_stream(self) -> Iterator[None]:
        with torch.inference_mode():
            if self.stream is None:
                yield
            else:
                with torch.cuda.stream(self.stream):
                    yield

    def step(self, codes: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """``rows`` is [2, batch]: slot ids, then 1 to continue a history or
        0 to start a fresh stream."""
        slots = rows[0]
        fresh = (rows[1] == 0).view(-1, 1, 1)
        state = [
            history.index_select(0, slots).masked_fill(fresh, 0.0)
            for history in self.histories
        ]
        audio, state = self.codec.stream(codes, state)
        for history, new in zip(self.histories, state, strict=True):
            history.index_copy_(0, slots, new)
        return audio

    def decode(
        self, codes: torch.Tensor, slots: Sequence[int], carry: Sequence[bool]
    ) -> torch.Tensor:
        """Decode [batch, frames, stacked] codes into [batch, samples] float32
        CPU audio, advancing each slot's history."""
        batch, frames, _ = codes.shape
        bucket = next((size for size in GRAPH_BATCH_SIZES if size >= batch), batch)
        captured = self.graphs.get((frames, bucket))
        width = batch if captured is None else bucket
        rows = torch.tensor(
            [
                [*slots, *[self.scratch] * (width - batch)],
                [*(int(c) for c in carry), *[0] * (width - batch)],
            ],
            dtype=torch.long,
        )
        with self.on_stream():
            if captured is None:
                audio = self.step(codes.to(self.device), rows.to(self.device))
            else:
                graph, static_codes, static_rows, output = captured
                static_codes[:batch].copy_(codes)
                static_rows.copy_(rows)
                graph.replay()
                audio = output[:batch]
            return audio.float().cpu()

    def capture(self, frames: Sequence[int], max_batch: int) -> None:
        """Capture one graph per chunk size and batch bucket up to
        ``max_batch``; other shapes run eagerly."""
        if self.stream is None:
            return
        else:
            pass
        start = time.perf_counter()
        sizes = [size for size in GRAPH_BATCH_SIZES if size <= max_batch]
        width = self.codec.config.num_stacked_codebooks
        pool = torch.cuda.graph_pool_handle()
        # Largest first, so smaller graphs reuse the shared pool's blocks.
        for chunk in sorted(set(frames), reverse=True):
            for batch in reversed(sizes):
                with self.on_stream():
                    codes = torch.zeros(
                        (batch, chunk, width), dtype=torch.long, device=self.device
                    )
                    rows = torch.zeros((2, batch), dtype=torch.long, device=self.device)
                    rows[0] = self.scratch
                    self.step(codes, rows)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                        output = self.step(codes, rows)
                self.graphs[(chunk, batch)] = (graph, codes, rows, output)
        self.stream.synchronize()
        logger.info(
            "EasyMagpie codec captured %d stream graphs (frames %s, batch <= %d) "
            "in %.1f s",
            len(self.graphs),
            sorted(set(frames)),
            sizes[-1],
            time.perf_counter() - start,
        )


__all__ = ["GRAPH_BATCH_SIZES", "StreamingCodecRunner"]
