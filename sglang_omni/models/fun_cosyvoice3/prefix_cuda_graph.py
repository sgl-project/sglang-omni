# SPDX-License-Identifier: Apache-2.0
"""The prefix hop's Euler solve replayed from CUDA graphs, one per frame tier."""

from __future__ import annotations

import bisect
import dataclasses
import gc
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import torch
from sglang.srt.utils.common import get_available_gpu_memory

from sglang_omni.models.fun_cosyvoice3.packed_dit import CONV_CONTEXT_FRAMES, PackedDiT
from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    PrefixCacheRow,
    PrefixKVPool,
    PrefixRowAttention,
    PrefixStepLayout,
    forward_prefix,
    run_prefix_solve,
)
from sglang_omni.platforms.device_graph import (
    DeviceGraphBackend,
    DeviceGraphPool,
    ReplayableGraph,
)

logger = logging.getLogger(__name__)

CAPTURE_WARMUP_RUNS = 2


class GraphSolve(Protocol):
    def __call__(self) -> torch.Tensor: ...


@contextmanager
def frozen_gc() -> Iterator[None]:
    # note (ratish): a collection during capture can free what a reference cycle holds,
    # an onnxruntime session among them, and invalidate the graph.
    gc.collect()
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()
        gc.collect()


def capture_solve_graph(
    solve: GraphSolve,
    *,
    backend: DeviceGraphBackend,
    graph_pool: DeviceGraphPool,
    stream: torch.Stream,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> tuple[ReplayableGraph, torch.Tensor]:
    """solve warmed on stream, then captured into graph_pool. Returns the graph
    and the output its replays write."""
    device_module = torch.get_device_module(device)
    stream.wait_stream(device_module.current_stream(device))
    with (
        device_module.stream(stream),
        torch.autocast(device_type=device.type, dtype=autocast_dtype),
    ):
        for _ in range(CAPTURE_WARMUP_RUNS):
            solve()
    device_module.current_stream(device).wait_stream(stream)
    device_module.synchronize(device)
    with (
        backend.capture(
            pool=graph_pool, stream=stream, thread_local_errors=True
        ) as graph,
        torch.autocast(device_type=device.type, dtype=autocast_dtype),
    ):
        output = solve()
    device_module.synchronize(device)
    return graph, output


@dataclass(kw_only=True)
class CapturedPrefixSolve:
    layout: PrefixStepLayout
    graph: ReplayableGraph
    noise: torch.Tensor
    time_span: torch.Tensor
    mu: torch.Tensor
    speaker_embeddings: torch.Tensor
    mel_conditioning: torch.Tensor
    attention: PrefixRowAttention
    context: torch.Tensor
    next_context: torch.Tensor
    output: torch.Tensor


class PrefixCudaGraphRunner:
    def __init__(
        self,
        estimator: PackedDiT,
        pool: PrefixKVPool,
        *,
        backend: DeviceGraphBackend,
        autocast_dtype: torch.dtype,
        frame_dtype: torch.dtype,
        speaker_dtype: torch.dtype,
        cfg_rate: float,
        mel_channels: int,
        speaker_channels: int,
        max_rows: int,
        hop_frames: int,
        min_hop_frames: int,
        max_frames: int,
    ) -> None:
        self.estimator = estimator
        self.pool = pool
        self.backend = backend
        self.device = pool.device
        self.device_module = torch.get_device_module(pool.device)
        self.autocast_dtype = autocast_dtype
        self.frame_dtype = frame_dtype
        self.speaker_dtype = speaker_dtype
        self.cfg_rate = cfg_rate
        self.euler_steps = len(pool.keys)
        self.mel_channels = mel_channels
        self.speaker_channels = speaker_channels
        row_ladder = sorted(
            {
                rows
                for rows in (1, 2, 4, 8, 12, *range(16, max_rows + 1, 8))
                if rows <= max_rows
            }
            | {max_rows}
        )
        self.tier_frames = [rows * hop_frames for rows in row_ladder]
        chunk_size = estimator.chunk_size
        self.layouts: list[PrefixStepLayout] = []
        for frames in self.tier_frames:
            row_slots = min(max_rows, frames // min_hop_frames)
            self.layouts.append(
                PrefixStepLayout(
                    half_frames=frames,
                    row_slots=row_slots,
                    # note(ratish): per CFG half, the frames' chunks, a partial chunk
                    # per row and one padding segment.
                    segment_count=2 * (-(-frames // chunk_size) + row_slots + 1),
                    page_table_width=max_frames,
                )
            )
        self.angles = estimator.rope_angles(max_frames)
        self.captured: list[CapturedPrefixSolve] = []

    @torch.inference_mode()
    def capture(self) -> None:
        """Largest tier first, so the smaller ones reuse its pool memory."""
        started = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device.type, self.device.index)
        graph_pool = self.backend.graph_pool_handle()
        stream = self.device_module.Stream(device=self.device)
        hidden_size = self.estimator.dit.input_embed.proj.out_features
        with frozen_gc():
            for layout in reversed(self.layouts):
                attention = PrefixRowAttention(
                    rows=[],
                    new_frames=[],
                    layout=layout,
                    padding_block=self.pool.padding_block,
                    chunk_size=self.estimator.chunk_size,
                    device=self.device,
                )
                if self.pool.forward is not forward_prefix:
                    attention.mark_dynamic()
                else:
                    pass
                noise = torch.zeros(
                    1,
                    layout.half_frames,
                    self.mel_channels,
                    device=self.device,
                    dtype=self.frame_dtype,
                )
                time_span = torch.zeros(
                    self.euler_steps + 1, device=self.device, dtype=self.frame_dtype
                )
                mu = torch.zeros_like(noise)
                speaker_embeddings = torch.zeros(
                    layout.row_slots,
                    self.speaker_channels,
                    device=self.device,
                    dtype=self.speaker_dtype,
                )
                mel_conditioning = torch.zeros_like(noise)
                context = torch.zeros(
                    self.euler_steps,
                    2 * layout.row_slots,
                    2,
                    CONV_CONTEXT_FRAMES,
                    hidden_size,
                    device=self.device,
                    dtype=self.speaker_dtype,
                )
                next_context = torch.empty_like(context)

                def solve() -> torch.Tensor:
                    return run_prefix_solve(
                        self.estimator,
                        self.pool,
                        noise,
                        time_span,
                        mu,
                        speaker_embeddings,
                        mel_conditioning,
                        attention,
                        self.angles,
                        context,
                        next_context,
                        cfg_rate=self.cfg_rate,
                    )

                graph, output = capture_solve_graph(
                    solve,
                    backend=self.backend,
                    graph_pool=graph_pool,
                    stream=stream,
                    device=self.device,
                    autocast_dtype=self.autocast_dtype,
                )
                self.captured.append(
                    CapturedPrefixSolve(
                        layout=layout,
                        graph=graph,
                        noise=noise,
                        time_span=time_span,
                        mu=mu,
                        speaker_embeddings=speaker_embeddings,
                        mel_conditioning=mel_conditioning,
                        attention=attention,
                        context=context,
                        next_context=next_context,
                        output=output,
                    )
                )
        self.captured.reverse()
        after_mem = get_available_gpu_memory(self.device.type, self.device.index)
        logger.info(
            f"Fun-CosyVoice3 prefix solve graphs captured: tiers={self.tier_frames} "
            f"frames, elapsed={time.perf_counter() - started:.2f} s, "
            f"mem usage={before_mem - after_mem:.2f} GB, avail mem={after_mem:.2f} GB."
        )

    @torch.inference_mode()
    def run(
        self,
        *,
        noise: torch.Tensor,
        time_span: torch.Tensor,
        mu: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        new_frames: list[int],
        caches: list[tuple[PrefixCacheRow, PrefixCacheRow]],
    ) -> torch.Tensor | None:
        """The solve replayed from the smallest tier holding the step's new frames;
        None above the largest tier."""
        frame_count = sum(new_frames)
        tier = bisect.bisect_left(self.tier_frames, frame_count)
        if tier == len(self.tier_frames):
            return None
        else:
            captured = self.captured[tier]
        row_count = len(new_frames)
        row_slots = captured.layout.row_slots
        assert row_count <= row_slots, "every row adds at least the shortest hop"
        assert (
            noise.dtype == self.frame_dtype
            and speaker_embeddings.dtype == self.speaker_dtype
        )
        twin_caches = [pair[0] for pair in caches] + [pair[1] for pair in caches]
        twin_new_frames = list(new_frames) * 2
        chunk_size = self.estimator.chunk_size
        # note(ratish): a segment reads pages only up to its end, a padding one at most a
        # chunk, so columns past this width stay stale and unread.
        page_table_width = max(
            chunk_size,
            *(
                row.committed_frames + new_frame_count
                for row, new_frame_count in zip(
                    twin_caches, twin_new_frames, strict=True
                )
            ),
        )
        attention = PrefixRowAttention(
            rows=twin_caches,
            new_frames=twin_new_frames,
            layout=dataclasses.replace(
                captured.layout, page_table_width=page_table_width
            ),
            padding_block=self.pool.padding_block,
            chunk_size=chunk_size,
            device=self.device,
        )
        contexts: list[torch.Tensor] = []
        for row in twin_caches:
            if row.conv_context is None:
                contexts.append(torch.zeros_like(captured.context[:, 0]))
            else:
                contexts.append(row.conv_context)
        static = captured.attention
        for destination, source in (
            (captured.noise[:, :frame_count], noise),
            (captured.mu[:, :frame_count], mu),
            (captured.mel_conditioning[:, :frame_count], mel_conditioning),
            (captured.speaker_embeddings[:row_count], speaker_embeddings),
            (captured.time_span, time_span),
            (captured.context[:, :row_count], torch.stack(contexts[:row_count], dim=1)),
            (
                captured.context[:, row_slots : row_slots + row_count],
                torch.stack(contexts[row_count:], dim=1),
            ),
            (static.positions, attention.positions),
            (static.conv_output_index, attention.conv_output_index),
            (static.speaker_index, attention.speaker_index),
            (static.extended_index, attention.extended_index),
            (static.tail_index, attention.tail_index),
            (static.write_index, attention.write_index),
            (static.cache_seqlens, attention.cache_seqlens),
            (static.cu_seqlens_q, attention.cu_seqlens_q),
            (static.page_table[:, :page_table_width], attention.page_table),
        ):
            destination.copy_(source)
        captured.graph.replay()
        # note(ratish): the next replay of any tier overwrites the shared pool.
        generated = captured.output[:, :frame_count].clone()
        for index, row in enumerate(twin_caches):
            if index < row_count:
                slot = index
            else:
                slot = row_slots + index - row_count
            row.committed_frames = attention.committed_frames[index]
            row.conv_context = captured.next_context[:, slot].clone()
        return generated
