# SPDX-License-Identifier: Apache-2.0
"""The prefix hop's Euler solve replayed from CUDA graphs, one per frame tier."""

from __future__ import annotations

import bisect
import dataclasses
from dataclasses import dataclass

import torch

from sglang_omni.models.fun_cosyvoice3.packed_dit import CONV_CONTEXT_FRAMES, PackedDiT
from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    PrefixCacheRow,
    PrefixKVPool,
    PrefixRowAttention,
    PrefixStepLayout,
    forward_prefix,
    run_prefix_solve,
)
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.platforms.device_graph import ReplayableGraph


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
        graphs: SolveGraphCapture,
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
        self.graphs = graphs
        self.device = pool.device
        self.frame_dtype = frame_dtype
        self.speaker_dtype = speaker_dtype
        self.cfg_rate = cfg_rate
        self.euler_steps = len(pool.keys)
        self.mel_channels = mel_channels
        self.speaker_channels = speaker_channels
        # note (ratish): a tier at every whole chunk from the shortest hop to max_rows
        # longest hops. Rows commit whole chunks and prompts pad to a hop, so at the
        # default hop a step is whole chunks; the prefill ladder padded it 6 %.
        chunk_size = estimator.chunk_size
        top_frames = max_rows * hop_frames
        self.tier_frames = [
            *(
                frames
                for frames in range(chunk_size, top_frames, chunk_size)
                if frames >= min_hop_frames
            ),
            top_frames,
        ]
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
        """Largest tier first, so the smaller ones reuse its pool memory. Every tier
        captures over views of one set of input buffers at the largest tier's size."""
        hidden_size = self.estimator.dit.input_embed.proj.out_features
        # note (ratish): replays run one at a time on the vocoder's stream, each copying
        # its inputs in and its outputs out, so the tiers can share their inputs; a set
        # per tier held 1.7 GB of contexts over the prefill ladder's tiers.
        top_tier_frames = max(layout.half_frames for layout in self.layouts)
        max_row_slots = max(layout.row_slots for layout in self.layouts)
        noise_buffer = torch.zeros(
            1,
            top_tier_frames,
            self.mel_channels,
            device=self.device,
            dtype=self.frame_dtype,
        )
        mu_buffer = torch.zeros_like(noise_buffer)
        mel_conditioning_buffer = torch.zeros_like(noise_buffer)
        time_span = torch.zeros(
            self.euler_steps + 1, device=self.device, dtype=self.frame_dtype
        )
        speaker_buffer = torch.zeros(
            max_row_slots,
            self.speaker_channels,
            device=self.device,
            dtype=self.speaker_dtype,
        )
        context_buffer = torch.zeros(
            self.euler_steps,
            2 * max_row_slots,
            2,
            CONV_CONTEXT_FRAMES,
            hidden_size,
            device=self.device,
            dtype=self.speaker_dtype,
        )
        next_context_buffer = torch.empty_like(context_buffer)
        with self.graphs.capture_session("prefix", self.tier_frames):
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
                noise = noise_buffer[:, : layout.half_frames]
                mu = mu_buffer[:, : layout.half_frames]
                mel_conditioning = mel_conditioning_buffer[:, : layout.half_frames]
                speaker_embeddings = speaker_buffer[: layout.row_slots]
                context = context_buffer[:, : 2 * layout.row_slots]
                next_context = next_context_buffer[:, : 2 * layout.row_slots]

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

                graph, output = self.graphs.capture(solve)
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
        # note (ratish): the next replay of any solve graph overwrites the shared pool.
        generated = captured.output[:, :frame_count].clone()
        for index, row in enumerate(twin_caches):
            if index < row_count:
                slot = index
            else:
                slot = row_slots + index - row_count
            row.committed_frames = attention.committed_frames[index]
            row.conv_context = captured.next_context[:, slot].clone()
        return generated
