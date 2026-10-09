# SPDX-License-Identifier: Apache-2.0
"""The Euler solve over each row's whole history replayed from CUDA graphs, one per
frame tier."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    PackedRows,
    pack_rows,
    solve_flow_euler_packed,
)
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import (
    SolveGraphCapture,
    replay_in_groups,
)
from sglang_omni.platforms.device_graph import ReplayableGraph


@dataclass(kw_only=True)
class CapturedWholeHistorySolve:
    tier_frames: int
    graph: ReplayableGraph
    noise: torch.Tensor
    time_span: torch.Tensor
    mu: torch.Tensor
    speaker_embeddings: torch.Tensor
    mel_conditioning: torch.Tensor
    twin_rows: PackedRows
    output: torch.Tensor


class WholeHistoryCudaGraphRunner:
    def __init__(
        self,
        estimator: PackedDiT,
        *,
        graphs: SolveGraphCapture,
        frame_dtype: torch.dtype,
        speaker_dtype: torch.dtype,
        cfg_rate: float,
        euler_steps: int,
        mel_channels: int,
        speaker_channels: int,
        max_rows: int,
        tier_frames: Sequence[int],
    ) -> None:
        self.estimator = estimator
        self.graphs = graphs
        self.device = graphs.device
        self.frame_dtype = frame_dtype
        self.speaker_dtype = speaker_dtype
        self.cfg_rate = cfg_rate
        self.euler_steps = euler_steps
        self.mel_channels = mel_channels
        self.speaker_channels = speaker_channels
        self.max_rows = max_rows
        self.tier_frames = sorted(tier_frames)
        # note (ratish): positions restart in every row, and no row a replay holds, the
        # padding row included, is longer than the largest tier.
        self.angles = estimator.rope_angles(self.tier_frames[-1])
        self.captured: list[CapturedWholeHistorySolve] = []

    def slot_lengths(self, lengths: Sequence[int], tier_frames: int) -> list[int]:
        """Frames per CFG half of a step's rows, empty rows up to the row slots,
        then one padding row over the rest of the tier."""
        return [
            *lengths,
            *[0] * (self.max_rows - len(lengths)),
            tier_frames - sum(lengths),
        ]

    @torch.inference_mode()
    def capture(self) -> None:
        """Largest tier first, so the smaller ones reuse its pool memory."""
        with self.graphs.capture_session("whole history", self.tier_frames):
            for tier_frames in reversed(self.tier_frames):
                twin_rows = pack_rows(
                    self.slot_lengths([], tier_frames) * 2, self.device
                )
                # note (ratish): captured over one padding row as wide as the tier, so the
                # baked widest row bounds every replay.
                attention = self.estimator.row_attention(
                    twin_rows, streaming=False, dtype=self.speaker_dtype
                )
                noise = torch.zeros(
                    1,
                    tier_frames,
                    self.mel_channels,
                    device=self.device,
                    dtype=self.frame_dtype,
                )
                time_span = torch.zeros(
                    self.euler_steps + 1, device=self.device, dtype=self.frame_dtype
                )
                mu = torch.zeros_like(noise)
                mel_conditioning = torch.zeros_like(noise)
                speaker_embeddings = torch.zeros(
                    self.max_rows + 1,
                    self.speaker_channels,
                    device=self.device,
                    dtype=self.speaker_dtype,
                )

                def solve() -> torch.Tensor:
                    return solve_flow_euler_packed(
                        self.estimator,
                        noise,
                        time_span,
                        mu,
                        speaker_embeddings,
                        mel_conditioning,
                        twin_rows,
                        attention,
                        self.angles,
                        cfg_rate=self.cfg_rate,
                    )

                graph, output = self.graphs.capture(solve)
                self.captured.append(
                    CapturedWholeHistorySolve(
                        tier_frames=tier_frames,
                        graph=graph,
                        noise=noise,
                        time_span=time_span,
                        mu=mu,
                        speaker_embeddings=speaker_embeddings,
                        mel_conditioning=mel_conditioning,
                        twin_rows=twin_rows,
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
        lengths: Sequence[int],
    ) -> torch.Tensor | None:
        """The solve replayed from the tier graphs, a step past the largest tier or the
        row slots as consecutive groups of rows that fit them; None when a row alone
        passes the largest tier."""
        if max(lengths) > self.tier_frames[-1]:
            return None
        else:
            pass

        def replay_rows(rows: slice, frames: slice) -> torch.Tensor:
            return self.replay(
                noise=noise[:, frames],
                time_span=time_span,
                mu=mu[:, frames],
                speaker_embeddings=speaker_embeddings[rows],
                mel_conditioning=mel_conditioning[:, frames],
                lengths=lengths[rows],
            )

        return replay_in_groups(
            replay_rows,
            lengths,
            max_frames=self.tier_frames[-1],
            max_rows=self.max_rows,
        )

    def replay(
        self,
        *,
        noise: torch.Tensor,
        time_span: torch.Tensor,
        mu: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        lengths: Sequence[int],
    ) -> torch.Tensor:
        """The solve of rows within the largest tier and the row slots, replayed from
        the smallest tier holding their frames."""
        frame_count = sum(lengths)
        captured = self.captured[bisect.bisect_left(self.tier_frames, frame_count)]
        assert (
            noise.dtype == self.frame_dtype
            and speaker_embeddings.dtype == self.speaker_dtype
        )
        twin_rows = pack_rows(
            self.slot_lengths(lengths, captured.tier_frames) * 2, self.device
        )
        static = captured.twin_rows
        for destination, source in (
            (captured.noise[:, :frame_count], noise),
            (captured.mu[:, :frame_count], mu),
            (captured.mel_conditioning[:, :frame_count], mel_conditioning),
            (captured.speaker_embeddings[: len(lengths)], speaker_embeddings),
            (captured.time_span, time_span),
            (static.starts, twin_rows.starts),
            (static.row_ids, twin_rows.row_ids),
            (static.positions, twin_rows.positions),
            (static.conv_input_index, twin_rows.conv_input_index),
            (static.conv_output_index, twin_rows.conv_output_index),
        ):
            destination.copy_(source)
        captured.graph.replay()
        # note (ratish): the next replay of any solve graph overwrites the shared pool.
        return captured.output[:, :frame_count].clone()
