# SPDX-License-Identifier: Apache-2.0
"""HiFT's decode replayed from CUDA graphs, one per row and frame tier."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.platforms.device_graph import ReplayableGraph


@dataclass(kw_only=True)
class CapturedHiftDecode:
    graph: ReplayableGraph
    mel: torch.Tensor
    source: torch.Tensor
    output: torch.Tensor


class HiftDecodeCudaGraphRunner:
    def __init__(
        self,
        hift: torch.nn.Module,
        *,
        graphs: SolveGraphCapture,
        mel_channels: int,
        samples_per_frame: int,
        row_tiers: Sequence[int],
        frame_tiers: Sequence[int],
    ) -> None:
        self.hift = hift
        self.graphs = graphs
        self.device = graphs.device
        self.mel_channels = mel_channels
        self.samples_per_frame = samples_per_frame
        self.row_tiers = sorted(row_tiers)
        self.frame_tiers = sorted(frame_tiers)
        largest_rows, largest_frames = self.row_tiers[-1], self.frame_tiers[-1]
        self.mel_storage = torch.zeros(
            largest_rows * mel_channels * largest_frames, device=self.device
        )
        self.source_storage = torch.zeros(
            largest_rows * largest_frames * samples_per_frame, device=self.device
        )
        self.captured: dict[tuple[int, int], CapturedHiftDecode] = {}

    def frame_tier(self, frames: int) -> int | None:
        """The smallest frame tier holding frames, None above the largest."""
        tier = bisect.bisect_left(self.frame_tiers, frames)
        return None if tier == len(self.frame_tiers) else self.frame_tiers[tier]

    @torch.inference_mode()
    def capture(self) -> None:
        """Largest tiers first, so the smaller ones reuse the pool's memory."""
        with self.graphs.capture_session("hift decode", self.frame_tiers):
            for frames in reversed(self.frame_tiers):
                for rows in reversed(self.row_tiers):
                    samples = frames * self.samples_per_frame
                    mel = self.mel_storage[: rows * self.mel_channels * frames].view(
                        rows, self.mel_channels, frames
                    )
                    source = self.source_storage[: rows * samples].view(
                        rows, 1, samples
                    )

                    def decode(
                        mel: torch.Tensor = mel, source: torch.Tensor = source
                    ) -> torch.Tensor:
                        return self.hift.decode(x=mel, s=source, finalize=True)

                    graph, output = self.graphs.capture(decode, autocast=False)
                    self.captured[(rows, frames)] = CapturedHiftDecode(
                        graph=graph, mel=mel, source=source, output=output
                    )

    @torch.inference_mode()
    def run(self, mel: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Replays the decode of mel (rows, channels, frames), frames a frame tier, and its
        source at the smallest row tier holding the rows. The samples are graph memory,
        consumed on this stream before the next replay of any graph in the pool."""
        rows, _, frames = mel.shape
        tier = bisect.bisect_left(self.row_tiers, rows)
        assert tier < len(self.row_tiers), f"{rows} rows past the largest row tier"
        captured = self.captured[(self.row_tiers[tier], frames)]
        # note (ratish): the tier's other rows keep an earlier step's inputs; each row
        # decodes alone, and their samples are never read.
        captured.mel[:rows].copy_(mel)
        captured.source[:rows].copy_(source)
        captured.graph.replay()
        return captured.output[:rows]
