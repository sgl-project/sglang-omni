# SPDX-License-Identifier: Apache-2.0
"""The final step's Euler solve replayed from CUDA graphs, one per frame tier."""

from __future__ import annotations

import bisect
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from sglang.srt.utils.common import get_available_gpu_memory

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    PackedRows,
    pack_rows,
    solve_flow_euler_packed,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import (
    capture_solve_graph,
    frozen_gc,
)
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph

logger = logging.getLogger(__name__)

# note (ratish): frames per CFG half, from measured replay times: coarse below 1,024
# where a replay costs its kernel count, 256 apart above, none past 4,096 where a final
# is device bound.
FINAL_TIER_FRAMES = (
    64,
    128,
    256,
    384,
    512,
    640,
    768,
    1024,
    *range(1280, 4097, 256),
)


@dataclass(kw_only=True)
class CapturedFinalSolve:
    tier_frames: int
    graph: ReplayableGraph
    noise: torch.Tensor
    time_span: torch.Tensor
    mu: torch.Tensor
    speaker_embeddings: torch.Tensor
    mel_conditioning: torch.Tensor
    twin_rows: PackedRows
    output: torch.Tensor


class FinalCudaGraphRunner:
    def __init__(
        self,
        estimator: PackedDiT,
        *,
        backend: DeviceGraphBackend,
        device: torch.device,
        autocast_dtype: torch.dtype,
        frame_dtype: torch.dtype,
        speaker_dtype: torch.dtype,
        cfg_rate: float,
        euler_steps: int,
        mel_channels: int,
        speaker_channels: int,
        max_rows: int,
        tier_frames: Sequence[int],
        max_frames: int,
    ) -> None:
        self.estimator = estimator
        self.backend = backend
        self.device = device
        self.device_module = torch.get_device_module(device)
        self.autocast_dtype = autocast_dtype
        self.frame_dtype = frame_dtype
        self.speaker_dtype = speaker_dtype
        self.cfg_rate = cfg_rate
        self.euler_steps = euler_steps
        self.mel_channels = mel_channels
        self.speaker_channels = speaker_channels
        self.max_rows = max_rows
        self.tier_frames = sorted(tier_frames)
        self.angles = estimator.rope_angles(max_frames)
        self.captured: list[CapturedFinalSolve] = []

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
        started = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device.type, self.device.index)
        graph_pool = self.backend.graph_pool_handle()
        stream = self.device_module.Stream(device=self.device)
        with frozen_gc():
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

                graph, output = capture_solve_graph(
                    solve,
                    backend=self.backend,
                    graph_pool=graph_pool,
                    stream=stream,
                    device=self.device,
                    autocast_dtype=self.autocast_dtype,
                )
                self.captured.append(
                    CapturedFinalSolve(
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
        after_mem = get_available_gpu_memory(self.device.type, self.device.index)
        logger.info(
            f"Fun-CosyVoice3 final solve graphs captured: tiers={self.tier_frames} "
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
        lengths: Sequence[int],
    ) -> torch.Tensor | None:
        """The solve replayed from the smallest tier holding the step's frames,
        None above the largest tier or past the row slots."""
        frame_count = sum(lengths)
        tier = bisect.bisect_left(self.tier_frames, frame_count)
        if tier == len(self.tier_frames) or len(lengths) > self.max_rows:
            return None
        else:
            captured = self.captured[tier]
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
        # note (ratish): the next replay of any tier overwrites the shared pool.
        return captured.output[:, :frame_count].clone()
