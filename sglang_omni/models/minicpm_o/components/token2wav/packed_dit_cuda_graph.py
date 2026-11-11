# SPDX-License-Identifier: Apache-2.0
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Capture and replay fixed-capacity MiniCPM-o packed DiT CUDA graphs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import torch
from torch._dynamo.exc import TorchDynamoException

from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    FixedPackedLayout,
    build_fixed_packed_layout,
)

# note (wirybeaver): SeedTTS EN packed batches stay below 784 mel frames; longer runs use eager.
PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH = 1024
logger = logging.getLogger(__name__)


class PackedDiTBlocks(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        sequence_ids: torch.Tensor,
        cumulative_sequence_lengths: torch.Tensor,
        maximum_sequence_length: int,
        real_frame_positions: torch.Tensor,
        real_frame_mask: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(kw_only=True)
class PackedDiTWorkspace:
    """Fixed-address buffers connecting independently keyed Flow and DiT graphs."""

    dense_input: torch.Tensor
    conditioning: torch.Tensor
    packed_output: torch.Tensor


@dataclass(kw_only=True)
class CapturedPackedDiTGraph:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor
    workspace: PackedDiTWorkspace


class PackedDiTCudaGraphRunner:
    """Replay capacity-dependent packing and DiT using shared boundary buffers."""

    def __init__(
        self,
        run_packed_blocks: PackedDiTBlocks,
        *,
        device: torch.device,
        hidden_size: int,
        output_channels: int,
        convolution_guard_frames: int,
        minimum_free_gibibytes: float = 3.0,
    ) -> None:
        self.run_packed_blocks: PackedDiTBlocks = run_packed_blocks
        self.device: torch.device = device
        self.hidden_size: int = hidden_size
        self.output_channels: int = output_channels
        self.convolution_guard_frames: int = convolution_guard_frames
        self.minimum_free_bytes: int = int(minimum_free_gibibytes * 1024**3)
        self.graphs: dict[tuple[int, int], CapturedPackedDiTGraph] = {}
        self.workspaces: dict[int, PackedDiTWorkspace] = {}
        self.pool: tuple[int, int] | None = None
        self.lock: Lock = Lock()
        self.graph_replays: int = 0
        self.graph_misses: int = 0

    @torch.inference_mode()
    def capture(self, shapes: tuple[tuple[int, int], ...]) -> None:
        if not shapes or len(set(shapes)) != len(shapes):
            raise ValueError("Packed DiT CUDA graph shapes must be nonempty and unique")
        else:
            pass
        if any(
            batch_size <= 1
            or packed_capacity <= 2 * batch_size
            or packed_capacity
            > (2 * batch_size + 1) * PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH
            for batch_size, packed_capacity in shapes
        ):
            raise ValueError(
                "Packed DiT graph capacities must exceed the CFG row count "
                "and fit the 1024-frame workspace and attention bound"
            )
        else:
            pass
        current_stream = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current_stream)
        graphs: dict[tuple[int, int], CapturedPackedDiTGraph] = {}
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            self.pool = torch.cuda.graph_pool_handle()
            self.workspaces = {}
            for batch_size, packed_capacity in sorted(
                shapes, key=lambda shape: shape[1], reverse=True
            ):
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                if free_bytes < self.minimum_free_bytes:
                    logger.warning(
                        f"MiniCPM-o packed DiT graph skipped batch={batch_size} "
                        f"capacity={packed_capacity}: free VRAM {free_bytes / 1024**3:.1f} GB "
                        f"is below {self.minimum_free_bytes / 1024**3:.1f} GB headroom"
                    )
                    continue
                else:
                    pass
                try:
                    hidden_size = self.hidden_size
                    # note (wirybeaver): Descending capacities allocate the largest workspace first.
                    if batch_size not in self.workspaces:
                        self.workspaces[batch_size] = PackedDiTWorkspace(
                            dense_input=torch.zeros(
                                2 * batch_size * PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH,
                                hidden_size,
                                device=self.device,
                                dtype=torch.bfloat16,
                            ),
                            conditioning=torch.zeros(
                                2 * batch_size + 1,
                                hidden_size,
                                device=self.device,
                                dtype=torch.bfloat16,
                            ),
                            packed_output=torch.zeros(
                                packed_capacity,
                                self.output_channels,
                                device=self.device,
                                dtype=torch.bfloat16,
                            ),
                        )
                    else:
                        pass
                    workspace = self.workspaces[batch_size]
                    row_length_frames = packed_capacity // (2 * batch_size + 1)
                    row_lengths = torch.full(
                        (2 * batch_size,),
                        row_length_frames,
                        device=self.device,
                        dtype=torch.int32,
                    )
                    # note (wirybeaver): Spread the remainder so the dummy row fits the attention bound.
                    remainder_frames = packed_capacity % (2 * batch_size + 1)
                    row_lengths[:remainder_frames] += 1
                    layout = build_fixed_packed_layout(
                        row_lengths,
                        row_length_frames + bool(remainder_frames),
                        packed_capacity,
                        self.convolution_guard_frames,
                    )
                    static_inputs = (
                        layout.source_positions,
                        layout.packed_padding_mask,
                        layout.row_ids,
                        layout.cumulative_sequence_lengths,
                        layout.convolution_positions,
                        layout.convolution_valid_mask,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    logger.warning(
                        f"MiniCPM-o packed DiT graph allocation failed for "
                        f"batch={batch_size} capacity={packed_capacity}: {exc}; using eager"
                    )
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    maximum_sequence_length = min(
                        packed_capacity, PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH
                    )

                    output = workspace.packed_output[:packed_capacity]

                    def run_packed_blocks() -> None:
                        packed = torch.index_select(
                            workspace.dense_input, 0, static_inputs[0]
                        )
                        packed.masked_fill_(static_inputs[1][:, None], 0)
                        output.copy_(
                            self.run_packed_blocks(
                                packed,
                                workspace.conditioning,
                                static_inputs[2],
                                static_inputs[3],
                                maximum_sequence_length,
                                static_inputs[4],
                                static_inputs[5],
                            )
                        )

                    try:
                        for _ in range(3):
                            run_packed_blocks()
                    except torch.cuda.OutOfMemoryError as exc:
                        logger.warning(
                            f"Packed DiT graph warmup failed for {(batch_size, packed_capacity)}: {exc}; using eager"
                        )
                        continue
                    try:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            cuda_graph=graph,
                            pool=self.pool,
                            stream=stream,
                            capture_error_mode="thread_local",
                        ):
                            run_packed_blocks()
                    except TorchDynamoException:
                        raise
                    except RuntimeError as exc:
                        logger.warning(
                            f"MiniCPM-o packed DiT graph capture failed for "
                            f"batch={batch_size} capacity={packed_capacity}: {exc}; using eager"
                        )
                        continue
                graphs[(batch_size, packed_capacity)] = CapturedPackedDiTGraph(
                    graph=graph,
                    inputs=static_inputs,
                    output=output,
                    workspace=workspace,
                )
        current_stream.wait_stream(stream)
        self.graphs = graphs
        logger.info(f"Captured {len(graphs)} MiniCPM-o packed DiT CUDA graph shapes")

    def fit(
        self, batch_size: int, mel_frames: int, packed_valid_frames: int | None
    ) -> int | None:
        packed_capacity = min(
            (
                captured_capacity
                for captured_batch_size, captured_capacity in self.graphs
                if captured_batch_size == batch_size
                and packed_valid_frames is not None
                and mel_frames
                <= min(captured_capacity, PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH)
                and captured_capacity > packed_valid_frames
                and captured_capacity - packed_valid_frames <= mel_frames
            ),
            default=None,
        )
        if packed_capacity is None:
            self.graph_misses += 1
        else:
            pass
        return packed_capacity

    def prepare(self, layout: FixedPackedLayout) -> CapturedPackedDiTGraph:
        """Install trajectory-invariant metadata while the solver holds the lock."""
        captured = self.graphs[
            (layout.padding_mask.shape[0] // 2, layout.capacity_frames)
        ]
        for target, source in zip(
            captured.inputs,
            (
                layout.source_positions,
                layout.packed_padding_mask,
                layout.row_ids,
                layout.cumulative_sequence_lengths,
                layout.convolution_positions,
                layout.convolution_valid_mask,
            ),
            strict=True,
        ):
            target.copy_(source)
        return captured

    def replay(
        self,
        captured: CapturedPackedDiTGraph,
    ) -> torch.Tensor:
        """Replay within the solver's graph lock."""
        captured.graph.replay()
        self.graph_replays += 1
        return captured.output
