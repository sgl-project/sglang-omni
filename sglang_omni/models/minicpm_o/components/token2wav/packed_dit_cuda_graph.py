# SPDX-License-Identifier: Apache-2.0
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Capture and replay fixed-capacity MiniCPM-o packed DiT CUDA graphs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock

import torch

from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    FixedPackedLayout,
    build_fixed_packed_layout,
)

# note (wirybeaver): SeedTTS EN packed batches stay below 784 mel frames; longer runs use eager.
PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH = 1024
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CapturedPackedDiTGraph:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor


class PackedDiTCudaGraphRunner:
    """Replay fixed-capacity DiT blocks without capturing mel-width-dependent work."""

    def __init__(
        self,
        estimator: torch.nn.Module,
        *,
        device: torch.device,
        min_free_gb: float = 3.0,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("Packed DiT CUDA graphs require a CUDA device")
        else:
            pass
        self.estimator: torch.nn.Module = estimator
        self.device: torch.device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
        self.min_free_bytes: int = int(min_free_gb * 1024**3)
        self.graphs: dict[tuple[int, int], CapturedPackedDiTGraph] = {}
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
            batch_size <= 1 or packed_capacity <= 2 * batch_size
            for batch_size, packed_capacity in shapes
        ):
            raise ValueError("Packed DiT graph capacities must exceed the row count")
        else:
            pass
        current_stream = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current_stream)
        graphs: dict[tuple[int, int], CapturedPackedDiTGraph] = {}
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            self.pool = torch.cuda.graph_pool_handle()
            for batch_size, packed_capacity in sorted(
                shapes, key=lambda shape: shape[1], reverse=True
            ):
                free, _ = torch.cuda.mem_get_info(self.device)
                if free < self.min_free_bytes:
                    logger.warning(
                        f"MiniCPM-o packed DiT graph skipped batch={batch_size} "
                        f"capacity={packed_capacity}: free VRAM {free / 1024**3:.1f} GB "
                        f"is below {self.min_free_bytes / 1024**3:.1f} GB headroom"
                    )
                    continue
                else:
                    pass
                try:
                    hidden_size = self.estimator.in_proj.out_features
                    row_length_frames = packed_capacity // (2 * batch_size + 1)
                    row_lengths = torch.full(
                        (2 * batch_size,),
                        row_length_frames,
                        device=self.device,
                        dtype=torch.int32,
                    )
                    layout = build_fixed_packed_layout(
                        row_lengths,
                        row_length_frames,
                        packed_capacity,
                        self.estimator.blocks[0].conv.kernel_size - 1,
                    )
                    static_inputs = (
                        torch.zeros(
                            packed_capacity,
                            hidden_size,
                            device=self.device,
                            dtype=torch.bfloat16,
                        ),
                        torch.zeros(
                            2 * batch_size + 1,
                            hidden_size,
                            device=self.device,
                            dtype=torch.bfloat16,
                        ),
                        layout.row_ids,
                        layout.cu_seqlens,
                        layout.conv_positions,
                        layout.conv_valid,
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        max_length = min(
                            packed_capacity, PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH
                        )

                        def run_packed_blocks() -> torch.Tensor:
                            return self.estimator.run_packed_blocks(
                                static_inputs[0],
                                static_inputs[1],
                                static_inputs[2],
                                static_inputs[3],
                                max_length,
                                static_inputs[4],
                                static_inputs[5],
                            )

                        for _ in range(3):
                            run_packed_blocks()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            cuda_graph=graph,
                            pool=self.pool,
                            stream=stream,
                            capture_error_mode="thread_local",
                        ):
                            output = run_packed_blocks()
                    graphs[(batch_size, packed_capacity)] = CapturedPackedDiTGraph(
                        graph=graph, inputs=static_inputs, output=output
                    )
                except Exception as exc:
                    logger.warning(
                        f"MiniCPM-o packed DiT graph capture failed for "
                        f"batch={batch_size} capacity={packed_capacity}: {exc}; using eager"
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
        captured = self.graphs[(layout.padding_mask.shape[0] // 2, layout.capacity)]
        for target, source in zip(
            captured.inputs[2:],
            (
                layout.row_ids,
                layout.cu_seqlens,
                layout.conv_positions,
                layout.conv_valid,
            ),
            strict=True,
        ):
            target.copy_(source)
        return captured

    def replay(
        self,
        captured: CapturedPackedDiTGraph,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Replay within the solver's graph lock."""
        captured.inputs[1][:-1].copy_(conditioning)
        captured.graph.replay()
        self.graph_replays += 1
        return captured.output
