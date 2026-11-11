# SPDX-License-Identifier: Apache-2.0
"""CUDA graph replay resources and precision validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, TypedDict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang_omni.models.firered_tts3.compiled_dit import FfnGemmMode
from sglang_omni.models.firered_tts3.conv_cache import Conv1dWeightCache
from sglang_omni.models.firered_tts3.flow_time_cache import (
    FlowPrecisionGate,
    TimeCacheUnavailable,
    TimeGemmTable,
)

logger = logging.getLogger(__name__)

# note (kermit): Warmup settles cuDNN/cuBLAS autotuning before graph capture.
WARMUP_ITERATIONS = 3


class FlowGraphCounters(TypedDict):
    calls: int
    hits: int
    captures: int
    eager: int
    padded: int
    optimized_captures: int
    optimization_fallbacks: int
    ffn_captures: int
    ffn_fallbacks: int


@dataclass
class FlowGraphEntry:
    """Captured graph and the tensors retained for replay."""

    graph: torch.cuda.CUDAGraph
    # note (kermit): Graph replay updates the flow-state tail in place.
    xt: torch.Tensor  # [B, history+patch, redae_dim]
    cond: torch.Tensor  # [B, history+patch, cond_channels]
    cfg: torch.Tensor  # [B, 1, 1] fp32 per-row CFG scale
    time_table: TimeGemmTable | None = None
    dispatch_modes: tuple[TorchDispatchMode, ...] = ()
    t_span: torch.Tensor | None = None
    replay_streams: dict[tuple[int | None, int], torch.cuda.Stream] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def record_replay_stream(self, stream: torch.cuda.Stream) -> None:
        """Register cached tensors once per replay stream; this does not serialize use."""
        stream_key = (stream.device.index, stream.cuda_stream)
        if stream_key in self.replay_streams:
            return
        else:
            pass
        for tensor in (self.xt, self.cond, self.cfg):
            tensor.record_stream(stream)
        if self.t_span is not None:
            self.t_span.record_stream(stream)
        else:
            pass
        if self.time_table is not None:
            for step in self.time_table.steps:
                for value in step.values():
                    value.output.record_stream(stream)
        else:
            pass
        for mode in self.dispatch_modes:
            if isinstance(mode, Conv1dWeightCache):
                for weight in mode.weights.values():
                    weight.packed.record_stream(stream)
                if mode.controlled_plan is not None:
                    mode.controlled_plan.workspace.record_stream(stream)
                else:
                    pass
            elif isinstance(mode, FfnGemmMode):
                mode.record_stream(stream)
            else:
                pass
        # note (Codex): replay bypasses Python dispatch, so retain each registered stream.
        self.replay_streams[stream_key] = stream


class FlowCaptureBody(Protocol):
    def __call__(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        scales: torch.Tensor,
        active_modes: tuple[TorchDispatchMode, ...],
        *,
        observed: bool = False,
    ) -> torch.Tensor: ...


def check_flow_precision(
    stage: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    gate: FlowPrecisionGate,
    probe: int,
    batch_size: int,
) -> None:
    result = gate.compare(actual, expected)
    logger.info(
        f"Flow capture gate: bucket={batch_size} probe={probe} stage={stage} "
        f"accepted={result.accepted} byte_exact={result.byte_exact} "
        f"schema_valid={result.schema_valid} finite={result.finite} "
        f"max_abs={result.max_abs} rmse={result.rmse} "
        f"worst_row_nrmse={result.worst_row_nrmse} "
        f"out_of_bounds={result.out_of_bounds} atol={gate.atol} rtol={gate.rtol}"
    )
    if not result.accepted:
        comparison = "exact" if gate.atol == gate.rtol == 0 else "tolerance"
        raise TimeCacheUnavailable(
            f"Complete Flow {comparison} gate failed for {stage}, probe {probe}"
        )
    else:
        pass


class FlowGraphResources:
    """Keep the graph pool and capture stream alive across bucket captures."""

    def __init__(self) -> None:
        self.graph_pool = torch.cuda.graph_pool_handle()
        # note (kermit): A dedicated capture stream prevents cuBLAS workspace races across graph families.
        self.capture_stream: torch.cuda.Stream | None = None

    def capture_graph(
        self,
        flow_state: torch.Tensor,
        conditioning: torch.Tensor,
        guidance_scales: torch.Tensor,
        active_modes: tuple[TorchDispatchMode, ...],
        gate: FlowPrecisionGate | None,
        probes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        patch_size: int,
        body: FlowCaptureBody,
    ) -> torch.cuda.CUDAGraph:
        for _ in range(WARMUP_ITERATIONS):
            body(
                flow_state,
                conditioning,
                guidance_scales,
                active_modes,
                observed=bool(active_modes),
            )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        if self.capture_stream is None:
            self.capture_stream = torch.cuda.Stream(device=flow_state.device)
        else:
            pass
        with torch.cuda.graph(
            graph,
            pool=self.graph_pool,
            stream=self.capture_stream,
            capture_error_mode="thread_local",
        ):
            body(
                flow_state,
                conditioning,
                guidance_scales,
                active_modes,
                observed=bool(active_modes),
            )
        if gate is not None:
            for probe, (state, condition, scales, expected) in enumerate(probes):
                flow_state.copy_(state)
                conditioning.copy_(condition)
                guidance_scales.copy_(scales)
                graph.replay()
                check_flow_precision(
                    "replay",
                    flow_state[:, -patch_size:],
                    expected,
                    gate,
                    probe,
                    flow_state.shape[0],
                )
        else:
            pass
        return graph
