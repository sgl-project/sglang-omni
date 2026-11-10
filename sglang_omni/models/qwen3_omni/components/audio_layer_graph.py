# SPDX-License-Identifier: Apache-2.0
"""CUDA graph runner for the Qwen3-Omni audio encoder's layer stack."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from sglang.srt.layers.attention.vision import (
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc

from sglang_omni.models.qwen3_omni.components.packed_attention import PackedAttention

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_BUCKETS: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)
WARMUP_ITERATIONS = 3


@dataclass(frozen=True, slots=True)
class Captured:
    graph: torch.cuda.CUDAGraph
    hidden_states: torch.Tensor
    cu_seqlens: torch.Tensor
    output: torch.Tensor
    segment_slots: int


def stack_qkv_projection(attention: nn.Module) -> None:
    """Give the attention one q, k and v projection; q_proj, k_proj and v_proj become views of it."""
    projections = (attention.q_proj, attention.k_proj, attention.v_proj)
    with torch.no_grad():
        weight = torch.cat([projection.weight for projection in projections])
        bias = torch.cat([projection.bias for projection in projections])
    stacked_projection = nn.Linear(weight.shape[1], weight.shape[0], device="meta")
    stacked_projection.weight = nn.Parameter(weight)
    stacked_projection.bias = nn.Parameter(bias)
    attention.qkv_proj = stacked_projection
    start_row = 0
    for projection in projections:
        end_row = start_row + projection.out_features
        projection.weight = nn.Parameter(weight[start_row:end_row])
        projection.bias = nn.Parameter(bias[start_row:end_row])
        start_row = end_row


def packed_attention_forward(
    attention: nn.Module,
    packed_attention: PackedAttention,
    hidden_states: torch.Tensor,
    attention_metadata: VisionAttentionMetadata,
) -> torch.Tensor:
    seq_length, _ = hidden_states.size()
    # note (ratish): the packed attention reads strided q, k and v, so one GEMM projects all three without copies.
    query_states, key_states, value_states = (
        attention.qkv_proj(hidden_states)
        .reshape(seq_length, 3, attention.num_heads, -1)
        .unbind(1)
    )
    attn_output = packed_attention(
        query_states,
        key_states,
        value_states,
        attention_metadata.cu_seqlens,
        bsz=1,
        seq_len=seq_length,
        softmax_scale=attention.scaling,
        forward_metadata=attention_metadata,
    )
    return attention.out_proj(attn_output.reshape(seq_length, -1).contiguous())


class AudioLayerGraphRunner:
    """Replays the audio encoder's layer stack from a captured CUDA graph.

    One instance owns one tower on one CUDA device and attends through the
    encoder's packed attention.
    """

    def __init__(
        self,
        tower: nn.Module,
        *,
        device: torch.device,
        window: int,
        packed_attention: PackedAttention,
        token_buckets: tuple[int, ...] = DEFAULT_TOKEN_BUCKETS,
        max_batch_rows: int = 32,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("audio layer CUDA graphs require a CUDA device")
        else:
            pass
        self.tower = tower
        # Note (wenyao): an indexless "cuda" never equals a tensor's "cuda:N",
        # so resolve the index or every replay silently declines.
        self.device = torch.device(
            "cuda",
            torch.cuda.current_device() if device.index is None else device.index,
        )
        self.window = int(window)
        self.token_buckets = tuple(sorted(token_buckets))
        self.max_batch_rows = int(max_batch_rows)
        self.graphs: dict[int, Captured] = {}
        self.pool = None
        self.disabled_reason: str | None = None
        self.owner_pid = os.getpid()
        self.dtype = next(tower.parameters()).dtype
        self.hidden = tower.config.d_model
        self.packed_attention: PackedAttention = packed_attention

    @property
    def has_graphs(self) -> bool:
        return bool(self.graphs) and self.disabled_reason is None

    def segment_slots(self, bucket: int) -> int:
        # Note (wenyao): a row contributes one window per full block plus a
        # remainder, so slack has to cover every row in the batch, not just
        # the bucket's own window count.
        return bucket // self.window + self.max_batch_rows + 2

    def window_segments(self, tokens: int) -> list[int]:
        full, remainder = divmod(tokens, self.window)
        segments = [self.window] * full
        if remainder:
            segments.append(remainder)
        else:
            pass
        return segments

    def capture_segments(self, bucket: int) -> list[int]:
        # Note (wenyao): every dummy segment must fit the window declared as
        # max_seqlen, or capture records attention kernels sized for a shorter
        # sequence than the tail segment actually is.
        segments = self.window_segments(bucket)
        segments.extend([0] * (self.segment_slots(bucket) - len(segments)))
        return segments

    def run_layers(self, hidden_states, cu_seqlens, max_seqlen: int):
        # An int max_seqlen avoids the device-to-host max() that would break capture.
        attention_metadata = prepare_vision_attention_metadata(
            cu_seqlens, cu_seqlens.device, max_seqlen=max_seqlen
        )
        for layer in self.tower.layers:
            residual = hidden_states
            hidden_states = layer.self_attn_layer_norm(hidden_states)
            hidden_states = packed_attention_forward(
                layer.self_attn,
                self.packed_attention,
                hidden_states,
                attention_metadata,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.fc2(layer.activation_fn(layer.fc1(hidden_states)))
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, -clamp_value, clamp_value)
            else:
                pass
        return hidden_states

    def capture(self, bucket: int) -> Captured | None:
        slots = self.segment_slots(bucket)
        hidden_states = torch.zeros(
            bucket, self.hidden, device=self.device, dtype=self.dtype
        )
        cu_seqlens = (
            torch.tensor([0, *self.capture_segments(bucket)], dtype=torch.int32)
            .cumsum(0)
            .to(torch.int32)
            .to(self.device)
        )
        try:
            with torch.no_grad():
                for _ in range(WARMUP_ITERATIONS):
                    self.run_layers(hidden_states, cu_seqlens, self.window)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                if self.pool is None:
                    with torch.cuda.graph(graph):
                        output = self.run_layers(hidden_states, cu_seqlens, self.window)
                    self.pool = graph.pool()
                else:
                    with torch.cuda.graph(graph, pool=self.pool):
                        output = self.run_layers(hidden_states, cu_seqlens, self.window)
            torch.cuda.synchronize()
        except Exception:
            logger.warning(
                "audio layer CUDA graph capture failed at bucket %d",
                bucket,
                exc_info=True,
            )
            return None
        return Captured(graph, hidden_states, cu_seqlens, output, slots)

    def capture_all(self) -> None:
        # note (ratish): a collection during capture can free what a reference cycle holds,
        # a dead runner's graphs among them, and invalidate the capture.
        with freeze_gc(enable_cudagraph_gc=False):
            for bucket in self.token_buckets:
                captured = self.capture(bucket)
                if captured is None:
                    self.disabled_reason = f"capture failed at bucket {bucket}"
                    self.graphs.clear()
                    return
                else:
                    pass
                self.graphs[bucket] = captured
        logger.info(
            f"audio layer CUDA graphs captured for buckets {list(self.graphs)} "
            f"with {type(self.packed_attention).__name__}"
        )

    def select(self, tokens: int, segments: list[int]) -> int | None:
        if sum(segments) != tokens or any(
            segment < 0 or segment > self.window for segment in segments
        ):
            return None
        else:
            pass
        for bucket in self.token_buckets:
            if bucket >= tokens and self.graphs.get(bucket) is not None:
                required_slots = len(segments) + len(
                    self.window_segments(bucket - tokens)
                )
                if required_slots <= self.graphs[bucket].segment_slots:
                    return bucket
                else:
                    pass
            else:
                pass
        return None

    def maybe_replay(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, segments: list[int]
    ) -> torch.Tensor | None:
        """Return the layer-stack output, or None when the caller must run eager."""
        if self.disabled_reason is not None or not self.graphs:
            return None
        else:
            pass
        if os.getpid() != self.owner_pid or hidden_states.device != self.device:
            return None
        else:
            pass
        tokens = hidden_states.shape[0]
        bucket = self.select(tokens, segments)
        if bucket is None:
            return None
        else:
            pass
        captured = self.graphs[bucket]
        padded = [*segments, *self.window_segments(bucket - tokens)]
        # Note (wenyao): real segments are never widened, only new ones added,
        # so padding rows form their own attention window and cannot reach the
        # real tokens sharing this replay.
        padded.extend([0] * (captured.segment_slots - len(padded)))
        cu = torch.tensor([0, *padded], dtype=torch.int32).cumsum(0).to(torch.int32)
        captured.hidden_states[:tokens].copy_(hidden_states)
        captured.hidden_states[tokens:].zero_()
        captured.cu_seqlens.copy_(cu.to(self.device, non_blocking=True))
        captured.graph.replay()
        return captured.output[:tokens]
