# SPDX-License-Identifier: Apache-2.0
"""CUDA graph runner for the Qwen3-Omni audio encoder's layer stack."""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from types import MethodType

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_BUCKETS: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)
WARMUP_ITERATIONS = 3


@dataclass(frozen=True, slots=True)
class _Captured:
    graph: torch.cuda.CUDAGraph
    hidden_states: torch.Tensor
    cu_seqlens: torch.Tensor
    output: torch.Tensor
    segment_slots: int


def _varlen_attention_forward(
    self, hidden_states, cu_seqlens, max_seqlen: int, **kwargs
):
    seq_length, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    varlen_config = self._omni_varlen_config
    attention_interface = hf_modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
        varlen_config._attn_implementation, hf_modeling.eager_attention_forward
    )
    stock_config, self.config = self.config, varlen_config
    try:
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    finally:
        self.config = stock_config
    return self.out_proj(attn_output.reshape(seq_length, -1).contiguous())


class AudioLayerGraphRunner:
    """Replays the audio encoder's layer stack from a captured CUDA graph.

    One instance owns one tower on one CUDA device. Replay is opt-in because
    it runs varlen flash attention where serving otherwise runs sdpa: outputs
    agree to bf16 kernel noise, not bitwise, so it needs the content gate
    rather than an equality check.
    """

    def __init__(
        self,
        tower: nn.Module,
        *,
        device: torch.device,
        window: int,
        token_buckets: tuple[int, ...] = DEFAULT_TOKEN_BUCKETS,
        max_batch_rows: int = 32,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("audio layer CUDA graphs require a CUDA device")
        self._tower = tower
        # Note (wenyao): an indexless "cuda" never equals a tensor's "cuda:N",
        # so resolve the index or every replay silently declines.
        self._device = torch.device(
            "cuda",
            torch.cuda.current_device() if device.index is None else device.index,
        )
        self._window = int(window)
        self._token_buckets = tuple(sorted(token_buckets))
        self._max_batch_rows = int(max_batch_rows)
        self._graphs: dict[int, _Captured] = {}
        self._pool = None
        self._disabled_reason: str | None = None
        self._owner_pid = os.getpid()
        self._dtype = next(tower.parameters()).dtype
        self._hidden = tower.config.d_model
        self._patched = False

    @property
    def has_graphs(self) -> bool:
        return bool(self._graphs) and self._disabled_reason is None

    def _segment_slots(self, bucket: int) -> int:
        # Note (wenyao): a row contributes one window per full block plus a
        # remainder, so slack has to cover every row in the batch, not just
        # the bucket's own window count.
        return bucket // self._window + self._max_batch_rows + 2

    def _window_segments(self, tokens: int) -> list[int]:
        full, remainder = divmod(tokens, self._window)
        segments = [self._window] * full
        if remainder:
            segments.append(remainder)
        return segments

    def _capture_segments(self, bucket: int) -> list[int]:
        # Note (wenyao): every dummy segment must fit the window declared as
        # max_seqlen, or capture records attention kernels sized for a shorter
        # sequence than the tail segment actually is.
        segments = self._window_segments(bucket)
        segments.extend([0] * (self._segment_slots(bucket) - len(segments)))
        return segments

    def _install_varlen(self) -> None:
        if self._patched:
            return
        for layer in self._tower.layers:
            attention = layer.self_attn
            # Note (wenyao): the config object is shared with the rest of the
            # thinker, so flipping it in place would silently switch attention
            # for every other component that loaded the same config.
            varlen_config = copy.copy(attention.config)
            varlen_config._attn_implementation = "flash_attention_2"
            attention._omni_varlen_config = varlen_config
            attention._omni_varlen_forward = MethodType(
                _varlen_attention_forward, attention
            )
        self._patched = True

    def _run_layers(self, hidden_states, cu_seqlens, max_seqlen: int):
        for layer in self._tower.layers:
            residual = hidden_states
            hidden_states = layer.self_attn_layer_norm(hidden_states)
            hidden_states = layer.self_attn._omni_varlen_forward(
                hidden_states, cu_seqlens, max_seqlen
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.fc2(layer.activation_fn(layer.fc1(hidden_states)))
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, -clamp_value, clamp_value)
        return hidden_states

    def _capture(self, bucket: int) -> _Captured | None:
        slots = self._segment_slots(bucket)
        hidden_states = torch.zeros(
            bucket, self._hidden, device=self._device, dtype=self._dtype
        )
        cu_seqlens = (
            torch.tensor([0, *self._capture_segments(bucket)], dtype=torch.int32)
            .cumsum(0)
            .to(torch.int32)
            .to(self._device)
        )
        try:
            with torch.no_grad():
                for _ in range(WARMUP_ITERATIONS):
                    self._run_layers(hidden_states, cu_seqlens, self._window)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                if self._pool is None:
                    with torch.cuda.graph(graph):
                        output = self._run_layers(
                            hidden_states, cu_seqlens, self._window
                        )
                    self._pool = graph.pool()
                else:
                    with torch.cuda.graph(graph, pool=self._pool):
                        output = self._run_layers(
                            hidden_states, cu_seqlens, self._window
                        )
            torch.cuda.synchronize()
        except Exception:
            logger.warning(
                "audio layer CUDA graph capture failed at bucket %d",
                bucket,
                exc_info=True,
            )
            return None
        return _Captured(graph, hidden_states, cu_seqlens, output, slots)

    def capture_all(self) -> None:
        self._install_varlen()
        for bucket in self._token_buckets:
            captured = self._capture(bucket)
            if captured is None:
                self._disabled_reason = f"capture failed at bucket {bucket}"
                self._graphs.clear()
                return
            self._graphs[bucket] = captured
        logger.info(
            "audio layer CUDA graphs captured for buckets %s", list(self._graphs)
        )

    def _select(self, tokens: int, segments: list[int]) -> int | None:
        if sum(segments) != tokens or any(
            segment < 0 or segment > self._window for segment in segments
        ):
            return None
        for bucket in self._token_buckets:
            if bucket >= tokens and self._graphs.get(bucket) is not None:
                required_slots = len(segments) + len(
                    self._window_segments(bucket - tokens)
                )
                if required_slots <= self._graphs[bucket].segment_slots:
                    return bucket
        return None

    def maybe_replay(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, segments: list[int]
    ) -> torch.Tensor | None:
        """Return the layer-stack output, or None when the caller must run eager."""
        if self._disabled_reason is not None or not self._graphs:
            return None
        if os.getpid() != self._owner_pid or hidden_states.device != self._device:
            return None
        tokens = hidden_states.shape[0]
        bucket = self._select(tokens, segments)
        if bucket is None:
            return None
        captured = self._graphs[bucket]
        padded = [*segments, *self._window_segments(bucket - tokens)]
        # Note (wenyao): real segments are never widened, only new ones added,
        # so padding rows form their own attention window and cannot reach the
        # real tokens sharing this replay.
        padded.extend([0] * (captured.segment_slots - len(padded)))
        cu = torch.tensor([0, *padded], dtype=torch.int32).cumsum(0).to(torch.int32)
        captured.hidden_states[:tokens].copy_(hidden_states)
        captured.hidden_states[tokens:].zero_()
        captured.cu_seqlens.copy_(cu.to(self._device, non_blocking=True))
        captured.graph.replay()
        return captured.output[:tokens]
