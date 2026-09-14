# SPDX-License-Identifier: Apache-2.0
"""CUDA graph runner for the Qwen3-Omni audio encoder's layer stack."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

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


_HOPPER = 9


def _packed_attention_backend(capability: tuple[int, int], *, is_hip: bool) -> str:
    """FA3 on NVIDIA Hopper, Triton on every other supported device.

    sglang's VisionAttention rule also picks FA4 on Blackwell. Omni validates
    Hopper only, and Triton is the arm sglang itself uses on the other parts.
    """
    if is_hip:
        # Note (zijiecode): HIP reports gfx950 as (9, 5), which the Hopper check
        # would send to FA3; FA3 does not exist on HIP.
        return "triton_attn"
    major, _ = capability
    return "fa3" if major == _HOPPER else "triton_attn"


def _resolve_packed_attention(device: torch.device) -> tuple[nn.Module, str]:
    # Local import: only a process that requests the graphs loads sglang's kernels.
    from sglang.srt.layers.attention import vision

    backend = _packed_attention_backend(
        torch.cuda.get_device_capability(device),
        is_hip=torch.version.hip is not None,
    )
    if backend == "fa3":
        impl_class = vision.VisionFlash3Attention
    else:
        impl_class = vision.VisionTritonAttention
    # The encoder process holds no sglang parallel state to read.
    return impl_class(use_data_parallel=True), backend


def _packed_attention_forward(
    attention: nn.Module,
    packed_attention: nn.Module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    seq_length, _ = hidden_states.size()
    heads = attention.num_heads
    query_states = attention.q_proj(hidden_states).reshape(seq_length, heads, -1)
    key_states = attention.k_proj(hidden_states).reshape(seq_length, heads, -1)
    value_states = attention.v_proj(hidden_states).reshape(seq_length, heads, -1)
    # An int max_seqlen avoids the device-to-host max() that would break capture.
    attn_output = packed_attention(
        query_states,
        key_states,
        value_states,
        cu_seqlens,
        bsz=1,
        seq_len=seq_length,
        softmax_scale=attention.scaling,
        max_seqlen=max_seqlen,
    )
    return attention.out_proj(attn_output.reshape(seq_length, -1).contiguous())


class AudioLayerGraphRunner:
    """Replays the audio encoder's layer stack from a captured CUDA graph.

    One instance owns one tower on one CUDA device. Replay is opt-in because
    it runs the packed attention kernel sglang picks for the device where
    serving otherwise runs sdpa: outputs agree to bf16 kernel noise, not
    bitwise, so it needs the content gate rather than an equality check.
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
        self._packed_attention: nn.Module | None = None
        self._backend: str | None = None

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

    def _resolve_attention(self) -> None:
        if self._packed_attention is not None or self._disabled_reason is not None:
            return
        try:
            self._packed_attention, self._backend = _resolve_packed_attention(
                self._device
            )
        except Exception as exc:
            # Like a failed capture: the encoder stays eager, the stage lives.
            self._disabled_reason = f"packed attention unavailable: {exc}"
            logger.warning(
                "audio layer CUDA graphs unavailable: %s",
                self._disabled_reason,
                exc_info=True,
            )

    def _run_layers(self, hidden_states, cu_seqlens, max_seqlen: int):
        for layer in self._tower.layers:
            residual = hidden_states
            hidden_states = layer.self_attn_layer_norm(hidden_states)
            hidden_states = _packed_attention_forward(
                layer.self_attn,
                self._packed_attention,
                hidden_states,
                cu_seqlens,
                max_seqlen,
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
        self._resolve_attention()
        if self._disabled_reason is not None:
            return
        for bucket in self._token_buckets:
            captured = self._capture(bucket)
            if captured is None:
                self._disabled_reason = f"capture failed at bucket {bucket}"
                self._graphs.clear()
                return
            self._graphs[bucket] = captured
        logger.info(
            "audio layer CUDA graphs captured for buckets %s with %s attention",
            list(self._graphs),
            self._backend,
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
