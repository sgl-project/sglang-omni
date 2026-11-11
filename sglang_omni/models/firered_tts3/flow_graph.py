# SPDX-License-Identifier: Apache-2.0
"""Capture and replay the DiT Euler flow loop with CUDA graphs."""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch

from sglang_omni.models.firered_tts3.compiled_dit import CompiledDiTExecutor
from sglang_omni.models.firered_tts3.flow_execution import (
    FLOW_MODES,
    FlowMode,
    euler_flow_loop,
    flow_dit_dtype,
    flow_ffn_optimization,
    flow_optimizations,
    flow_t_span,
    prepare_flow_dit,
)
from sglang_omni.models.firered_tts3.flow_graph_capture import FlowGraphCapture
from sglang_omni.models.firered_tts3.flow_graph_resources import (
    FlowGraphCounters,
    FlowGraphEntry,
)
from sglang_omni.models.firered_tts3.flow_precision import FlowPrecisionGate

logger = logging.getLogger(__name__)

# note (kermit): Batch rows are independent, so unused bucket rows can be zero-padded.
DEFAULT_CAPTURE_SIZES = (1, 2, 4, 8, 16)


class FlowGraphStats(FlowGraphCounters):
    cache_size: int
    precompute_time_conditions: bool
    optimize_convolution: bool
    optimize_ffn: bool
    optimization_enabled: bool


class FlowGraphRunner:
    """Replay batched Euler integration using per-bucket CUDA graphs."""

    def __init__(
        self,
        dit: torch.nn.Module | CompiledDiTExecutor,
        t_span: torch.Tensor,
        *,
        n_timesteps: int,
        patch_size: int,
        history_length: int,
        max_graphs: int = 8,
        capture_sizes: list[int] | tuple[int, ...] | None = None,
        mode: FlowMode = "fp32",
        dit_fallback: torch.nn.Module | None = None,
        precompute_time_conditions: bool | None = None,
    ) -> None:
        self.mode = mode
        self.precision_gate = FlowPrecisionGate.from_environment()
        self.dit = dit
        self.dit_dtype = flow_dit_dtype(mode)
        self.dit_fallback = dit_fallback if dit_fallback is not None else dit
        time_cache, self.optimize_convolution = flow_optimizations(mode)
        self.precompute_time_conditions = (
            time_cache
            if precompute_time_conditions is None
            else precompute_time_conditions
        )
        self.optimize_ffn = (
            flow_ffn_optimization(mode)
            and t_span.device.type == "cuda"
            and torch.cuda.get_device_capability(t_span.device) == (12, 0)
        )
        self.optimize = (
            self.precompute_time_conditions
            or self.optimize_convolution
            or self.optimize_ffn
        )
        if self.optimize and not mode.startswith("compile"):
            raise ValueError("Flow time/convolution caches require a compile flow mode")
        else:
            pass
        if isinstance(dit, CompiledDiTExecutor):
            self.executor = dit
        elif self.optimize:
            self.executor = CompiledDiTExecutor(self.dit_fallback, original=dit)
            self.dit = self.executor
        else:
            self.executor = None
        self.t_span = t_span.clone() if self.optimize else t_span
        self.n_timesteps = int(n_timesteps)
        self.patch_size = int(patch_size)
        self.history_length = int(history_length)
        self.capture_sizes = (
            tuple(sorted({int(size) for size in capture_sizes if int(size) > 0}))
            or None
            if capture_sizes
            else None
        )
        self.max_graphs = max(int(max_graphs), len(self.capture_sizes or ()))
        self.enabled = True
        self.cache: OrderedDict[
            tuple[int | None, int, tuple[int, ...]], FlowGraphEntry
        ] = OrderedDict()
        self.counters: FlowGraphCounters = {
            "calls": 0,
            "hits": 0,
            "captures": 0,
            "eager": 0,
            "padded": 0,
            "optimized_captures": 0,
            "optimization_fallbacks": 0,
            "ffn_captures": 0,
            "ffn_fallbacks": 0,
        }
        self.graph_capture = FlowGraphCapture(
            dit_fallback=self.dit_fallback,
            executor=self.executor,
            dit_dtype=self.dit_dtype,
            t_span=self.t_span,
            n_timesteps=self.n_timesteps,
            patch_size=self.patch_size,
            optimize=self.optimize,
            precompute_time_conditions=self.precompute_time_conditions,
            optimize_convolution=self.optimize_convolution,
            optimize_ffn=self.optimize_ffn,
            precision_gate=self.precision_gate,
            counters=self.counters,
        )

    def stats(self) -> FlowGraphStats:
        return {
            **self.counters,
            "cache_size": len(self.cache),
            "precompute_time_conditions": self.precompute_time_conditions,
            "optimize_convolution": self.optimize_convolution,
            "optimize_ffn": self.optimize_ffn,
            "optimization_enabled": self.optimize,
        }

    def run(
        self,
        hist_latents: torch.Tensor,  # [B, history, D]
        x0: torch.Tensor,  # [B, patch, D]
        cond: torch.Tensor,  # [B, history+patch, C]
        cfgs: torch.Tensor,  # [B] or [B, 1, 1] fp32
        n_timesteps: int,
    ) -> torch.Tensor | None:
        """One batched flow step via graph replay; None requests eager."""
        history_latents, initial_noise = hist_latents, x0
        conditioning, guidance_scales = cond, cfgs
        self.counters["calls"] += 1
        if (
            not self.enabled
            or int(n_timesteps) != self.n_timesteps
            or history_latents.device.type != "cuda"
            or torch.cuda.is_current_stream_capturing()
        ):
            self.counters["eager"] += 1
            return None
        else:
            pass
        batch_size = history_latents.shape[0]
        guidance_scales = guidance_scales.reshape(batch_size, 1, 1)
        capture_batch_size = batch_size
        if self.capture_sizes:
            capture_batch_size = next(
                (size for size in self.capture_sizes if size >= batch_size), batch_size
            )
        else:
            pass
        if capture_batch_size != batch_size:
            padding_size = capture_batch_size - batch_size
            history_latents = self.pad_batch(history_latents, padding_size)
            initial_noise = self.pad_batch(initial_noise, padding_size)
            conditioning = self.pad_batch(conditioning, padding_size)
            guidance_scales = self.pad_batch(guidance_scales, padding_size)
            self.counters["padded"] += 1
        else:
            pass
        cache_key = (
            history_latents.device.index,
            capture_batch_size,
            tuple(conditioning.shape),
        )
        entry = self.cache.get(cache_key)
        if entry is None:
            entry = self.capture(history_latents, initial_noise, conditioning)
            if entry is None and self.dit is not self.dit_fallback:
                # The compiled stack failed to compile or capture; degrade
                # to the same dtype without torch.compile (permanently —
                # later buckets share the conclusion) before giving up on
                # the graphs entirely.
                logger.warning(
                    f"FireRedTTS3 flow mode {self.mode}: capture with the "
                    "compiled DiT failed; retrying without torch.compile"
                )
                self.dit = self.dit_fallback
                entry = self.capture(history_latents, initial_noise, conditioning)
            else:
                pass
            if entry is None:
                self.enabled = False
                self.counters["eager"] += 1
                return None
            else:
                pass
            self.counters["captures"] += 1
            self.cache[cache_key] = entry
            while len(self.cache) > self.max_graphs:
                _, evicted = self.cache.popitem(last=False)
                del evicted
        else:
            self.cache.move_to_end(cache_key)
            self.counters["hits"] += 1
        entry.xt[:, : self.history_length].copy_(history_latents)
        entry.xt[:, self.history_length :].copy_(initial_noise)
        entry.cond.copy_(conditioning)
        entry.cfg.copy_(guidance_scales)
        if entry.dispatch_modes:
            entry.record_replay_stream(torch.cuda.current_stream(entry.xt.device))
        else:
            pass
        entry.graph.replay()
        # The tail aliases the static buffer the next replay overwrites.
        return entry.xt[:, -self.patch_size :][:batch_size].clone()

    @staticmethod
    def pad_batch(t: torch.Tensor, pad: int) -> torch.Tensor:
        batch, padding_size = t, pad
        return torch.cat(
            [batch, batch.new_zeros(padding_size, *batch.shape[1:])], dim=0
        )

    def capture(
        self,
        hist_latents: torch.Tensor,
        x0: torch.Tensor,
        cond: torch.Tensor,
    ) -> FlowGraphEntry | None:
        """Capture one whole-loop graph for the input shape, or return None."""
        return self.graph_capture.capture(hist_latents, x0, cond, dit=self.dit)


__all__ = [
    "DEFAULT_CAPTURE_SIZES",
    "FLOW_MODES",
    "FlowGraphEntry",
    "FlowGraphRunner",
    "FlowMode",
    "euler_flow_loop",
    "flow_dit_dtype",
    "flow_t_span",
    "prepare_flow_dit",
]
