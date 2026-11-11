# SPDX-License-Identifier: Apache-2.0
"""Capture optimized flow graphs and retain their replay resources."""

from __future__ import annotations

import logging

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang_omni.models.firered_tts3.compiled_dit import CompiledDiTExecutor
from sglang_omni.models.firered_tts3.conv_cache import (
    INPUT_SHAPE as CONTROLLED_CONV_INPUT_SHAPE,
)
from sglang_omni.models.firered_tts3.conv_cache import Conv1dWeightCache
from sglang_omni.models.firered_tts3.ffn_gemm import QKV_CONFIGS, FfnGemmMode
from sglang_omni.models.firered_tts3.flow_execution import euler_flow_loop
from sglang_omni.models.firered_tts3.flow_graph_resources import (
    FlowGraphCounters,
    FlowGraphEntry,
    FlowGraphResources,
    check_flow_precision,
    prepare_flow_probes,
)
from sglang_omni.models.firered_tts3.flow_precision import FlowPrecisionGate
from sglang_omni.models.firered_tts3.flow_time_cache import (
    TimeCacheUnavailable,
    TimeConditionCache,
)

logger = logging.getLogger(__name__)


class FlowGraphCapture:
    """Own capture streams, cached optimization weights, and precision gates."""

    def __init__(
        self,
        *,
        dit_fallback: torch.nn.Module | CompiledDiTExecutor,
        executor: CompiledDiTExecutor | None,
        dit_dtype: torch.dtype,
        t_span: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        optimize: bool,
        precompute_time_conditions: bool,
        optimize_convolution: bool,
        optimize_ffn: bool,
        precision_gate: FlowPrecisionGate,
        counters: FlowGraphCounters,
    ) -> None:
        self.dit_fallback = dit_fallback
        self.executor = executor
        self.dit_dtype = dit_dtype
        self.t_span = t_span
        self.n_timesteps = n_timesteps
        self.patch_size = patch_size
        self.optimize = optimize
        self.precompute_time_conditions = precompute_time_conditions
        self.optimize_convolution = optimize_convolution
        self.optimize_ffn = optimize_ffn
        self.precision_gate = precision_gate
        self.counters = counters
        self.time_cache: TimeConditionCache | None = None
        self.conv_cache: Conv1dWeightCache | None = None
        self.ffn_mode: FfnGemmMode | None = None
        self.resources = FlowGraphResources()

    def capture(
        self,
        history_latents: torch.Tensor,
        initial_noise: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        dit: torch.nn.Module | CompiledDiTExecutor,
    ) -> FlowGraphEntry | None:
        """Capture one whole-loop graph for the inputs' shape; None on failure."""
        # note (kermit): Clones prevent replay placeholders from aliasing caller inputs.
        flow_state = (
            torch.cat([history_latents, initial_noise], dim=1).contiguous().clone()
        )
        conditioning = conditioning.contiguous().clone()
        guidance_scales = torch.zeros(
            flow_state.shape[0], 1, 1, device=flow_state.device, dtype=torch.float32
        )

        executor = self.executor if dit is not self.dit_fallback else None
        time_table = None
        time_mode = None
        conv_mode = None
        ffn_mode = None
        modes: tuple[TorchDispatchMode, ...] = ()

        def body(
            state: torch.Tensor,
            condition: torch.Tensor,
            scales: torch.Tensor,
            active_modes: tuple[TorchDispatchMode, ...],
            *,
            observed: bool = False,
        ) -> torch.Tensor:
            active_time = time_mode if time_mode in active_modes else None
            active_conv = conv_mode if conv_mode in active_modes else None
            conv_hits = active_conv.hits if active_conv is not None else 0
            active_ffn = ffn_mode if ffn_mode in active_modes else None
            ffn_hits = active_ffn.hits if active_ffn is not None else 0
            result = euler_flow_loop(
                dit,
                state,
                condition,
                self.t_span,
                scales,
                self.n_timesteps,
                self.patch_size,
                dit_dtype=self.dit_dtype,
                executor=executor if observed else None,
                dispatch_modes=active_modes,
                time_mode=active_time,
            )
            if active_conv is not None:
                expected = len(active_conv.weights) * self.n_timesteps
                if expected == 0 or active_conv.hits - conv_hits != expected:
                    raise TimeCacheUnavailable(
                        "Compiled convolution schema was not fully handled"
                    )
                else:
                    pass
            else:
                pass
            if active_ffn is not None:
                expected = (
                    active_ffn.expected_hits(2 * state.shape[0] * state.shape[1])
                    * self.n_timesteps
                )
                if expected == 0 or active_ffn.hits - ffn_hits != expected:
                    raise RuntimeError("Compiled DiT GEMM schema was not fully handled")
                else:
                    pass
            else:
                pass
            return result

        probes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def capture_graph(
            active_modes: tuple[TorchDispatchMode, ...],
            gate: FlowPrecisionGate | None = None,
        ) -> torch.cuda.CUDAGraph:
            return self.resources.capture_graph(
                flow_state,
                conditioning,
                guidance_scales,
                active_modes,
                gate,
                probes,
                self.patch_size,
                body,
            )

        def check_precision(
            stage: str,
            actual: torch.Tensor,
            expected: torch.Tensor,
            gate: FlowPrecisionGate,
            probe: int,
        ) -> None:
            check_flow_precision(
                stage, actual, expected, gate, probe, flow_state.shape[0]
            )

        def capture_convolution(
            active_conv: Conv1dWeightCache,
            time_modes: tuple[TorchDispatchMode, ...],
        ) -> tuple[torch.cuda.CUDAGraph, tuple[TorchDispatchMode, ...]]:
            nonlocal conv_mode
            conv_mode = active_conv
            candidate_modes = time_modes + (active_conv,)
            for probe, (state, condition, scales, expected) in enumerate(probes):
                actual = body(
                    state.clone(), condition, scales, candidate_modes, observed=True
                )
                check_precision(
                    "convolution", actual, expected, self.precision_gate, probe
                )
            return capture_graph(candidate_modes, self.precision_gate), candidate_modes

        with torch.inference_mode():
            if self.optimize and executor is not None:
                try:
                    if self.precompute_time_conditions:
                        if self.time_cache is None:
                            self.time_cache = TimeConditionCache(executor.source)
                        else:
                            pass
                        dit_inputs = torch.cat([flow_state, conditioning], dim=2)
                        x_uncond = torch.cat([flow_state, conditioning * 0], dim=2)
                        example = torch.cat([dit_inputs, x_uncond], dim=0).to(
                            self.dit_dtype
                        )
                        time_table = self.time_cache.prepare(
                            executor, example, self.t_span, self.n_timesteps
                        )
                        time_mode = self.time_cache.mode(time_table)
                        modes += (time_mode,)
                    else:
                        pass
                    strict_gate = FlowPrecisionGate()
                    probes = prepare_flow_probes(
                        flow_state,
                        conditioning,
                        guidance_scales,
                        time_mode,
                        strict_gate,
                        body,
                    )
                    graph = capture_graph(modes, strict_gate)
                except Exception:
                    logger.warning(
                        f"Flow optimization rejected for bucket={flow_state.shape[0]}; "
                        "retrying the original compiled DiT",
                        exc_info=True,
                    )
                    self.counters["optimization_fallbacks"] += 1
                    modes, time_table, time_mode = (), None, None
                    try:
                        graph = capture_graph(())
                    except Exception:
                        logger.warning(
                            "Original compiled Flow capture failed", exc_info=True
                        )
                        return None
                else:
                    if self.optimize_convolution:
                        time_modes = modes
                        try:
                            if self.conv_cache is None:
                                self.conv_cache = Conv1dWeightCache(executor.source)
                            else:
                                pass
                            self.conv_cache.validate_weights()
                            try:
                                graph, modes = capture_convolution(
                                    self.conv_cache, time_modes
                                )
                            except Exception:
                                if not (
                                    self.dit_dtype is torch.bfloat16
                                    and flow_state.device.type == "cuda"
                                    and 2 * flow_state.shape[0]
                                    == CONTROLLED_CONV_INPUT_SHAPE[0]
                                    and flow_state.shape[1]
                                    == CONTROLLED_CONV_INPUT_SHAPE[-1]
                                    and torch.cuda.get_device_capability(
                                        flow_state.device
                                    )
                                    == (12, 0)
                                ):
                                    raise
                                else:
                                    pass
                                logger.info(
                                    f"Generic convolution rejected for bucket={flow_state.shape[0]}; "
                                    "trying the controlled convolution plan"
                                )
                                controlled_conv = Conv1dWeightCache(
                                    executor.source, use_controlled_plan=True
                                )
                                graph, modes = capture_convolution(
                                    controlled_conv, time_modes
                                )
                        except Exception:
                            logger.warning(
                                f"Flow optimization rejected for bucket={flow_state.shape[0]}: "
                                "convolution failed; retaining validated time cache",
                                exc_info=True,
                            )
                            self.counters["optimization_fallbacks"] += 1
                            modes = time_modes
                    else:
                        pass
                    if (
                        self.optimize_ffn
                        and conv_mode is not None
                        and conv_mode in modes
                        and flow_state.shape[0] in (1, 2, 4, 8, 16)
                        and flow_state.shape[1] == 12
                    ):
                        base_modes = modes
                        try:
                            if self.ffn_mode is None:
                                self.ffn_mode = FfnGemmMode(executor.source)
                            else:
                                pass
                            ffn_mode = self.ffn_mode
                            rows = 2 * flow_state.shape[0] * flow_state.shape[1]
                            if (
                                ffn_mode.enabled
                                and rows in QKV_CONFIGS
                                and ffn_mode.packed_compiled is None
                            ):
                                ffn_mode.prepare_packed(
                                    executor.compile_backend, flow_state.device
                                )
                            else:
                                pass
                            if not ffn_mode.supports(rows):
                                raise RuntimeError(ffn_mode.reason)
                            else:
                                pass
                            ffn_mode.validate_weights()
                            modes = (*base_modes, ffn_mode)
                            for probe, (
                                state,
                                condition,
                                scales,
                                expected,
                            ) in enumerate(probes):
                                actual = body(
                                    state.clone(),
                                    condition,
                                    scales,
                                    modes,
                                    observed=True,
                                )
                                if (
                                    flow_state.shape[0] in (8, 16)
                                    and time_mode is not None
                                ):
                                    uncached = body(
                                        state.clone(),
                                        condition,
                                        scales,
                                        tuple(
                                            mode
                                            for mode in modes
                                            if mode is not time_mode
                                        ),
                                        observed=True,
                                    )
                                    check_precision(
                                        "packed_time",
                                        actual,
                                        uncached,
                                        strict_gate,
                                        probe,
                                    )
                                else:
                                    pass
                                check_precision(
                                    "ffn",
                                    actual,
                                    expected,
                                    self.precision_gate,
                                    probe,
                                )
                            graph = capture_graph(modes, self.precision_gate)
                            self.counters["ffn_captures"] += 1
                        except Exception:
                            logger.warning(
                                f"FFN optimization rejected for bucket={flow_state.shape[0]}; "
                                "retaining validated time/convolution optimizations",
                                exc_info=True,
                            )
                            self.counters["ffn_fallbacks"] += 1
                            modes = base_modes
                            try:
                                graph = capture_graph(modes, self.precision_gate)
                            except Exception:
                                logger.warning(
                                    "Validated Flow baseline capture failed",
                                    exc_info=True,
                                )
                                return None
                    else:
                        pass
                    if modes:
                        self.counters["optimized_captures"] += 1
                    else:
                        pass
            else:
                try:
                    graph = capture_graph(())
                except Exception:
                    logger.warning(
                        f"FireRedTTS3 flow CUDA graph capture failed for flow_state={tuple(flow_state.shape)} "
                        f"conditioning={tuple(conditioning.shape)}",
                        exc_info=True,
                    )
                    return None
        logger.info(
            f"Captured FireRedTTS3 flow: flow_state={tuple(flow_state.shape)} conditioning={tuple(conditioning.shape)} "
            f"steps={self.n_timesteps} optimized={bool(modes)}"
        )
        return FlowGraphEntry(
            graph=graph,
            xt=flow_state,
            cond=conditioning,
            cfg=guidance_scales,
            time_table=time_table,
            dispatch_modes=modes,
            t_span=self.t_span,
        )
