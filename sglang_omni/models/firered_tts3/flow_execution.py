# SPDX-License-Identifier: Apache-2.0
"""DiT execution modes and Euler integration for FireRedTTS3."""

from __future__ import annotations

import copy
import os
from contextlib import nullcontext
from typing import Literal

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang_omni.models.firered_tts3.compiled_dit import CompiledDiTExecutor
from sglang_omni.models.firered_tts3.flow_time_cache import TimeGemmMode

# note (kermit): Euler state and guidance stay fp32 even when DiT uses bf16.
FLOW_MODES = ("fp32", "compile_fp32", "bf16", "compile_bf16")
FlowMode = Literal["fp32", "compile_fp32", "bf16", "compile_bf16"]


def flow_dit_dtype(mode: str) -> torch.dtype:
    if mode not in FLOW_MODES:
        raise ValueError(f"unknown flow mode {mode!r}; expected one of {FLOW_MODES}")
    else:
        pass
    return torch.bfloat16 if mode.endswith("bf16") else torch.float32


def flow_optimizations(mode: FlowMode) -> tuple[bool, bool]:
    is_compiled = mode.startswith("compile")
    time_cache_setting = os.environ.get(
        "FIRERED_TTS3_PRECOMPUTE_TIME", "1" if is_compiled else "0"
    )
    convolution_layout = os.environ.get(
        "FIREREDTTS3_CONV_LAYOUT", "conv2d_channels_last" if is_compiled else "conv1d"
    )
    if time_cache_setting not in ("0", "1"):
        raise ValueError("FIRERED_TTS3_PRECOMPUTE_TIME must be 0 or 1")
    else:
        pass
    if convolution_layout not in ("conv1d", "conv2d_channels_last"):
        raise ValueError(
            "FIREREDTTS3_CONV_LAYOUT must be conv1d or conv2d_channels_last"
        )
    else:
        pass
    return time_cache_setting == "1", convolution_layout == "conv2d_channels_last"


def flow_ffn_optimization(mode: FlowMode) -> bool:
    value = os.environ.get("FIRERED_TTS3_FFN_GEMM", "1")
    if value not in ("0", "1"):
        raise ValueError("FIRERED_TTS3_FFN_GEMM must be 0 or 1")
    else:
        return value == "1" and mode == "compile_bf16"


def prepare_flow_dit(
    dit: torch.nn.Module, mode: FlowMode
) -> tuple[torch.nn.Module | CompiledDiTExecutor, torch.nn.Module]:
    """Build the DiT execution stack and its same-dtype eager fallback."""
    if mode not in FLOW_MODES:
        raise ValueError(f"unknown flow mode {mode!r}; expected one of {FLOW_MODES}")
    else:
        pass
    time_cache, conv_cache = flow_optimizations(mode)
    ffn_gemm = flow_ffn_optimization(mode)
    if (time_cache or conv_cache) and not mode.startswith("compile"):
        raise ValueError("Flow time/convolution caches require a compile flow mode")
    else:
        pass
    if mode == "fp32":
        return dit, dit
    else:
        pass
    # note (kermit): Module.to mutates in place, so preserve the fp32 fallback.
    if mode.endswith("bf16"):
        dit = copy.deepcopy(dit).to(dtype=torch.bfloat16)
    else:
        pass
    # note (kermit): Disable epilogue fusion to preserve the DiT precision gate.
    if mode.startswith("compile"):
        if time_cache or conv_cache or ffn_gemm:
            return CompiledDiTExecutor(dit), dit
        else:
            pass
        return (
            torch.compile(dit, dynamic=False, options={"epilogue_fusion": False}),
            dit,
        )
    else:
        pass
    return dit, dit


def flow_t_span(n_timesteps: int, device: torch.device) -> torch.Tensor:
    """The cosine t_span schedule (fp32 red line), n_timesteps + 1 knots."""
    assert n_timesteps > 0, "Flow requires a positive timestep count"
    t = torch.linspace(0.0, 1.0, n_timesteps + 1, device=device, dtype=torch.float32)
    return 1.0 - torch.cos(t * 0.5 * torch.pi)


def euler_flow_loop(
    dit: torch.nn.Module | CompiledDiTExecutor,
    xt: torch.Tensor,
    cond: torch.Tensor,
    t_span: torch.Tensor,
    cfg: torch.Tensor,
    n_timesteps: int,
    patch_size: int,
    *,
    dit_dtype: torch.dtype = torch.float32,
    executor: CompiledDiTExecutor | None = None,
    dispatch_modes: tuple[TorchDispatchMode, ...] = (),
    time_mode: TimeGemmMode | None = None,
) -> torch.Tensor:
    """Integrate the flow state in place and return its patch tail."""
    assert n_timesteps > 0, "Flow requires a positive timestep count"
    flow_state, conditioning, guidance_scales = xt, cond, cfg
    batch_size = flow_state.shape[0]
    for timestep_index in range(n_timesteps):
        t = t_span[timestep_index]
        dt = t_span[timestep_index + 1] - t
        dit_inputs = torch.cat([flow_state, conditioning], dim=2)
        dit_timesteps = t.view(-1, 1, 1).expand(batch_size, -1, -1)
        unconditional_inputs = torch.cat([flow_state, conditioning * 0], dim=2)
        dit_inputs = torch.cat([dit_inputs, unconditional_inputs], dim=0)
        dit_timesteps = torch.cat([dit_timesteps, dit_timesteps], dim=0)
        with (
            (
                executor.dispatch(dispatch_modes)
                if executor is not None
                else nullcontext()
            ),
            time_mode.step(timestep_index) if time_mode is not None else nullcontext(),
        ):
            run_dit = executor.run if executor is not None else dit
            if dit_dtype is torch.float32:
                velocity = run_dit(x=dit_inputs, t=dit_timesteps)
            else:
                velocity = run_dit(
                    x=dit_inputs.to(dit_dtype), t=dit_timesteps.to(dit_dtype)
                ).float()
        conditional_velocity, unconditional_velocity = velocity.chunk(2, dim=0)
        velocity = (
            1.0 + guidance_scales
        ) * conditional_velocity - guidance_scales * unconditional_velocity
        flow_state[:, -patch_size:] = (
            flow_state[:, -patch_size:] + dt.view(-1, 1, 1) * velocity[:, -patch_size:]
        )
    return flow_state[:, -patch_size:]
