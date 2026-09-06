# SPDX-License-Identifier: Apache-2.0
"""Shared ServerArgs construction for SGLang AR engines."""
from __future__ import annotations

from typing import Any

from sglang.srt.server_args import ServerArgs

from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend
from sglang_omni.vendor.sglang.server_args import override_server_args

_DECODE_CUDA_GRAPH_ALIASES = {
    "cuda_graph_max_bs": "cuda_graph_max_bs_decode",
    "cuda_graph_bs": "cuda_graph_bs_decode",
}


def _platform_device_type() -> str:
    from sglang_omni.platforms import current_platform

    return current_platform.device_type


def _normalize_decode_cuda_graph_overrides(kwargs: dict[str, Any]) -> None:
    """Translate Omni's legacy public knobs to SGLang's decode fields."""
    for legacy_name, decode_name in _DECODE_CUDA_GRAPH_ALIASES.items():
        if legacy_name not in kwargs:
            continue
        legacy_value = kwargs.pop(legacy_name)
        if decode_name in kwargs and kwargs[decode_name] != legacy_value:
            raise ValueError(
                f"Conflicting {legacy_name} and {decode_name} values: "
                f"{legacy_value!r} != {kwargs[decode_name]!r}"
            )
        kwargs[decode_name] = legacy_value


def _apply_platform_decode_cuda_graph_backend(kwargs: dict[str, Any]) -> None:
    """SGLang applies this after its disable switches, and a stage may name cpu
    on an accelerator host, so both are checked before it is set."""
    from sglang_omni.platforms import current_platform

    backend = current_platform.get_decode_cuda_graph_backend()
    if backend is None:
        return
    device = str(kwargs.get("device") or "").split(":")[0]
    if device != current_platform.device_type:
        return
    if kwargs.get("disable_cuda_graph") or kwargs.get("disable_decode_cuda_graph"):
        return
    kwargs.setdefault("cuda_graph_backend_decode", backend)


def build_sglang_server_args(
    model_path: str,
    context_length: int,
    *,
    chunked_prefill_size: int | None = None,
    max_prefill_tokens: int = 16384,
    max_running_requests: int = 16,
    mem_fraction_static: float | None = None,
    **overrides: Any,
) -> ServerArgs:
    """Build ServerArgs with shared defaults for all SGLang AR engines."""
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": True,
        "tp_size": 1,
        "pp_size": 1,
        "chunked_prefill_size": chunked_prefill_size,
        "max_prefill_tokens": max_prefill_tokens,
        "max_running_requests": max_running_requests,
        "random_seed": 123,
        "context_length": context_length,
    }
    if mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = mem_fraction_static
    kwargs.update(overrides)
    _normalize_decode_cuda_graph_overrides(kwargs)
    # Existing Omni models remain eager-prefill by default. Models that have
    # adapted SGLang's phase-specific prefill contract opt in explicitly
    # through their generation defaults / server overrides.
    kwargs.setdefault("cuda_graph_backend_prefill", CudaGraphBackend.DISABLED)
    if kwargs.get("mem_fraction_static") is None:
        kwargs.pop("mem_fraction_static", None)
    kwargs.setdefault("device", _platform_device_type())
    _apply_platform_decode_cuda_graph_backend(kwargs)
    server_args = ServerArgs(**kwargs)
    # DP attention is unsupported; reject at configuration time. Mixed
    # chunked prefill stays allowed (the bridge handles it natively).
    if server_args.enable_dp_attention:
        raise ValueError("sglang-omni does not support enable_dp_attention")
    # Overlapped startup weight load leaves sentinel weights until the scheduler
    # calls finalize_startup_weight_load after capture; omni's bootstrap never
    # does, so profiling, weight sharing and capture would run on the sentinels.
    if server_args.startup_weight_load_mode == "overlap":
        raise ValueError(
            "sglang-omni does not support startup_weight_load_mode='overlap'"
        )
    return server_args


def apply_encoder_mem_reserve(
    server_args: ServerArgs,
    encoder_mem_reserve: float,
) -> None:
    """Subtract Qwen external encoder headroom from an auto-selected SGLang budget."""
    if not 0.0 <= encoder_mem_reserve < 1.0:
        raise ValueError("encoder_mem_reserve must be in [0, 1)")
    if encoder_mem_reserve == 0:
        return

    current = server_args.mem_fraction_static
    if current is None:
        return

    reserved = current - encoder_mem_reserve
    if reserved < 0.1:
        raise ValueError(
            f"auto mem_fraction_static {current:.3f} minus encoder_mem_reserve "
            f"{encoder_mem_reserve:.3f} = {reserved:.3f} is below the safe "
            "floor 0.1; lower encoder_mem_reserve or pin mem_fraction_static "
            "explicitly."
        )
    override_server_args(
        server_args,
        "sglang_omni.encoder_mem_reserve",
        mem_fraction_static=round(reserved, 3),
    )
