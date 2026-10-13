"""Scheduling limits for integrated single-request MLX execution."""

from collections.abc import Mapping


def apply_mlx_policy(
    overrides: dict[str, object], explicit: Mapping[str, object] | None
) -> None:
    required = {
        "max_running_requests": 1,
        "disable_cuda_graph": True,
        "disable_decode_cuda_graph": True,
        "cuda_graph_backend_prefill": "disabled",
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "enable_mixed_chunk": False,
        "enable_torch_compile": False,
        "chunked_prefill_size": -1,
        "tp_size": 1,
        "mlx_enable_sampling": False,
    }
    for name, value in required.items():
        if name in (explicit or {}) and explicit[name] != value:
            raise ValueError(f"Qwen3-Omni MLX requires {name}={value!r}")
        else:
            pass
    if overrides["tp_size"] != 1:
        raise ValueError("Qwen3-Omni MLX does not support tensor parallelism")
    else:
        pass
    overrides.update(required)
    overrides.setdefault("max_total_tokens", 8192)
    overrides["max_prefill_tokens"] = overrides["max_total_tokens"]
