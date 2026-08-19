# SPDX-License-Identifier: Apache-2.0
"""Generation-stage batch policy helpers for SGLang-backed stages."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any

from sglang.srt.model_executor.cuda_graph_config import Backend as CudaGraphBackend
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig

logger = logging.getLogger(__name__)

_MISSING = object()

# A prefill replay falls back to eager when the padded bucket exceeds this
# multiple of the real token count.
_PREFILL_PADDING_FACTOR = 2


def get_decode_cuda_graph_max_bs(server_args: Any) -> Any:
    """Read the resolved SGLang decode CUDA Graph batch cap."""
    return server_args.cuda_graph_config.decode.max_bs


def get_decode_cuda_graph_bs(server_args: Any) -> Any:
    """Read the resolved SGLang decode CUDA Graph batch buckets."""
    return server_args.cuda_graph_config.decode.bs


def get_prefill_cuda_graph_backend(server_args: Any) -> str:
    """Read the resolved SGLang prefill CUDA graph backend."""
    return server_args.cuda_graph_config.prefill.backend


def build_default_cuda_graph_bs(max_bs: int) -> list[int]:
    max_bs = int(max_bs)
    if max_bs < 1:
        raise ValueError("max_bs must be >= 1")

    values = [1, 2, 4, 8, 12]
    values.extend(range(16, 257, 8))
    values.extend(range(272, 512, 16))
    values.extend(range(512, max_bs + 1, 32))
    values = [bs for bs in values if bs <= max_bs]
    if not values or values[-1] != max_bs:
        values.append(max_bs)
    return values


def build_default_prefill_cuda_graph_bs(max_num_tokens: int) -> list[int]:
    """Prefill token-count ladder for a declared token budget: fine-grained
    at the bottom so short extends pad minimally, coarsening upward. The cap
    is appended when off-grid so max(bs) matches the budget."""
    max_num_tokens = int(max_num_tokens)
    if max_num_tokens < 1:
        raise ValueError("max_num_tokens must be >= 1")

    values = [1, 2]
    values.extend(range(4, 33, 4))
    values.extend(range(48, 257, 16))
    values.extend(range(288, 513, 32))
    values.extend(range(576, 1025, 64))
    values.extend(range(1280, 4097, 256))
    values.extend(range(4608, max_num_tokens + 1, 512))
    values = [size for size in values if size <= max_num_tokens]
    if not values or values[-1] != max_num_tokens:
        values.append(max_num_tokens)
    return values


def resolve_prefill_cuda_graph_capture_cap(
    *,
    default_capture_budget: int,
    overrides: Mapping[str, Any],
    context_length: int | None = None,
) -> int:
    caps = [
        _normalize_positive_int(
            "prefill_cuda_graph_capture_budget", default_capture_budget
        )
    ]
    for field in (
        "cuda_graph_max_bs_prefill",
        "max_prefill_tokens",
        "max_total_tokens",
    ):
        value = overrides.get(field)
        if value is not None:
            caps.append(_normalize_positive_int(field, value))

    chunked_prefill_size = overrides.get("chunked_prefill_size")
    if chunked_prefill_size is not None and int(chunked_prefill_size) > 0:
        caps.append(
            _normalize_positive_int("chunked_prefill_size", chunked_prefill_size)
        )

    resolved_context_length = overrides.get("context_length", context_length)
    if resolved_context_length is not None:
        caps.append(_normalize_positive_int("context_length", resolved_context_length))
    return min(caps)


def resolve_prefill_cuda_graph_buckets(
    *,
    default_capture_budget: int | None,
    overrides: dict[str, Any],
    context_length: int | None = None,
    operator_supplied_buckets: bool = False,
) -> None:
    if overrides.get("cuda_graph_backend_prefill") == CudaGraphBackend.DISABLED:
        overrides.pop("cuda_graph_bs_prefill", None)
        overrides.pop("cuda_graph_max_bs_prefill", None)
        return

    buckets = overrides.get("cuda_graph_bs_prefill")
    if operator_supplied_buckets:
        normalized = _require_valid_cuda_graph_bs(
            buckets, field="cuda_graph_bs_prefill"
        )
        overrides["cuda_graph_max_bs_prefill"] = max(normalized)
        return

    if default_capture_budget is None:
        if buckets is None:
            return
        normalized = _require_valid_cuda_graph_bs(
            buckets, field="cuda_graph_bs_prefill"
        )
        default_capture_budget = max(normalized)

    capture_cap = resolve_prefill_cuda_graph_capture_cap(
        default_capture_budget=default_capture_budget,
        overrides=overrides,
        context_length=context_length,
    )
    resolved_buckets = build_default_prefill_cuda_graph_bs(capture_cap)
    overrides["cuda_graph_bs_prefill"] = resolved_buckets
    overrides["cuda_graph_max_bs_prefill"] = max(resolved_buckets)


def nested_prefill_overrides(overrides: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the prefill section of a nested cuda_graph_config override."""
    config = overrides.get("cuda_graph_config")
    if isinstance(config, CudaGraphConfig):
        config = config.to_dict()
    if not isinstance(config, Mapping):
        return {}
    prefill_config = config.get("prefill")
    return prefill_config if isinstance(prefill_config, Mapping) else {}


def build_generation_batch_overrides(
    *,
    max_running_requests: int,
    cuda_graph_max_bs: int | None = None,
    torch_compile_max_bs: int | None = None,
    prefill_cuda_graph_capture_budget: int | None = None,
    context_length: int | None = None,
    server_args_overrides: Mapping[str, Any] | None = None,
    **stage_defaults: Any,
) -> dict[str, Any]:
    incoming = dict(server_args_overrides or {})
    cuda_graph_config = incoming.get("cuda_graph_config")
    if isinstance(cuda_graph_config, CudaGraphConfig):
        cuda_graph_config = cuda_graph_config.to_dict()
    if isinstance(cuda_graph_config, Mapping):
        cuda_graph_config = dict(cuda_graph_config)
        prefill_config = cuda_graph_config.get("prefill")
        if isinstance(prefill_config, Mapping):
            cuda_graph_config["prefill"] = dict(prefill_config)
        incoming["cuda_graph_config"] = cuda_graph_config

    # note(ratish): the nested form wins in sglang; mirror its prefill
    # fields into the flat keys.
    nested_prefill = nested_prefill_overrides(incoming)
    nested_prefill_keys = frozenset(nested_prefill)
    for nested_key, flat_key in (
        ("backend", "cuda_graph_backend_prefill"),
        ("bs", "cuda_graph_bs_prefill"),
        ("max_bs", "cuda_graph_max_bs_prefill"),
    ):
        if nested_key not in nested_prefill:
            continue
        nested_value = nested_prefill[nested_key]
        if flat_key in incoming and incoming[flat_key] != nested_value:
            raise ValueError(
                f"Conflicting {flat_key} and cuda_graph_config prefill "
                f"{nested_key} values: "
                f"{incoming[flat_key]!r} != {nested_value!r}"
            )
        incoming[flat_key] = nested_value
    max_running_requests = _normalize_positive_int(
        "max_running_requests",
        incoming.pop("max_running_requests", max_running_requests),
    )
    cuda_graph_max_bs = (
        max_running_requests if cuda_graph_max_bs is None else cuda_graph_max_bs
    )
    cuda_graph_max_bs = _normalize_positive_int(
        "cuda_graph_max_bs",
        incoming.pop("cuda_graph_max_bs", cuda_graph_max_bs),
    )
    torch_compile_max_bs = (
        max_running_requests if torch_compile_max_bs is None else torch_compile_max_bs
    )
    torch_compile_max_bs = _normalize_positive_int(
        "torch_compile_max_bs",
        incoming.pop("torch_compile_max_bs", torch_compile_max_bs),
    )
    cuda_graph_bs = incoming.pop("cuda_graph_bs", _MISSING)

    overrides = {
        **stage_defaults,
        **incoming,
        "max_running_requests": max_running_requests,
        "cuda_graph_max_bs": cuda_graph_max_bs,
        "torch_compile_max_bs": torch_compile_max_bs,
    }
    if cuda_graph_bs is _MISSING:
        overrides["cuda_graph_bs"] = build_default_cuda_graph_bs(cuda_graph_max_bs)
    else:
        overrides["cuda_graph_bs"] = cuda_graph_bs

    # sglang resolves an explicit prefill backend after the disable flags;
    # without this, a stage-default backend overrides a deployment's disable.
    disables_prefill = bool(incoming.get("disable_cuda_graph")) or bool(
        incoming.get("disable_prefill_cuda_graph")
    )
    if disables_prefill and "cuda_graph_backend_prefill" not in incoming:
        overrides["cuda_graph_backend_prefill"] = CudaGraphBackend.DISABLED
        overrides.pop("cuda_graph_bs_prefill", None)
        overrides.pop("cuda_graph_max_bs_prefill", None)

    resolve_prefill_cuda_graph_buckets(
        default_capture_budget=prefill_cuda_graph_capture_budget,
        overrides=overrides,
        context_length=context_length,
        operator_supplied_buckets="cuda_graph_bs_prefill" in incoming,
    )

    # Keep fields supplied through the authoritative nested config aligned
    # with the resolved flat aliases that SGLang would otherwise overwrite.
    if isinstance(nested_prefill, dict):
        for nested_key, flat_key in (
            ("backend", "cuda_graph_backend_prefill"),
            ("bs", "cuda_graph_bs_prefill"),
            ("max_bs", "cuda_graph_max_bs_prefill"),
        ):
            if nested_key not in nested_prefill_keys:
                continue
            if flat_key in overrides:
                nested_prefill[nested_key] = overrides[flat_key]
            else:
                nested_prefill.pop(nested_key, None)

    return overrides


def validate_generation_batch_policy(
    *,
    model_name: str,
    server_args: Any,
    model_buffer_bs: int | None = None,
) -> None:
    errors: list[str] = []

    max_running_requests = _validate_positive_int(
        "max_running_requests",
        server_args.max_running_requests,
        errors,
    )
    cuda_graph_enabled = not bool(server_args.disable_cuda_graph)

    cuda_graph_max_bs: int | None = None
    cuda_graph_bs: tuple[int, ...] | None = None
    if cuda_graph_enabled:
        cuda_graph_max_bs = _validate_positive_int(
            "cuda_graph_max_bs",
            get_decode_cuda_graph_max_bs(server_args),
            errors,
            required=True,
        )
        cuda_graph_bs_value = get_decode_cuda_graph_bs(server_args)
        if cuda_graph_bs_value is None:
            errors.append("cuda_graph_bs must be explicit when CUDA graph is enabled")
        else:
            cuda_graph_bs = _normalize_cuda_graph_bs(
                cuda_graph_bs_value, errors, field="cuda_graph_bs"
            )

        if cuda_graph_max_bs is not None and cuda_graph_bs is not None:
            if max(cuda_graph_bs) != cuda_graph_max_bs:
                errors.append(
                    "max(cuda_graph_bs) must match cuda_graph_max_bs "
                    f"({max(cuda_graph_bs)} != {cuda_graph_max_bs})"
                )

        if (
            max_running_requests is not None
            and cuda_graph_max_bs is not None
            and cuda_graph_max_bs < max_running_requests
        ):
            errors.append(
                "cuda_graph_max_bs must cover max_running_requests "
                f"({cuda_graph_max_bs} < {max_running_requests})"
            )

    _validate_prefill_graph_policy(server_args, cuda_graph_enabled, errors)

    torch_compile_enabled = bool(server_args.enable_torch_compile)
    torch_compile_max_bs = _validate_positive_int(
        "torch_compile_max_bs",
        server_args.torch_compile_max_bs,
        errors,
        required=torch_compile_enabled,
    )
    normalized_model_buffer_bs: int | None = None
    if model_buffer_bs is not None:
        normalized_model_buffer_bs = int(model_buffer_bs)
        if normalized_model_buffer_bs < 1:
            errors.append("model_buffer_bs must be >= 1")
        if (
            max_running_requests is not None
            and normalized_model_buffer_bs < max_running_requests
        ):
            errors.append(
                "model_buffer_bs must cover max_running_requests "
                f"({normalized_model_buffer_bs} < {max_running_requests})"
            )

    if errors:
        raise ValueError(
            f"{model_name} invalid generation batch policy: " + "; ".join(errors)
        )


def _validate_prefill_graph_policy(
    server_args: Any,
    cuda_graph_enabled: bool,
    errors: list[str],
) -> None:
    """Validate the resolved prefill CUDA graph policy."""
    backend = get_prefill_cuda_graph_backend(server_args)
    if backend == CudaGraphBackend.DISABLED:
        return

    if not cuda_graph_enabled:
        errors.append(
            "prefill CUDA graphs require CUDA graphs enabled "
            f"(backend={backend!r} with disable_cuda_graph)"
        )
        return
    if backend != CudaGraphBackend.BREAKABLE:
        errors.append(
            "prefill CUDA graph backend must be 'breakable' or 'disabled', "
            f"got {backend!r}"
        )
        return

    incompatibilities = (
        ("context parallel (attn_cp_size > 1)", server_args.attn_cp_size > 1),
        ("decode context parallel (dcp_size > 1)", server_args.dcp_size > 1),
        ("LoRA", bool(server_args.lora_paths) or bool(server_args.enable_lora)),
        ("MoE A2A", server_args.moe_a2a_backend != "none"),
    )
    for feature, is_active in incompatibilities:
        if is_active:
            errors.append(
                f"breakable prefill CUDA graphs are incompatible with {feature}; "
                "set cuda_graph_backend_prefill='disabled'"
            )

    prefill_cfg = server_args.cuda_graph_config.prefill
    buckets = _normalize_cuda_graph_bs(
        prefill_cfg.bs, errors, field="cuda_graph_bs_prefill"
    )
    if buckets is None:
        return

    max_bs = prefill_cfg.max_bs
    if max_bs is not None and max(buckets) != int(max_bs):
        logger.warning(
            "max(cuda_graph_bs_prefill) must match cuda_graph_max_bs_prefill "
            "(%d != %s); the resolved bucket list determines replay coverage",
            max(buckets),
            max_bs,
        )

    # Buckets above either per-forward token cap can never replay.
    for cap_name, cap_value in (
        ("chunked_prefill_size", server_args.chunked_prefill_size),
        ("max_prefill_tokens", server_args.max_prefill_tokens),
    ):
        if (
            cap_value is not None
            and int(cap_value) > 0
            and max(buckets) > int(cap_value)
        ):
            logger.warning(
                "cuda_graph_bs_prefill buckets above %s are unreachable " "(%d > %s)",
                cap_name,
                max(buckets),
                cap_value,
            )

    valleys = []
    for previous, next_bucket in zip(buckets, buckets[1:]):
        eager_end = (next_bucket - 1) // _PREFILL_PADDING_FACTOR
        if eager_end > previous:
            valleys.append((previous + 1, eager_end))
    if valleys:
        logger.warning(
            "prefill CUDA graph bucket gaps exceed the %dx padding factor; "
            "prompt lengths inside %s fall back to eager",
            _PREFILL_PADDING_FACTOR,
            valleys,
        )


def _validate_positive_int(
    field: str,
    value: Any,
    errors: list[str],
    *,
    required: bool = True,
) -> int | None:
    if value is None:
        if required:
            errors.append(f"{field} must be explicit")
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        errors.append(f"{field} must be an integer")
        return None
    if normalized < 1:
        errors.append(f"{field} must be >= 1")
        return None
    return normalized


def _normalize_positive_int(field: str, value: Any) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if normalized < 1:
        raise ValueError(f"{field} must be >= 1")
    return normalized


def _normalize_cuda_graph_bs(
    value: Iterable[Any],
    errors: list[str],
    *,
    field: str,
) -> tuple[int, ...] | None:
    if isinstance(value, (str, bytes)):
        errors.append(f"{field} must be a sequence of positive integers")
        return None

    try:
        items = tuple(value)
    except TypeError:
        errors.append(f"{field} must be a sequence of positive integers")
        return None
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in items):
        errors.append(f"{field} must be a sequence of positive integers")
        return None
    normalized = tuple(int(item) for item in items)

    if not normalized:
        errors.append(f"{field} must be non-empty")
        return None
    if any(item < 1 for item in normalized):
        errors.append(f"{field} values must be >= 1")
        return None
    if tuple(sorted(set(normalized))) != normalized:
        errors.append(f"{field} must be strictly increasing")
        return None
    return normalized


def _require_valid_cuda_graph_bs(
    value: Iterable[Any], *, field: str
) -> tuple[int, ...]:
    errors: list[str] = []
    normalized = _normalize_cuda_graph_bs(value, errors, field=field)
    if normalized is None:
        raise ValueError("; ".join(errors))
    return normalized
