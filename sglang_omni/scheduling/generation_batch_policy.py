# SPDX-License-Identifier: Apache-2.0
"""Generation-stage batch policy helpers for SGLang-backed stages."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationBatchPolicyReport:
    model_name: str
    max_running_requests: int
    cuda_graph_enabled: bool
    cuda_graph_max_bs: int | None
    cuda_graph_bs: tuple[int, ...] | None
    torch_compile_enabled: bool
    torch_compile_max_bs: int | None
    model_buffer_bs: int | None


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


def sync_cuda_graph_bs_with_max_bs(
    overrides: dict[str, Any],
    server_args_overrides: dict[str, Any] | None,
) -> None:
    if not server_args_overrides:
        return
    if (
        "cuda_graph_max_bs" in server_args_overrides
        and "cuda_graph_bs" not in server_args_overrides
    ):
        overrides["cuda_graph_bs"] = build_default_cuda_graph_bs(
            int(overrides["cuda_graph_max_bs"])
        )


def validate_generation_batch_policy(
    *,
    model_name: str,
    server_args: Any,
    model_buffer_bs: int | None = None,
) -> GenerationBatchPolicyReport:
    errors: list[str] = []

    max_running_requests = _read_positive_int(
        server_args,
        "max_running_requests",
        errors,
    )
    cuda_graph_enabled = not bool(getattr(server_args, "disable_cuda_graph", False))

    cuda_graph_max_bs: int | None = None
    cuda_graph_bs: tuple[int, ...] | None = None
    if cuda_graph_enabled:
        cuda_graph_max_bs = _read_positive_int(
            server_args,
            "cuda_graph_max_bs",
            errors,
            required=True,
        )
        cuda_graph_bs_value = getattr(server_args, "cuda_graph_bs", None)
        if cuda_graph_bs_value is None:
            errors.append("cuda_graph_bs must be explicit when CUDA graph is enabled")
        else:
            cuda_graph_bs = _normalize_cuda_graph_bs(cuda_graph_bs_value, errors)

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

    torch_compile_enabled = bool(getattr(server_args, "enable_torch_compile", False))
    torch_compile_max_bs = _read_positive_int(
        server_args,
        "torch_compile_max_bs",
        errors,
        required=torch_compile_enabled,
    )
    if (
        torch_compile_enabled
        and max_running_requests is not None
        and torch_compile_max_bs is not None
        and torch_compile_max_bs < max_running_requests
    ):
        errors.append(
            "torch_compile_max_bs must cover max_running_requests "
            f"({torch_compile_max_bs} < {max_running_requests})"
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

    assert max_running_requests is not None
    return GenerationBatchPolicyReport(
        model_name=model_name,
        max_running_requests=max_running_requests,
        cuda_graph_enabled=cuda_graph_enabled,
        cuda_graph_max_bs=cuda_graph_max_bs,
        cuda_graph_bs=cuda_graph_bs,
        torch_compile_enabled=torch_compile_enabled,
        torch_compile_max_bs=torch_compile_max_bs,
        model_buffer_bs=normalized_model_buffer_bs,
    )


def _read_positive_int(
    obj: Any,
    field: str,
    errors: list[str],
    *,
    required: bool = True,
) -> int | None:
    value = getattr(obj, field, None)
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


def _normalize_cuda_graph_bs(
    value: Iterable[Any],
    errors: list[str],
) -> tuple[int, ...] | None:
    if isinstance(value, (str, bytes)):
        errors.append("cuda_graph_bs must be a sequence of positive integers")
        return None

    try:
        normalized = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        errors.append("cuda_graph_bs must be a sequence of positive integers")
        return None

    if not normalized:
        errors.append("cuda_graph_bs must be non-empty")
        return None
    if any(item < 1 for item in normalized):
        errors.append("cuda_graph_bs values must be >= 1")
        return None
    if tuple(sorted(set(normalized))) != normalized:
        errors.append("cuda_graph_bs must be strictly increasing")
        return None
    return normalized
