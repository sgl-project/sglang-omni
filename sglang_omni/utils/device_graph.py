# SPDX-License-Identifier: Apache-2.0
"""Small compatibility layer for CUDA/MUSA graph capture call sites."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch


def _backend(device: torch.device | str):
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda, "thread_local"
    if device.type == "musa":
        return torch.musa, "relaxed"
    raise RuntimeError(f"Device graphs require CUDA or MUSA, got {device.type!r}")


def graph_pool_handle(device: torch.device | str) -> Any:
    backend, _ = _backend(device)
    return backend.graph_pool_handle()


def new_graph(device: torch.device | str) -> Any:
    backend, _ = _backend(device)
    graph_cls = getattr(backend, "CUDAGraph", None) or getattr(backend, "MUSAGraph")
    return graph_cls()


def new_event(device: torch.device | str | None = None) -> Any:
    backend, _ = _backend(device or torch.device("cuda"))
    return backend.Event()


def new_stream(
    device: torch.device | str | None = None,
    *,
    priority: int | None = None,
) -> Any:
    backend, _ = _backend(device or torch.device("cuda"))
    kwargs: dict[str, Any] = {}
    if priority is not None:
        kwargs["priority"] = priority
    if device is None:
        return backend.Stream(**kwargs)
    return backend.Stream(device=torch.device(device), **kwargs)


def current_stream(device: torch.device | str | None = None) -> Any:
    backend, _ = _backend(device or torch.device("cuda"))
    if device is None:
        return backend.current_stream()
    return backend.current_stream(torch.device(device))


def stream_priority_range(device: torch.device | str) -> tuple[int, int]:
    backend, _ = _backend(device)
    priority_range = getattr(backend.Stream, "priority_range", None)
    if callable(priority_range):
        return priority_range()
    return (0, 0)


@contextmanager
def stream(stream_obj: Any):
    backend, _ = _backend(stream_obj.device)
    with backend.stream(stream_obj):
        yield


def synchronize(device: torch.device | str | None = None) -> None:
    backend, _ = _backend(device or torch.device("cuda"))
    if device is None:
        backend.synchronize()
    else:
        backend.synchronize(torch.device(device))


@contextmanager
def graph(graph_obj: Any, *, device: torch.device | str, **kwargs: Any):
    backend, capture_error_mode = _backend(device)
    kwargs.setdefault("capture_error_mode", capture_error_mode)
    with backend.graph(graph_obj, **kwargs):
        yield


@contextmanager
def device_context(device: torch.device | str):
    backend, _ = _backend(device)
    with backend.device(torch.device(device)):
        yield


def mem_get_info(device: torch.device | str) -> tuple[int, int]:
    backend, _ = _backend(device)
    return backend.mem_get_info(torch.device(device))


def is_current_stream_capturing(device: torch.device | str) -> bool:
    backend, _ = _backend(device)
    is_capturing = getattr(backend, "is_current_stream_capturing", None)
    if not callable(is_capturing):
        return False
    return bool(is_capturing())
