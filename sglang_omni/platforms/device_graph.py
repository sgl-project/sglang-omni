# SPDX-License-Identifier: Apache-2.0
"""Graph capture for the model-owned graph paths, one backend per accelerator.

A model that captures its own graphs asks its platform for the backend instead of
naming torch.cuda. The graph object and the capture context's keywords differ per
accelerator, and SGLang resolves them explicitly for the same reason, so the
choice belongs on the platform rather than in a per-model branch.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

import torch


class DeviceGraphBackend(Protocol):
    """Records a model-owned graph on one accelerator."""

    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
        allow_host_input_update: bool = False,
    ) -> AbstractContextManager[Any]:
        """Open a capture and yield the graph it records into."""
        ...

    def replay(
        self,
        graph: Any,
        *,
        host_input_updates: list[dict[str, Any]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Replay a graph, applying backend-supported host input updates first."""
        ...


class CudaDeviceGraphBackend:
    """CUDA, and the backends that present through torch.cuda: HIP and MUSA."""

    @contextmanager
    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
        allow_host_input_update: bool = False,
    ) -> Iterator[Any]:
        if allow_host_input_update:
            raise ValueError("CUDA graphs do not support host input updates")
        graph = torch.cuda.CUDAGraph()
        kwargs: dict[str, Any] = {}
        if pool is not None:
            kwargs["pool"] = pool
        if stream is not None:
            kwargs["stream"] = stream
        if thread_local_errors:
            kwargs["capture_error_mode"] = "thread_local"
        with torch.cuda.graph(cuda_graph=graph, **kwargs):
            yield graph

    def replay(
        self,
        graph: Any,
        *,
        host_input_updates: list[dict[str, Any]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        del device
        if host_input_updates is not None:
            raise ValueError("CUDA graphs do not support host input updates")
        graph.replay()


class NpuDeviceGraphBackend:
    """Ascend NPU."""

    @contextmanager
    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
        allow_host_input_update: bool = False,
    ) -> Iterator[Any]:
        graph = torch.npu.NPUGraph()
        kwargs: dict[str, Any] = {}
        if pool is not None:
            kwargs["pool"] = pool
        if stream is not None:
            kwargs["stream"] = stream
        if thread_local_errors:
            kwargs["capture_error_mode"] = "thread_local"
        if allow_host_input_update:
            kwargs["auto_dispatch_capture"] = True
        with torch.npu.graph(npu_graph=graph, **kwargs):
            yield graph

    def replay(
        self,
        graph: Any,
        *,
        host_input_updates: list[dict[str, Any]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        if host_input_updates is None:
            graph.replay()
            return
        if device is None:
            raise ValueError("NPU host input updates require the graph device")

        update_error: list[BaseException] = []

        def update() -> None:
            try:
                # Device selection is thread-local in torch_npu.
                torch.npu.set_device(device)
                graph.update(cpu_update_input=host_input_updates)
            except BaseException as exc:  # noqa: BLE001
                update_error.append(exc)

        # torch_npu pairs update and replay concurrently; completing update first
        # can block waiting for replay to consume the new host parameters.
        update_thread = threading.Thread(target=update)
        update_thread.start()
        try:
            graph.replay()
        finally:
            update_thread.join()
        if update_error:
            raise RuntimeError("NPU graph host input update failed") from update_error[
                0
            ]


class XpuDeviceGraphBackend:
    """Intel XPU."""

    @contextmanager
    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
        allow_host_input_update: bool = False,
    ) -> Iterator[Any]:
        # Note (siju): XPU's graph context declares no capture_error_mode and
        # rejects it as a TypeError, so the request is dropped, not translated.
        del thread_local_errors
        if allow_host_input_update:
            raise ValueError("XPU graphs do not support host input updates")
        graph = torch.xpu.XPUGraph()
        kwargs: dict[str, Any] = {}
        if pool is not None:
            kwargs["pool"] = pool
        if stream is not None:
            kwargs["stream"] = stream
        with torch.xpu.graph(xpu_graph=graph, **kwargs):
            yield graph

    def replay(
        self,
        graph: Any,
        *,
        host_input_updates: list[dict[str, Any]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        del device
        if host_input_updates is not None:
            raise ValueError("XPU graphs do not support host input updates")
        graph.replay()


__all__ = [
    "CudaDeviceGraphBackend",
    "DeviceGraphBackend",
    "NpuDeviceGraphBackend",
    "XpuDeviceGraphBackend",
]
