# SPDX-License-Identifier: Apache-2.0
"""Graph capture for the model-owned graph paths, one backend per accelerator.

A model that captures its own graphs asks its platform for the backend instead of
naming torch.cuda. The graph object and the capture context's keywords differ per
accelerator, and SGLang resolves them explicitly for the same reason, so the
choice belongs on the platform rather than in a per-model branch.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any, Iterator, Protocol

import torch


class DeviceGraphBackend(Protocol):
    """Records a model-owned graph on one accelerator."""

    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
    ) -> AbstractContextManager[Any]:
        """Open a capture and yield the graph it records into."""
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
    ) -> Iterator[Any]:
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


class NpuDeviceGraphBackend:
    """Ascend NPU."""

    @contextmanager
    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
    ) -> Iterator[Any]:
        graph = torch.npu.NPUGraph()
        kwargs: dict[str, Any] = {}
        if pool is not None:
            kwargs["pool"] = pool
        if stream is not None:
            kwargs["stream"] = stream
        if thread_local_errors:
            kwargs["capture_error_mode"] = "thread_local"
        with torch.npu.graph(npu_graph=graph, **kwargs):
            yield graph


class XpuDeviceGraphBackend:
    """Intel XPU."""

    @contextmanager
    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
    ) -> Iterator[Any]:
        # Note (siju): XPU's graph context declares no capture_error_mode and
        # rejects it as a TypeError, so the request is dropped, not translated.
        del thread_local_errors
        graph = torch.xpu.XPUGraph()
        kwargs: dict[str, Any] = {}
        if pool is not None:
            kwargs["pool"] = pool
        if stream is not None:
            kwargs["stream"] = stream
        capture = torch.xpu.graph(xpu_graph=graph, **kwargs)
        with ExitStack() as stack:
            # The XPU generator allocates its seed and offset staging tensors on
            # the process's first capture and refills them on every later one, so
            # a capture under inference_mode would leave them inference tensors
            # and a later capture under no_grad could not refill them.
            with torch.inference_mode(False):
                stack.enter_context(capture)
            yield graph


__all__ = [
    "CudaDeviceGraphBackend",
    "DeviceGraphBackend",
    "NpuDeviceGraphBackend",
    "XpuDeviceGraphBackend",
]
