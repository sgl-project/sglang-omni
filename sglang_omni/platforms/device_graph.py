# SPDX-License-Identifier: Apache-2.0
"""Graph capture for the model-owned graph paths, one backend per accelerator.

A model that captures its own graphs asks its platform for the backend instead of
naming torch.cuda. The graph object and the capture context's keywords differ per
accelerator, and SGLang resolves them explicitly for the same reason, so the
choice belongs on the platform rather than in a per-model branch.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Literal, Protocol, TypedDict

import torch
from torch.cuda import _POOL_HANDLE as CudaGraphPoolHandle
from torch.xpu import _POOL_HANDLE as XpuGraphPoolHandle


class CudaCaptureKwargs(TypedDict, total=False):
    pool: CudaGraphPoolHandle
    stream: torch.cuda.Stream
    capture_error_mode: Literal["thread_local"]


class XpuCaptureKwargs(TypedDict, total=False):
    pool: XpuGraphPoolHandle
    stream: torch.xpu.Stream


class ReplayableGraph(Protocol):
    def replay(self) -> None: ...


class DeviceGraphBackend(Protocol):
    """Records a model-owned graph on one accelerator."""

    def capture(
        self,
        *,
        pool: CudaGraphPoolHandle | XpuGraphPoolHandle | tuple[int, int] | None = None,
        stream: torch.cuda.Stream | torch.xpu.Stream | torch.Stream | None = None,
        thread_local_errors: bool = False,
    ) -> AbstractContextManager[ReplayableGraph]:
        """Open a capture and yield the graph it records into."""
        ...


class CudaDeviceGraphBackend:
    """CUDA, and the backends that present through torch.cuda: HIP and MUSA."""

    @contextmanager
    def capture(
        self,
        *,
        pool: CudaGraphPoolHandle | None = None,
        stream: torch.cuda.Stream | None = None,
        thread_local_errors: bool = False,
    ) -> Iterator[torch.cuda.CUDAGraph]:
        graph = torch.cuda.CUDAGraph()
        kwargs: CudaCaptureKwargs = {}
        if pool is not None:
            kwargs["pool"] = pool
        else:
            pass
        if stream is not None:
            kwargs["stream"] = stream
        else:
            pass
        if thread_local_errors:
            kwargs["capture_error_mode"] = "thread_local"
        else:
            pass
        with torch.cuda.graph(cuda_graph=graph, **kwargs):
            yield graph


class NpuDeviceGraphBackend:
    """Ascend NPU."""

    @contextmanager
    def capture(
        self,
        *,
        pool: tuple[int, int] | None = None,
        stream: torch.Stream | None = None,
        thread_local_errors: bool = False,
    ) -> Iterator[ReplayableGraph]:
        graph = torch.npu.NPUGraph()
        kwargs: dict[str, object] = {}
        if pool is not None:
            kwargs["pool"] = pool
        else:
            pass
        if stream is not None:
            kwargs["stream"] = stream
        else:
            pass
        if thread_local_errors:
            kwargs["capture_error_mode"] = "thread_local"
        else:
            pass
        with torch.npu.graph(npu_graph=graph, **kwargs):
            yield graph


class XpuDeviceGraphBackend:
    """Intel XPU."""

    @contextmanager
    def capture(
        self,
        *,
        pool: XpuGraphPoolHandle | None = None,
        stream: torch.xpu.Stream | None = None,
        thread_local_errors: bool = False,
    ) -> Iterator[torch.xpu.XPUGraph]:
        # Note (siju): XPU's graph context declares no capture_error_mode and
        # rejects it as a TypeError, so the request is dropped, not translated.
        del thread_local_errors
        graph = torch.xpu.XPUGraph()
        kwargs: XpuCaptureKwargs = {}
        if pool is not None:
            kwargs["pool"] = pool
        else:
            pass
        if stream is not None:
            kwargs["stream"] = stream
        else:
            pass
        with torch.xpu.graph(xpu_graph=graph, **kwargs):
            yield graph


__all__ = [
    "CudaDeviceGraphBackend",
    "DeviceGraphBackend",
    "NpuDeviceGraphBackend",
    "ReplayableGraph",
    "XpuDeviceGraphBackend",
]
