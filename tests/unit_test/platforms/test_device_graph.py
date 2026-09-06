# SPDX-License-Identifier: Apache-2.0
"""Each backend records into its own graph type with its own context keywords."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.platforms.device_graph import (
    CudaDeviceGraphBackend,
    XpuDeviceGraphBackend,
)


def _recording_module(graph_attr: str) -> SimpleNamespace:
    """A torch.cuda / torch.xpu stand-in that records how graph() was called."""
    calls: list[dict[str, object]] = []

    class _Graph:
        pass

    def graph(**kwargs):
        calls.append(kwargs)
        return nullcontext()

    return SimpleNamespace(calls=calls, graph=graph, **{graph_attr: _Graph})


def test_cuda_backend_records_into_a_cuda_graph(monkeypatch) -> None:
    module = _recording_module("CUDAGraph")
    monkeypatch.setattr(torch, "cuda", module)
    pool = object()

    with CudaDeviceGraphBackend().capture(
        pool=pool, stream="s", thread_local_errors=True
    ) as graph:
        pass

    assert isinstance(graph, module.CUDAGraph)
    assert module.calls[-1] == {
        "cuda_graph": graph,
        "pool": pool,
        "stream": "s",
        # CUDA scopes a capture failure to the process unless asked otherwise.
        "capture_error_mode": "thread_local",
    }


def test_cuda_backend_asks_for_nothing_it_was_not_given(monkeypatch) -> None:
    module = _recording_module("CUDAGraph")
    monkeypatch.setattr(torch, "cuda", module)

    with CudaDeviceGraphBackend().capture() as graph:
        pass

    assert module.calls[-1] == {"cuda_graph": graph}


def test_xpu_backend_records_into_an_xpu_graph_without_error_mode(monkeypatch) -> None:
    """XPU's context declares no capture_error_mode and rejects it as TypeError."""
    module = _recording_module("XPUGraph")
    monkeypatch.setattr(torch, "xpu", module)
    pool = object()

    with XpuDeviceGraphBackend().capture(
        pool=pool, stream="s", thread_local_errors=True
    ) as graph:
        pass

    assert isinstance(graph, module.XPUGraph)
    assert module.calls[-1] == {"xpu_graph": graph, "pool": pool, "stream": "s"}


def test_each_backend_uses_the_keyword_its_torch_context_declares() -> None:
    """The stub tests above accept any keyword, so pin the real ones here.

    Both contexts are plain Python classes that a build without the device still
    exposes, so this runs anywhere.
    """
    cuda = inspect.signature(torch.cuda.graph).parameters
    xpu = inspect.signature(torch.xpu.graph).parameters

    assert "cuda_graph" in cuda and "capture_error_mode" in cuda
    assert "xpu_graph" in xpu and "capture_error_mode" not in xpu


@pytest.mark.parametrize("backend", [CudaDeviceGraphBackend(), XpuDeviceGraphBackend()])
def test_a_capture_that_raises_still_closes_its_context(backend, monkeypatch) -> None:
    """The graph context must exit on the body's exception, not swallow it."""
    exited: list[bool] = []

    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            exited.append(True)
            return False

    module = _recording_module("CUDAGraph")
    module.graph = lambda **kwargs: _Ctx()
    module.XPUGraph = module.CUDAGraph
    monkeypatch.setattr(torch, "cuda", module)
    monkeypatch.setattr(torch, "xpu", module)

    with pytest.raises(ValueError):
        with backend.capture():
            raise ValueError("capture body failed")

    assert exited == [True]
