# SPDX-License-Identifier: Apache-2.0
"""Each backend records into its own graph type with its own context keywords."""

from __future__ import annotations

import inspect
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.platforms.device_graph import (
    CudaDeviceGraphBackend,
    NpuDeviceGraphBackend,
    XpuDeviceGraphBackend,
    get_npu_graph_update_stream,
)


def _recording_module(graph_attr: str) -> SimpleNamespace:
    """A torch.cuda / torch.xpu stand-in that records how graph() was called."""
    calls: list[dict[str, object]] = []

    class _Graph:
        def __init__(self):
            self.updates = []
            self.replays = 0

        def update(self, **kwargs):
            self.updates.append(kwargs)

        def replay(self):
            self.replays += 1

    def graph(**kwargs):
        calls.append(kwargs)
        return nullcontext()

    return SimpleNamespace(
        calls=calls,
        graph=graph,
        set_device=lambda device: None,
        **{graph_attr: _Graph},
    )


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


@pytest.mark.parametrize("thread_local_errors", [False, True])
def test_npu_backend_records_into_an_npu_graph(
    monkeypatch, thread_local_errors
) -> None:
    module = _recording_module("NPUGraph")
    monkeypatch.setattr(torch, "npu", module, raising=False)
    pool = object()
    stream = object()

    with NpuDeviceGraphBackend().capture(
        pool=pool, stream=stream, thread_local_errors=thread_local_errors
    ) as graph:
        pass

    assert isinstance(graph, module.NPUGraph)
    expected = {"npu_graph": graph, "pool": pool, "stream": stream}
    if thread_local_errors:
        expected["capture_error_mode"] = "thread_local"
    assert module.calls == [expected]


def test_npu_graph_update_stream_is_shared(monkeypatch) -> None:
    class GraphDispatchMode:
        update_stream = None

        def __new__(cls):
            if cls.update_stream is None:
                cls.update_stream = object()
            return super().__new__(cls)

    graphs = types.ModuleType("torch_npu.npu.graphs")
    graphs._GraphDispatchMode = GraphDispatchMode
    npu = types.ModuleType("torch_npu.npu")
    npu.__path__ = []
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.__path__ = []
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "torch_npu.npu", npu)
    monkeypatch.setitem(sys.modules, "torch_npu.npu.graphs", graphs)

    first = get_npu_graph_update_stream()
    second = get_npu_graph_update_stream()

    assert first is GraphDispatchMode.update_stream
    assert second is first


def test_cuda_graph_context_declares_the_expected_keywords() -> None:
    """The stub tests accept any keyword, so pin the real CUDA context here."""
    cuda = inspect.signature(torch.cuda.graph).parameters
    assert "cuda_graph" in cuda and "capture_error_mode" in cuda


def test_xpu_graph_context_declares_the_expected_keywords() -> None:
    """Some non-XPU PyTorch distributions do not expose the XPU graph API."""
    try:
        xpu_graph = torch.xpu.graph
    except AttributeError:
        pytest.skip("this PyTorch distribution does not expose torch.xpu.graph")

    xpu = inspect.signature(xpu_graph).parameters
    assert "xpu_graph" in xpu and "capture_error_mode" not in xpu


@pytest.mark.parametrize(
    "backend",
    [CudaDeviceGraphBackend(), XpuDeviceGraphBackend(), NpuDeviceGraphBackend()],
)
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
    module.NPUGraph = module.CUDAGraph
    monkeypatch.setattr(torch, "cuda", module)
    monkeypatch.setattr(torch, "xpu", module)
    monkeypatch.setattr(torch, "npu", module, raising=False)

    with pytest.raises(ValueError), backend.capture():
        raise ValueError("capture body failed")

    assert exited == [True]
