# SPDX-License-Identifier: Apache-2.0
"""Each backend records into its own graph type with its own context keywords."""

from __future__ import annotations

import inspect
import runpy
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Literal

import pytest
import torch

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import (
    CudaDeviceGraphBackend,
    NpuDeviceGraphBackend,
    XpuDeviceGraphBackend,
)
from tests.unit_test.fixtures.accelerator import require_device_streams


def test_graph_backend_import_without_xpu_pool_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(torch.xpu, "_POOL_HANDLE", raising=False)
    module = runpy.run_path(inspect.getfile(CudaDeviceGraphBackend))

    assert module["XpuGraphPoolHandle"] is module["CudaGraphPoolHandle"]


def recording_module(graph_attr: str) -> SimpleNamespace:
    """A torch.cuda / torch.xpu stand-in that records how graph() was called."""
    calls: list[dict[str, object]] = []

    class Graph:
        pass

    def graph(**kwargs):
        calls.append(kwargs)
        return nullcontext()

    return SimpleNamespace(calls=calls, graph=graph, **{graph_attr: Graph})


def test_cuda_backend_records_into_a_cuda_graph(monkeypatch) -> None:
    module = recording_module("CUDAGraph")
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
    module = recording_module("CUDAGraph")
    monkeypatch.setattr(torch, "cuda", module)

    with CudaDeviceGraphBackend().capture() as graph:
        pass

    assert module.calls[-1] == {"cuda_graph": graph}


def test_xpu_backend_records_into_an_xpu_graph_without_error_mode(monkeypatch) -> None:
    """XPU's context declares no capture_error_mode and rejects it as TypeError."""
    module = recording_module("XPUGraph")
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
    module = recording_module("NPUGraph")
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


@pytest.mark.parametrize(
    "backend,graph_keyword,has_capture_error_mode",
    [
        ("cuda", "cuda_graph", True),
        pytest.param(
            "xpu",
            "xpu_graph",
            False,
            marks=pytest.mark.skipif(
                not hasattr(torch.xpu, "graph"),
                reason="This PyTorch build does not provide XPU graph contexts",
            ),
        ),
    ],
)
def test_each_backend_uses_the_keyword_its_torch_context_declares(
    backend: Literal["cuda", "xpu"],
    graph_keyword: str,
    has_capture_error_mode: bool,
) -> None:
    """Verify the real graph context keywords for each available backend."""
    graph_context = torch.cuda.graphs.graph if backend == "cuda" else torch.xpu.graph
    parameters = inspect.signature(graph_context).parameters

    assert graph_keyword in parameters
    assert ("capture_error_mode" in parameters) == has_capture_error_mode


@pytest.mark.parametrize(
    "backend",
    [CudaDeviceGraphBackend(), XpuDeviceGraphBackend(), NpuDeviceGraphBackend()],
)
def test_a_capture_that_raises_still_closes_its_context(backend, monkeypatch) -> None:
    """The graph context must see the body's exception, and must not swallow it."""
    seen: list[tuple] = []

    class Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            seen.append(args)
            return False

    module = recording_module("CUDAGraph")
    module.graph = lambda **kwargs: Ctx()
    module.XPUGraph = module.CUDAGraph
    module.NPUGraph = module.CUDAGraph
    monkeypatch.setattr(torch, "cuda", module)
    monkeypatch.setattr(torch, "xpu", module)
    monkeypatch.setattr(torch, "npu", module, raising=False)

    with pytest.raises(ValueError, match="capture body failed"):
        with backend.capture():
            raise ValueError("capture body failed")

    assert len(seen) == 1
    exc_type, exc_value, traceback = seen[0]
    assert exc_type is ValueError
    assert str(exc_value) == "capture body failed"
    assert traceback is not None


@pytest.mark.parametrize(
    "backend",
    [CudaDeviceGraphBackend(), XpuDeviceGraphBackend(), NpuDeviceGraphBackend()],
)
def test_a_graph_context_that_suppresses_is_honored(backend, monkeypatch) -> None:
    """A context that reports the exception handled leaves the caller running."""

    class Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return True

    module = recording_module("CUDAGraph")
    module.graph = lambda **kwargs: Ctx()
    module.XPUGraph = module.CUDAGraph
    module.NPUGraph = module.CUDAGraph
    monkeypatch.setattr(torch, "cuda", module)
    monkeypatch.setattr(torch, "xpu", module)
    monkeypatch.setattr(torch, "npu", module, raising=False)

    with backend.capture():
        raise ValueError("capture body failed")


@pytest.mark.parametrize(
    "backend, graph_attr, module_name",
    [
        (CudaDeviceGraphBackend(), "CUDAGraph", "cuda"),
        (NpuDeviceGraphBackend(), "NPUGraph", "npu"),
        (XpuDeviceGraphBackend(), "XPUGraph", "xpu"),
    ],
)
def test_each_backend_pools_through_its_own_module(
    backend, graph_attr, module_name, monkeypatch
) -> None:
    """The pool must come from the runtime that records the graph: MUSA records
    through torch.cuda while answering streams from its own module."""
    handle = (7, 11)
    module = recording_module(graph_attr)
    module.graph_pool_handle = lambda: handle
    monkeypatch.setattr(torch, module_name, module, raising=False)

    assert backend.graph_pool_handle() is handle


def test_xpu_backend_opens_the_capture_outside_inference_mode(monkeypatch) -> None:
    """XPU's capture_begin updates the device generator state in place. Inside
    inference mode that state becomes an inference tensor, and an inference tensor
    cannot be updated in place afterwards, so every later capture in the process is
    refused."""
    begin_modes: list[bool] = []

    class Ctx:
        def __enter__(self):
            begin_modes.append(torch.is_inference_mode_enabled())
            return None

        def __exit__(self, *args):
            return False

    module = recording_module("XPUGraph")
    module.graph = lambda **kwargs: Ctx()
    monkeypatch.setattr(torch, "xpu", module)

    with torch.inference_mode():
        with XpuDeviceGraphBackend().capture():
            body_mode = torch.is_inference_mode_enabled()

    assert begin_modes == [False]
    assert body_mode is True


@pytest.mark.accelerator
@pytest.mark.parametrize("model_owned_first", [True, False], ids=["model", "engine"])
def test_captures_in_either_mode_leave_later_captures_working(
    model_owned_first: bool,
) -> None:
    """Model-owned runners capture under inference mode and SGLang captures under
    no_grad, in whichever order a stage happens to build them. Neither may leave
    the device unable to open the next capture."""
    device = require_device_streams()
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    static = torch.zeros(8, device=device)
    with torch.inference_mode():
        for _ in range(2):
            static.sin().sum()
            static.cos().sum()

    if model_owned_first:
        modes = [torch.inference_mode(), torch.no_grad()]
    else:
        modes = [torch.no_grad(), torch.inference_mode()]

    outputs = []
    graphs = []
    for mode in modes:
        with mode, backend.capture() as graph:
            outputs.append(static.cos().sum())
        graphs.append(graph)

    static.fill_(0.0)
    for graph in graphs:
        graph.replay()
    torch.get_device_module(device).synchronize(device)

    for output in outputs:
        assert float(output) == pytest.approx(8.0, abs=1e-4)
