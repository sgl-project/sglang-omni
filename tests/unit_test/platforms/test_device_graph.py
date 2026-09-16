# SPDX-License-Identifier: Apache-2.0
"""Each backend records into its own graph type with its own context keywords."""

from __future__ import annotations

import inspect
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.platforms.device_graph import (
    CudaDeviceGraphBackend,
    NpuDeviceGraphBackend,
    XpuDeviceGraphBackend,
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


def test_npu_backend_enables_and_applies_host_input_updates(monkeypatch) -> None:
    module = _recording_module("NPUGraph")
    monkeypatch.setattr(torch, "npu", module, raising=False)
    backend = NpuDeviceGraphBackend()

    with backend.capture(allow_host_input_update=True) as graph:
        pass

    updates = [{"actual_seq_lengths": [3, 8]}]
    device = SimpleNamespace(type="npu", index=3)
    backend.replay(graph, host_input_updates=updates, device=device)

    assert module.calls == [{"npu_graph": graph, "auto_dispatch_capture": True}]
    assert graph.updates == [{"cpu_update_input": updates}]
    assert graph.replays == 1


def test_npu_backend_pairs_host_update_with_replay(monkeypatch) -> None:
    update_started = threading.Event()
    replay_started = threading.Event()
    devices = []

    class Graph:
        def update(self, **kwargs):
            update_started.set()
            assert replay_started.wait(timeout=1)

        def replay(self):
            assert update_started.wait(timeout=1)
            replay_started.set()

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(set_device=devices.append),
        raising=False,
    )
    device = SimpleNamespace(type="npu", index=2)

    NpuDeviceGraphBackend().replay(
        Graph(), host_input_updates=[{"value": [1]}], device=device
    )

    assert devices == [device]


def test_npu_backend_propagates_host_update_failure(monkeypatch) -> None:
    class Graph:
        def update(self, **kwargs):
            raise ValueError("bad host input")

        def replay(self):
            pass

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(set_device=lambda device: None),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="host input update failed") as exc_info:
        NpuDeviceGraphBackend().replay(
            Graph(),
            host_input_updates=[{"value": [1]}],
            device=SimpleNamespace(type="npu", index=0),
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("backend", "module_name", "graph_attr"),
    [
        (CudaDeviceGraphBackend(), "cuda", "CUDAGraph"),
        (XpuDeviceGraphBackend(), "xpu", "XPUGraph"),
    ],
)
def test_non_npu_backends_reject_host_input_updates(
    monkeypatch, backend, module_name, graph_attr
) -> None:
    module = _recording_module(graph_attr)
    monkeypatch.setattr(torch, module_name, module)

    with (
        pytest.raises(ValueError, match="do not support host input updates"),
        backend.capture(allow_host_input_update=True),
    ):
        pass

    graph = getattr(module, graph_attr)()
    with pytest.raises(ValueError, match="do not support host input updates"):
        backend.replay(graph, host_input_updates=[{"value": [1]}])


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
