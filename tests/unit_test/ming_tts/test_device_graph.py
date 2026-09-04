# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS reaches its accelerator by name, not through torch.cuda."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend

from sglang_omni import platforms
from sglang_omni.models.ming_tts.device_graph import (
    accelerator_graph_class,
    accelerator_module,
    graph_capture,
    graph_capture_attention,
)


def test_cpu_has_no_accelerator_module() -> None:
    """The CPU decoder is the verification path: no streams, no graphs."""
    assert accelerator_module(torch.device("cpu")) is None


def test_an_accelerator_resolves_to_its_own_torch_module() -> None:
    device = torch.device(platforms.current_platform.device_type, 0)
    if device.type == "cpu":
        pytest.skip("no accelerator on this host")

    assert accelerator_module(device) is torch.get_device_module(device)


@pytest.mark.parametrize("graph_attr", ["CUDAGraph", "XPUGraph"])
def test_the_graph_type_is_found_under_each_backend_name(graph_attr: str) -> None:
    class _Graph:
        pass

    module = SimpleNamespace(__name__="fake", **{graph_attr: _Graph})

    assert accelerator_graph_class(module) is _Graph


def test_a_module_with_no_graph_type_is_refused_by_name() -> None:
    module = SimpleNamespace(__name__="torch.fake")

    with pytest.raises(RuntimeError, match="exposes no graph type"):
        accelerator_graph_class(module)


class _RecordingGraphModule:
    """Stand-in for a torch device module's graph context."""

    __name__ = "fake"

    def __init__(self, *, accepts_error_mode: bool) -> None:
        self.calls: list[dict[str, object]] = []
        if accepts_error_mode:

            def graph(graph, pool=None, stream=None, capture_error_mode="global"):
                self.calls.append(
                    {
                        "graph": graph,
                        "stream": stream,
                        "capture_error_mode": capture_error_mode,
                    }
                )
                return nullcontext()

        else:

            def graph(graph, pool=None, stream=None):
                self.calls.append({"graph": graph, "stream": stream})
                return nullcontext()

        self.graph = graph


def test_thread_local_capture_errors_are_requested_where_they_exist() -> None:
    module = _RecordingGraphModule(accepts_error_mode=True)
    sentinel = object()

    graph_capture(module, sentinel, stream="s", thread_local_errors=True)

    assert module.calls == [
        {"graph": sentinel, "stream": "s", "capture_error_mode": "thread_local"}
    ]


def test_a_backend_without_capture_error_mode_still_captures() -> None:
    """Passing it there is a TypeError, not an ignored keyword."""
    module = _RecordingGraphModule(accepts_error_mode=False)
    sentinel = object()

    graph_capture(module, sentinel, stream="s", thread_local_errors=True)

    assert module.calls == [{"graph": sentinel, "stream": "s"}]


def test_a_caller_that_wants_the_backend_default_asks_for_nothing() -> None:
    module = _RecordingGraphModule(accepts_error_mode=True)
    sentinel = object()

    graph_capture(module, sentinel)

    assert module.calls == [
        {"graph": sentinel, "stream": None, "capture_error_mode": "global"}
    ]


def test_a_platform_that_names_no_sdpa_backend_keeps_its_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(
        platforms.current_platform,
        "get_graph_capture_sdpa_backends",
        lambda: (),
    )

    assert type(graph_capture_attention()) is type(nullcontext())


def test_a_platform_that_names_sdpa_backends_pins_them(monkeypatch) -> None:
    """Entering the pin narrows SDPA; the flags prove it rather than the type."""
    monkeypatch.setattr(
        platforms.current_platform,
        "get_graph_capture_sdpa_backends",
        lambda: (SDPBackend.MATH,),
    )

    with graph_capture_attention():
        assert torch.backends.cuda.math_sdp_enabled() is True
        assert torch.backends.cuda.flash_sdp_enabled() is False
        assert torch.backends.cuda.mem_efficient_sdp_enabled() is False
