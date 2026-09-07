# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for Ming talker graph capture."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang_omni.models.ming_omni.talker import (
    modeling_ming_omni_talker as talker_model,
)
from sglang_omni.models.ming_omni.talker.configuration_bailing_talker import (
    MingOmniTalkerConfig,
)
from sglang_omni.models.ming_omni.talker.device_runtime import TalkerDeviceRuntime


def test_cfm_graph_capture_uses_device_runtime(monkeypatch) -> None:
    events: list[object] = []
    graph = object()

    class _Runtime:
        def __init__(self, device):
            events.append(("runtime", device))

        def new_graph(self):
            events.append("new_graph")
            return graph

        @contextmanager
        def graph_context(self, captured_graph):
            events.append(("capture", captured_graph))
            yield

    class _CFM:
        def sample(self, _hidden, _history, noise, *_args, **_kwargs):
            return noise + 1

    monkeypatch.setattr(talker_model, "TalkerDeviceRuntime", _Runtime)
    executor = talker_model.CFMGraphExecutor(
        SimpleNamespace(steps=2, patch_size=2),
        _CFM(),
        lambda latents: latents + 2,
        lambda hidden: torch.stack((hidden[:, 0], hidden[:, 0] + 1), dim=-1),
    )
    input_tensor = torch.randn(1, 1, 4)
    history = torch.randn(1, 2, 4)
    noise = torch.randn(1, 2, 4)
    sde_noise = torch.randn(2, 1, 2, 4)

    executor._initialize_graph(input_tensor, history, noise, sde_noise)

    assert executor.initialized is True
    assert executor.graph is graph
    assert events == [
        ("runtime", input_tensor.device),
        "new_graph",
        ("capture", graph),
    ]


def test_use_torch_attention_overrides_both_talker_backends() -> None:
    config = object.__new__(MingOmniTalkerConfig)
    config.flowmodel = {"attn_backend": "flash_attn"}
    config.aggregator = {"attn_backend": "flash_attn"}

    config.use_torch_attention()

    assert config.flowmodel["attn_backend"] == "torch"
    assert config.aggregator["attn_backend"] == "torch"


def test_accelerator_device_runtime_delegates_stream_and_graph(monkeypatch) -> None:
    stream = object()
    graph = object()
    synchronize = Mock()
    module = SimpleNamespace(
        Stream=Mock(return_value=stream),
        stream=Mock(return_value=nullcontext()),
        current_stream=Mock(return_value=SimpleNamespace(synchronize=synchronize)),
        NPUGraph=Mock(return_value=graph),
        graph=Mock(return_value=nullcontext()),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device: module)

    runtime = TalkerDeviceRuntime("npu:2")
    with runtime.stream_context(runtime.new_stream()):
        pass
    runtime.synchronize()
    with runtime.graph_context(runtime.new_graph()):
        pass

    device = torch.device("npu:2")
    module.Stream.assert_called_once_with(device=device)
    module.stream.assert_called_once_with(stream)
    module.current_stream.assert_called_once_with(device)
    synchronize.assert_called_once_with()
    module.NPUGraph.assert_called_once_with()
    module.graph.assert_called_once_with(graph, capture_error_mode="thread_local")
