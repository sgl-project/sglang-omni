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


def test_cfm_graph_capture_uses_platform_backend(monkeypatch) -> None:
    events: list[object] = []
    graph = object()

    class _FakeGraphBackend:
        @contextmanager
        def capture(self, *, thread_local_errors):
            events.append(("capture", thread_local_errors))
            yield graph

    class _CFM:
        def sample(self, _hidden, _history, noise, *_args, **_kwargs):
            return noise + 1

    get_backend = Mock(return_value=_FakeGraphBackend())
    monkeypatch.setattr(
        talker_model,
        "current_platform",
        SimpleNamespace(get_device_graph_backend=get_backend),
    )
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
    get_backend.assert_called_once_with(input_tensor.device)
    assert events == [("capture", True)]


def test_use_torch_attention_overrides_both_talker_backends() -> None:
    config = object.__new__(MingOmniTalkerConfig)
    config.flowmodel = {"attn_backend": "flash_attn"}
    config.aggregator = {"attn_backend": "flash_attn"}

    config.use_torch_attention()

    assert config.flowmodel["attn_backend"] == "torch"
    assert config.aggregator["attn_backend"] == "torch"


def test_device_runtime_delegates_stream_and_synchronization(monkeypatch) -> None:
    stream = object()
    synchronize = Mock()
    device_module = SimpleNamespace(
        Stream=Mock(return_value=stream),
        stream=Mock(return_value=nullcontext()),
        current_stream=Mock(return_value=SimpleNamespace(synchronize=synchronize)),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device: device_module)

    device_runtime = TalkerDeviceRuntime("cuda:2")
    with device_runtime.create_stream_context(device_runtime.create_stream()):
        pass
    device_runtime.synchronize()

    device = torch.device("cuda:2")
    device_module.Stream.assert_called_once_with(device=device)
    device_module.stream.assert_called_once_with(stream)
    device_module.current_stream.assert_called_once_with(device)
    synchronize.assert_called_once_with()
