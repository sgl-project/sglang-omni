# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for Ming talker graph capture."""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
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

    class FakeGraphBackend:
        @contextmanager
        def capture(self, *, thread_local_errors):
            events.append(("capture", thread_local_errors))
            yield graph

    class CFM:
        def sample(self, _hidden, _history, noise, *args, **_kwargs):
            return noise + 1

    get_backend = Mock(return_value=FakeGraphBackend())
    monkeypatch.setattr(
        talker_model,
        "current_platform",
        SimpleNamespace(get_device_graph_backend=get_backend, is_cuda=lambda: False),
    )
    executor = talker_model.CFMGraphExecutor(
        SimpleNamespace(steps=2, patch_size=2),
        CFM(),
        lambda latents: latents + 2,
        lambda hidden: torch.stack((hidden[:, 0], hidden[:, 0] + 1), dim=-1),
    )
    input_tensor = torch.randn(1, 1, 4)
    history = torch.randn(1, 2, 4)
    noise = torch.randn(1, 2, 4)
    sde_noise = torch.randn(2, 1, 2, 4)

    executor.initialize_graph(
        input_tensor,
        history,
        noise,
        torch.tensor([0.0, 1.0]),
        (2.0, 0.25, 0.0),
        sde_noise,
    )

    assert executor.initialized is True
    assert executor.graph is graph
    get_backend.assert_called_once_with(input_tensor.device)
    assert events == [("capture", True)]


def test_cfm_graph_warms_up_full_tail_before_capture(monkeypatch) -> None:
    events: list[str] = []

    class FakeCFM:
        def sample(
            self, hidden, history, noise, timesteps, sde_args, sde_noise, *, abort_event
        ):
            assert abort_event is None
            events.append("sample")
            return noise + hidden[:, :1, :1] + history[:, :1, :1]

    class FakeGraphBackend:
        @contextmanager
        def capture(self, *, thread_local_errors):
            assert thread_local_errors is True
            events.append("capture_start")
            yield SimpleNamespace(replay=lambda: None)
            events.append("capture_end")

    class FakeRuntime:
        def __init__(self, device) -> None:
            assert device.type == "cpu"

        def synchronize(self) -> None:
            events.append("synchronize")

        def create_stream(self):
            return "stream"

        @contextmanager
        def create_stream_context(self, stream):
            assert stream == "stream"
            events.append("stream_start")
            yield
            events.append("stream_end")

    monkeypatch.setattr(talker_model, "TalkerDeviceRuntime", FakeRuntime)
    monkeypatch.setattr(
        talker_model,
        "current_platform",
        SimpleNamespace(
            get_device_graph_backend=lambda device: FakeGraphBackend(),
            is_cuda=lambda: True,
        ),
    )
    executor = talker_model.CFMGraphExecutor(
        SimpleNamespace(steps=2, patch_size=2),
        FakeCFM(),
        lambda latents: events.append("aggregate") or latents,
        lambda hidden: events.append("stop")
        or torch.stack((hidden[:, 0], hidden[:, 0] + 1), dim=-1),
    )
    executor.initialize_graph(
        torch.ones(1, 1, 4),
        torch.ones(1, 2, 4),
        torch.ones(1, 2, 4),
        torch.tensor([0.0, 1.0]),
        (2.0, 0.25, 0.0),
        torch.ones(2, 1, 2, 4),
    )

    capture_index = events.index("capture_start")
    warmup_events = events[:capture_index]
    assert warmup_events.count("sample") == 2
    assert warmup_events.count("aggregate") == 2
    assert warmup_events.count("stop") == 2
    assert warmup_events.index("stream_start") < warmup_events.index("sample")
    assert warmup_events[-2:] == ["synchronize", "stream_end"]
    assert events[capture_index:] == [
        "capture_start",
        "sample",
        "aggregate",
        "stop",
        "capture_end",
    ]
    assert executor.initialized is True


def test_cfm_graph_replay_uses_new_inputs_and_checks_abort(monkeypatch) -> None:
    replay_count = 0
    executor = talker_model.CFMGraphExecutor(
        SimpleNamespace(steps=2, patch_size=2),
        SimpleNamespace(
            sample=lambda hidden, history, noise, timesteps, sde_args, sde_noise, *, abort_event: (
                noise + hidden[:, :1, :1] + history[:, :1, :1]
            )
        ),
        lambda latents: latents + 2,
        lambda hidden: torch.stack(
            (hidden[:, 0], torch.ones_like(hidden[:, 0])), dim=-1
        ),
    )

    class FakeGraph:
        def replay(self) -> None:
            nonlocal replay_count
            replay_count += 1
            latents, embeds, stop = executor.compute_tail()
            executor.gen_lat_placeholder.copy_(latents)
            executor.inputs_embeds_placeholder.copy_(embeds)
            executor.stop_out_placeholder.copy_(stop)

    class FakeGraphBackend:
        @contextmanager
        def capture(self, *, thread_local_errors):
            yield FakeGraph()

    monkeypatch.setattr(
        talker_model,
        "current_platform",
        SimpleNamespace(
            get_device_graph_backend=lambda device: FakeGraphBackend(),
            is_cuda=lambda: False,
        ),
    )
    history = torch.ones(1, 2, 4)
    torch.manual_seed(1)
    first = executor.execute(torch.zeros(1, 1, 4), history)
    torch.manual_seed(1)
    second = executor.execute(torch.ones(1, 1, 4), history)
    assert replay_count == 2
    torch.testing.assert_close(second[0] - first[0], torch.ones_like(first[0]))
    assert not torch.equal(first[2], second[2])

    abort_event = threading.Event()
    abort_event.set()
    with pytest.raises(asyncio.CancelledError):
        executor.execute(torch.ones(1, 1, 4), history, abort_event=abort_event)
    assert replay_count == 2


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
