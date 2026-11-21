# SPDX-License-Identifier: Apache-2.0
"""Exercise Speech graph ownership without an NPU."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang_omni.models.minicpm_o.components import flow_npu_graph


def test_graph_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SPEECH_FLOW_NPU_GRAPH", raising=False)
    forward_chunk = Mock()
    decoder = SimpleNamespace(forward_chunk=forward_chunk)
    flow_npu_graph.install_flow_npu_graph(
        SimpleNamespace(flow=SimpleNamespace(decoder=decoder))
    )
    assert decoder.forward_chunk is forward_chunk


def test_graph_replays_on_owner_stream_and_returns_owned_outputs(monkeypatch):
    class FakeNpuTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("npu")

    owner_stream = Mock()
    other_stream = Mock()
    npu = SimpleNamespace(
        current_stream=Mock(return_value=owner_stream),
        Stream=Mock(),
        stream=lambda stream: nullcontext(),
        synchronize=Mock(),
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: False)
    monkeypatch.setenv("SPEECH_FLOW_NPU_GRAPH", "1")
    monkeypatch.setenv("SPEECH_NPU_DISABLE_INTERNAL_FORMAT", "1")
    monkeypatch.setenv("SPEECH_FLOW_NPU_GRAPH_VERIFY", "0")
    monkeypatch.setattr(
        flow_npu_graph,
        "importlib",
        SimpleNamespace(
            import_module=lambda name: SimpleNamespace(
                _C=SimpleNamespace(_npu_getOption=lambda option: b"disable")
            )
        ),
    )

    capture_calls = []
    graph = SimpleNamespace(replay=Mock())
    captured_inputs = None
    captured_output = None

    @contextmanager
    def capture(*, pool):
        capture_calls.append(pool)
        yield graph

    def eager(*inputs):
        nonlocal captured_inputs, captured_output
        mu, _, _, _, _, _, attention_window, _ = inputs
        attention_window.add_(1)
        output = mu + 10
        if capture_calls and captured_inputs is None:
            captured_inputs, captured_output = inputs, output
        return output

    def replay():
        assert captured_inputs is not None and captured_output is not None
        captured_inputs[6].add_(1)
        captured_output.copy_(captured_inputs[0] + 10)

    graph.replay.side_effect = replay
    monkeypatch.setattr(
        flow_npu_graph,
        "NpuDeviceGraphBackend",
        lambda: SimpleNamespace(graph_pool_handle=lambda: (1, 2), capture=capture),
    )
    decoder = SimpleNamespace(forward_chunk=Mock(side_effect=eager), training=False)
    eager_forward = decoder.forward_chunk
    model = SimpleNamespace(
        device=torch.device("npu"), flow=SimpleNamespace(decoder=decoder)
    )
    flow_npu_graph.install_flow_npu_graph(model)

    def run(value, frames=6):
        mu = torch.full((1, 1, frames), value).as_subclass(FakeNpuTensor)
        attention_window = torch.zeros((1, 1, 1, 1, 408))
        with torch.no_grad():
            output = decoder.forward_chunk(
                mu,
                None,
                None,
                n_timesteps=5,
                convolution_cache=torch.zeros(1),
                attention_window=attention_window,
            )
        assert attention_window.eq(1).all()
        return output

    first_output = run(1.0)
    assert run(2.0).eq(12).all()
    assert first_output.eq(11).all()
    assert capture_calls == [(1, 2)]
    assert graph.replay.call_count == 2

    eager_forward.reset_mock()
    npu.current_stream.return_value = other_stream
    assert run(3.0).eq(13).all()
    eager_forward.assert_called_once()
    assert graph.replay.call_count == 2

    eager_forward.reset_mock()
    npu.current_stream.return_value = owner_stream
    assert run(4.0, frames=7).eq(14).all()
    eager_forward.assert_called_once()
    assert graph.replay.call_count == 2
