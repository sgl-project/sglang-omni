# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.flow_npugraph import (
    FlowNPUGraphRunner,
    _graph_safe_non_streaming_mask,
    _GraphEntry,
    _replay_after_capture,
    enable_flow_npugraph,
    prepare_flow_npugraph_environment,
)


class _Estimator(torch.nn.Module):
    def forward(self, x, mask, mu, t, spks, cond, streaming=False):
        del mask, mu, spks, cond, streaming
        return x + t.reshape(-1, 1, 1)


class _ReplayGraph:
    def __init__(self, entry_inputs, output):
        self.inputs = entry_inputs
        self.output = output

    def replay(self):
        self.output.copy_(self.inputs["x"] + self.inputs["t"].reshape(-1, 1, 1))


def test_capture_result_is_materialized_by_first_replay(monkeypatch):
    calls = []
    current = SimpleNamespace(
        wait_stream=lambda stream: calls.append(("current", stream))
    )
    capture = object()
    graph = SimpleNamespace(replay=lambda: calls.append(("replay", None)))
    monkeypatch.setattr(torch.npu, "current_stream", lambda: current)
    monkeypatch.setattr(torch.npu, "synchronize", lambda: calls.append(("sync", None)))

    _replay_after_capture(graph, capture)

    assert calls == [
        ("current", capture),
        ("replay", None),
        ("current", capture),
        ("sync", None),
    ]


def _inputs(length=4):
    return {
        "x": torch.ones(2, 1, length),
        "mask": torch.ones(2, 1, length),
        "mu": torch.zeros(2, 1, length),
        "t": torch.tensor([0.0, 1.0]),
        "spks": torch.zeros(2, 1),
        "cond": torch.zeros(2, 1, length),
    }


def _install_fake_capture(monkeypatch, runner):
    capture_stream = SimpleNamespace(wait_stream=lambda stream: None)

    def capture(inputs):
        static = {name: value.clone() for name, value in inputs.items()}
        output = static["x"] + static["t"].reshape(-1, 1, 1)
        return _GraphEntry(static, _ReplayGraph(static, output), output, capture_stream)

    monkeypatch.setattr(runner, "_capture", capture)
    monkeypatch.setattr(
        torch.npu,
        "current_stream",
        lambda: SimpleNamespace(wait_stream=lambda stream: None),
    )


def test_graph_replay_updates_every_input(monkeypatch):
    runner = FlowNPUGraphRunner(_Estimator())
    _install_fake_capture(monkeypatch, runner)
    first = _inputs()
    runner(**first)

    second = _inputs()
    second["x"].fill_(3)
    second["t"] = torch.tensor([2.0, 4.0])
    actual = runner(**second)

    torch.testing.assert_close(actual, _Estimator()(**second))


def test_graph_cache_is_shape_keyed_and_bounded(monkeypatch):
    estimator = _Estimator()
    runner = FlowNPUGraphRunner(estimator, max_graphs=1)
    _install_fake_capture(monkeypatch, runner)
    runner(**_inputs(length=4))

    actual = runner(**_inputs(length=5))

    assert len(runner._graphs) == 1
    torch.testing.assert_close(actual, estimator(**_inputs(length=5)))


def test_streaming_uses_eager_path(monkeypatch):
    runner = FlowNPUGraphRunner(_Estimator())
    monkeypatch.setattr(
        runner, "_capture", lambda inputs: pytest.fail("capture should not run")
    )

    actual = runner(**_inputs(), streaming=True)

    torch.testing.assert_close(actual, _Estimator()(**_inputs(), streaming=True))


def test_graph_safe_mask_preserves_rows_and_repairs_empty_rows():
    masks = torch.tensor([[[True, False]], [[False, False]]])

    actual = _graph_safe_non_streaming_mask(
        torch.empty(2, 2, 1), masks, False, False, 0, 0, -1
    )

    assert actual.tolist() == [[[True, False]], [[True, True]]]


def test_enable_is_npu_only(monkeypatch):
    from sglang_omni import platforms

    flow = SimpleNamespace(decoder=SimpleNamespace(estimator=_Estimator()))
    monkeypatch.setattr(platforms.current_platform, "device_type", "cuda")
    assert enable_flow_npugraph(flow) is False
    assert not isinstance(flow.decoder.estimator.forward, FlowNPUGraphRunner)

    monkeypatch.setattr(platforms.current_platform, "device_type", "npu")
    assert enable_flow_npugraph(flow, max_graphs=2) is True
    assert isinstance(flow.decoder.estimator.forward, FlowNPUGraphRunner)


def test_prepare_environment_disables_internal_format_on_npu(monkeypatch):
    import sys

    from sglang_omni import platforms

    config = SimpleNamespace(allow_internal_format=True)
    torch_npu = SimpleNamespace(npu=SimpleNamespace(config=config))
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setattr(platforms.current_platform, "device_type", "npu")

    assert prepare_flow_npugraph_environment() is True
    assert config.allow_internal_format is False
