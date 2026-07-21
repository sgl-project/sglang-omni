# SPDX-License-Identifier: Apache-2.0
"""Role plumbing tests for SGLANG_OMNI_WEIGHT_SHARE in SGLModelRunner.

Runs the real ``SGLModelRunner.load_model`` / ``init_device_graphs`` overrides
with the upstream ``ModelRunner`` methods mocked out, on CPU tensors. Verifies
role dispatch, the dummy load-format toggle (set for the super() call, restored
after), attach-before-capture ordering, and the weight-update guards.

Requires sglang to be importable (login node: stub sgl_kernel first — see the
scratchpad ``run_test_login_node.py`` pattern).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

sglang_model_runner = pytest.importorskip(
    "sglang_omni.model_runner.sglang_model_runner",
    reason="sglang (and its sgl_kernel dependency) not importable here",
)
from sglang.srt.model_executor.model_runner import ModelRunner  # noqa: E402

from sglang_omni.utils import ipc_weights  # noqa: E402

SGLModelRunner = sglang_model_runner.SGLModelRunner


class SmallModel(nn.Module):
    def __init__(self, fill: float):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        with torch.no_grad():
            self.linear.weight.fill_(fill)
            self.linear.bias.fill_(fill)


def _bare_runner(load_format="auto"):
    runner = SGLModelRunner.__new__(SGLModelRunner)
    runner.server_args = SimpleNamespace(
        load_format=load_format,
        max_total_tokens=1000,
        tp_size=1,
        pp_size=1,
        model_path="m",
        revision="r",
    )
    # Satisfy the architecture gate: these tests exercise role plumbing with a
    # stand-in model, not architecture enforcement, so override to an audited
    # arch. (The gate itself is covered in test_ipc_weights.py.)
    runner._model_arch_override = "HiggsMultimodalQwen3ForConditionalGeneration"
    runner._weight_share_config = None
    runner._weight_share_record = None
    runner._weight_ipc_leader_monitor = None
    return runner


def _fake_upstream_load(fill_by_format):
    """Upstream ModelRunner.load_model stand-in: materializes a model whose
    values depend on the load format actually in effect during the call."""

    def fake_load(self):
        fmt = self.server_args.load_format
        assert fmt in fill_by_format, f"unexpected load_format {fmt!r}"
        self.model = SmallModel(fill=fill_by_format[fmt])

    return fake_load


def test_env_unset_is_stock_path(tmp_path, monkeypatch):
    monkeypatch.delenv(ipc_weights.ENV_WEIGHT_SHARE, raising=False)
    runner = _bare_runner()
    with mock.patch.object(
        ModelRunner, "load_model", _fake_upstream_load({"auto": 1.0})
    ):
        runner.load_model()
    assert runner._weight_share_config is None
    assert runner._weight_share_record is None
    assert not os.listdir(tmp_path)  # nothing exported anywhere
    # Weight updates stay allowed on the stock path.
    assert runner._weight_update_blocked_reason() is None


def test_leader_loads_normally_then_exports(tmp_path, monkeypatch):
    monkeypatch.setenv(ipc_weights.ENV_WEIGHT_SHARE, f"leader:{tmp_path}")
    runner = _bare_runner()
    with mock.patch.object(
        ModelRunner, "load_model", _fake_upstream_load({"auto": 1.0})
    ):
        runner.load_model()
    assert runner.server_args.load_format == "auto"  # untouched for leader
    handle = tmp_path / "SmallModel.weights-ipc"
    assert handle.exists()
    # Leader-side record kept for the pre-capture identity check (empty here:
    # a CPU model has no IPC-shareable tensors, everything rode the value path).
    assert runner._weight_share_record is not None
    assert runner._weight_update_blocked_reason() is not None


def test_follower_dummy_loads_waits_and_attaches(tmp_path, monkeypatch):
    # Leader publishes first (CPU tensors ride the by-value path; role
    # plumbing and ordering are what's under test, not CUDA IPC).
    leader_model = SmallModel(fill=7.0)
    ipc_weights.export_weights(leader_model, str(tmp_path / "SmallModel.weights-ipc"))

    monkeypatch.setenv(ipc_weights.ENV_WEIGHT_SHARE, f"follower:{tmp_path}")
    seen_formats = []

    def fake_load(self):
        seen_formats.append(self.server_args.load_format)
        self.model = SmallModel(fill=0.0)  # dummy values

    with mock.patch.object(ModelRunner, "load_model", fake_load):
        runner = _bare_runner(load_format="auto")
        runner.load_model()
    runner._weight_ipc_leader_monitor.stop()  # don't leak the poller thread

    # Dummy format was in effect exactly during the super() call, restored after.
    assert seen_formats == ["dummy"]
    assert runner.server_args.load_format == "auto"
    # Values came from the leader export, not the dummy init.
    assert torch.all(runner.model.linear.weight == 7.0)
    assert runner._weight_share_record is not None
    assert runner._weight_update_blocked_reason() is not None


def test_follower_verifies_attachment_before_graph_capture(tmp_path, monkeypatch):
    ipc_weights.export_weights(
        SmallModel(fill=7.0), str(tmp_path / "SmallModel.weights-ipc")
    )
    monkeypatch.setenv(ipc_weights.ENV_WEIGHT_SHARE, f"follower:{tmp_path}")

    def fake_load(self):
        self.model = SmallModel(fill=0.0)

    calls = []
    with mock.patch.object(ModelRunner, "load_model", fake_load):
        runner = _bare_runner()
        runner.load_model()
    runner._weight_ipc_leader_monitor.stop()  # don't leak the poller thread
    with (
        mock.patch.object(
            ModelRunner, "init_device_graphs", lambda self: calls.append("capture")
        ),
        mock.patch.object(
            ipc_weights,
            "verify_attachment",
            side_effect=lambda *a, **k: calls.append("verify"),
        ),
    ):
        runner.init_device_graphs()
    assert calls == ["verify", "capture"]  # attach guard precedes capture


def test_follower_requires_explicit_kv_cap(tmp_path, monkeypatch):
    monkeypatch.setenv(ipc_weights.ENV_WEIGHT_SHARE, f"follower:{tmp_path}")
    runner = _bare_runner()
    runner.server_args.max_total_tokens = None
    with pytest.raises(ipc_weights.WeightShareError, match="max-total-tokens"):
        runner.load_model()


def test_weight_update_guard_blocks_all_three(tmp_path, monkeypatch):
    monkeypatch.setenv(ipc_weights.ENV_WEIGHT_SHARE, f"leader:{tmp_path}")
    runner = _bare_runner()
    with mock.patch.object(
        ModelRunner, "load_model", _fake_upstream_load({"auto": 1.0})
    ):
        runner.load_model()
    for method in (
        runner.update_weights_from_disk,
        runner.update_weights_from_tensor,
        runner.update_weights_from_distributed,
    ):
        ok, message = method()
        assert ok is False
        assert "weight sharing" in message
