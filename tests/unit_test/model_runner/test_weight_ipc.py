# SPDX-License-Identifier: Apache-2.0
"""Model-runner policy tests for weight IPC."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.distributed.weight_ipc.types import WeightIpcConfig, WeightIpcRole
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.sglang_model_runner import (
    SGLModelRunner,
    _validate_weight_ipc_kv_cap,
)
from sglang_omni.models.higgs_tts.model import HiggsTTSModel


def _runner(role: WeightIpcRole, store_dir: str | None = None) -> SGLModelRunner:
    runner = object.__new__(SGLModelRunner)
    runner._weight_ipc_config = WeightIpcConfig(role=role, store_dir=store_dir)
    runner.server_args = SimpleNamespace(
        model_path="model",
        revision="revision",
        load_format="auto",
    )
    runner.model = object()
    runner._weight_ipc_leader_monitor = None
    return runner


def _worker(role: WeightIpcRole) -> ModelWorker:
    worker = object.__new__(ModelWorker)
    worker.weight_ipc = WeightIpcConfig(role=role, store_dir="/tmp/weight-ipc")
    worker.server_args = SimpleNamespace(
        model_path="old",
        load_format="auto",
        weight_version="old",
        tp_size=1,
    )
    worker.model_arch_override = None
    worker.tp_rank = 0
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(),
        server_args=SimpleNamespace(weight_version="old"),
        model_config=SimpleNamespace(model_path="old"),
        update_weights_from_disk=lambda *args, **kwargs: (True, "updated"),
        update_weights_from_tensor=lambda *args, **kwargs: (True, "updated"),
        update_weights_from_distributed=lambda *args, **kwargs: (True, "updated"),
        init_weights_update_group=lambda *args, **kwargs: (True, "updated"),
        destroy_weights_update_group=lambda *args, **kwargs: (True, "updated"),
    )
    return worker


def test_load_model_normally_when_ipc_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner("off")
    calls: list[str] = []

    monkeypatch.setattr(
        SGLModelRunner.__bases__[0],
        "load_model",
        lambda self: calls.append("load"),
    )

    SGLModelRunner.load_model(runner)

    assert calls == ["load"]


def test_follower_requires_explicit_kv_cap() -> None:
    with pytest.raises(ValueError, match="explicit --max-total-tokens"):
        _validate_weight_ipc_kv_cap("follower", None)


def test_leader_and_ipc_off_allow_automatic_kv_cap() -> None:
    _validate_weight_ipc_kv_cap("leader", None)
    _validate_weight_ipc_kv_cap("off", None)


def test_higgs_load_weights_rejects_follower_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEIGHT_IPC_ROLE", "follower")
    monkeypatch.setenv("WEIGHT_IPC_STORE", str(tmp_path))
    model = object.__new__(HiggsTTSModel)

    with pytest.raises(RuntimeError, match="must not be called"):
        HiggsTTSModel.load_weights(model, [])


def test_leader_loads_then_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner("leader", str(tmp_path))
    calls: list[str] = []

    monkeypatch.setattr(
        SGLModelRunner.__bases__[0],
        "load_model",
        lambda self: calls.append("load"),
    )

    def export(model, config, *, model_path, model_revision) -> None:
        assert model is runner.model
        assert config.role == "leader"
        calls.append(f"export:{model_path}:{model_revision}")

    monkeypatch.setattr(
        "sglang_omni.model_runner.sglang_model_runner.export_leader_weights",
        export,
    )

    SGLModelRunner.load_model(runner)

    assert calls == ["load", "export:model:revision"]


def test_follower_uses_dummy_load_then_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner("follower", str(tmp_path))
    calls: list[str] = []
    monitor = object()

    monkeypatch.setattr(
        SGLModelRunner.__bases__[0],
        "load_model",
        lambda self: calls.append(f"load:{self.server_args.load_format}"),
    )

    def materialize(model, config, *, model_path, model_revision) -> object:
        assert model is runner.model
        assert config.role == "follower"
        calls.append(f"alias:{model_path}:{model_revision}")
        return monitor

    monkeypatch.setattr(
        "sglang_omni.model_runner.sglang_model_runner.materialize_follower_weights",
        materialize,
    )

    SGLModelRunner.load_model(runner)

    assert calls == ["load:dummy", "alias:model:revision"]
    assert runner.server_args.load_format == "auto"
    assert runner._weight_ipc_leader_monitor is monitor


@pytest.mark.parametrize("role", ["leader", "follower"])
def test_ipc_mode_does_not_advertise_weight_updates(role: WeightIpcRole) -> None:
    assert ModelWorker.model_info(_worker(role))["supports_weight_update"] is False


def test_ipc_mode_rejects_every_weight_update_entrypoint() -> None:
    worker = _worker("follower")
    rejection = (
        False,
        "weight update is unsupported while weight IPC is enabled; "
        "disable weight IPC before updating weights",
    )
    results = [
        ModelWorker.update_weights_from_disk(worker, {"model_path": "new"}),
        ModelWorker.update_weights_from_tensor(worker, {}),
        ModelWorker.update_weights_from_distributed(worker, {}),
        ModelWorker.init_weights_update_group(worker, {}),
        ModelWorker.destroy_weights_update_group(worker, {}),
    ]

    assert results == [rejection] * len(results)


def test_weight_updates_remain_enabled_when_ipc_is_off() -> None:
    worker = _worker("off")

    result = ModelWorker.update_weights_from_disk(
        worker,
        {"model_path": "new", "load_format": "safetensors"},
    )

    assert result == (True, "updated")
    assert worker.server_args.model_path == "new"
    assert worker.server_args.load_format == "safetensors"
