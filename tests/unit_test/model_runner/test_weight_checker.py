# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.weight_checker import StrictWeightChecker, _tensor_bytes


def test_strict_weight_checker_snapshot_compare_and_checksum() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    runner = SimpleNamespace(model=model)
    checker = StrictWeightChecker(runner)

    snapshot = checker.run("snapshot")
    checksum = checker.run("checksum")
    compare_before = checker.run("compare")

    with torch.no_grad():
        model.weight[0, 0] += 1.0

    compare_after = checker.run("compare")

    assert snapshot["tensor_count"] == 1
    assert set(snapshot["checksums"]) == {"weight"}
    assert checksum["per_gpu_checksum"] == snapshot["per_gpu_checksum"]
    assert compare_before["matched"] is True
    assert compare_after["matched"] is False
    assert compare_after["changed"] == ["weight"]


def test_strict_weight_checker_checksums_bfloat16_tensor_bytes() -> None:
    model = torch.nn.Module()
    model.register_buffer(
        "low_precision_weight",
        torch.tensor([1.0, -2.0, 3.5], dtype=torch.bfloat16),
    )
    checker = StrictWeightChecker(SimpleNamespace(model=model))

    summary = checker.run("checksum")

    assert summary["tensor_count"] == 1
    assert set(summary["checksums"]) == {"low_precision_weight"}
    assert summary["tensor_metadata"]["low_precision_weight"]["dtype"] == (
        "torch.bfloat16"
    )


def test_tensor_bytes_supports_bfloat16_fallback_path() -> None:
    tensor = torch.tensor([1.0, -2.0, 3.5], dtype=torch.bfloat16)

    raw = _tensor_bytes(tensor)

    assert isinstance(raw, bytes)
    assert len(raw) == tensor.numel() * tensor.element_size()


def test_tensor_bytes_supports_float8_fallback_path() -> None:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("torch does not expose float8_e4m3fn")
    tensor = torch.tensor([1.0, -2.0, 0.5], dtype=dtype)

    raw = _tensor_bytes(tensor)

    assert isinstance(raw, bytes)
    assert len(raw) == tensor.numel() * tensor.element_size()


def test_model_worker_update_weights_from_disk_updates_visible_model_info() -> None:
    calls: list[tuple[str, str, bool]] = []

    def update_weights_from_disk(
        model_path: str,
        load_format: str,
        *,
        recapture_cuda_graph: bool,
    ) -> tuple[bool, str]:
        calls.append((model_path, load_format, recapture_cuda_graph))
        return True, "ok"

    worker_args = SimpleNamespace(
        model_path="/tmp/old-model",
        load_format="auto",
        weight_version="old",
    )
    runner_args = SimpleNamespace(
        model_path="/tmp/old-model",
        load_format="auto",
        weight_version="old",
    )
    runner = SimpleNamespace(
        server_args=runner_args,
        model_config=SimpleNamespace(model_path="/tmp/old-model"),
        update_weights_from_disk=update_weights_from_disk,
    )
    worker = object.__new__(ModelWorker)
    worker.server_args = worker_args
    worker.model_runner = runner

    success, message = ModelWorker.update_weights_from_disk(
        worker,
        {
            "model_path": "/tmp/new-model",
            "load_format": "safetensors",
            "weight_version": "v2",
            "recapture_cuda_graph": True,
        },
    )

    assert success is True
    assert message == "ok"
    assert calls == [("/tmp/new-model", "safetensors", True)]
    assert worker_args.model_path == "/tmp/new-model"
    assert worker_args.load_format == "safetensors"
    assert worker_args.weight_version == "v2"
    assert runner_args.model_path == "/tmp/new-model"
    assert runner_args.load_format == "safetensors"
    assert runner_args.weight_version == "v2"
    assert runner.model_config.model_path == "/tmp/new-model"


def test_model_worker_init_weights_update_group_passes_positional_args() -> None:
    calls: list[tuple[Any, ...]] = []

    def init_weights_update_group(
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> tuple[bool, str]:
        calls.append(
            (master_address, master_port, rank_offset, world_size, group_name, backend)
        )
        return True, "group ready"

    runner = SimpleNamespace(init_weights_update_group=init_weights_update_group)
    worker = object.__new__(ModelWorker)
    worker.server_args = SimpleNamespace()
    worker.model_runner = runner

    success, message = ModelWorker.init_weights_update_group(
        worker,
        {
            "master_address": "10.0.0.1",
            "master_port": "12355",
            "rank_offset": 1,
            "world_size": 2,
            "group_name": "talker_group",
        },
    )

    assert success is True
    assert message == "group ready"
    assert calls == [("10.0.0.1", 12355, 1, 2, "talker_group", "nccl")]


def test_model_worker_init_weights_update_group_rejects_non_integer_fields() -> None:
    def init_weights_update_group(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        raise AssertionError("init should not run for a non-integer payload")

    runner = SimpleNamespace(init_weights_update_group=init_weights_update_group)
    worker = object.__new__(ModelWorker)
    worker.server_args = SimpleNamespace()
    worker.model_runner = runner

    success, message = ModelWorker.init_weights_update_group(
        worker,
        {
            "master_address": "10.0.0.1",
            "master_port": "not-a-port",
            "world_size": 2,
        },
    )

    assert success is False
    assert "integers" in message


def test_model_worker_destroy_weights_update_group_passes_group_name() -> None:
    calls: list[str] = []

    def destroy_weights_update_group(group_name: str) -> tuple[bool, str]:
        calls.append(group_name)
        return True, "group destroyed"

    runner = SimpleNamespace(destroy_weights_update_group=destroy_weights_update_group)
    worker = object.__new__(ModelWorker)
    worker.server_args = SimpleNamespace()
    worker.model_runner = runner

    success, message = ModelWorker.destroy_weights_update_group(
        worker,
        {"group_name": "talker_group"},
    )

    assert success is True
    assert message == "group destroyed"
    assert calls == ["talker_group"]


def test_model_worker_update_weights_from_distributed_passes_positional_args() -> None:
    calls: list[tuple[Any, ...]] = []

    def update_weights_from_distributed(
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
        load_format: str | None = None,
    ) -> tuple[bool, str]:
        calls.append((names, dtypes, shapes, group_name, load_format))
        return True, "ok"

    runner_args = SimpleNamespace(weight_version="old")
    runner = SimpleNamespace(
        server_args=runner_args,
        update_weights_from_distributed=update_weights_from_distributed,
    )
    worker = object.__new__(ModelWorker)
    worker.server_args = SimpleNamespace(weight_version="old")
    worker.model_runner = runner

    success, message = ModelWorker.update_weights_from_distributed(
        worker,
        {
            "names": ["model.embed.weight"],
            "dtypes": ["bfloat16"],
            "shapes": [[4, 8]],
            "group_name": "talker_group",
            "weight_version": "v2",
        },
    )

    assert success is True
    assert message == "ok"
    assert calls == [
        (["model.embed.weight"], ["bfloat16"], [[4, 8]], "talker_group", None)
    ]
    assert worker.server_args.weight_version == "v2"
    assert runner_args.weight_version == "v2"


def test_model_worker_update_weights_from_distributed_requires_names() -> None:
    runner = SimpleNamespace(
        update_weights_from_distributed=lambda *a, **k: (True, "should not be called"),
    )
    worker = object.__new__(ModelWorker)
    worker.server_args = SimpleNamespace()
    worker.model_runner = runner

    success, message = ModelWorker.update_weights_from_distributed(
        worker,
        {"names": [], "dtypes": [], "shapes": []},
    )

    assert success is False
    assert "names" in message


def test_model_worker_update_weights_from_distributed_rejects_mismatched_metadata() -> (
    None
):
    calls = 0

    def update_weights_from_distributed(*args, **kwargs) -> tuple[bool, str]:
        nonlocal calls
        calls += 1
        return True, "should not be called"

    runner = SimpleNamespace(
        update_weights_from_distributed=update_weights_from_distributed,
    )
    worker = object.__new__(ModelWorker)
    worker.server_args = SimpleNamespace()
    worker.model_runner = runner

    success, message = ModelWorker.update_weights_from_distributed(
        worker,
        {
            "names": ["model.embed.weight"],
            "dtypes": ["bfloat16"],
            "shapes": [[4, 8], [8, 4]],
        },
    )

    assert success is False
    assert "same length" in message
    assert calls == 0
