# SPDX-License-Identifier: Apache-2.0
"""prewarm_cuda_ipc_export shares one tensor per GPU process and holds it."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.pipeline import stage_workers


class FakeStorage:
    def __init__(self, record: list[int], index: int) -> None:
        self.record = record
        self.index = index

    def _share_cuda_(
        self,
    ) -> tuple[int, bytes, int, int, str, int, bytes, bool]:  # noqa: leading-underscore
        self.record.append(self.index)
        return (0, b"", 0, 0, f"/torch_fake_{self.index}", 0, b"", False)


class FakeTensor:
    def __init__(self, record: list[int], index: int) -> None:
        self.storage = FakeStorage(record, index)

    def untyped_storage(self) -> FakeStorage:
        return self.storage


def make_spec(stages: list[tuple[int | None, bool]]) -> SimpleNamespace:
    specs = [
        SimpleNamespace(gpu_id=gpu_id, allow_child_processes=is_child)
        for gpu_id, is_child in stages
    ]
    return SimpleNamespace(stage_specs=specs)


def install_fake_cuda(monkeypatch: pytest.MonkeyPatch, record: list[int]) -> None:
    def _empty(size: int, dtype: torch.dtype, device: str) -> FakeTensor:
        return FakeTensor(record, int(device.split(":")[1]))

    monkeypatch.setattr(stage_workers, "_cuda_ipc_warm_shares", [])
    monkeypatch.setattr(stage_workers, "_cuda_ipc_warm_handles", [])
    monkeypatch.setattr(stage_workers.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(stage_workers.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(stage_workers.torch, "empty", _empty)


def test_warms_once_on_the_lowest_host_gpu_and_retains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GPU process shares once on its lowest host device and holds the share."""
    record: list[int] = []
    install_fake_cuda(monkeypatch, record)
    spec = make_spec([(1, False), (1, False), (0, False)])
    stage_workers.prewarm_cuda_ipc_export(spec, logging.getLogger("test"))
    assert record == [0]
    held = stage_workers._cuda_ipc_warm_shares  # noqa: leading-underscore
    assert len(held) == 1
    assert all(isinstance(tensor, FakeTensor) for tensor in held)


def test_cpu_and_child_process_stages_warm_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU stage and a child process GPU stage do not share anything."""
    record: list[int] = []
    install_fake_cuda(monkeypatch, record)
    spec = make_spec([(None, False), (2, True)])
    stage_workers.prewarm_cuda_ipc_export(spec, logging.getLogger("test"))
    assert record == []
    assert stage_workers._cuda_ipc_warm_shares == []  # noqa: leading-underscore


def test_non_cuda_non_rocm_platform_warms_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A platform that is neither CUDA nor ROCm, such as MUSA, shares nothing."""
    record: list[int] = []
    install_fake_cuda(monkeypatch, record)
    monkeypatch.setattr(stage_workers.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(stage_workers.current_platform, "is_rocm", lambda: False)
    spec = make_spec([(0, False)])
    stage_workers.prewarm_cuda_ipc_export(spec, logging.getLogger("test"))
    assert record == []


def test_release_unlinks_the_counter_file_and_drops_the_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown unlinks the prewarm share's ref counter file and drops the tensor."""
    record: list[int] = []
    install_fake_cuda(monkeypatch, record)
    unlinked: list[str] = []
    monkeypatch.setattr(stage_workers.os, "unlink", lambda path: unlinked.append(path))
    spec = make_spec([(0, False)])
    stage_workers.prewarm_cuda_ipc_export(spec, logging.getLogger("test"))
    stage_workers.release_cuda_ipc_warm_shares()
    assert unlinked == ["/dev/shm/torch_fake_0"]
    assert stage_workers._cuda_ipc_warm_shares == []  # noqa: leading-underscore
    assert stage_workers._cuda_ipc_warm_handles == []  # noqa: leading-underscore
