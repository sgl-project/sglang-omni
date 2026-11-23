# SPDX-License-Identifier: Apache-2.0
"""The prewarm CUDA IPC share is held for the process and its file unlinked at exit."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.pipeline import stage_workers
from sglang_omni.utils import process
from tests.unit_test.fixtures.accelerator import require_cuda


class FakeStorage:
    def __init__(self, record: list[int], index: int) -> None:
        self.record = record
        self.index = index

    def _share_cuda_(
        self,
    ) -> tuple[
        int, bytes, int, int, bytes, int, bytes, bool
    ]:  # noqa: leading-underscore
        self.record.append(self.index)
        return (0, b"", 0, 0, f"/torch_fake_{self.index}".encode(), 0, b"", False)


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
    monkeypatch.setattr(process, "_prewarm_cuda_ipc_handles", [])
    monkeypatch.setattr(stage_workers.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(stage_workers.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(stage_workers.torch, "empty", _empty)


def test_warms_once_on_the_lowest_host_gpu_and_registers_its_file(
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
    assert process._prewarm_cuda_ipc_handles == [  # noqa: leading-underscore
        b"/torch_fake_0"
    ]


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
    assert process._prewarm_cuda_ipc_handles == []  # noqa: leading-underscore


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


def test_teardown_exit_unlinks_the_prewarm_counter_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exit path that skips teardown unlinks the prewarm share's bytes named file."""
    record: list[int] = []
    install_fake_cuda(monkeypatch, record)
    unlinked: list[bytes] = []
    monkeypatch.setattr(process.os, "unlink", lambda path: unlinked.append(path))
    spec = make_spec([(0, False)])
    stage_workers.prewarm_cuda_ipc_export(spec, logging.getLogger("test"))
    process.release_cuda_ipc_file()
    assert unlinked == [b"/dev/shm/torch_fake_0"]
    assert process._prewarm_cuda_ipc_handles == []  # noqa: leading-underscore


PREWARM_SCRIPT = textwrap.dedent(
    """
    import logging
    import sys
    from types import SimpleNamespace

    from sglang_omni.pipeline import stage_workers
    from sglang_omni.utils import process

    spec = SimpleNamespace(
        stage_specs=[SimpleNamespace(gpu_id=0, allow_child_processes=False)]
    )
    stage_workers.prewarm_cuda_ipc_export(spec, logging.getLogger("prewarm"))
    print(process._prewarm_cuda_ipc_handles[0].decode(), flush=True)
    """
)


def run_prewarm_then(exit_statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", PREWARM_SCRIPT + exit_statement],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def shm_path(completed: subprocess.CompletedProcess[str]) -> str:
    handle = completed.stdout.strip().splitlines()[-1]
    return "/dev/shm/" + handle.lstrip("/")


@pytest.mark.accelerator
def test_real_prewarm_share_leaves_no_file_after_the_worker_exit() -> None:
    """A real prewarm share's counter file is gone after the worker exit, with no crash."""
    require_cuda()
    completed = run_prewarm_then("process.exit_without_teardown(0)\n")
    assert completed.returncode == 0, completed.stderr
    assert not os.path.exists(shm_path(completed))


@pytest.mark.accelerator
def test_real_prewarm_share_does_not_abort_a_normal_exit() -> None:
    """A normal exit after a real prewarm share does not abort on a double unlink."""
    require_cuda()
    completed = run_prewarm_then("sys.exit(0)\n")
    assert completed.returncode == 0, completed.stderr
    assert not os.path.exists(shm_path(completed))
