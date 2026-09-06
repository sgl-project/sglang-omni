# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import signal

import pytest

from sglang_omni.pipeline import stage_workers

EXPECTED_PARENT = 4321


class _FakeLibc:
    def __init__(self, rc: int = 0) -> None:
        self.rc = rc
        self.calls: list[tuple] = []

    def prctl(self, *args) -> int:
        self.calls.append(args)
        return self.rc


def test_parent_death_signal_is_noop_off_linux(monkeypatch) -> None:
    monkeypatch.setattr(stage_workers.sys, "platform", "darwin")

    def _fail(*_a, **_k):  # pragma: no cover
        raise AssertionError("ctypes must not be touched off Linux")

    import ctypes

    monkeypatch.setattr(ctypes, "CDLL", _fail)
    stage_workers._install_parent_death_signal(EXPECTED_PARENT)


def test_parent_death_signal_requests_sigkill_on_parent_exit(monkeypatch) -> None:
    fake = _FakeLibc()
    import ctypes

    monkeypatch.setattr(stage_workers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    monkeypatch.setattr(stage_workers.os, "getppid", lambda: EXPECTED_PARENT)

    stage_workers._install_parent_death_signal(EXPECTED_PARENT)

    assert fake.calls, "prctl was not called"
    option, sig = fake.calls[0][0], fake.calls[0][1]
    assert option == 1
    assert sig == getattr(signal, "SIGKILL", 9)


def test_parent_death_signal_exits_when_parent_died_before_install(monkeypatch) -> None:
    fake = _FakeLibc()
    import ctypes

    monkeypatch.setattr(stage_workers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    monkeypatch.setattr(stage_workers.os, "getppid", lambda: 1)

    exit_codes: list[int] = []

    def _fake_exit(code: int) -> None:
        exit_codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(stage_workers.os, "_exit", _fake_exit)

    with pytest.raises(SystemExit):
        stage_workers._install_parent_death_signal(EXPECTED_PARENT)

    assert exit_codes == [1]
    assert not fake.calls, "prctl must not run once the parent is already gone"


def test_parent_death_signal_exits_when_parent_died_during_install(monkeypatch) -> None:
    fake = _FakeLibc()
    import ctypes

    monkeypatch.setattr(stage_workers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    ppids = iter([EXPECTED_PARENT, 1])
    monkeypatch.setattr(stage_workers.os, "getppid", lambda: next(ppids))

    exit_codes: list[int] = []

    def _fake_exit(code: int) -> None:
        exit_codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(stage_workers.os, "_exit", _fake_exit)

    with pytest.raises(SystemExit):
        stage_workers._install_parent_death_signal(EXPECTED_PARENT)

    assert exit_codes == [1]
    assert fake.calls, "prctl should have run before the second check"


def test_parent_death_signal_survives_prctl_failure(monkeypatch) -> None:
    fake = _FakeLibc(rc=-1)
    import ctypes

    monkeypatch.setattr(stage_workers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    monkeypatch.setattr(ctypes, "get_errno", lambda: 1)
    monkeypatch.setattr(stage_workers.os, "getppid", lambda: EXPECTED_PARENT)

    stage_workers._install_parent_death_signal(EXPECTED_PARENT)


def test_spawn_threads_expected_parent_pid_into_install(monkeypatch) -> None:
    from sglang_omni.pipeline.stage_workers import (
        StageGroup,
        StageLaunchConfig,
        StageWorkerProcessSpec,
    )

    installed_with: list[int] = []
    monkeypatch.setattr(
        stage_workers,
        "_install_parent_death_signal",
        lambda pid: installed_with.append(pid),
    )
    monkeypatch.setattr(stage_workers, "_prepare_cuda_environment", lambda *a, **k: None)
    monkeypatch.setattr(stage_workers, "apply_gpu_compat_env_defaults", lambda *a, **k: None)
    monkeypatch.setattr(stage_workers, "_run_process", lambda *a, **k: None)

    captured: dict = {}

    class _FakeEvent:
        def set(self) -> None: ...

        def is_set(self) -> bool:
            return True

    class _FakeQueue:
        def put(self, *a) -> None: ...

        def close(self) -> None: ...

        def join_thread(self) -> None: ...

    class _FakeProcess:
        def __init__(self, target, args, **_) -> None:
            self.target, self.args = target, args
            self.pid = 1234

        def start(self) -> None:
            self.target(*self.args)

    class _FakeCtx:
        def Event(self) -> _FakeEvent:
            return _FakeEvent()

        def Queue(self) -> _FakeQueue:
            return _FakeQueue()

        def Process(self, target, args, name=None, daemon=None) -> _FakeProcess:
            captured["target"], captured["args"] = target, args
            return _FakeProcess(target, args)

    group = StageGroup(
        "g",
        [
            StageWorkerProcessSpec(
                process_name="p",
                stage_specs=[StageLaunchConfig(stage_name="s")],
            )
        ],
    )
    group.spawn(_FakeCtx())

    assert captured["target"] is stage_workers.stage_process_main
    assert captured["args"][2] == os.getpid()
    assert installed_with == [os.getpid()]
