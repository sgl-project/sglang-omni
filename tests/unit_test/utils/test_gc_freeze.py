# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc

from sglang_omni.utils import gc_freeze


def test_freezes_once_per_process(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(gc_freeze, "frozen_pids", set())
    monkeypatch.setattr(gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(gc, "freeze", lambda: calls.append("freeze"))

    assert gc_freeze.freeze_gc_after_warmup("vocoder") is True
    assert gc_freeze.freeze_gc_after_warmup("talker") is False
    assert calls == ["collect", "freeze"]
