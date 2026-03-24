# SPDX-License-Identifier: Apache-2.0
"""Tests for model worker port selection."""

from __future__ import annotations

from sglang_omni.engines.ar.sglang_backend.model_worker import _resolve_nccl_port


def test_resolve_nccl_port_prefers_master_port(monkeypatch) -> None:
    monkeypatch.setenv("MASTER_PORT", "45678")
    assert _resolve_nccl_port() == 45678


def test_resolve_nccl_port_sets_master_port_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("MASTER_PORT", raising=False)
    port = _resolve_nccl_port()
    assert isinstance(port, int)
    assert port > 0
    assert port < 65536
