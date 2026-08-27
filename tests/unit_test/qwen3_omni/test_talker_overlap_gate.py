# SPDX-License-Identifier: Apache-2.0
"""Talker overlap disable is gated by SGLANG_OMNI_TALKER_OVERLAP.

Feedback-enabled talkers force `disable_overlap_schedule` today; the env flag
lets an experiment keep the caller's value. CUDA-graph handling is unaffected
either way.
"""
from __future__ import annotations

import pytest

from sglang_omni.models.qwen3_omni.talker_scheduler import configure_talker_server_args
from tests.unit_test.fakes import FakeServerArgs

_ENV = "SGLANG_OMNI_TALKER_OVERLAP"


def _server_args(*, disable_overlap_schedule=False, disable_cuda_graph=False):
    return FakeServerArgs(
        disable_overlap_schedule=disable_overlap_schedule,
        disable_cuda_graph=disable_cuda_graph,
        disable_radix_cache=False,
        chunked_prefill_size=8192,
    )


def test_default_forces_overlap_disabled(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    args = _server_args()
    want_cuda_graph = configure_talker_server_args(args, feedback_enabled=True)
    assert args.disable_overlap_schedule is True
    assert want_cuda_graph is True
    assert args.disable_cuda_graph is True


@pytest.mark.parametrize("caller_value", [True, False])
def test_env_flag_keeps_caller_overlap_value(monkeypatch, caller_value):
    monkeypatch.setenv(_ENV, "1")
    args = _server_args(disable_overlap_schedule=caller_value)
    want_cuda_graph = configure_talker_server_args(args, feedback_enabled=True)
    assert args.disable_overlap_schedule is caller_value
    assert want_cuda_graph is True
    assert args.disable_cuda_graph is True


def test_env_flag_leaves_cuda_graph_disabled_when_not_requested(monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    args = _server_args(disable_cuda_graph=True)
    want_cuda_graph = configure_talker_server_args(args, feedback_enabled=True)
    assert want_cuda_graph is False
    assert args.disable_cuda_graph is True


def test_env_flag_zero_forces_overlap_disabled(monkeypatch):
    monkeypatch.setenv(_ENV, "0")
    args = _server_args()
    configure_talker_server_args(args, feedback_enabled=True)
    assert args.disable_overlap_schedule is True


@pytest.mark.parametrize("env_value", [None, "0", "1"])
def test_feedback_disabled_ignores_env(monkeypatch, env_value):
    if env_value is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, env_value)
    args = _server_args()
    want_cuda_graph = configure_talker_server_args(args, feedback_enabled=False)
    assert args.disable_overlap_schedule is False
    assert args.disable_cuda_graph is False
    assert want_cuda_graph is True
    assert args.disable_radix_cache is True
    assert args.chunked_prefill_size == 0
