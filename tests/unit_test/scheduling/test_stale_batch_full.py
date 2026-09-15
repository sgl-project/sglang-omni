# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

import sglang.srt.managers.scheduler as upstream_scheduler  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.schedule_policy import AddReqResult  # noqa: E402

from tests.unit_test.pipeline.test_scheduler import (  # noqa: E402
    _construct_omni_scheduler,
)


class _WaitingReq:
    beam_group = None

    def __init__(self, rid: str) -> None:
        self.rid = rid

    def init_next_round_input(self, _tree_cache) -> None:
        pass


class _PrefillBatch:
    return_logprob = False
    input_embeds = None

    def __init__(self, reqs) -> None:
        self.reqs = reqs

    def prepare_for_extend(self) -> None:
        pass


class _PrefillAdder:
    def __init__(self, *_args, **_kwargs) -> None:
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None

    def add_one_req(self, req, **_kwargs):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


def _latched_scheduler(monkeypatch, *, allocatable_rows: int):
    scheduler = _construct_omni_scheduler(monkeypatch)
    waiters = [_WaitingReq("waiting-0"), _WaitingReq("waiting-1")]
    scheduler.waiting_queue = waiters.copy()
    scheduler.running_batch.batch_is_full = True
    scheduler.chunked_req = None
    scheduler.get_num_allocatable_reqs = lambda *_args, **_kwargs: allocatable_rows
    scheduler.policy.calc_priority = lambda *_args: None
    scheduler.tp_worker.model_runner.attn_backend = SimpleNamespace()
    scheduler.tp_worker.model_runner.prefill_aware_swa = False
    scheduler.server_args.prefill_max_requests = None
    scheduler.server_args.enable_flexkv = False
    monkeypatch.setattr(upstream_scheduler, "get_memory", lambda: scheduler.server_args)
    monkeypatch.setattr(
        upstream_scheduler, "get_schedule", lambda: scheduler.server_args
    )
    scheduler.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda batch, **_kwargs: batch,
        maybe_convert_decode_to_extend=lambda batch: batch,
    )
    scheduler.ngram_embedding_manager = SimpleNamespace(
        prepare_for_forward=lambda batch, **_kwargs: batch
    )
    scheduler.load_inquirer = SimpleNamespace(
        _get_num_pending_tokens=lambda **_kwargs: 0
    )
    monkeypatch.setattr(upstream_scheduler, "PrefillAdder", _PrefillAdder)
    monkeypatch.setattr(
        upstream_scheduler.PrefillStats,
        "from_adder",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        upstream_scheduler, "set_schedule_time_batch", lambda _batch: None
    )
    monkeypatch.setattr(upstream_scheduler, "set_time_batch", lambda *_args: None)
    monkeypatch.setattr(
        ScheduleBatch,
        "init_new",
        classmethod(lambda _cls, reqs, *_args, **_kwargs: _PrefillBatch(reqs)),
    )
    return scheduler, waiters


def test_empty_stale_full_batch_admits_waiter_when_row_is_allocatable(
    monkeypatch,
) -> None:
    scheduler, waiters = _latched_scheduler(monkeypatch, allocatable_rows=1)

    batch = scheduler.get_next_batch_to_run()

    assert batch is not None
    assert batch.reqs == [waiters[0]]
    assert scheduler.waiting_queue == [waiters[1]]


def test_empty_stale_full_batch_keeps_backoff_without_allocatable_row(
    monkeypatch,
) -> None:
    scheduler, waiters = _latched_scheduler(monkeypatch, allocatable_rows=0)

    batch = scheduler.get_next_batch_to_run()

    assert batch is None
    assert scheduler.running_batch.batch_is_full is True
    assert scheduler.waiting_queue == waiters
