# SPDX-License-Identifier: Apache-2.0
"""Lifetime of ``feedback_slot_idx`` across retract, and the depth-1 replay limit.

``feedback_slot_idx`` records the ``req_pool_idx`` a feedback row was emitted into.
Retract frees that pool index, so the retract snapshot is the last legal read of the
recorded slot: any later read can land on a row another request now owns. These tests
drive a second retract against a recycled slot on CPU, without a server or a GPU.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sglang.srt.managers.scheduler import Scheduler as _Upstream

from sglang_omni.models.qwen3_omni.components.feedback_slots import feedback_slot_rows
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler

MAX_RUNNING_REQUESTS = 4
POOL_IDX = 2
HIDDEN = 3
CPU = torch.device("cpu")
STRANGER_VALUE = 99.0
TEXT_ROW = torch.full((HIDDEN,), 10.0)


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_slots=torch.zeros(
            feedback_slot_rows(MAX_RUNNING_REQUESTS), HIDDEN, dtype=torch.float32
        ),
        _feedback_buffer=torch.zeros(1, HIDDEN, dtype=torch.float32),
        _feedback_mask=torch.zeros(1, dtype=torch.bool),
        _output_embeds=torch.zeros(1, HIDDEN, dtype=torch.float32),
        _output_codes=torch.tensor([[0, 100]], dtype=torch.long),
    )


def _runner(model: SimpleNamespace) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner._feedback_enabled = True
    runner._code2wav_target = "code2wav"
    runner._outbox = SimpleNamespace(sent=[])
    runner._outbox.put = runner._outbox.sent.append
    return runner


def _req(rid: str, pool_idx: int | None) -> SimpleNamespace:
    return SimpleNamespace(rid=rid, req_pool_idx=pool_idx)


def _data(req: SimpleNamespace) -> SimpleNamespace:
    data = SimpleNamespace(
        pending_feedback_count=0,
        feedback_slot_idx=None,
        retracted_feedback_embed=None,
        pending_text_queue=deque(),
        thinker_chunks_done=False,
        tts_pad_embed=None,
        stage_payload=None,
        decode_input_embeds=[],
        req=req,
    )
    req._omni_data = data
    return data


def _emit(
    runner: QwenTalkerModelRunner,
    req: SimpleNamespace,
    data: SimpleNamespace,
    value: float,
) -> None:
    runner.model._output_embeds[0] = torch.full((HIDDEN,), value)
    runner._emit_code_chunks_and_feedback(
        schedule_batch=SimpleNamespace(reqs=[req]),
        requests=[SimpleNamespace(data=data)],
        pool_indices=torch.tensor([req.req_pool_idx], dtype=torch.long),
    )


def _retract_scheduler(
    runner: QwenTalkerModelRunner, monkeypatch: Any
) -> QwenTalkerScheduler:
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._model_runner = runner
    scheduler.queued = []
    monkeypatch.setattr(
        _Upstream,
        "_add_request_to_queue",
        lambda self, req, is_retracted=False: self.queued.append(
            (req.rid, is_retracted)
        ),
    )
    return scheduler


def _retract(scheduler: QwenTalkerScheduler, req: SimpleNamespace) -> None:
    req.is_retracted = True
    req.req_pool_idx = None
    scheduler._add_request_to_queue(req, is_retracted=True)


def _consume_one_frame(runner: QwenTalkerModelRunner, data: SimpleNamespace) -> None:
    data.pending_text_queue.append(TEXT_ROW)
    runner._write_feedback_buffers([SimpleNamespace(data=data)])


def _drive_to_second_retract(
    runner: QwenTalkerModelRunner,
    scheduler: QwenTalkerScheduler,
    req: SimpleNamespace,
    data: SimpleNamespace,
) -> SimpleNamespace:
    # Note (wenyao): two pending frames exceed the one-row snapshot depth.
    _emit(runner, req, data, 5.0)
    _emit(runner, req, data, 6.0)
    assert data.pending_feedback_count == 2
    assert data.feedback_slot_idx == POOL_IDX

    _retract(scheduler, req)
    assert torch.equal(data.retracted_feedback_embed, torch.full((HIDDEN,), 6.0))

    _consume_one_frame(runner, data)
    assert data.pending_feedback_count == 1
    assert data.retracted_feedback_embed is None

    other = _req("rB", POOL_IDX)
    _emit(runner, other, _data(other), STRANGER_VALUE)
    assert torch.equal(
        runner.model._feedback_slots[POOL_IDX], torch.full((HIDDEN,), STRANGER_VALUE)
    )
    return other


def test_second_retract_cannot_read_a_recycled_slot(monkeypatch: Any) -> None:
    model = _model()
    runner = _runner(model)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    scheduler = _retract_scheduler(runner, monkeypatch)

    _drive_to_second_retract(runner, scheduler, req, data)

    req.is_retracted = True
    req.req_pool_idx = None
    with pytest.raises(RuntimeError, match="no recorded slot"):
        runner.snapshot_feedback_for_retract(req)

    assert data.retracted_feedback_embed is None
    assert (
        QwenTalkerModelRunner._take_next_decode_input_embed(
            sched_req=SimpleNamespace(data=data),
            device=CPU,
            dtype=torch.float32,
        )
        is None
    )
    own_row = torch.full((HIDDEN,), 6.0) + TEXT_ROW
    for row in data.decode_input_embeds:
        assert torch.equal(row, own_row)


def test_snapshot_clears_the_recorded_slot_index(monkeypatch: Any) -> None:
    model = _model()
    runner = _runner(model)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    _emit(runner, req, data, 5.0)

    _retract(_retract_scheduler(runner, monkeypatch), req)

    assert torch.equal(data.retracted_feedback_embed, torch.full((HIDDEN,), 5.0))
    assert data.feedback_slot_idx is None


def test_retract_clears_the_slot_index_when_a_snapshot_is_already_held(
    monkeypatch: Any,
) -> None:
    model = _model()
    runner = _runner(model)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    _emit(runner, req, data, 5.0)
    held = torch.full((HIDDEN,), 42.0)
    data.retracted_feedback_embed = held

    _retract(_retract_scheduler(runner, monkeypatch), req)

    assert data.retracted_feedback_embed is held
    assert data.feedback_slot_idx is None


def test_retract_without_pending_feedback_clears_the_slot_index(
    monkeypatch: Any,
) -> None:
    model = _model()
    runner = _runner(model)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    _emit(runner, req, data, 5.0)
    data.pending_feedback_count = 0

    _retract(_retract_scheduler(runner, monkeypatch), req)

    assert data.retracted_feedback_embed is None
    assert data.feedback_slot_idx is None


def test_emit_after_retract_rearms_the_slot_index(monkeypatch: Any) -> None:
    model = _model()
    runner = _runner(model)
    scheduler = _retract_scheduler(runner, monkeypatch)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    _emit(runner, req, data, 5.0)
    _retract(scheduler, req)
    data.retracted_feedback_embed = None
    data.pending_feedback_count = 0

    new_pool_idx = POOL_IDX + 1
    req.req_pool_idx = new_pool_idx
    _emit(runner, req, data, 7.0)
    assert data.feedback_slot_idx == new_pool_idx

    _retract(scheduler, req)

    assert torch.equal(data.retracted_feedback_embed, torch.full((HIDDEN,), 7.0))
    assert data.feedback_slot_idx is None


def test_pause_path_retract_retires_the_slot_index(monkeypatch: Any) -> None:
    model = _model()
    runner = _runner(model)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    _emit(runner, req, data, 5.0)

    scheduler = _retract_scheduler(runner, monkeypatch)
    req.is_retracted = True
    req.req_pool_idx = None
    scheduler._add_request_to_queue(req)

    assert scheduler.queued == [("rA", False)]
    assert torch.equal(data.retracted_feedback_embed, torch.full((HIDDEN,), 5.0))
    assert data.feedback_slot_idx is None


def _containment_scheduler(
    runner: QwenTalkerModelRunner, monkeypatch: Any
) -> QwenTalkerScheduler:
    """A scheduler with enough state for the real ``OmniScheduler.abort`` to run."""
    scheduler = _retract_scheduler(runner, monkeypatch)
    scheduler.is_entry_rank = True
    scheduler.errors = []
    scheduler.outbox = SimpleNamespace(put=scheduler.errors.append)
    scheduler.inbox = queue.Queue()
    scheduler._request_admission_lock = threading.Lock()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._pending_request_builds = {}
    scheduler._pending_request_admissions = {}
    scheduler._backlogged_request_build_payloads = []
    scheduler.waiting_queue = []
    scheduler._abort_callback = None
    scheduler._pending_stream_ingress = {}
    scheduler._completed_request_ids = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.running_batch = None
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._async_pending = None
    return scheduler


def test_failed_retract_snapshot_fails_one_request_not_the_stage(
    monkeypatch: Any,
) -> None:
    model = _model()
    runner = _runner(model)
    scheduler = _containment_scheduler(runner, monkeypatch)

    doomed = _req("rA", POOL_IDX)
    doomed_data = _data(doomed)
    _drive_to_second_retract(runner, scheduler, doomed, doomed_data)
    scheduler.queued.clear()

    healthy = _req("rC", POOL_IDX + 1)
    healthy_data = _data(healthy)
    _emit(runner, healthy, healthy_data, 3.0)

    _retract(scheduler, doomed)
    _retract(scheduler, healthy)

    assert [(m.request_id, m.type) for m in scheduler.errors] == [("rA", "error")]
    assert isinstance(scheduler.errors[0].data, RuntimeError)
    assert scheduler._aborted_request_ids == {"rA"}
    assert doomed_data.retracted_feedback_embed is None

    assert scheduler.queued == [("rC", True)]
    assert torch.equal(
        healthy_data.retracted_feedback_embed, torch.full((HIDDEN,), 3.0)
    )


def test_replay_past_the_single_snapshot_raises() -> None:
    model = _model()
    runner = _runner(model)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    _emit(runner, req, data, 5.0)
    _emit(runner, req, data, 6.0)
    data.retracted_feedback_embed = model._feedback_slots[POOL_IDX].clone()
    data.pending_text_queue.extend(
        [torch.full((HIDDEN,), 10.0), torch.full((HIDDEN,), 20.0)]
    )

    with pytest.raises(RuntimeError, match="at most one feedback row"):
        QwenTalkerModelRunner._generated_prefill_slice(
            sched_req=SimpleNamespace(data=data),
            gen_start=0,
            gen_end=2,
            device=CPU,
            dtype=torch.float32,
            take_next_decode_input_embed=QwenTalkerModelRunner._take_next_decode_input_embed,
        )

    assert data.pending_feedback_count == 1
