# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sglang.srt.managers.scheduler import Scheduler as _Upstream

from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import ModelRunnerOutput

POOL_SIZE = 8
# Note (wenyao): pool rows deliberately differ from batch order.
POOL_BY_RID = {"r0": 3, "r1": 5, "r2": 0}


def _init_terminal_output_state(scheduler: OmniScheduler) -> None:
    scheduler._request_admission_lock = threading.RLock()
    scheduler.is_entry_rank = True
    scheduler._model_runner = None
    scheduler._stream_output_builder = None
    scheduler._request_finished_callback = None
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}
    scheduler._prefill_end_done = set()


def _fake_model(n: int, hidden: int, code_groups: int) -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_buffer=torch.zeros(n, hidden, dtype=torch.float32),
        _feedback_mask=torch.zeros(n, dtype=torch.bool),
        _feedback_slots=torch.zeros(POOL_SIZE, hidden, dtype=torch.float32),
        _output_codes=torch.stack(
            [torch.tensor([i, i + 100], dtype=torch.long) for i in range(n)]
        )[:, :code_groups],
        _output_embeds=torch.stack(
            [torch.full((hidden,), float(i * 7 + 1)) for i in range(n)]
        ),
    )


def _runner(model: SimpleNamespace) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner._feedback_enabled = True
    runner._code2wav_target = "code2wav"
    runner._outbox = SimpleNamespace(sent=[])
    runner._outbox.put = runner._outbox.sent.append
    return runner


def _make_req(rid: str) -> SimpleNamespace:
    return SimpleNamespace(rid=rid, req_pool_idx=POOL_BY_RID[rid])


def _data(
    feedback: torch.Tensor | None,
    text: torch.Tensor | None,
    *,
    req: SimpleNamespace,
    thinker_done: bool = False,
    pad: torch.Tensor | None = None,
    stage_payload: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_count=0 if feedback is None else 1,
        feedback_slot_idx=None,
        retracted_feedback_embed=None,
        pending_text_queue=deque([text]) if text is not None else deque(),
        thinker_chunks_done=thinker_done,
        tts_pad_embed=pad,
        stage_payload=stage_payload,
        decode_input_embeds=[],
        req=req,
    )


def _req_wrap(data: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(data=data)


def _sched_batch(reqs: list) -> SimpleNamespace:
    reqs = list(reqs)
    return SimpleNamespace(
        reqs=reqs,
        req_pool_indices=torch.tensor(
            [0 if r.req_pool_idx is None else r.req_pool_idx for r in reqs],
            dtype=torch.long,
        ),
    )


def _pool_indices(requests: list) -> torch.Tensor:
    return torch.tensor(
        [
            0 if r.data.req.req_pool_idx is None else int(r.data.req.req_pool_idx)
            for r in requests
        ],
        dtype=torch.long,
    )


def _emit_step(runner, requests: list) -> torch.Tensor:
    codes_snap = runner._emit_code_chunks_and_feedback(
        requests=requests, pool_indices=_pool_indices(requests)
    )
    runner._put_code_chunks(requests, codes_snap)
    return codes_snap


def _pool_req(rid: str, pool_idx: int | None) -> SimpleNamespace:
    return SimpleNamespace(rid=rid, req_pool_idx=pool_idx)


def _retract_scheduler(
    runner: QwenTalkerModelRunner, monkeypatch: Any
) -> SimpleNamespace:
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


def _emit(
    runner: QwenTalkerModelRunner,
    req: SimpleNamespace,
    data: SimpleNamespace,
    embed: torch.Tensor,
) -> None:
    assert data.req is req
    runner.model._output_embeds[0] = embed
    _emit_step(runner, [_req_wrap(data)])


def test_recycled_pool_slot_cannot_feed_stale_feedback() -> None:
    n, hidden, code_groups = 2, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    old_req = _make_req("r0")
    old_data = _data(None, torch.full((hidden,), 10.0), req=old_req)
    _emit(runner, old_req, old_data, torch.full((hidden,), 5.0))
    assert torch.equal(
        model._feedback_slots[POOL_BY_RID["r0"]], torch.full((hidden,), 5.0)
    )

    new_req = _pool_req("r9", POOL_BY_RID["r0"])
    new_data = _data(None, torch.full((hidden,), 20.0), req=new_req)
    assert not QwenTalkerModelRunner._data_has_next_decode_input(new_data)

    _emit(runner, new_req, new_data, torch.full((hidden,), 7.0))
    runner._write_feedback_buffers(
        [_req_wrap(new_data)],
        _pool_indices([_req_wrap(new_data)]),
    )

    assert torch.equal(model._feedback_buffer[0], torch.full((hidden,), 27.0))


def test_retract_replay_reproduces_history_rows() -> None:
    n, hidden, code_groups = 1, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    req = _make_req("r0")
    data = _data(None, None, req=req)
    for step in range(2):
        model._feedback_slots[POOL_BY_RID["r0"]] = torch.full(
            (hidden,), float(step + 1)
        )
        data.pending_feedback_count = 1
        data.pending_text_queue.append(torch.full((hidden,), float(10 * (step + 1))))
        runner._write_feedback_buffers(
            [_req_wrap(data)],
            _pool_indices([_req_wrap(data)]),
        )
    history = [row.clone() for row in data.decode_input_embeds]
    assert len(history) == 2

    req.req_pool_idx = None
    model._feedback_slots[POOL_BY_RID["r0"]] = torch.full((hidden,), -1.0)

    replayed = QwenTalkerModelRunner._generated_prefill_slice(
        sched_req=_req_wrap(data),
        gen_start=0,
        gen_end=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        take_next_decode_input_embed=QwenTalkerModelRunner._take_next_decode_input_embed,
    )

    assert torch.equal(replayed, torch.stack(history))


def test_retract_snapshots_slot_before_another_request_reuses_it(
    monkeypatch: Any,
) -> None:
    n, hidden, code_groups = 2, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    req = _make_req("r0")
    data = _data(None, torch.full((hidden,), 10.0), req=req)
    req._omni_data = data
    _emit(runner, req, data, torch.full((hidden,), 5.0))
    assert data.pending_feedback_count == 1

    scheduler = _retract_scheduler(runner, monkeypatch)
    req.is_retracted = True
    req.req_pool_idx = None
    scheduler._add_request_to_queue(req, is_retracted=True)

    assert scheduler.queued == [("r0", True)]
    assert data.pending_feedback_count == 1

    other_req = _pool_req("r9", POOL_BY_RID["r0"])
    other_data = _data(None, None, req=other_req)
    _emit(runner, other_req, other_data, torch.full((hidden,), 99.0))
    assert torch.equal(
        model._feedback_slots[POOL_BY_RID["r0"]], torch.full((hidden,), 99.0)
    )

    combined = QwenTalkerModelRunner._take_next_decode_input_embed(
        sched_req=_req_wrap(data),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(combined, torch.full((hidden,), 15.0))
    assert data.pending_feedback_count == 0
    assert data.retracted_feedback_embed is None


def test_retract_snapshot_runs_on_the_flag_only_path(monkeypatch: Any) -> None:
    n, hidden, code_groups = 1, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    req = _make_req("r0")
    data = _data(None, torch.full((hidden,), 10.0), req=req)
    req._omni_data = data
    _emit(runner, req, data, torch.full((hidden,), 5.0))

    scheduler = _retract_scheduler(runner, monkeypatch)
    req.is_retracted = True
    req.req_pool_idx = None
    scheduler._add_request_to_queue(req)

    assert scheduler.queued == [("r0", False)]
    assert torch.equal(data.retracted_feedback_embed, torch.full((hidden,), 5.0))


def test_upstream_add_request_to_queue_still_takes_is_retracted() -> None:
    parameters = inspect.signature(_Upstream._add_request_to_queue).parameters
    assert "is_retracted" in parameters


def test_pending_feedback_without_recorded_slot_raises() -> None:
    n, hidden, code_groups = 1, 3, 2
    runner = _runner(_fake_model(n, hidden, code_groups))

    req = _make_req("r0")
    data = _data(torch.full((hidden,), 5.0), torch.full((hidden,), 10.0), req=req)
    data.feedback_slot_idx = None
    req._omni_data = data

    with pytest.raises(RuntimeError, match="no recorded slot"):
        runner.snapshot_feedback_for_retract(req)


def test_request_data_defaults_feedback_slot_idx_to_none() -> None:
    assert SGLangARRequestData().feedback_slot_idx is None


def test_retract_without_pending_feedback_takes_no_snapshot(monkeypatch: Any) -> None:
    n, hidden, code_groups = 1, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    req = _make_req("r0")
    data = _data(None, torch.full((hidden,), 10.0), req=req)
    req._omni_data = data
    req.is_retracted = True

    scheduler = _retract_scheduler(runner, monkeypatch)
    scheduler._add_request_to_queue(req, is_retracted=True)

    assert data.retracted_feedback_embed is None
    assert scheduler.queued == [("r0", True)]


def test_finish_clears_pending_feedback_state() -> None:
    hidden = 3
    scheduler = object.__new__(OmniScheduler)
    _init_terminal_output_state(scheduler)
    scheduler._aborted_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler.server_args = SimpleNamespace(weight_version="w0")
    scheduler._result_adapter = lambda data: data
    scheduler.outbox = SimpleNamespace(put=lambda message: None)

    data = _data(None, None, req=None)
    data.pending_feedback_count = 2
    data.retracted_feedback_embed = torch.zeros(hidden)
    req = SimpleNamespace(
        rid="r0",
        output_ids=[1, 2],
        finished_reason=None,
        _omni_data=data,
        finished=lambda: True,
        _omni_terminal_claimed=False,
    )

    scheduler.stream_output([req])

    assert data.pending_feedback_count == 0
    assert data.retracted_feedback_embed is None
    assert data.decode_input_embeds is None


def test_row_ownership_survives_prep_then_emit() -> None:
    n, hidden, code_groups = 3, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    reqs = [_make_req(f"r{i}") for i in range(n)]
    requests = [_req_wrap(_data(feedbacks[i], texts[i], req=reqs[i])) for i in range(n)]
    for i in range(n):
        model._feedback_slots[POOL_BY_RID[f"r{i}"]] = feedbacks[i]
    schedule_batch = _sched_batch(reqs)
    for i in range(n):
        assert requests[i].data.req is schedule_batch.reqs[i]

    runner._write_feedback_buffers(requests, _pool_indices(requests))

    assert torch.equal(model._feedback_mask, torch.ones(n, dtype=torch.bool))
    for i in range(n):
        assert torch.equal(model._feedback_buffer[i], feedbacks[i] + texts[i])

    _emit_step(runner, requests)

    sent = runner._outbox.sent
    assert [m.request_id for m in sent] == [f"r{i}" for i in range(n)]
    for i, msg in enumerate(sent):
        assert msg.target == "code2wav"
        assert msg.metadata == {"stream": False}
        assert torch.equal(msg.data, model._output_codes[i])
        assert requests[i].data.pending_feedback_count == 1
        assert torch.equal(
            model._feedback_slots[POOL_BY_RID[f"r{i}"]], model._output_embeds[i]
        )


def test_sparse_feedback_row_stays_unwritten() -> None:
    n, hidden, code_groups = 3, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    requests = [
        _req_wrap(_data(feedbacks[0], texts[0], req=_make_req("r0"))),
        _req_wrap(_data(feedbacks[1], None, req=_make_req("r1"), thinker_done=False)),
        _req_wrap(_data(feedbacks[2], texts[2], req=_make_req("r2"))),
    ]
    for i in range(n):
        model._feedback_slots[POOL_BY_RID[f"r{i}"]] = feedbacks[i]

    runner._write_feedback_buffers(requests, _pool_indices(requests))

    assert model._feedback_mask.tolist() == [True, False, True]
    assert torch.equal(model._feedback_buffer[1], torch.zeros(hidden))
    assert torch.equal(model._feedback_buffer[0], feedbacks[0] + texts[0])
    assert torch.equal(model._feedback_buffer[2], feedbacks[2] + texts[2])


def test_stale_mask_cannot_leak_into_reused_slot() -> None:
    n, hidden, code_groups = 2, 3, 2
    model = _fake_model(n, hidden, code_groups)
    model._feedback_mask[:n] = True
    runner = _runner(model)

    feedback1 = torch.full((hidden,), 5.0)
    text1 = torch.full((hidden,), 50.0)
    requests = [
        _req_wrap(
            _data(
                torch.full((hidden,), 1.0),
                None,
                req=_make_req("r0"),
                thinker_done=False,
            )
        ),
        _req_wrap(_data(feedback1, text1, req=_make_req("r1"))),
    ]
    model._feedback_slots[POOL_BY_RID["r0"]] = torch.full((hidden,), 1.0)
    model._feedback_slots[POOL_BY_RID["r1"]] = feedback1

    runner._write_feedback_buffers(requests, _pool_indices(requests))

    assert model._feedback_mask.tolist() == [False, True]
    assert torch.equal(model._feedback_buffer[0], torch.zeros(hidden))
    assert torch.equal(model._feedback_buffer[1], feedback1 + text1)


def test_row_ownership_tracks_current_batch_order_across_steps() -> None:
    n, hidden, code_groups = 2, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    reqs = {rid: _make_req(rid) for rid in ("r0", "r1")}
    request_data = {
        "r0": _data(
            torch.full((hidden,), 1.0),
            torch.full((hidden,), 10.0),
            req=reqs["r0"],
        ),
        "r1": _data(
            torch.full((hidden,), 2.0),
            torch.full((hidden,), 20.0),
            req=reqs["r1"],
        ),
    }
    model._feedback_slots[POOL_BY_RID["r0"]] = torch.full((hidden,), 1.0)
    model._feedback_slots[POOL_BY_RID["r1"]] = torch.full((hidden,), 2.0)
    request_data["r0"].pending_text_queue.extend(
        [torch.full((hidden,), 11.0), torch.full((hidden,), 12.0)]
    )
    request_data["r1"].pending_text_queue.append(torch.full((hidden,), 21.0))

    previous_feedback = {
        "r0": torch.full((hidden,), 1.0),
        "r1": torch.full((hidden,), 2.0),
    }
    text_by_request = {
        "r0": [10.0, 11.0, 12.0],
        "r1": [20.0, 21.0],
    }
    step_orders = [("r0", "r1"), ("r1", "r0"), ("r0",)]
    expected_messages: list[tuple[str, torch.Tensor]] = []
    expected_pending_feedback: dict[str, torch.Tensor] = {}

    for step, order in enumerate(step_orders):
        requests = [_req_wrap(request_data[rid]) for rid in order]
        schedule_batch = _sched_batch([reqs[rid] for rid in order])
        schedule_batch.output_ids = None
        for row, rid in enumerate(order):
            assert request_data[rid].req is schedule_batch.reqs[row]
        expected_inputs = [
            previous_feedback[rid] + torch.full((hidden,), text_by_request[rid].pop(0))
            for rid in order
        ]

        runner._write_feedback_buffers(requests, _pool_indices(requests))

        assert model._feedback_mask.tolist() == [True] * len(order) + [False] * (
            n - len(order)
        )
        for row, expected in enumerate(expected_inputs):
            assert torch.equal(model._feedback_buffer[row], expected)

        # Match the real forward, which consumes and clears the active mask.
        model._feedback_mask[: len(order)] = False
        tokens = torch.tensor(
            [step * 10 + int(rid[-1]) for rid in order], dtype=torch.long
        )
        codes = torch.stack(
            [
                torch.tensor(
                    [step * 100 + int(rid[-1]), step * 100 + int(rid[-1]) + 1000],
                    dtype=torch.long,
                )
                for rid in order
            ]
        )
        embeds = torch.stack(
            [
                torch.full((hidden,), float(step * 100 + int(rid[-1]) + 1))
                for rid in order
            ]
        )
        model._output_codes[: len(order)] = codes
        model._output_embeds[: len(order)] = embeds

        result = SimpleNamespace()
        runner._stage_token_ids(result, tokens)
        _emit_step(runner, requests)

        emitted = runner._outbox.sent[-len(order) :]
        assert [message.request_id for message in emitted] == list(order)
        for row, rid in enumerate(order):
            assert torch.equal(emitted[row].data, codes[row])
            assert torch.equal(model._feedback_slots[POOL_BY_RID[rid]], embeds[row])
            previous_feedback[rid] = embeds[row].clone()
            expected_messages.append((rid, codes[row].clone()))
            expected_pending_feedback[rid] = embeds[row].clone()

        assert len(runner._outbox.sent) == len(expected_messages)
        for message, (expected_rid, expected_code) in zip(
            runner._outbox.sent, expected_messages
        ):
            assert message.request_id == expected_rid
            assert torch.equal(message.data, expected_code)
        for rid, expected_feedback in expected_pending_feedback.items():
            assert request_data[rid].pending_feedback_count == 1
            assert torch.equal(
                model._feedback_slots[POOL_BY_RID[rid]], expected_feedback
            )

        model_runner_output = ModelRunnerOutput(
            outputs={},
            can_run_cuda_graph=False,
            host_token_ids=runner._resolve_host_token_ids(result),
        )
        batch_result = OmniScheduler._make_batch_result(model_runner_output)
        assert batch_result.next_token_ids is model_runner_output.host_token_ids
        assert batch_result.next_token_ids.tolist() == tokens.tolist()


def test_run_batch_resolve_hands_upstream_the_staged_host_copy() -> None:
    # Async sibling of _make_batch_result. Upstream's process_batch_result calls
    # .tolist() on whatever it is handed, and on the device tensor that copy is
    # enqueued BEHIND the forward this iteration's launch already submitted — so
    # the host waits a whole step for a value the launch staged before it.
    device_ids = torch.tensor([11, 22], dtype=torch.long)
    host_ids = torch.tensor([11, 22], dtype=torch.long)
    pending = SimpleNamespace(batch_result=SimpleNamespace(next_token_ids=device_ids))
    scheduler = object.__new__(OmniScheduler)
    scheduler._model_runner = SimpleNamespace(
        execute_resolve=lambda step: ModelRunnerOutput(
            outputs={}, can_run_cuda_graph=False, host_token_ids=host_ids
        )
    )
    scheduler._emit_stream_output = lambda *args, **kwargs: None

    result = scheduler._run_batch_resolve(None, None, pending)

    assert result.next_token_ids is host_ids
    assert result.next_token_ids is not device_ids


def test_run_batch_resolve_keeps_device_ids_when_nothing_was_staged() -> None:
    device_ids = torch.tensor([5], dtype=torch.long)
    scheduler = object.__new__(OmniScheduler)
    scheduler._model_runner = SimpleNamespace(
        execute_resolve=lambda step: SimpleNamespace(
            next_token_ids=device_ids, can_run_cuda_graph=False, host_token_ids=None
        )
    )
    scheduler._emit_stream_output = lambda *args, **kwargs: None

    result = scheduler._run_batch_resolve(None, None, None)

    assert result.next_token_ids is device_ids


def test_run_batch_resolve_prefers_the_staged_host_copy() -> None:
    device_ids = torch.tensor([5], dtype=torch.long)
    host_ids = [5]
    scheduler = object.__new__(OmniScheduler)
    scheduler._model_runner = SimpleNamespace(
        execute_resolve=lambda step: SimpleNamespace(
            next_token_ids=device_ids, can_run_cuda_graph=False, host_token_ids=host_ids
        )
    )
    scheduler._emit_stream_output = lambda *args, **kwargs: None

    result = scheduler._run_batch_resolve(None, None, None)

    assert result.next_token_ids is host_ids


def test_make_batch_result_requires_declared_host_token_ids() -> None:
    malformed_output = SimpleNamespace(next_token_ids=None, can_run_cuda_graph=False)

    with pytest.raises(AttributeError, match="host_token_ids"):
        OmniScheduler._make_batch_result(malformed_output)


class _FakeReq:
    def __init__(self, rid: str, finished: bool, retracted: bool = False) -> None:
        self.rid = rid
        self._finished = finished
        self.is_retracted = retracted

    def finished(self) -> bool:
        return self._finished


def _resolve_scheduler(result: SimpleNamespace) -> tuple[OmniScheduler, list]:
    scheduler = object.__new__(OmniScheduler)
    captured: list = []
    scheduler._run_batch_resolve = (
        lambda batch, sched_output, pending_step, skip_rids=(): result
    )
    scheduler.process_batch_result = lambda batch, res: captured.append(
        ([r.rid for r in batch.reqs], res.next_token_ids)
    )
    return scheduler, captured


def test_overrun_drop_keeps_reqs_and_tokens_index_aligned() -> None:
    reqs = [
        _FakeReq("r0", finished=False),
        _FakeReq("r1", finished=True),
        _FakeReq("r2", finished=False),
        _FakeReq("r3", finished=True),
    ]
    batch = SimpleNamespace(reqs=list(reqs))
    result = SimpleNamespace(next_token_ids=torch.tensor([100, 101, 102, 103]))
    scheduler, captured = _resolve_scheduler(result)

    scheduler._resolve_and_process(batch, None, None)

    assert len(captured) == 1
    rids, tokens = captured[0]
    assert rids == ["r0", "r2"]
    assert tokens.tolist() == [100, 102]


def test_overrun_drop_retracted_row_is_dropped() -> None:
    reqs = [
        _FakeReq("r0", finished=False),
        _FakeReq("r1", finished=False, retracted=True),
        _FakeReq("r2", finished=False),
    ]
    batch = SimpleNamespace(reqs=list(reqs))
    result = SimpleNamespace(next_token_ids=torch.tensor([10, 11, 12]))
    scheduler, captured = _resolve_scheduler(result)

    scheduler._resolve_and_process(batch, None, None)

    rids, tokens = captured[0]
    assert rids == ["r0", "r2"]
    assert tokens.tolist() == [10, 12]


def test_overrun_drop_noop_keeps_full_alignment() -> None:
    reqs = [_FakeReq(f"r{i}", finished=False) for i in range(3)]
    batch = SimpleNamespace(reqs=list(reqs))
    result = SimpleNamespace(next_token_ids=torch.tensor([7, 8, 9]))
    scheduler, captured = _resolve_scheduler(result)

    scheduler._resolve_and_process(batch, None, None)

    rids, tokens = captured[0]
    assert rids == ["r0", "r1", "r2"]
    assert tokens.tolist() == [7, 8, 9]


def test_overrun_drop_all_finished_skips_process() -> None:
    reqs = [_FakeReq("r0", finished=True), _FakeReq("r1", finished=True)]
    batch = SimpleNamespace(reqs=list(reqs))
    result = SimpleNamespace(next_token_ids=torch.tensor([1, 2]))
    scheduler, captured = _resolve_scheduler(result)

    scheduler._resolve_and_process(batch, None, None)

    assert captured == []
    assert batch.reqs == []
