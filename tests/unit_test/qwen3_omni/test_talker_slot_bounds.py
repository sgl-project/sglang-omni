# SPDX-License-Identifier: Apache-2.0
"""Bounds coverage for the req_pool_idx-keyed talker feedback table.

``ReqToTokenPool`` reserves row 0 and allocates ``req_pool_idx`` from ``[1, size]``
inclusive, so the top allocatable index equals the pool size. Every test here sizes
the table through the production helper, so under-sizing it by the reserved row makes
these fail on CPU instead of only under sustained slot churn on a GPU.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.feedback_slots import feedback_slot_rows
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner

MAX_RUNNING_REQUESTS = 4
TOP_POOL_IDX = MAX_RUNNING_REQUESTS
HIDDEN = 3


def _model(bs: int) -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_slots=torch.zeros(
            feedback_slot_rows(MAX_RUNNING_REQUESTS), HIDDEN, dtype=torch.float32
        ),
        _feedback_buffer=torch.zeros(bs, HIDDEN, dtype=torch.float32),
        _feedback_mask=torch.zeros(bs, dtype=torch.bool),
        _output_embeds=torch.stack(
            [torch.full((HIDDEN,), float(i + 1)) for i in range(bs)]
        ),
        _output_codes=torch.stack(
            [torch.tensor([i, i + 100], dtype=torch.long) for i in range(bs)]
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


def _sched_batch(pool_ids: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(rid=f"r{i}", req_pool_idx=pool_idx)
            for i, pool_idx in enumerate(pool_ids)
        ],
        req_pool_indices=torch.tensor(pool_ids, dtype=torch.long),
    )


def _emit_requests(n: int) -> list:
    return [
        SimpleNamespace(
            data=SimpleNamespace(pending_feedback_count=0, stage_payload=None)
        )
        for _ in range(n)
    ]


def _emit_frame(
    runner: QwenTalkerModelRunner, schedule_batch: SimpleNamespace, requests: list
) -> None:
    runner._emit_code_chunks_and_feedback(
        schedule_batch=schedule_batch,
        requests=requests,
        pool_indices=QwenTalkerModelRunner._batch_pool_indices(
            schedule_batch, len(requests)
        ),
    )


def _consume_data(
    pool_idx: int | None,
    text: torch.Tensor,
    *,
    override: torch.Tensor | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_count=1,
        retracted_feedback_embed=override,
        pending_text_queue=deque([text]),
        thinker_chunks_done=False,
        tts_pad_embed=None,
        decode_input_embeds=[],
        req=SimpleNamespace(req_pool_idx=pool_idx),
        feedback_slot_idx=pool_idx,
    )


def test_emit_scatter_lands_at_top_pool_index() -> None:
    model = _model(bs=1)
    runner = _runner(model)
    requests = _emit_requests(1)

    _emit_frame(runner, _sched_batch([TOP_POOL_IDX]), requests)

    assert torch.equal(model._feedback_slots[TOP_POOL_IDX], model._output_embeds[0])
    assert requests[0].data.feedback_slot_idx == TOP_POOL_IDX


def test_emit_scatter_covers_every_allocatable_index() -> None:
    pool_ids = list(range(1, MAX_RUNNING_REQUESTS + 1))
    model = _model(bs=len(pool_ids))
    runner = _runner(model)

    _emit_frame(runner, _sched_batch(pool_ids), _emit_requests(len(pool_ids)))

    for i, pool_idx in enumerate(pool_ids):
        assert torch.equal(model._feedback_slots[pool_idx], model._output_embeds[i])
    assert torch.equal(model._feedback_slots[0], torch.zeros(HIDDEN))


def test_emit_ignores_cuda_graph_padded_rows() -> None:
    # Note (wenyao): CUDA-graph padding uses row 0, not a request row.
    real_pool_ids = [1, TOP_POOL_IDX]
    model = _model(bs=len(real_pool_ids))
    runner = _runner(model)

    schedule_batch = _sched_batch(real_pool_ids + [0, 0])
    _emit_frame(runner, schedule_batch, _emit_requests(len(real_pool_ids)))

    for i, pool_idx in enumerate(real_pool_ids):
        assert torch.equal(model._feedback_slots[pool_idx], model._output_embeds[i])
    assert torch.equal(model._feedback_slots[0], torch.zeros(HIDDEN))
    assert len(runner._outbox.sent) == len(real_pool_ids)


def test_consume_gather_reads_top_pool_index() -> None:
    model = _model(bs=1)
    runner = _runner(model)
    feedback = torch.full((HIDDEN,), 5.0)
    model._feedback_slots[TOP_POOL_IDX] = feedback
    text = torch.full((HIDDEN,), 10.0)
    data = _consume_data(TOP_POOL_IDX, text)

    runner._write_feedback_buffers([SimpleNamespace(data=data)])

    assert torch.equal(model._feedback_buffer[0], feedback + text)
    assert model._feedback_mask.tolist() == [True]
    assert data.pending_feedback_count == 0


def test_retract_snapshot_reads_top_pool_index() -> None:
    model = _model(bs=1)
    runner = _runner(model)
    feedback = torch.full((HIDDEN,), 7.0)
    model._feedback_slots[TOP_POOL_IDX] = feedback
    data = _consume_data(TOP_POOL_IDX, torch.zeros(HIDDEN))
    data.retracted_feedback_embed = None

    runner.snapshot_feedback_for_retract(SimpleNamespace(_omni_data=data))

    assert torch.equal(data.retracted_feedback_embed, feedback)
    # Note (wenyao): the snapshot must survive pool-row reuse.
    model._feedback_slots[TOP_POOL_IDX] = torch.zeros(HIDDEN)
    assert torch.equal(data.retracted_feedback_embed, feedback)


def test_consume_gather_row_zero_fallback_is_discarded() -> None:
    # Note (wenyao): a snapshot must override the reserved-row gather fallback.
    model = _model(bs=1)
    runner = _runner(model)
    model._feedback_slots[0] = torch.full((HIDDEN,), 99.0)
    override = torch.full((HIDDEN,), 1.0)
    text = torch.full((HIDDEN,), 10.0)
    data = _consume_data(None, text, override=override)

    runner._write_feedback_buffers([SimpleNamespace(data=data)])

    assert torch.equal(model._feedback_buffer[0], override + text)


def _runner_through_init(
    slot_rows: int,
    pool_size: int,
    *,
    feedback_enabled: bool = True,
    expose_pool: bool = True,
) -> QwenTalkerModelRunner:
    model = SimpleNamespace(_feedback_slots=torch.zeros(slot_rows, HIDDEN))
    pool = SimpleNamespace(
        size=pool_size,
        req_to_token=torch.zeros(pool_size + 1, 1, dtype=torch.int32),
    )
    inner_runner = SimpleNamespace(model=model)
    if expose_pool:
        inner_runner.req_to_token_pool = pool
    tp_worker = SimpleNamespace(gpu_id=0, model_runner=inner_runner)
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner.tp_worker = tp_worker
    runner._feedback_enabled = feedback_enabled
    if feedback_enabled:
        runner._check_feedback_slots_cover_pool()
    return runner


def test_startup_guard_accepts_pool_sized_slots() -> None:
    runner = _runner_through_init(
        feedback_slot_rows(MAX_RUNNING_REQUESTS), MAX_RUNNING_REQUESTS
    )

    assert runner.model._feedback_slots.shape[0] == MAX_RUNNING_REQUESTS + 1


def test_startup_guard_rejects_slots_missing_the_reserved_row() -> None:
    with pytest.raises(RuntimeError, match="too small for the request pool"):
        _runner_through_init(MAX_RUNNING_REQUESTS, MAX_RUNNING_REQUESTS)


def test_startup_guard_rejects_tables_undersized_by_many_rows() -> None:
    with pytest.raises(RuntimeError, match="too small for the request pool"):
        _runner_through_init(4, 16)


def test_startup_guard_requires_the_request_pool_contract() -> None:
    with pytest.raises(AttributeError, match="req_to_token_pool"):
        _runner_through_init(
            MAX_RUNNING_REQUESTS, MAX_RUNNING_REQUESTS, expose_pool=False
        )


def test_startup_guard_skipped_when_feedback_disabled() -> None:
    runner = _runner_through_init(
        MAX_RUNNING_REQUESTS, MAX_RUNNING_REQUESTS, feedback_enabled=False
    )

    assert runner._feedback_enabled is False
