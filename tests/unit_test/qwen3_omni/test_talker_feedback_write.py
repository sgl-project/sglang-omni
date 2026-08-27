# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.models.qwen3_omni import talker_model_runner
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner

POOL_SIZE = 8
POOL_IDS = [6, 1, 4]


def _fake_model(n: int, hidden: int) -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_buffer=torch.zeros(n, hidden, dtype=torch.float32),
        _feedback_mask=torch.zeros(n, dtype=torch.bool),
        _feedback_slots=torch.zeros(POOL_SIZE, hidden, dtype=torch.float32),
    )


def _runner(model: SimpleNamespace) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    return runner


def _data(
    text: torch.Tensor | None,
    *,
    pool_idx: int,
    count: int = 1,
    thinker_done: bool = False,
    pad: torch.Tensor | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_count=count,
        retracted_feedback_embed=None,
        pending_text_queue=deque([text]) if text is not None else deque(),
        thinker_chunks_done=thinker_done,
        tts_pad_embed=pad,
        decode_input_embeds=[],
        req=SimpleNamespace(req_pool_idx=pool_idx),
    )


def _req_wrap(data: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(data=data)


def _seed_slots(model: SimpleNamespace, feedbacks: list[torch.Tensor]) -> None:
    for i, feedback in enumerate(feedbacks):
        model._feedback_slots[POOL_IDS[i]] = feedback


def _pool_indices(pool_ids: list[int]) -> torch.Tensor:
    return torch.tensor(pool_ids, dtype=torch.long)


def test_dense_write_skips_row_index_tensor(monkeypatch: Any) -> None:
    n, hidden = 3, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    _seed_slots(model, feedbacks)
    requests = [_req_wrap(_data(texts[i], pool_idx=POOL_IDS[i])) for i in range(n)]

    calls: list = []
    real_tensor = torch.tensor

    def _spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(args)
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(talker_model_runner.torch, "tensor", _spy)

    runner._write_feedback_buffers(requests)

    # Note (wenyao): the dense path may build only its pool-id gather tensor.
    assert len(calls) == 1
    assert list(calls[0][0]) == POOL_IDS[:n]
    assert torch.equal(model._feedback_mask, torch.ones(n, dtype=torch.bool))
    for i in range(n):
        assert torch.equal(model._feedback_buffer[i], feedbacks[i] + texts[i])
        assert requests[i].data.pending_feedback_count == 0


def test_dense_write_gathers_through_the_batch_pool_indices(monkeypatch: Any) -> None:
    n, hidden = 3, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    _seed_slots(model, feedbacks)
    requests = [_req_wrap(_data(texts[i], pool_idx=POOL_IDS[i])) for i in range(n)]

    pool_indices = _pool_indices(POOL_IDS[:n])
    calls: list = []
    real_tensor = torch.tensor
    monkeypatch.setattr(
        talker_model_runner.torch,
        "tensor",
        lambda *a, **kw: (calls.append(a), real_tensor(*a, **kw))[1],
    )

    runner._write_feedback_buffers(requests, pool_indices)

    assert calls == []
    for i in range(n):
        assert torch.equal(model._feedback_buffer[i], feedbacks[i] + texts[i])


def test_retract_override_ignores_the_batch_pool_indices() -> None:
    # Note (wenyao): a snapshotted request no longer owns the batch pool row.
    n, hidden = 2, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    model._feedback_slots[0] = torch.full((hidden,), 99.0)
    model._feedback_slots[POOL_IDS[1]] = torch.full((hidden,), 2.0)
    override_data = _data(torch.full((hidden,), 10.0), pool_idx=POOL_IDS[0])
    override_data.req.req_pool_idx = None
    override_data.retracted_feedback_embed = torch.full((hidden,), 1.0)
    requests = [
        _req_wrap(override_data),
        _req_wrap(_data(torch.full((hidden,), 20.0), pool_idx=POOL_IDS[1])),
    ]

    runner._write_feedback_buffers(requests, _pool_indices([0, 0]))

    assert torch.equal(model._feedback_buffer[0], torch.full((hidden,), 11.0))
    assert torch.equal(model._feedback_buffer[1], torch.full((hidden,), 22.0))


def test_sparse_write_ignores_the_batch_pool_indices() -> None:
    # Note (wenyao): sparse gather rows do not align with batch rows.
    n, hidden = 3, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    model._feedback_slots[0] = torch.full((hidden,), 99.0)
    model._feedback_slots[POOL_IDS[0]] = torch.full((hidden,), 1.0)
    model._feedback_slots[POOL_IDS[2]] = torch.full((hidden,), 3.0)
    requests = [
        _req_wrap(_data(torch.full((hidden,), 10.0), pool_idx=POOL_IDS[0])),
        _req_wrap(_data(None, pool_idx=POOL_IDS[1])),
        _req_wrap(_data(torch.full((hidden,), 30.0), pool_idx=POOL_IDS[2])),
    ]

    runner._write_feedback_buffers(requests, _pool_indices([0, 0, 0]))

    assert torch.equal(model._feedback_buffer[0], torch.full((hidden,), 11.0))
    assert torch.equal(model._feedback_buffer[2], torch.full((hidden,), 33.0))


def test_sparse_write_leaves_starved_row_unwritten() -> None:
    n, hidden = 3, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    _seed_slots(model, feedbacks)
    requests = [
        _req_wrap(_data(texts[0], pool_idx=POOL_IDS[0])),
        _req_wrap(_data(None, pool_idx=POOL_IDS[1])),
        _req_wrap(_data(texts[2], pool_idx=POOL_IDS[2])),
    ]

    runner._write_feedback_buffers(requests)

    assert model._feedback_mask.tolist() == [True, False, True]
    assert torch.equal(model._feedback_buffer[0], feedbacks[0] + texts[0])
    assert torch.equal(model._feedback_buffer[1], torch.zeros(hidden))
    assert torch.equal(model._feedback_buffer[2], feedbacks[2] + texts[2])
    assert requests[1].data.pending_feedback_count == 1


def test_write_skips_rows_without_pending_feedback() -> None:
    n, hidden = 2, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    _seed_slots(model, feedbacks)
    requests = [
        _req_wrap(_data(texts[0], pool_idx=POOL_IDS[0], count=0)),
        _req_wrap(_data(texts[1], pool_idx=POOL_IDS[1])),
    ]

    runner._write_feedback_buffers(requests)

    assert model._feedback_mask.tolist() == [False, True]
    assert torch.equal(model._feedback_buffer[0], torch.zeros(hidden))
    assert torch.equal(model._feedback_buffer[1], feedbacks[1] + texts[1])
    assert len(requests[0].data.decode_input_embeds) == 0


def test_write_reads_feedback_from_the_requests_own_slot() -> None:
    n, hidden = 2, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    # Note (wenyao): batch and pool order deliberately differ.
    model._feedback_slots[POOL_IDS[0]] = torch.full((hidden,), 1.0)
    model._feedback_slots[POOL_IDS[1]] = torch.full((hidden,), 2.0)
    requests = [
        _req_wrap(_data(torch.full((hidden,), 10.0), pool_idx=POOL_IDS[1])),
        _req_wrap(_data(torch.full((hidden,), 20.0), pool_idx=POOL_IDS[0])),
    ]

    runner._write_feedback_buffers(requests)

    assert torch.equal(model._feedback_buffer[0], torch.full((hidden,), 12.0))
    assert torch.equal(model._feedback_buffer[1], torch.full((hidden,), 21.0))


def test_write_prefers_retracted_snapshot_over_slot() -> None:
    n, hidden = 1, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    model._feedback_slots[POOL_IDS[0]] = torch.full((hidden,), 99.0)
    data = _data(torch.full((hidden,), 10.0), pool_idx=POOL_IDS[0])
    data.retracted_feedback_embed = torch.full((hidden,), 1.0)

    runner._write_feedback_buffers([_req_wrap(data)])

    assert torch.equal(model._feedback_buffer[0], torch.full((hidden,), 11.0))
    assert data.retracted_feedback_embed is None
    assert data.pending_feedback_count == 0


def test_write_records_decode_input_history() -> None:
    n, hidden = 1, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    model._feedback_slots[POOL_IDS[0]] = torch.full((hidden,), 1.0)
    data = _data(torch.full((hidden,), 20.0), pool_idx=POOL_IDS[0])

    runner._write_feedback_buffers([_req_wrap(data)])

    assert len(data.decode_input_embeds) == 1
    assert torch.equal(data.decode_input_embeds[0], torch.full((hidden,), 21.0))


def test_consumed_rows_survive_later_buffer_writes() -> None:
    n, hidden = 2, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    _seed_slots(model, feedbacks)
    requests = [_req_wrap(_data(texts[i], pool_idx=POOL_IDS[i])) for i in range(n)]

    runner._write_feedback_buffers(requests)

    expected = [feedbacks[i] + texts[i] for i in range(n)]
    model._feedback_buffer.copy_(model._feedback_buffer + 999.0)
    model._feedback_slots.copy_(model._feedback_slots + 999.0)

    for i in range(n):
        assert torch.equal(requests[i].data.decode_input_embeds[0], expected[i])


def test_pending_feedback_without_pool_slot_raises() -> None:
    n, hidden = 1, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    data = _data(torch.full((hidden,), 10.0), pool_idx=POOL_IDS[0])
    data.req.req_pool_idx = None

    with pytest.raises(RuntimeError, match="no pool slot"):
        runner._write_feedback_buffers([_req_wrap(data)])


def test_readiness_requires_pending_feedback_count() -> None:
    hidden = 3
    pad = torch.full((hidden,), 7.0)
    text = torch.full((hidden,), 20.0)

    no_feedback = _data(text, pool_idx=POOL_IDS[0], count=0)
    with_text = _data(text, pool_idx=POOL_IDS[0])
    with_pad = _data(None, pool_idx=POOL_IDS[0], thinker_done=True, pad=pad)
    no_text = _data(None, pool_idx=POOL_IDS[0])

    assert not QwenTalkerModelRunner._data_has_next_decode_input(no_feedback)
    assert QwenTalkerModelRunner._data_has_next_decode_input(with_text)
    assert QwenTalkerModelRunner._data_has_next_decode_input(with_pad)
    assert not QwenTalkerModelRunner._data_has_next_decode_input(no_text)
