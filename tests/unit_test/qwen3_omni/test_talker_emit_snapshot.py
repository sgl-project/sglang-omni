# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner

POOL_SIZE = 8
POOL_IDS = [5, 2, 7, 0, 3]


def _fake_model(n: int, hidden: int, code_groups: int) -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_slots=torch.zeros(POOL_SIZE, hidden),
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


def _data() -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_count=0,
        stage_payload=None,
    )


def _requests(n: int) -> list:
    return [SimpleNamespace(data=_data()) for _ in range(n)]


def _sched_batch(n: int) -> SimpleNamespace:
    return SimpleNamespace(
        reqs=[SimpleNamespace(rid=f"r{i}", req_pool_idx=POOL_IDS[i]) for i in range(n)],
        req_pool_indices=torch.tensor(POOL_IDS[:n], dtype=torch.long),
    )


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


def test_emitted_rows_survive_next_step_inplace_write() -> None:
    n, hidden, code_groups = 4, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    codes_before = model._output_codes.clone()
    embeds_before = model._output_embeds.clone()

    requests = _requests(n)
    _emit_frame(runner, _sched_batch(n), requests)

    model._output_codes.copy_(model._output_codes + 999)
    model._output_embeds.copy_(model._output_embeds + 999.0)

    for i, msg in enumerate(runner._outbox.sent):
        assert torch.equal(msg.data, codes_before[i])
        assert torch.equal(model._feedback_slots[POOL_IDS[i]], embeds_before[i])


def test_emit_writes_feedback_to_pool_indexed_slots() -> None:
    n, hidden, code_groups = 3, 4, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    requests = _requests(n)
    _emit_frame(runner, _sched_batch(n), requests)

    for i in range(n):
        assert torch.equal(model._feedback_slots[POOL_IDS[i]], model._output_embeds[i])
        assert requests[i].data.pending_feedback_count == 1
        assert not hasattr(requests[i].data, "pending_feedback_queue")

    for row in set(range(POOL_SIZE)) - set(POOL_IDS[:n]):
        assert torch.equal(model._feedback_slots[row], torch.zeros(hidden))


def test_emit_keeps_one_batched_clone_for_codes() -> None:
    n, hidden, code_groups = 5, 4, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    clones: list = []
    orig_clone = torch.Tensor.clone

    def _counting_clone(self, *args, **kwargs):
        out = orig_clone(self, *args, **kwargs)
        clones.append(out)
        return out

    requests = _requests(n)
    torch.Tensor.clone = _counting_clone
    try:
        _emit_frame(runner, _sched_batch(n), requests)
    finally:
        torch.Tensor.clone = orig_clone

    assert len(clones) == 1

    code_ptrs = {msg.data.untyped_storage().data_ptr() for msg in runner._outbox.sent}
    assert len(code_ptrs) == 1


def test_emit_counts_accumulate_across_steps() -> None:
    n, hidden, code_groups = 2, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    requests = _requests(n)
    schedule_batch = _sched_batch(n)

    _emit_frame(runner, schedule_batch, requests)
    model._output_embeds.copy_(model._output_embeds + 1.0)
    _emit_frame(runner, schedule_batch, requests)

    for i in range(n):
        assert requests[i].data.pending_feedback_count == 2
        assert torch.equal(model._feedback_slots[POOL_IDS[i]], model._output_embeds[i])
