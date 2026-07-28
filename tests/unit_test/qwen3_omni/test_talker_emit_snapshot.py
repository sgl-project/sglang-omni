# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner


def _fake_model(n: int, hidden: int, code_groups: int) -> SimpleNamespace:
    return SimpleNamespace(
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
        pending_feedback_queue=deque(),
        stage_payload=None,
    )


def _requests(n: int) -> list:
    return [SimpleNamespace(data=_data()) for _ in range(n)]


def _sched_batch(n: int) -> SimpleNamespace:
    return SimpleNamespace(reqs=[SimpleNamespace(rid=f"r{i}") for i in range(n)])


def test_emitted_rows_survive_next_step_inplace_write() -> None:
    n, hidden, code_groups = 4, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    codes_before = model._output_codes.clone()
    embeds_before = model._output_embeds.clone()

    requests = _requests(n)
    runner._emit_code_chunks_and_feedback(
        schedule_batch=_sched_batch(n), requests=requests
    )

    model._output_codes.copy_(model._output_codes + 999)
    model._output_embeds.copy_(model._output_embeds + 999.0)

    for i, msg in enumerate(runner._outbox.sent):
        assert torch.equal(msg.data, codes_before[i])
        fb_queue = requests[i].data.pending_feedback_queue
        assert torch.equal(fb_queue[0], embeds_before[i])


def test_two_batched_clones_rows_share_storage() -> None:
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
        runner._emit_code_chunks_and_feedback(
            schedule_batch=_sched_batch(n), requests=requests
        )
    finally:
        torch.Tensor.clone = orig_clone

    assert len(clones) == 2

    code_ptrs = {msg.data.untyped_storage().data_ptr() for msg in runner._outbox.sent}
    embed_ptrs = {
        req.data.pending_feedback_queue[0].untyped_storage().data_ptr()
        for req in requests
    }
    assert len(code_ptrs) == 1
    assert len(embed_ptrs) == 1
