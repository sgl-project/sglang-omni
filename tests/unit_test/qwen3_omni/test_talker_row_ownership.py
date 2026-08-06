# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import ModelRunnerOutput


def _fake_model(n: int, hidden: int, code_groups: int) -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_buffer=torch.zeros(n, hidden, dtype=torch.float32),
        _feedback_mask=torch.zeros(n, dtype=torch.bool),
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


def _data(
    feedback: torch.Tensor | None,
    text: torch.Tensor | None,
    *,
    thinker_done: bool = False,
    pad: torch.Tensor | None = None,
    stage_payload: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_queue=deque([feedback]) if feedback is not None else deque(),
        pending_text_queue=deque([text]) if text is not None else deque(),
        thinker_chunks_done=thinker_done,
        tts_pad_embed=pad,
        stage_payload=stage_payload,
    )


def _req_wrap(data: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(data=data)


def _sched_batch(n: int) -> SimpleNamespace:
    return SimpleNamespace(reqs=[SimpleNamespace(rid=f"r{i}") for i in range(n)])


def test_row_ownership_survives_prep_then_emit() -> None:
    n, hidden, code_groups = 3, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    requests = [_req_wrap(_data(feedbacks[i], texts[i])) for i in range(n)]
    schedule_batch = _sched_batch(n)

    runner._write_feedback_buffers(requests)

    assert torch.equal(model._feedback_mask, torch.ones(n, dtype=torch.bool))
    for i in range(n):
        assert torch.equal(model._feedback_buffer[i], feedbacks[i] + texts[i])

    runner._emit_code_chunks_and_feedback(
        schedule_batch=schedule_batch, requests=requests
    )

    sent = runner._outbox.sent
    assert [m.request_id for m in sent] == [f"r{i}" for i in range(n)]
    for i, msg in enumerate(sent):
        assert msg.target == "code2wav"
        assert msg.metadata == {"stream": False}
        assert torch.equal(msg.data, model._output_codes[i])
        fb_queue = requests[i].data.pending_feedback_queue
        assert len(fb_queue) == 1
        assert torch.equal(fb_queue[0], model._output_embeds[i])


def test_sparse_feedback_row_stays_unwritten() -> None:
    n, hidden, code_groups = 3, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    requests = [
        _req_wrap(_data(feedbacks[0], texts[0])),
        _req_wrap(_data(feedbacks[1], None, thinker_done=False)),
        _req_wrap(_data(feedbacks[2], texts[2])),
    ]

    runner._write_feedback_buffers(requests)

    assert model._feedback_mask.tolist() == [True, False, True]
    assert torch.equal(model._feedback_buffer[1], torch.zeros(hidden))
    assert torch.equal(model._feedback_buffer[0], feedbacks[0] + texts[0])
    assert torch.equal(model._feedback_buffer[2], feedbacks[2] + texts[2])


def test_stale_mask_cannot_leak_into_reused_slot() -> None:
    # Note (wenyao): forward-side mask reset (talker.py:422) needs a real forward; integration-level only
    n, hidden, code_groups = 2, 3, 2
    model = _fake_model(n, hidden, code_groups)
    model._feedback_mask[:n] = True
    runner = _runner(model)

    feedback1 = torch.full((hidden,), 5.0)
    text1 = torch.full((hidden,), 50.0)
    requests = [
        _req_wrap(_data(torch.full((hidden,), 1.0), None, thinker_done=False)),
        _req_wrap(_data(feedback1, text1)),
    ]

    runner._write_feedback_buffers(requests)

    assert model._feedback_mask.tolist() == [False, True]
    assert torch.equal(model._feedback_buffer[0], torch.zeros(hidden))
    assert torch.equal(model._feedback_buffer[1], feedback1 + text1)


def test_row_ownership_tracks_current_batch_order_across_steps() -> None:
    n, hidden, code_groups = 2, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    request_data = {
        "r0": _data(
            torch.full((hidden,), 1.0),
            torch.full((hidden,), 10.0),
        ),
        "r1": _data(
            torch.full((hidden,), 2.0),
            torch.full((hidden,), 20.0),
        ),
    }
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
        schedule_batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid=rid) for rid in order],
            output_ids=None,
        )
        expected_inputs = [
            previous_feedback[rid] + torch.full((hidden,), text_by_request[rid].pop(0))
            for rid in order
        ]

        runner._write_feedback_buffers(requests)

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
        runner._emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
        )

        emitted = runner._outbox.sent[-len(order) :]
        assert [message.request_id for message in emitted] == list(order)
        for row, rid in enumerate(order):
            assert torch.equal(emitted[row].data, codes[row])
            assert torch.equal(request_data[rid].pending_feedback_queue[0], embeds[row])
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
            pending_feedback = request_data[rid].pending_feedback_queue
            assert len(pending_feedback) == 1
            assert torch.equal(pending_feedback[0], expected_feedback)

        model_runner_output = ModelRunnerOutput(
            outputs={},
            can_run_cuda_graph=False,
            host_token_ids=runner._resolve_host_token_ids(result),
        )
        batch_result = OmniScheduler._make_batch_result(model_runner_output)
        assert batch_result.next_token_ids is model_runner_output.host_token_ids
        assert batch_result.next_token_ids.tolist() == tokens.tolist()


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
