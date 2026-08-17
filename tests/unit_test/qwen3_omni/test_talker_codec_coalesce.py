# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel, make_qwen_payload


def _fake_model(
    n: int, hidden: int, code_groups: int, step: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        _output_codes=torch.stack(
            [
                torch.tensor([i * 1000 + step, i + 100 + step], dtype=torch.long)
                for i in range(n)
            ]
        )[:, :code_groups],
        _output_embeds=torch.stack(
            [torch.full((hidden,), float(i * 7 + 1 + step)) for i in range(n)]
        ),
    )


def _runner(
    model: SimpleNamespace, coalesce: int, first_frames: int = 0
) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner._feedback_enabled = True
    runner._code2wav_target = "code2wav"
    runner._codec_coalesce_frames = coalesce
    runner._codec_coalesce_first_frames = first_frames
    runner._outbox = SimpleNamespace(sent=[])
    runner._outbox.put = runner._outbox.sent.append
    return runner


def _data() -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_queue=deque(),
        pending_codec_rows=[],
        codec_first_flush_done=False,
        stage_payload=None,
        finish_reason=None,
    )


def _requests(n: int) -> list:
    return [SimpleNamespace(data=_data()) for _ in range(n)]


def _sched_batch(n: int) -> SimpleNamespace:
    return SimpleNamespace(reqs=[SimpleNamespace(rid=f"r{i}") for i in range(n)])


def _run_steps(runner, requests, batch, steps: int) -> list[torch.Tensor]:
    seen = []
    for step in range(steps):
        runner.model._output_codes += 1
        runner.model._output_embeds += 1.0
        seen.append(runner.model._output_codes[0].clone())
        runner._emit_code_chunks_and_feedback(schedule_batch=batch, requests=requests)
    return seen


def test_coalesce_disabled_emits_one_message_per_frame() -> None:
    n = 3
    runner = _runner(_fake_model(n, 4, 2), coalesce=0)
    requests, batch = _requests(n), _sched_batch(n)
    _run_steps(runner, requests, batch, steps=2)
    assert len(runner._outbox.sent) == 2 * n
    assert all(m.data.ndim == 1 for m in runner._outbox.sent)
    assert all(not r.data.pending_codec_rows for r in requests)


def test_coalesce_buffers_until_threshold_then_emits_stacked_rows() -> None:
    n, k = 2, 3
    runner = _runner(_fake_model(n, 4, 2), coalesce=k)
    requests, batch = _requests(n), _sched_batch(n)

    seen = _run_steps(runner, requests, batch, steps=k)
    assert runner._outbox.sent == []
    assert all(len(r.data.pending_codec_rows) == k for r in requests)

    seen += _run_steps(runner, requests, batch, steps=1)
    assert len(runner._outbox.sent) == n
    assert all(len(r.data.pending_codec_rows) == 1 for r in requests)
    msg = next(m for m in runner._outbox.sent if m.request_id == "r0")
    assert msg.type == "stream"
    assert msg.target == "code2wav"
    assert msg.metadata == {"stream": False}
    assert msg.data.shape == (k, 2)
    assert torch.equal(msg.data, torch.stack(seen[:k], dim=0))


def test_coalesced_rows_survive_next_step_inplace_write() -> None:
    n, k = 2, 2
    runner = _runner(_fake_model(n, 4, 2), coalesce=k)
    requests, batch = _requests(n), _sched_batch(n)
    runner._emit_code_chunks_and_feedback(schedule_batch=batch, requests=requests)
    buffered = requests[0].data.pending_codec_rows[0].clone()
    runner.model._output_codes.copy_(runner.model._output_codes + 999)
    assert torch.equal(requests[0].data.pending_codec_rows[0], buffered)


def test_on_request_finished_flushes_partial_tail() -> None:
    n, k = 1, 5
    runner = _runner(_fake_model(n, 4, 2), coalesce=k)
    requests, batch = _requests(n), _sched_batch(n)
    _run_steps(runner, requests, batch, steps=2)
    assert runner._outbox.sent == []

    runner.on_request_finished("r0", requests[0].data)
    assert len(runner._outbox.sent) == 1
    assert runner._outbox.sent[0].data.shape == (2, 2)
    assert not requests[0].data.pending_codec_rows

    runner.on_request_finished("r0", requests[0].data)
    assert len(runner._outbox.sent) == 1


def test_single_row_flush_keeps_legacy_1d_shape() -> None:
    n, k = 1, 5
    runner = _runner(_fake_model(n, 4, 2), coalesce=k)
    requests, batch = _requests(n), _sched_batch(n)
    _run_steps(runner, requests, batch, steps=1)
    runner.on_request_finished("r0", requests[0].data)
    assert runner._outbox.sent[0].data.ndim == 1


def test_feedback_queue_fills_regardless_of_coalescing() -> None:
    n, k = 2, 4
    runner = _runner(_fake_model(n, 4, 2), coalesce=k)
    requests, batch = _requests(n), _sched_batch(n)
    _run_steps(runner, requests, batch, steps=2)
    assert all(len(r.data.pending_feedback_queue) == 2 for r in requests)


def _make_scheduler(model: FakeCode2WavModel) -> Code2WavScheduler:
    return Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )


def test_ingest_unbinds_coalesced_chunk_and_decodes() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = _make_scheduler(model)
    scheduler._stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    scheduler._handle_stream_chunk(
        "req-1",
        StreamItem(
            0,
            torch.tensor([[1, 10], [2, 20]]),
            "talker",
            metadata={"stream": False},
        ),
    )
    assert model.calls == [(1, 2, 2)]
    state = scheduler._stream_states["req-1"]
    assert all(chunk.ndim == 1 for chunk in state.chunks)


class _SyncGuardTensor(torch.Tensor):
    def item(self):
        raise AssertionError("D2H .item() on coalesced chunk")

    def tolist(self):
        raise AssertionError("D2H .tolist() on coalesced chunk")


def test_ingest_2d_chunk_does_not_sync() -> None:
    scheduler = _make_scheduler(FakeCode2WavModel(total_upsample=2))
    state = scheduler.create_stream_state("req-1")
    codes = torch.Tensor._make_subclass(
        _SyncGuardTensor, torch.tensor([[1, 10], [2, 20]])
    )
    scheduler.ingest("req-1", state, codes)
    assert len(state.chunks) == 2


def test_ingest_1d_row_still_drops_eos() -> None:
    scheduler = _make_scheduler(FakeCode2WavModel(total_upsample=2))
    state = scheduler.create_stream_state("req-1")
    scheduler.ingest("req-1", state, torch.tensor([scheduler._codec_eos_token_id, 0]))
    assert state.chunks == []


def test_stop_finish_pops_trailing_eos_row_before_tail_flush() -> None:
    n, k = 1, 5
    runner = _runner(_fake_model(n, 4, 2), coalesce=k)
    requests, batch = _requests(n), _sched_batch(n)
    seen = _run_steps(runner, requests, batch, steps=3)
    requests[0].data.finish_reason = "stop"
    runner.on_request_finished("r0", requests[0].data)
    assert len(runner._outbox.sent) == 1
    msg = runner._outbox.sent[0]
    assert msg.data.shape == (2, 2)
    assert torch.equal(msg.data, torch.stack(seen[:2], dim=0))
    assert not requests[0].data.pending_codec_rows


def test_stop_finish_at_threshold_never_leaks_eos_into_chunk() -> None:
    n, k = 1, 3
    runner = _runner(_fake_model(n, 4, 2), coalesce=k)
    requests, batch = _requests(n), _sched_batch(n)
    seen = _run_steps(runner, requests, batch, steps=k + 1)
    requests[0].data.finish_reason = "stop"
    runner.on_request_finished("r0", requests[0].data)
    assert len(runner._outbox.sent) == 1
    assert torch.equal(runner._outbox.sent[0].data, torch.stack(seen[:k], dim=0))
    assert not requests[0].data.pending_codec_rows


def test_length_finish_flushes_tail_unpopped() -> None:
    n, k = 1, 5
    runner = _runner(_fake_model(n, 4, 2), coalesce=k)
    requests, batch = _requests(n), _sched_batch(n)
    seen = _run_steps(runner, requests, batch, steps=3)
    requests[0].data.finish_reason = "length"
    runner.on_request_finished("r0", requests[0].data)
    assert torch.equal(runner._outbox.sent[0].data, torch.stack(seen, dim=0))


def test_first_flush_uses_smaller_threshold_then_steady_state() -> None:
    n, k, first = 2, 5, 2
    runner = _runner(_fake_model(n, 4, 2), coalesce=k, first_frames=first)
    requests, batch = _requests(n), _sched_batch(n)

    _run_steps(runner, requests, batch, steps=first + 1)
    assert len(runner._outbox.sent) == n
    assert all(m.data.shape[0] == first for m in runner._outbox.sent)
    assert all(r.data.codec_first_flush_done for r in requests)

    _run_steps(runner, requests, batch, steps=k - 1)
    assert len(runner._outbox.sent) == n

    _run_steps(runner, requests, batch, steps=1)
    assert len(runner._outbox.sent) == 2 * n
    assert all(m.data.shape[0] == k for m in runner._outbox.sent[n:])


def test_first_frames_of_one_emits_legacy_row_then_stacked() -> None:
    n, k = 1, 3
    runner = _runner(_fake_model(n, 4, 2), coalesce=k, first_frames=1)
    requests, batch = _requests(n), _sched_batch(n)

    _run_steps(runner, requests, batch, steps=2)
    assert len(runner._outbox.sent) == 1
    assert runner._outbox.sent[0].data.ndim == 1

    _run_steps(runner, requests, batch, steps=k)
    assert len(runner._outbox.sent) == 2
    assert runner._outbox.sent[1].data.shape[0] == k


def test_first_frames_zero_keeps_uniform_threshold() -> None:
    n, k = 1, 3
    runner = _runner(_fake_model(n, 4, 2), coalesce=k, first_frames=0)
    requests, batch = _requests(n), _sched_batch(n)
    _run_steps(runner, requests, batch, steps=k)
    assert runner._outbox.sent == []
    _run_steps(runner, requests, batch, steps=1)
    assert len(runner._outbox.sent) == 1
    assert runner._outbox.sent[0].data.shape[0] == k
