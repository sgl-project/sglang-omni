# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
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
    runner._codec_coalesce_early_frames = 0
    runner._codec_coalesce_first_frames = first_frames
    runner._outbox = SimpleNamespace(sent=[])
    runner._outbox.put = runner._outbox.sent.append
    return runner


def _data() -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_queue=deque(),
        pending_codec_rows=[],
        codec_first_flush_done=False,
        codec_frames_seen=0,
        stage_payload=None,
        finish_reason=None,
    )


def _requests(n: int) -> list:
    return [SimpleNamespace(data=_data()) for _ in range(n)]


def _sched_batch(n: int) -> SimpleNamespace:
    return SimpleNamespace(reqs=[SimpleNamespace(rid=f"r{i}") for i in range(n)])


def _run_steps(runner, requests, batch, steps: int) -> list[torch.Tensor]:
    seen = []
    for _ in range(steps):
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


def test_early_frames_preserve_code2wav_window_cadence() -> None:
    runner = _runner(_fake_model(1, 4, 2), coalesce=10)
    runner._codec_coalesce_early_frames = 12
    requests, batch = _requests(1), _sched_batch(1)
    ready_at = {}
    received_frames = 0
    for step in range(1, 42):
        sent_before = len(runner._outbox.sent)
        _run_steps(runner, requests, batch, 1)
        for message in runner._outbox.sent[sent_before:]:
            received_frames += 1 if message.data.ndim == 1 else len(message.data)
        for window_end in (2, 12, 22, 32):
            if received_frames >= window_end:
                ready_at.setdefault(window_end, step)

    # Note (wenyao): Code2Wav consumes two frames, then ten per window; only
    # the newest row stays pending until the next step can rule out EOS.
    assert ready_at == {2: 2, 12: 12, 22: 23, 32: 33}


@pytest.mark.parametrize("steps", [13, 20, 21, 22, 23, 30, 31, 32, 33])
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_default_early_frames_preserve_order_and_final_tail(steps, finish_reason):
    """The early single-frame prefix and coalesced tail emit each non-EOS row once."""
    runner = _runner(_fake_model(1, 4, 2), coalesce=10)
    runner._codec_coalesce_early_frames = 12
    requests, batch = _requests(1), _sched_batch(1)
    expected = _run_steps(runner, requests, batch, steps)
    data = requests[0].data
    data.finish_reason = finish_reason
    runner.on_request_finished("r0", data)

    messages = runner._outbox.sent
    assert all(message.data.ndim == 1 for message in messages[:12])
    actual = torch.cat(
        [
            message.data.unsqueeze(0) if message.data.ndim == 1 else message.data
            for message in messages
        ]
    )
    assert torch.equal(actual, torch.stack(expected))
    assert not data.pending_codec_rows
    assert len(data.pending_feedback_queue) == steps


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
    assert len(runner._outbox.sent) == 2
    assert all(msg.data.shape == (2,) for msg in runner._outbox.sent)
    assert not requests[0].data.pending_codec_rows

    runner.on_request_finished("r0", requests[0].data)
    assert len(runner._outbox.sent) == 2


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


def _make_scheduler(
    model: FakeCode2WavModel, *, enable_output_overlap: bool = True
) -> Code2WavScheduler:
    return Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_output_overlap=enable_output_overlap,
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
    assert scheduler.should_decode(state, is_final=False)


def test_ingest_1d_row_eager_path_drops_eos_immediately() -> None:
    scheduler = _make_scheduler(
        FakeCode2WavModel(total_upsample=2), enable_output_overlap=False
    )
    state = scheduler.create_stream_state("req-1")
    scheduler.ingest("req-1", state, torch.tensor([scheduler._codec_eos_token_id, 0]))
    assert state.chunks == []


def test_ingest_1d_row_lazy_path_drops_eos_at_final_scan() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = _make_scheduler(model, enable_output_overlap=True)
    state = scheduler.create_stream_state("req-1")
    scheduler.ingest("req-1", state, torch.tensor([scheduler._codec_eos_token_id, 0]))

    assert len(state.chunks) == 1
    assert state.checked == 0
    assert scheduler.decode_delta("req-1", state, is_final=True) is None
    assert state.chunks == []
    assert model.calls == []


@pytest.mark.parametrize("finish_reason", [None, "stop", "length"])
@pytest.mark.parametrize("steps", [1, 3, 4])
def test_finish_sends_uncertain_last_row_separately(finish_reason, steps) -> None:
    runner = _runner(_fake_model(1, 4, 2), coalesce=3)
    requests, batch = _requests(1), _sched_batch(1)
    seen = _run_steps(runner, requests, batch, steps=steps)
    requests[0].data.finish_reason = finish_reason
    runner.on_request_finished("r0", requests[0].data)

    messages = runner._outbox.sent
    assert messages[-1].data.ndim == 1
    assert torch.equal(messages[-1].data, seen[-1])
    if steps > 1:
        assert torch.equal(messages[0].data, torch.stack(seen[:-1]))
    assert not requests[0].data.pending_codec_rows
    runner.on_request_finished("r0", requests[0].data)
    assert len(messages) == (1 if steps == 1 else 2)


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("last_is_eos", [False, True])
@pytest.mark.parametrize("enable_output_overlap", [False, True])
@pytest.mark.parametrize("steps", [1, 3, 4])
def test_finish_tail_is_filtered_by_vocoder(
    finish_reason, last_is_eos, enable_output_overlap, steps
) -> None:
    runner = _runner(_fake_model(1, 4, 2), coalesce=3)
    requests, batch = _requests(1), _sched_batch(1)
    seen = _run_steps(runner, requests, batch, steps=steps)
    scheduler = _make_scheduler(
        FakeCode2WavModel(total_upsample=2),
        enable_output_overlap=enable_output_overlap,
    )
    data = requests[0].data
    if last_is_eos:
        data.pending_codec_rows[-1][0] = scheduler._codec_eos_token_id
    data.finish_reason = finish_reason
    # The sender must never inspect tensor values to decide whether this is EOS.
    data.pending_codec_rows[:] = [
        torch.Tensor._make_subclass(_SyncGuardTensor, row)
        for row in data.pending_codec_rows
    ]
    runner.on_request_finished("r0", data)

    state = scheduler.create_stream_state("r0")
    for message in runner._outbox.sent:
        scheduler.ingest("r0", state, message.data.as_subclass(torch.Tensor))
    scheduler.decode_delta("r0", state, is_final=True)
    expected = seen[:-1] if last_is_eos else seen
    assert len(state.chunks) == len(expected)
    assert all(
        torch.equal(actual, wanted) for actual, wanted in zip(state.chunks, expected)
    )


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
