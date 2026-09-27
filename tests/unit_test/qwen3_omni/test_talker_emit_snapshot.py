# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.scheduling.message import OutgoingMessage


def fake_model(n: int, hidden: int, code_groups: int) -> SimpleNamespace:
    return SimpleNamespace(
        output_codes=torch.stack(
            [torch.tensor([i, i + 100], dtype=torch.long) for i in range(n)]
        )[:, :code_groups],
        output_embeds=torch.stack(
            [torch.full((hidden,), float(i * 7 + 1)) for i in range(n)]
        ),
    )


def make_runner(model: SimpleNamespace) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner.feedback_enabled = True
    runner.code2wav_target = "code2wav"
    runner.code2wav_in_process = True
    runner.codec_coalesce_frames = 0
    runner.outbox = SimpleNamespace(sent=[])
    runner.outbox.put = runner.outbox.sent.append
    return runner


def data() -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_queue=deque(),
        stage_payload=None,
    )


def make_requests(n: int) -> list:
    return [SimpleNamespace(data=data()) for _ in range(n)]


def sched_batch(n: int) -> SimpleNamespace:
    return SimpleNamespace(reqs=[SimpleNamespace(rid=f"r{i}") for i in range(n)])


def test_emitted_rows_survive_next_step_inplace_write() -> None:
    n, hidden, code_groups = 4, 3, 2
    model = fake_model(n, hidden, code_groups)
    runner = make_runner(model)

    codes_before = model.output_codes.clone()
    embeds_before = model.output_embeds.clone()

    requests = make_requests(n)
    runner.emit_code_chunks_and_feedback(
        schedule_batch=sched_batch(n), requests=requests
    )

    model.output_codes.copy_(model.output_codes + 999)
    model.output_embeds.copy_(model.output_embeds + 999.0)

    for i, msg in enumerate(runner.outbox.sent):
        assert torch.equal(msg.data, codes_before[i])
        fb_queue = requests[i].data.pending_feedback_queue
        assert torch.equal(fb_queue[0], embeds_before[i])


def test_two_batched_clones_rows_share_storage() -> None:
    n, hidden, code_groups = 5, 4, 2
    model = fake_model(n, hidden, code_groups)
    runner = make_runner(model)

    clones: list = []
    orig_clone = torch.Tensor.clone

    def counting_clone(self, *args, **kwargs):
        out = orig_clone(self, *args, **kwargs)
        clones.append(out)
        return out

    requests = make_requests(n)
    torch.Tensor.clone = counting_clone
    try:
        runner.emit_code_chunks_and_feedback(
            schedule_batch=sched_batch(n), requests=requests
        )
    finally:
        torch.Tensor.clone = orig_clone

    assert len(clones) == 2

    code_ptrs = {msg.data.untyped_storage().data_ptr() for msg in runner.outbox.sent}
    embed_ptrs = {
        req.data.pending_feedback_queue[0].untyped_storage().data_ptr()
        for req in requests
    }
    assert len(code_ptrs) == 1
    assert len(embed_ptrs) == 1


class DeviceCodesTensor(torch.Tensor):
    """CPU tensor that reports a CUDA device to the sender."""

    @property
    def device(self) -> torch.device:
        return torch.device("cuda")


def test_every_code_message_carries_one_event_recorded_after_the_snapshot(
    monkeypatch,
) -> None:
    log: list[object] = []
    talker_stream = object()

    class _RecordingEvent:
        def record(self, stream: object) -> None:
            log.append(("record", stream))

    original_clone = torch.Tensor.clone

    def _logging_clone(self, *args, **kwargs):
        log.append("clone")
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "Event", _RecordingEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: talker_stream)
    monkeypatch.setattr(torch.Tensor, "clone", _logging_clone)
    n = 3
    model = fake_model(n, 3, 2)
    model.output_codes = torch.Tensor._make_subclass(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        DeviceCodesTensor, model.output_codes
    )
    runner = make_runner(model)

    def put_after_record(message: OutgoingMessage) -> None:
        assert log[-1] == ("record", talker_stream)
        assert message.metadata["codes_ready_event"] is not None
        runner.outbox.sent.append(message)

    runner.outbox.put = put_after_record

    runner.emit_code_chunks_and_feedback(
        schedule_batch=sched_batch(n), requests=make_requests(n)
    )

    assert len(runner.outbox.sent) == n
    events = [msg.metadata["codes_ready_event"] for msg in runner.outbox.sent]
    assert all(event is events[0] for event in events)
    assert log.count(("record", talker_stream)) == 1
    assert log[-1] == ("record", talker_stream)
    assert "clone" in log


def test_cpu_code_messages_carry_no_event() -> None:
    n = 2
    runner = make_runner(fake_model(n, 3, 2))

    runner.emit_code_chunks_and_feedback(
        schedule_batch=sched_batch(n), requests=make_requests(n)
    )

    assert len(runner.outbox.sent) == n
    assert all(msg.metadata == {"stream": False} for msg in runner.outbox.sent)


def test_code_messages_for_another_process_carry_no_event(monkeypatch) -> None:
    def record_forbidden() -> None:
        raise AssertionError("no event may be recorded for another process")

    monkeypatch.setattr(torch.cuda, "Event", record_forbidden)
    n = 2
    model = fake_model(n, 3, 2)
    model.output_codes = torch.Tensor._make_subclass(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        DeviceCodesTensor, model.output_codes
    )
    runner = make_runner(model)
    runner.code2wav_in_process = False

    runner.emit_code_chunks_and_feedback(
        schedule_batch=sched_batch(n), requests=make_requests(n)
    )

    assert len(runner.outbox.sent) == n
    assert all(msg.metadata == {"stream": False} for msg in runner.outbox.sent)
