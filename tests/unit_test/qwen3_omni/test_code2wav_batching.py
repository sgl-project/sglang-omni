# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavRunResult,
    GraphKey,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
    Code2WavStreamState,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.fixtures.qwen_fakes import (
    FakeCode2WavModel,
    deliver_code2wav_chunk,
    make_qwen_payload,
)

CHUNK_FRAMES = 2
CONTEXT_FRAMES = 1


class PublishedSizesRunner:
    """Replays through the model and reports a graph only for the row counts it publishes."""

    def __init__(self, model: FakeCode2WavModel, sizes: tuple[int, ...]) -> None:
        self.model = model
        self.sizes = sizes
        self.rows: list[int] = []
        self.modes: list[str] = []

    def available_batch_sizes(self, frames: int) -> tuple[int, ...]:
        del frames
        return self.sizes

    def run(self, codes: torch.Tensor, *, eligible: bool = True) -> Code2WavRunResult:
        key = GraphKey(batch_size=int(codes.shape[0]), frames=int(codes.shape[2]))
        mode = "cuda_graph" if eligible and key.batch_size in self.sizes else "eager"
        self.rows.append(key.batch_size)
        self.modes.append(mode)
        return Code2WavRunResult(self.model(codes), mode, key, None)


def make_scheduler(
    model: FakeCode2WavModel | None = None,
    *,
    runner: PublishedSizesRunner | None = None,
    left_context_size: int = CONTEXT_FRAMES,
    **kwargs,
) -> Code2WavScheduler:
    return Code2WavScheduler(
        model or FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=CHUNK_FRAMES,
        left_context_size=left_context_size,
        sample_rate=24000,
        enable_output_overlap=False,
        enable_cuda_graph=runner is not None,
        cuda_graph_runner=runner,
        **kwargs,
    )


def frames(*codes: int) -> torch.Tensor:
    return torch.tensor([[code, code * 10] for code in codes])


def item(codes: torch.Tensor) -> StreamItem:
    return StreamItem(0, codes, "talker", metadata={"stream": True})


def open_request(scheduler: Code2WavScheduler, request_id: str) -> None:
    scheduler.stream_payloads[request_id] = make_qwen_payload(request_id=request_id)


def queue_chunk(
    scheduler: Code2WavScheduler, request_id: str, codes: torch.Tensor
) -> None:
    """A chunk taken off the inbox while more remain queued: ingested, not decoded."""
    scheduler.handle_stream_chunk(request_id, item(codes))


def run_steps(scheduler: Code2WavScheduler) -> None:
    while scheduler.has_ready_work():
        scheduler.run_ready_step()


def stream_audio(scheduler: Code2WavScheduler) -> dict[str, list[list[float]]]:
    audio: dict[str, list[list[float]]] = {}
    while not scheduler.outbox.empty():
        message = scheduler.outbox.get_nowait()
        if message.type == "stream":
            waveform = np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
            audio.setdefault(message.request_id, []).append(waveform.tolist())
        else:
            pass
    return audio


def test_the_serving_loop_takes_every_queued_chunk_before_one_replay() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = make_scheduler(model)
    for index, request_id in enumerate(("a", "b", "c")):
        open_request(scheduler, request_id)
        scheduler.inbox.put(
            IncomingMessage(
                request_id=request_id,
                type="stream_chunk",
                data=item(frames(index + 1, index + 2)),
            )
        )
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        sent: set[str] = set()
        while len(sent) < 3 and time.monotonic() < deadline:
            try:
                sent.add(scheduler.outbox.get(timeout=0.1).request_id)
            except queue.Empty:
                pass
    finally:
        scheduler.stop()
        thread.join(timeout=2)
    assert sent == {"a", "b", "c"}
    # the three first windows were queued together, so they are one replay of three rows
    assert model.calls == [(3, 2, CHUNK_FRAMES)]


def test_every_row_of_a_replay_is_its_window_decoded_alone() -> None:
    codes = {
        "a": (1, 2, 3, 4, 5, 6),
        "b": (7, 8, 9, 10, 11, 12),
        "c": (2, 4, 6, 8, 1, 3),
    }

    alone_model = FakeCode2WavModel(total_upsample=2)
    alone = make_scheduler(alone_model)
    for request_id, sequence in codes.items():
        open_request(alone, request_id)
        for start in range(0, len(sequence), CHUNK_FRAMES):
            deliver_code2wav_chunk(
                alone, request_id, item(frames(*sequence[start : start + CHUNK_FRAMES]))
            )

    together_model = FakeCode2WavModel(total_upsample=2)
    together = make_scheduler(together_model)
    for request_id in codes:
        open_request(together, request_id)
    for start in range(0, 6, CHUNK_FRAMES):
        for request_id, sequence in codes.items():
            queue_chunk(
                together, request_id, frames(*sequence[start : start + CHUNK_FRAMES])
            )
        run_steps(together)

    assert stream_audio(together) == stream_audio(alone)
    assert {shape[0] for shape in alone_model.calls} == {1}
    assert [shape[0] for shape in together_model.calls] == [3, 3, 3]


@pytest.mark.parametrize("left_context_size", [CONTEXT_FRAMES, 0])
def test_first_windows_go_ahead_of_later_windows_that_waited_longer(
    left_context_size: int,
) -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = make_scheduler(model, left_context_size=left_context_size)
    open_request(scheduler, "early")
    open_request(scheduler, "late")
    deliver_code2wav_chunk(scheduler, "early", item(frames(1, 2)))
    queue_chunk(scheduler, "early", frames(3, 4))
    scheduler.has_ready_work()
    queue_chunk(scheduler, "late", frames(5, 6))

    # without left context both windows have one length, and still never share a replay
    scheduler.run_ready_step()
    assert model.calls[-1] == (1, 2, CHUNK_FRAMES), "the newcomer's first window"
    scheduler.run_ready_step()
    assert model.calls[-1] == (1, 2, left_context_size + CHUNK_FRAMES)
    assert not scheduler.has_ready_work()


def test_windows_of_different_lengths_never_share_a_replay() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = make_scheduler(model, left_context_size=3)
    for request_id in ("ahead", "behind"):
        open_request(scheduler, request_id)
    deliver_code2wav_chunk(scheduler, "ahead", item(frames(1, 2)))
    deliver_code2wav_chunk(scheduler, "ahead", item(frames(3, 4)))
    deliver_code2wav_chunk(scheduler, "behind", item(frames(5, 6)))
    queue_chunk(scheduler, "ahead", frames(7, 8))
    queue_chunk(scheduler, "behind", frames(9, 10))
    calls_before = len(model.calls)

    run_steps(scheduler)

    # ahead's context saturated at 3 frames (window 5), behind's holds 2 (window 4)
    assert sorted(model.calls[calls_before:]) == [(1, 2, 4), (1, 2, 5)]


@pytest.mark.parametrize(
    ("queued", "sizes", "expected_rows"),
    [
        (3, (4, 2, 1), [2, 1]),
        (5, (4, 2, 1), [4, 1]),
        (7, (8, 4, 2, 1), [4, 2, 1]),
        (7, (8, 7, 6, 5, 4, 3, 2, 1), [7]),
        (2, (1,), [1, 1]),
    ],
)
def test_a_replay_takes_the_largest_captured_row_count_the_queue_fills(
    queued: int, sizes: tuple[int, ...], expected_rows: list[int]
) -> None:
    model = FakeCode2WavModel(total_upsample=2)
    runner = PublishedSizesRunner(model, sizes)
    scheduler = make_scheduler(model, runner=runner)
    for index in range(queued):
        open_request(scheduler, f"r{index}")
        queue_chunk(scheduler, f"r{index}", frames(index + 1, index + 2))

    run_steps(scheduler)

    assert runner.rows == expected_rows
    assert set(runner.modes) == {"cuda_graph"}, "no replay falls back for its row count"


def test_eager_replays_take_at_most_max_replay_rows() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = make_scheduler(model, max_replay_rows=2)
    for index in range(3):
        open_request(scheduler, f"r{index}")
        queue_chunk(scheduler, f"r{index}", frames(index + 1, index + 2))

    run_steps(scheduler)

    assert [shape[0] for shape in model.calls] == [2, 1]


def test_a_backlog_decodes_one_chunk_per_window_in_frame_order() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = make_scheduler(model)
    open_request(scheduler, "r")
    deliver_code2wav_chunk(scheduler, "r", item(frames(1, 2)))
    queue_chunk(scheduler, "r", frames(3, 4))
    queue_chunk(scheduler, "r", frames(5, 6))

    run_steps(scheduler)

    reference = make_scheduler()
    open_request(reference, "r")
    for codes in (frames(1, 2), frames(3, 4), frames(5, 6)):
        deliver_code2wav_chunk(reference, "r", item(codes))
    # two windows of the captured length, never one window of both chunks
    assert model.calls[1:] == [(1, 2, 3), (1, 2, 3)]
    assert stream_audio(scheduler) == stream_audio(reference)


def test_stream_done_sends_its_ready_window_before_the_final_one() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = make_scheduler(model)
    open_request(scheduler, "r")
    deliver_code2wav_chunk(scheduler, "r", item(frames(1, 2)))
    stream_audio(scheduler)
    queue_chunk(scheduler, "r", frames(3, 4, 5))

    scheduler.handle_stream_done("r")

    messages = [scheduler.outbox.get_nowait() for _ in range(scheduler.outbox.qsize())]
    assert [message.type for message in messages] == ["stream", "stream", "result"]
    # the threshold window keeps its captured length; the final window holds the rest
    assert model.calls[1:] == [(1, 2, 3), (1, 2, 2)]
    second, final = (
        np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
        for message in messages[:2]
    )
    assert len(second) == CHUNK_FRAMES * 2 and len(final) == 1 * 2


def test_a_failing_replay_aborts_its_rows_and_spares_other_lengths() -> None:
    class FailingTwoRows(FakeCode2WavModel):
        def __call__(self, codes: torch.Tensor) -> torch.Tensor:
            if codes.shape[0] == 2:
                raise RuntimeError("replay failed")
            else:
                return super().__call__(codes)

    model = FailingTwoRows(total_upsample=2)
    scheduler = make_scheduler(model)
    aborted: list[str] = []
    scheduler.abort_callback = aborted.append
    for request_id in ("a", "b", "c"):
        open_request(scheduler, request_id)
    deliver_code2wav_chunk(scheduler, "c", item(frames(9, 9)))
    stream_audio(scheduler)
    queue_chunk(scheduler, "a", frames(1, 2))
    queue_chunk(scheduler, "b", frames(3, 4))
    queue_chunk(scheduler, "c", frames(5, 6))

    run_steps(scheduler)

    messages = [scheduler.outbox.get_nowait() for _ in range(scheduler.outbox.qsize())]
    assert sorted((m.request_id, m.type) for m in messages) == [
        ("a", "error"),
        ("b", "error"),
        ("c", "stream"),
    ]
    assert sorted(aborted) == ["a", "b"]
    assert "a" not in scheduler.stream_states and "b" not in scheduler.stream_states


def test_an_aborted_request_window_is_neither_decoded_nor_sent() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = make_scheduler(model)
    for request_id in ("kept", "dropped"):
        open_request(scheduler, request_id)
        queue_chunk(scheduler, request_id, frames(1, 2))
    scheduler.abort("dropped")

    run_steps(scheduler)

    assert model.calls == [(1, 2, CHUNK_FRAMES)]
    assert set(stream_audio(scheduler)) == {"kept"}


def test_a_window_keeps_its_full_chunk_when_the_model_returns_fewer_samples() -> None:
    scheduler = make_scheduler(FakeCode2WavModel(total_upsample=2, output_deficit=1))
    state = Code2WavStreamState(stream_enabled=True)
    state.chunks = [torch.tensor([code, code * 10]) for code in (1, 2, 3)]
    state.emitted = 1
    state.audio_parts = [np.zeros(1, dtype=np.float32)]
    scheduler.stream_states["r"] = state

    (message,) = scheduler.decode_windows([("r", state)])

    assert np.frombuffer(message.data["audio_waveform"], dtype=np.float32).shape == (4,)
    assert state.emitted == 3


def test_profile_events_carry_each_row_and_the_replay_rows(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler._get_event_recorder",
        lambda: SimpleNamespace(is_active=lambda: True, active_run_id=lambda: "run"),
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler._emit_event",
        lambda **event: events.append(event),
    )
    scheduler = make_scheduler()
    for request_id in ("a", "b"):
        open_request(scheduler, request_id)
        queue_chunk(scheduler, request_id, frames(1, 2))

    run_steps(scheduler)

    starts = [e for e in events if e["event_name"] == "code2wav_decode_start"]
    ends = [e for e in events if e["event_name"] == "code2wav_decode_end"]
    assert sorted(e["request_id"] for e in starts) == ["a", "b"]
    assert sorted(e["request_id"] for e in ends) == ["a", "b"]
    assert {e["metadata"]["rows"] for e in starts + ends} == {2}
    assert {e["metadata"]["window_frames"] for e in ends} == {CHUNK_FRAMES}


def test_first_window_ingest_events_are_bounded_and_exclude_eos(monkeypatch) -> None:
    events = []
    recorder = SimpleNamespace(is_active=lambda: True, active_run_id=lambda: "run-a")
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler"
        "._get_event_recorder",
        lambda: recorder,
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler._emit_event",
        lambda **kw: events.append(kw),
    )
    scheduler = make_scheduler(initial_codec_chunk_frames=2)
    state = Code2WavStreamState()
    for code in (2150, 1, 2, 3, 4):
        scheduler.ingest("req-a", state, torch.tensor([code, code * 10]))

    assert [event["event_name"] for event in events] == [
        "code2wav_first_ingest",
        "code2wav_first_window_ready",
    ]
    first, ready = events
    assert first["metadata"]["accepted_frames"] == 0
    assert ready["metadata"]["messages"] == 3
    assert ready["metadata"]["accepted_frames"] == 2
    assert ready["metadata"]["threshold_frames"] == 2
    assert ready["metadata"]["eos_checks"] == 3
    assert [row[0].item() for row in state.chunks] == [1, 2, 3, 4]


def test_first_window_profile_resets_on_new_run(monkeypatch) -> None:
    events = []
    current = SimpleNamespace(run_id="run-a")
    recorder = SimpleNamespace(
        is_active=lambda: True, active_run_id=lambda: current.run_id
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler"
        "._get_event_recorder",
        lambda: recorder,
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler._emit_event",
        lambda **kw: events.append(kw),
    )
    scheduler = make_scheduler(initial_codec_chunk_frames=2)
    state = Code2WavStreamState()
    scheduler.ingest("req-a", state, torch.tensor([[1, 10], [2, 20]]))
    assert events[-1]["metadata"]["messages"] == 1
    assert events[-1]["metadata"]["accepted_frames"] == 2

    current.run_id = "run-b"
    scheduler.ingest("req-a", state, torch.tensor([[3, 30]]))
    assert len(events) == 4
    assert events[-1]["metadata"]["started_with_frames"] == 2
    assert state.checked == 3


def test_ingest_without_recorder_does_not_read_profile_clocks(monkeypatch) -> None:
    def unexpected():
        raise AssertionError("inactive profiling must not read a profile clock")

    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler"
        "._get_event_recorder",
        lambda: SimpleNamespace(is_active=lambda: False),
    )
    scheduler = make_scheduler()
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler.time",
        SimpleNamespace(time_ns=unexpected, perf_counter_ns=unexpected),
    )
    state = Code2WavStreamState()
    scheduler.ingest("req-a", state, torch.tensor([1, 10]))
    scheduler.ingest("req-a", state, torch.tensor([2150, 0]))
    assert len(state.chunks) == 1
