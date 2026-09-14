# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from sglang_omni.models.ming_tts.payload_types import MingTTSState
from sglang_omni.models.ming_tts.streaming_vocoder import (
    MingTTSStreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage

_WAIT_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class _StreamCall:
    slot_ids: tuple[int, ...]
    patch_values: tuple[tuple[float, ...], ...]
    terminal_flags: tuple[bool, ...]
    thread_id: int


class _ScriptedAudioDecoder:
    sample_rate = 44100

    def __init__(self, *, capacity: int) -> None:
        self.stream_capacity = capacity
        self.streaming_ready = True
        self.stream_actions: deque[tuple[torch.Tensor, ...] | Exception] = deque()
        self.full_actions: deque[torch.Tensor | Exception] = deque()
        self.stream_calls: list[_StreamCall] = []
        self.full_calls: list[torch.Tensor] = []
        self.reset_rows_calls: list[tuple[int, ...]] = []
        self.reset_all_calls = 0
        self.prepare_calls = 0
        self.close_calls = 0
        self.reset_rows_error: Exception | None = None
        self.reset_all_error: Exception | None = None
        self.trace: list[tuple[str, object, int]] = []
        self.block_stream_call: int | None = None
        self.block_started = threading.Event()
        self.block_release = threading.Event()
        self.block_full_call: int | None = None
        self.full_block_started = threading.Event()
        self.full_block_release = threading.Event()
        self._condition = threading.Condition()

    def run_streaming(
        self,
        *,
        slot_ids: tuple[int, ...],
        patch_groups: tuple[tuple[torch.Tensor, ...], ...],
        terminal_flags: tuple[bool, ...],
    ) -> tuple[torch.Tensor, ...]:
        values = tuple(
            tuple(float(patch[0, 0]) for patch in patches) for patches in patch_groups
        )
        thread_id = threading.get_ident()
        with self._condition:
            self.stream_calls.append(
                _StreamCall(slot_ids, values, terminal_flags, thread_id)
            )
            call_index = len(self.stream_calls)
            self.trace.append(("stream", values, thread_id))
            self._condition.notify_all()

        if self.block_stream_call == call_index:
            self.block_started.set()
            if not self.block_release.wait(_WAIT_TIMEOUT_S):
                raise AssertionError("timed out waiting to release blocked stream call")

        action = self.stream_actions.popleft() if self.stream_actions else None
        if isinstance(action, Exception):
            raise action
        if action is not None:
            return action
        return tuple(
            torch.tensor(row_values, dtype=torch.float32) for row_values in values
        )

    def decode_full(self, latents: torch.Tensor) -> torch.Tensor:
        with self._condition:
            self.full_calls.append(latents.clone())
            call_index = len(self.full_calls)
            self.trace.append(("full", call_index, threading.get_ident()))
            self._condition.notify_all()

        if self.block_full_call == call_index:
            self.full_block_started.set()
            if not self.full_block_release.wait(_WAIT_TIMEOUT_S):
                raise AssertionError("timed out waiting to release blocked full call")

        action = (
            self.full_actions.popleft()
            if self.full_actions
            else torch.tensor([0.5, -0.5], dtype=torch.float32)
        )
        if isinstance(action, Exception):
            raise action
        return action

    def reset_stream_rows(self, slot_ids) -> None:
        slots = tuple(slot_ids)
        thread_id = threading.get_ident()
        with self._condition:
            self.reset_rows_calls.append(slots)
            self.trace.append(("reset_rows", slots, thread_id))
            self._condition.notify_all()
        if self.reset_rows_error is not None:
            raise self.reset_rows_error

    def reset_all_stream_rows(self) -> None:
        self.reset_all_calls += 1
        self.trace.append(("reset_all", self.reset_all_calls, threading.get_ident()))
        if self.reset_all_error is not None:
            raise self.reset_all_error

    def prepare_streaming(self) -> None:
        self.prepare_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def wait_for_stream_calls(self, count: int) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: len(self.stream_calls) >= count,
                timeout=_WAIT_TIMEOUT_S,
            )

    def wait_for_full_calls(self, count: int) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: len(self.full_calls) >= count,
                timeout=_WAIT_TIMEOUT_S,
            )


def _payload(
    request_id: str,
    *,
    stream: bool,
    generated_latents: torch.Tensor | None = None,
) -> StagePayload:
    state = MingTTSState(
        text="hello",
        input_ids=[1, 2, 3],
        max_decode_steps=8,
        generated_latents=generated_latents,
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": stream}),
        data=state.to_dict(),
    )


def _stream_item(
    chunk_id: int,
    value: float,
    *,
    is_last: bool,
    data: torch.Tensor | None = None,
    metadata: dict[str, Any] | None = None,
) -> StreamItem:
    if data is None:
        data = torch.full((2, 3), value, dtype=torch.float32)
    if metadata is None:
        metadata = {
            "modality": "audio_latents",
            "stream": True,
            "is_last": is_last,
        }
    return StreamItem(
        chunk_id=chunk_id,
        data=data,
        from_stage="tts_engine",
        metadata=metadata,
    )


def _scheduler(
    decoder: _ScriptedAudioDecoder,
    *,
    initial: int = 1,
    steady: int = 2,
) -> MingTTSStreamingVocoderScheduler:
    return MingTTSStreamingVocoderScheduler(
        decoder,
        patch_size=2,
        latent_dim=3,
        initial_chunk_patches=initial,
        steady_chunk_patches=steady,
    )


def _drain(scheduler: MingTTSStreamingVocoderScheduler):
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def _take(scheduler: MingTTSStreamingVocoderScheduler, count: int):
    return [scheduler.outbox.get(timeout=_WAIT_TIMEOUT_S) for _ in range(count)]


def _start(scheduler: MingTTSStreamingVocoderScheduler) -> threading.Thread:
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    return thread


def _stop(
    scheduler: MingTTSStreamingVocoderScheduler,
    thread: threading.Thread,
) -> None:
    scheduler.stop()
    thread.join(timeout=_WAIT_TIMEOUT_S)
    assert not thread.is_alive()


def _start_external_abort_after_tombstone(
    scheduler: MingTTSStreamingVocoderScheduler,
    request_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> threading.Thread:
    recorded = threading.Event()
    original = scheduler._record_aborted_request_id

    def record_and_signal(aborted_request_id: str) -> None:
        original(aborted_request_id)
        recorded.set()

    monkeypatch.setattr(scheduler, "_record_aborted_request_id", record_and_signal)
    thread = threading.Thread(target=scheduler.abort, args=(request_id,), daemon=True)
    thread.start()
    assert recorded.wait(_WAIT_TIMEOUT_S)
    return thread


def test_stream_capacity_bounds_chunk_collection_not_full_batching() -> None:
    decoder = _ScriptedAudioDecoder(capacity=2)
    scheduler = _scheduler(decoder)
    chunk_messages = [
        IncomingMessage(
            f"stream-{index}",
            "stream_chunk",
            _stream_item(0, index, is_last=False),
        )
        for index in range(3)
    ]
    scheduler.inbox.put(chunk_messages[1])
    scheduler.inbox.put(chunk_messages[2])

    collected_chunks = scheduler._collect_stream_chunk_batch(chunk_messages[0])

    assert collected_chunks == chunk_messages[:2]
    assert scheduler.inbox.get_nowait() == chunk_messages[2]

    full_messages = [
        IncomingMessage(
            f"full-{index}",
            "new_request",
            _payload(f"full-{index}", stream=False),
        )
        for index in range(2)
    ]
    scheduler.inbox.put(full_messages[1])

    assert scheduler._collect_new_request_batch(full_messages[0]) == full_messages[:1]
    assert scheduler.inbox.get_nowait() == full_messages[1]


@pytest.mark.parametrize(
    "invalid_case",
    ["metadata", "chunk_id", "is_last", "dtype", "shape", "device"],
)
def test_ming_stream_ingress_rejects_owned_contract_errors(
    invalid_case: str,
) -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    scheduler = _scheduler(decoder)
    item = _stream_item(0, 1.0, is_last=False)
    if invalid_case == "metadata":
        item.metadata = None
    elif invalid_case == "chunk_id":
        item.chunk_id = 1
    elif invalid_case == "is_last":
        item.metadata = {**item.metadata, "is_last": "false"}
    elif invalid_case == "dtype":
        item.data = item.data.to(torch.float64)
    elif invalid_case == "shape":
        item.data = torch.ones((1, 3), dtype=torch.float32)
    elif invalid_case == "device":
        item.data = torch.ones((2, 3), device="meta")
    scheduler.on_stream_chunk_batch([("invalid-ingress", item)])

    messages = _drain(scheduler)
    assert [(message.request_id, message.type) for message in messages] == [
        ("invalid-ingress", "error")
    ]
    assert scheduler._is_aborted("invalid-ingress")
    assert "invalid-ingress" not in scheduler._stream_states
    assert decoder.stream_calls == []


def test_real_inbox_preserves_mk_terminal_backlog_exact_once() -> None:
    decoder = _ScriptedAudioDecoder(capacity=2)
    decoder.stream_actions.append((torch.empty(0, dtype=torch.float32),))
    scheduler = _scheduler(decoder, initial=2, steady=4)
    request_id = "stream-mk"
    full_id = "healthy-full"
    for chunk_id in range(2):
        scheduler.inbox.put(
            IncomingMessage(
                request_id,
                "stream_chunk",
                _stream_item(chunk_id, chunk_id, is_last=False),
            )
        )
    scheduler.inbox.put(
        IncomingMessage(
            full_id,
            "new_request",
            _payload(
                full_id,
                stream=False,
                generated_latents=torch.ones((1, 2, 3)),
            ),
        )
    )
    scheduler.inbox.put(IncomingMessage(full_id, "stream_done"))
    for chunk_id in range(2, 9):
        scheduler.inbox.put(
            IncomingMessage(
                request_id,
                "stream_chunk",
                _stream_item(chunk_id, chunk_id, is_last=chunk_id == 8),
            )
        )
    scheduler.inbox.put(IncomingMessage(request_id, "stream_done"))
    scheduler.inbox.put(
        IncomingMessage(
            request_id,
            "new_request",
            _payload(request_id, stream=True),
        )
    )

    thread = _start(scheduler)
    messages = _take(scheduler, 4)

    assert [call.patch_values for call in decoder.stream_calls] == [
        ((0.0, 1.0),),
        ((2.0, 3.0, 4.0, 5.0),),
        ((6.0, 7.0, 8.0),),
    ]
    assert [call.terminal_flags for call in decoder.stream_calls] == [
        (False,),
        (False,),
        (True,),
    ]
    assert len({call.slot_ids for call in decoder.stream_calls}) == 1
    assert [
        value
        for call in decoder.stream_calls
        for row in call.patch_values
        for value in row
    ] == [float(value) for value in range(9)]
    assert len(decoder.full_calls) == 1
    stream_types = [
        message.type for message in messages if message.request_id == request_id
    ]
    assert stream_types == ["stream", "stream", "result"]
    assert [message.type for message in messages if message.request_id == full_id] == [
        "result"
    ]
    stream_result = next(
        message.data
        for message in messages
        if message.request_id == request_id and message.type == "result"
    )
    restored = MingTTSState.from_dict(stream_result.data)
    assert restored.sample_rate == 44100
    assert restored.duration_s == pytest.approx(7 / 44100)
    assert stream_result.data["modality"] == "audio"

    _stop(scheduler, thread)


def test_initial_cadence_can_exceed_steady_cadence() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    scheduler = _scheduler(decoder, initial=3, steady=1)
    scheduler.on_stream_chunk_batch(
        [
            (
                "initial-larger",
                _stream_item(index, index, is_last=index == 4),
            )
            for index in range(5)
        ]
    )

    assert [call.patch_values for call in decoder.stream_calls] == [
        ((0.0, 1.0, 2.0),),
        ((3.0,),),
        ((4.0,),),
    ]
    assert [call.terminal_flags for call in decoder.stream_calls] == [
        (False,),
        (False,),
        (True,),
    ]


def test_mk_mixed_wave_preserves_request_slot_and_output_mapping() -> None:
    decoder = _ScriptedAudioDecoder(capacity=3)
    scheduler = _scheduler(decoder, initial=2, steady=4)
    scheduler.on_stream_chunk_batch(
        [
            ("step", _stream_item(0, 10, is_last=False)),
            ("step", _stream_item(1, 11, is_last=False)),
        ]
    )
    step_slot = scheduler._slot_bindings.slot_for("step")
    first_messages = _drain(scheduler)

    scheduler.on_stream_chunk_batch(
        [
            ("step", _stream_item(2, 12, is_last=False)),
            ("step", _stream_item(3, 13, is_last=False)),
            ("step", _stream_item(4, 14, is_last=False)),
            ("step", _stream_item(5, 15, is_last=False)),
            ("open-terminal", _stream_item(0, 20, is_last=False)),
            ("open-terminal", _stream_item(1, 21, is_last=True)),
            ("short-terminal", _stream_item(0, 30, is_last=True)),
            ("waiting", _stream_item(0, 40, is_last=False)),
        ]
    )
    mixed_messages = _drain(scheduler)
    mixed_call = decoder.stream_calls[1]
    terminal_slots = set(mixed_call.slot_ids[1:])

    assert decoder.stream_calls[0].patch_values == ((10.0, 11.0),)
    assert decoder.stream_calls[0].terminal_flags == (False,)
    assert mixed_call.patch_values == (
        (12.0, 13.0, 14.0, 15.0),
        (20.0, 21.0),
        (30.0,),
    )
    assert mixed_call.terminal_flags == (False, True, True)
    assert mixed_call.slot_ids[0] == step_slot
    assert scheduler._slot_bindings.slot_for("step") == step_slot
    assert scheduler._slot_bindings.slot_for("open-terminal") is None
    assert scheduler._slot_bindings.slot_for("short-terminal") is None
    assert scheduler._slot_bindings.slot_for("waiting") is None

    expected_payloads = {
        "step": (12.0, 13.0, 14.0, 15.0),
        "open-terminal": (20.0, 21.0),
        "short-terminal": (30.0,),
    }
    assert [message.request_id for message in first_messages] == ["step"]
    assert (
        first_messages[0].data["audio_waveform"]
        == torch.tensor([10.0, 11.0], dtype=torch.float32).numpy().tobytes()
    )
    assert {message.request_id for message in mixed_messages} == set(expected_payloads)
    for message in mixed_messages:
        assert message.type == "stream"
        assert (
            message.data["audio_waveform"]
            == torch.tensor(expected_payloads[message.request_id], dtype=torch.float32)
            .numpy()
            .tobytes()
        )

    scheduler.on_stream_chunk_batch([("waiting", _stream_item(1, 41, is_last=True))])
    waiting_messages = _drain(scheduler)
    waiting_call = decoder.stream_calls[2]

    assert waiting_call.patch_values == ((40.0, 41.0),)
    assert waiting_call.terminal_flags == (True,)
    assert waiting_call.slot_ids[0] in terminal_slots
    assert scheduler._slot_bindings.slot_for("step") == step_slot
    assert scheduler._slot_bindings.slot_for("waiting") is None
    assert [message.request_id for message in waiting_messages] == ["waiting"]
    assert (
        waiting_messages[0].data["audio_waveform"]
        == torch.tensor([40.0, 41.0], dtype=torch.float32).numpy().tobytes()
    )


def test_terminal_patch_bypasses_opening_threshold() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    scheduler = _scheduler(decoder, initial=3, steady=2)

    scheduler.on_stream_chunk_batch(
        [("short-terminal", _stream_item(0, 1, is_last=True))]
    )

    assert len(decoder.stream_calls) == 1
    assert decoder.stream_calls[0].patch_values == ((1.0,),)
    assert decoder.stream_calls[0].terminal_flags == (True,)


def test_late_chunk_after_terminal_is_request_local_error() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    scheduler = _scheduler(decoder, initial=1, steady=1)
    scheduler.on_stream_chunk_batch(
        [("late-terminal", _stream_item(0, 1, is_last=True))]
    )
    _drain(scheduler)

    scheduler.on_stream_chunk_batch(
        [("late-terminal", _stream_item(1, 2, is_last=False))]
    )

    messages = _drain(scheduler)
    assert [(message.request_id, message.type) for message in messages] == [
        ("late-terminal", "error")
    ]
    assert "after the terminal patch" in str(messages[0].data)
    assert len(decoder.stream_calls) == 1


def test_one_pass_admission_keeps_holder_and_uses_registry_order() -> None:
    decoder = _ScriptedAudioDecoder(capacity=2)
    scheduler = _scheduler(decoder, initial=1, steady=1)
    scheduler.on_stream_chunk_batch(
        [
            ("holder", _stream_item(0, 10, is_last=False)),
            ("releasing", _stream_item(0, 20, is_last=False)),
        ]
    )
    holder_slot = scheduler._slot_bindings.slot_for("holder")
    assert holder_slot is not None
    scheduler.on_stream_chunk_batch(
        [
            ("waiter-first", _stream_item(0, 30, is_last=True)),
            ("waiter-second", _stream_item(0, 40, is_last=True)),
        ]
    )
    assert len(decoder.stream_calls) == 1

    scheduler.on_stream_chunk_batch([("releasing", _stream_item(1, 21, is_last=True))])

    assert [call.patch_values for call in decoder.stream_calls[1:]] == [
        ((21.0,),),
        ((30.0,),),
        ((40.0,),),
    ]
    assert scheduler._slot_bindings.slot_for("holder") == holder_slot
    assert scheduler._slot_bindings.slot_for("waiter-first") is None
    assert scheduler._slot_bindings.slot_for("waiter-second") is None


def test_successful_mixed_wave_commits_terminal_before_releasing_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _ScriptedAudioDecoder(capacity=2)
    scheduler = _scheduler(decoder, initial=1, steady=1)
    scheduler.on_stream_chunk_batch(
        [
            ("live", _stream_item(0, 10, is_last=False)),
            ("terminal", _stream_item(0, 20, is_last=False)),
        ]
    )
    live_slot = scheduler._slot_bindings.slot_for("live")
    terminal_slot = scheduler._slot_bindings.slot_for("terminal")
    _drain(scheduler)

    release_observations = []
    release_clean = scheduler._slot_bindings.release_clean

    def observe_release(request_ids) -> None:
        release_observations.append(
            tuple(
                (
                    request_id,
                    scheduler._stream_states[request_id].terminal_committed,
                    len(scheduler._stream_states[request_id].pending_patches),
                    scheduler._stream_states[request_id].emitted_samples,
                    scheduler._slot_bindings.slot_for(request_id),
                )
                for request_id in request_ids
            )
        )
        release_clean(request_ids)

    monkeypatch.setattr(scheduler._slot_bindings, "release_clean", observe_release)

    scheduler.on_stream_chunk_batch([("waiter", _stream_item(0, 30, is_last=True))])
    assert len(decoder.stream_calls) == 1

    scheduler.on_stream_chunk_batch(
        [
            ("live", _stream_item(1, 11, is_last=False)),
            ("terminal", _stream_item(1, 21, is_last=True)),
        ]
    )

    assert decoder.stream_calls[1].patch_values == ((11.0,), (21.0,))
    assert decoder.stream_calls[1].terminal_flags == (False, True)
    assert decoder.stream_calls[2].patch_values == ((30.0,),)
    assert decoder.stream_calls[2].terminal_flags == (True,)
    assert decoder.stream_calls[2].slot_ids == (terminal_slot,)
    assert release_observations == [
        (("terminal", True, 0, 2, terminal_slot),),
        (("waiter", True, 0, 1, terminal_slot),),
    ]
    assert scheduler._slot_bindings.slot_for("live") == live_slot
    assert scheduler._slot_bindings.slot_for("terminal") is None
    assert scheduler._slot_bindings.slot_for("waiter") is None


def test_step_error_resets_participant_before_next_wave() -> None:
    decoder = _ScriptedAudioDecoder(capacity=2)
    scheduler = _scheduler(decoder, initial=1, steady=1)
    scheduler.on_stream_chunk_batch([("holder", _stream_item(0, 10, is_last=False))])
    holder_slot = scheduler._slot_bindings.slot_for("holder")
    _drain(scheduler)
    decoder.stream_actions.append(ValueError("invalid staged input"))

    scheduler.on_stream_chunk_batch([("offender", _stream_item(0, 20, is_last=False))])
    scheduler.on_stream_chunk_batch([("waiter", _stream_item(0, 30, is_last=True))])
    assert len(decoder.stream_calls) == 3
    errors = [message for message in _drain(scheduler) if message.type == "error"]
    assert [message.request_id for message in errors] == ["offender"]
    assert scheduler._slot_bindings.slot_for("holder") == holder_slot
    assert len(decoder.reset_rows_calls) == 1
    reset_index = next(
        index for index, entry in enumerate(decoder.trace) if entry[0] == "reset_rows"
    )
    waiter_index = next(
        index
        for index, entry in enumerate(decoder.trace)
        if entry[0] == "stream" and entry[1] == ((30.0,),)
    )
    assert reset_index < waiter_index
    assert decoder.stream_calls[-1].patch_values == ((30.0,),)
    assert scheduler._slot_bindings.slot_for("holder") == holder_slot


def test_wave_failure_aborts_only_participants_and_future_wave_succeeds() -> None:
    decoder = _ScriptedAudioDecoder(capacity=3)
    scheduler = _scheduler(decoder, initial=1, steady=1)
    scheduler.on_stream_chunk_batch(
        [("inactive-holder", _stream_item(0, 10, is_last=False))]
    )
    holder_slot = scheduler._slot_bindings.slot_for("inactive-holder")
    _drain(scheduler)
    decoder.stream_actions.append(RuntimeError("wave materialization failed"))

    scheduler.on_stream_chunk_batch(
        [
            ("failed-a", _stream_item(0, 1, is_last=False)),
            ("failed-b", _stream_item(0, 2, is_last=True)),
        ]
    )

    messages = _drain(scheduler)
    assert {(message.request_id, message.type) for message in messages} == {
        ("failed-a", "error"),
        ("failed-b", "error"),
    }
    assert scheduler._slot_bindings.slot_for("inactive-holder") == holder_slot
    assert scheduler._is_aborted("failed-a")
    assert scheduler._is_aborted("failed-b")
    assert len(decoder.reset_rows_calls) == 1
    failed_call_count = len(decoder.stream_calls)
    scheduler.on_stream_chunk_batch(
        [
            ("failed-a", _stream_item(1, 3, is_last=True)),
            ("failed-b", _stream_item(1, 4, is_last=True)),
        ]
    )
    assert len(decoder.stream_calls) == failed_call_count
    assert _drain(scheduler) == []

    scheduler.on_stream_chunk_batch([("future-row", _stream_item(0, 3, is_last=True))])
    assert decoder.stream_calls[-1].patch_values == ((3.0,),)


def test_terminal_without_audio_is_request_local_error() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    decoder.stream_actions.append((torch.empty(0, dtype=torch.float32),))
    scheduler = _scheduler(decoder, initial=2, steady=4)
    scheduler.on_stream_chunk_batch(
        [("silent-terminal", _stream_item(0, 1, is_last=True))]
    )
    thread = _start(scheduler)
    scheduler.inbox.put(
        IncomingMessage(
            "silent-terminal",
            "new_request",
            _payload("silent-terminal", stream=True),
        )
    )
    scheduler.inbox.put(IncomingMessage("silent-terminal", "stream_done"))

    messages = _take(scheduler, 1)
    assert [(message.request_id, message.type) for message in messages] == [
        ("silent-terminal", "error")
    ]
    assert "completed without audio" in str(messages[0].data)
    assert decoder.reset_rows_calls == []
    _stop(scheduler, thread)


def test_slotless_terminal_full_stacks_stream_patches_exactly_once() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    scheduler = _scheduler(decoder, initial=1, steady=2)
    thread = _start(scheduler)
    scheduler.inbox.put(
        IncomingMessage("holder", "stream_chunk", _stream_item(0, 9, is_last=False))
    )
    assert _take(scheduler, 1)[0].request_id == "holder"

    for chunk_id in range(3):
        scheduler.inbox.put(
            IncomingMessage(
                "slotless-full",
                "stream_chunk",
                _stream_item(chunk_id, chunk_id, is_last=chunk_id == 2),
            )
        )
    scheduler.inbox.put(IncomingMessage("slotless-full", "stream_done"))
    scheduler.inbox.put(
        IncomingMessage(
            "slotless-full",
            "new_request",
            _payload("slotless-full", stream=True),
        )
    )

    decoder.wait_for_full_calls(1)
    messages = _take(scheduler, 2)

    assert len(decoder.full_calls) == 1
    expected_latents = torch.stack(
        [torch.full((2, 3), value, dtype=torch.float32) for value in range(3)]
    )
    assert torch.equal(decoder.full_calls[0], expected_latents)
    assert [call.patch_values for call in decoder.stream_calls] == [((9.0,),)]
    assert [message.type for message in messages] == ["stream", "result"]
    assert all(message.request_id == "slotless-full" for message in messages)
    result = MingTTSState.from_dict(messages[1].data.data)
    assert result.duration_s == pytest.approx(2 / decoder.sample_rate)
    assert scheduler._slot_bindings.slot_for("holder") == 0
    assert scheduler._slot_bindings.slot_for("slotless-full") is None
    _stop(scheduler, thread)


@pytest.mark.parametrize(
    ("failure", "error_pattern"),
    [
        ("exception", "full device failed"),
        ("empty", "completed without audio"),
    ],
)
def test_slotless_full_failure_is_request_local(
    failure: str,
    error_pattern: str,
) -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    scheduler = _scheduler(decoder, initial=1, steady=1)
    thread = _start(scheduler)
    scheduler.inbox.put(
        IncomingMessage("holder", "stream_chunk", _stream_item(0, 9, is_last=False))
    )
    assert _take(scheduler, 1)[0].request_id == "holder"
    if failure == "exception":
        decoder.full_actions.append(RuntimeError("full device failed"))
    else:
        decoder.full_actions.append(torch.empty(0, dtype=torch.float32))

    for chunk_id in range(2):
        scheduler.inbox.put(
            IncomingMessage(
                "failed-full",
                "stream_chunk",
                _stream_item(chunk_id, chunk_id, is_last=chunk_id == 1),
            )
        )
    scheduler.inbox.put(IncomingMessage("failed-full", "stream_done"))
    scheduler.inbox.put(
        IncomingMessage(
            "failed-full",
            "new_request",
            _payload("failed-full", stream=True),
        )
    )

    decoder.wait_for_full_calls(1)
    error = _take(scheduler, 1)

    assert [(message.request_id, message.type) for message in error] == [
        ("failed-full", "error")
    ]
    assert error_pattern in str(error[0].data)
    assert scheduler._is_aborted("failed-full")
    assert "failed-full" not in scheduler._stream_states
    assert len(decoder.full_calls) == 1
    assert thread.is_alive()

    scheduler.inbox.put(
        IncomingMessage(
            "failed-full", "stream_chunk", _stream_item(2, 10, is_last=True)
        )
    )
    scheduler.inbox.put(
        IncomingMessage("holder", "stream_chunk", _stream_item(1, 11, is_last=True))
    )
    scheduler.inbox.put(
        IncomingMessage("future", "stream_chunk", _stream_item(0, 12, is_last=True))
    )
    decoder.wait_for_stream_calls(3)
    future_messages = _take(scheduler, 2)

    assert len(decoder.full_calls) == 1
    assert [call.patch_values for call in decoder.stream_calls] == [
        ((9.0,),),
        ((11.0,),),
        ((12.0,),),
    ]
    assert {(message.request_id, message.type) for message in future_messages} == {
        ("holder", "stream"),
        ("future", "stream"),
    }
    assert thread.is_alive()
    _stop(scheduler, thread)


def test_external_abort_is_lazy_and_next_streaming_turn_resets_before_reuse() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    scheduler = _scheduler(decoder, initial=1, steady=2)
    thread = _start(scheduler)
    scheduler.inbox.put(
        IncomingMessage("holder", "stream_chunk", _stream_item(0, 1, is_last=False))
    )
    assert _take(scheduler, 1)[0].request_id == "holder"
    scheduler.inbox.put(
        IncomingMessage("waiter", "stream_chunk", _stream_item(0, 2, is_last=True))
    )
    scheduler.inbox.put(
        IncomingMessage(
            "barrier-full",
            "new_request",
            _payload(
                "barrier-full",
                stream=False,
                generated_latents=torch.ones((1, 2, 3)),
            ),
        )
    )
    decoder.wait_for_full_calls(1)

    caller_thread_id = threading.get_ident()
    scheduler.abort("holder")

    assert decoder.reset_rows_calls == []
    assert scheduler.inbox.empty()
    assert scheduler._slot_bindings.slot_for("holder") == 0

    scheduler.inbox.put(IncomingMessage("waiter", "stream_done"))
    scheduler.inbox.put(
        IncomingMessage("waiter", "new_request", _payload("waiter", stream=True))
    )
    decoder.wait_for_full_calls(2)
    messages = _take(scheduler, 3)

    assert decoder.full_calls[1].shape == (1, 2, 3)
    assert [message.type for message in messages if message.request_id == "waiter"] == [
        "stream",
        "result",
    ]
    assert decoder.reset_rows_calls == []

    scheduler.inbox.put(
        IncomingMessage("future", "stream_chunk", _stream_item(0, 3, is_last=True))
    )
    decoder.wait_for_stream_calls(2)
    future_message = _take(scheduler, 1)

    assert decoder.reset_rows_calls == [(0,)]
    assert [call.patch_values for call in decoder.stream_calls] == [
        ((1.0,),),
        ((3.0,),),
    ]
    reset_index = next(
        index for index, entry in enumerate(decoder.trace) if entry[0] == "reset_rows"
    )
    future_stream_index = next(
        index
        for index, entry in enumerate(decoder.trace)
        if entry[0] == "stream" and entry[1] == ((3.0,),)
    )
    assert reset_index < future_stream_index
    reset_entry = decoder.trace[reset_index]
    assert reset_entry[2] == thread.ident
    assert reset_entry[2] != caller_thread_id
    assert future_message[0].request_id == "future"
    assert scheduler._slot_bindings.slot_for("holder") is None
    assert scheduler._slot_bindings.slot_for("future") is None
    _stop(scheduler, thread)


def test_external_abort_during_fixed_wave_suppresses_output_and_resets_after_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    decoder.block_stream_call = 1
    scheduler = _scheduler(decoder, initial=1, steady=1)
    scheduler_thread = _start(scheduler)
    scheduler.inbox.put(
        IncomingMessage(
            "aborted-in-flight",
            "stream_chunk",
            _stream_item(0, 1, is_last=False),
        )
    )
    assert decoder.block_started.wait(_WAIT_TIMEOUT_S)

    abort_thread = _start_external_abort_after_tombstone(
        scheduler, "aborted-in-flight", monkeypatch
    )

    assert scheduler._is_aborted("aborted-in-flight")
    assert abort_thread.is_alive()
    assert decoder.reset_rows_calls == []
    decoder.block_release.set()
    abort_thread.join(timeout=_WAIT_TIMEOUT_S)
    assert not abort_thread.is_alive()
    assert _drain(scheduler) == []
    assert decoder.reset_rows_calls == []

    scheduler.inbox.put(
        IncomingMessage("future", "stream_chunk", _stream_item(0, 2, is_last=True))
    )
    decoder.wait_for_stream_calls(2)
    future_message = _take(scheduler, 1)

    assert [(message.request_id, message.type) for message in future_message] == [
        ("future", "stream")
    ]
    assert decoder.reset_rows_calls == [(0,)]
    reset_index = next(
        index for index, entry in enumerate(decoder.trace) if entry[0] == "reset_rows"
    )
    future_stream_index = next(
        index
        for index, entry in enumerate(decoder.trace)
        if entry[0] == "stream" and entry[1] == ((2.0,),)
    )
    assert reset_index < future_stream_index
    assert decoder.trace[reset_index][2] == scheduler_thread.ident
    _stop(scheduler, scheduler_thread)


def test_external_abort_during_slotless_full_suppresses_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    decoder.block_full_call = 1
    scheduler = _scheduler(decoder, initial=1, steady=1)
    scheduler_thread = _start(scheduler)
    scheduler.inbox.put(
        IncomingMessage("holder", "stream_chunk", _stream_item(0, 9, is_last=False))
    )
    assert _take(scheduler, 1)[0].request_id == "holder"
    scheduler.inbox.put(
        IncomingMessage(
            "aborted-full", "stream_chunk", _stream_item(0, 1, is_last=True)
        )
    )
    scheduler.inbox.put(IncomingMessage("aborted-full", "stream_done"))
    scheduler.inbox.put(
        IncomingMessage(
            "aborted-full",
            "new_request",
            _payload("aborted-full", stream=True),
        )
    )
    assert decoder.full_block_started.wait(_WAIT_TIMEOUT_S)

    abort_thread = _start_external_abort_after_tombstone(
        scheduler, "aborted-full", monkeypatch
    )

    assert scheduler._is_aborted("aborted-full")
    assert abort_thread.is_alive()
    decoder.full_block_release.set()
    abort_thread.join(timeout=_WAIT_TIMEOUT_S)
    assert not abort_thread.is_alive()

    assert _drain(scheduler) == []
    assert len(decoder.full_calls) == 1
    assert decoder.full_calls[0].shape == (1, 2, 3)
    assert decoder.trace[-1][0] == "full"
    assert decoder.trace[-1][2] == scheduler_thread.ident
    assert decoder.reset_rows_calls == []
    assert "aborted-full" not in scheduler._stream_states
    assert scheduler._slot_bindings.slot_for("holder") == 0
    _stop(scheduler, scheduler_thread)


def test_reset_failure_unbinds_request_and_keeps_slot_unavailable() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    scheduler = _scheduler(decoder, initial=1, steady=1)
    decoder.stream_actions.append(RuntimeError("row copy failed"))
    decoder.reset_rows_error = RuntimeError("reset fence failed")

    scheduler.on_stream_chunk_batch([("dirty-row", _stream_item(0, 1, is_last=False))])

    dirty_error = _drain(scheduler)
    assert [(message.request_id, message.type) for message in dirty_error] == [
        ("dirty-row", "error")
    ]
    assert len(decoder.reset_rows_calls) == 1
    assert scheduler._slot_bindings.slot_for("dirty-row") is None

    thread = _start(scheduler)
    scheduler.inbox.put(
        IncomingMessage("waiter", "stream_chunk", _stream_item(0, 2, is_last=True))
    )
    scheduler.inbox.put(IncomingMessage("waiter", "stream_done"))
    scheduler.inbox.put(
        IncomingMessage("waiter", "new_request", _payload("waiter", stream=True))
    )
    decoder.wait_for_full_calls(1)
    waiter_messages = _take(scheduler, 2)

    scheduler.on_stream_chunk_batch([("future", _stream_item(0, 3, is_last=True))])

    assert [message.type for message in waiter_messages] == ["stream", "result"]
    assert len(decoder.full_calls) == 1
    assert _drain(scheduler) == []
    assert len(decoder.reset_rows_calls) == 1
    assert len(decoder.stream_calls) == 1
    assert scheduler._slot_bindings.slot_for("dirty-row") is None
    assert scheduler._slot_bindings.slot_for("future") is None
    _stop(scheduler, thread)


def test_full_failure_does_not_affect_live_or_future_streaming() -> None:
    decoder = _ScriptedAudioDecoder(capacity=2)
    scheduler = _scheduler(decoder, initial=1, steady=1)
    thread = _start(scheduler)
    scheduler.inbox.put(
        IncomingMessage(
            "holder",
            "stream_chunk",
            _stream_item(0, 1, is_last=False),
        )
    )
    decoder.wait_for_stream_calls(1)
    _take(scheduler, 1)
    decoder.full_actions.append(RuntimeError("full device failed"))
    scheduler.inbox.put(
        IncomingMessage(
            "failing-full",
            "new_request",
            _payload(
                "failing-full",
                stream=False,
                generated_latents=torch.ones((1, 2, 3)),
            ),
        )
    )
    decoder.wait_for_full_calls(1)

    error = _take(scheduler, 1)
    assert [(message.request_id, message.type) for message in error] == [
        ("failing-full", "error")
    ]
    assert thread.is_alive()
    scheduler.inbox.put(
        IncomingMessage(
            "future-full",
            "new_request",
            _payload(
                "future-full",
                stream=False,
                generated_latents=torch.empty((0, 2, 3)),
            ),
        )
    )
    scheduler.inbox.put(
        IncomingMessage(
            "future-stream",
            "stream_chunk",
            _stream_item(0, 3, is_last=False),
        )
    )
    future_results = _take(scheduler, 2)
    assert {(message.request_id, message.type) for message in future_results} == {
        ("future-full", "result"),
        ("future-stream", "stream"),
    }
    assert len(decoder.full_calls) == 2
    assert len(decoder.stream_calls) == 2
    assert scheduler._slot_bindings.slot_for("holder") == 0
    assert thread.is_alive()
    _stop(scheduler, thread)
    assert decoder.reset_all_calls == 1
    assert decoder.close_calls == 1


def test_done_without_terminal_is_request_local_error() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    scheduler = _scheduler(decoder)
    thread = _start(scheduler)
    scheduler.inbox.put(
        IncomingMessage(
            "missing-terminal",
            "new_request",
            _payload("missing-terminal", stream=True),
        )
    )
    scheduler.inbox.put(IncomingMessage("missing-terminal", "stream_done"))

    messages = _take(scheduler, 1)
    assert len(messages) == 1
    assert messages[0].request_id == "missing-terminal"
    assert messages[0].type == "error"
    assert "without a terminal latent patch" in str(messages[0].data)
    _stop(scheduler, thread)


def test_stop_before_start_is_idempotent_and_consumes_no_business() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    scheduler = _scheduler(decoder)
    scheduler.inbox.put(
        IncomingMessage(
            "never-run",
            "stream_chunk",
            _stream_item(0, 1, is_last=True),
        )
    )

    scheduler.stop()
    thread = _start(scheduler)
    thread.join(timeout=_WAIT_TIMEOUT_S)

    assert not thread.is_alive()
    assert decoder.stream_calls == []
    assert decoder.full_calls == []
    assert decoder.reset_all_calls == 1
    assert decoder.close_calls == 1
    assert scheduler.outbox.empty()


def test_healthy_stop_resets_full_bank_once_with_live_bound_state() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    scheduler = _scheduler(decoder, initial=1, steady=2)
    scheduler.on_stream_chunk_batch(
        [("live-holder", _stream_item(0, 1, is_last=False))]
    )
    assert scheduler._slot_bindings.slot_for("live-holder") == 0

    scheduler.stop()

    assert scheduler._stream_states == {}
    assert scheduler._slot_bindings.slot_for("live-holder") is None
    assert decoder.reset_rows_calls == []
    assert decoder.reset_all_calls == 1
    assert decoder.close_calls == 1


def test_start_rejects_unprepared_graph_and_still_closes() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    decoder.streaming_ready = False
    scheduler = _scheduler(decoder)

    with pytest.raises(RuntimeError, match="backend is not prepared"):
        scheduler.start()

    assert decoder.reset_all_calls == 1
    assert decoder.close_calls == 1


def test_shutdown_reset_failure_still_closes_decoder() -> None:
    decoder = _ScriptedAudioDecoder(capacity=1)
    decoder.reset_all_error = RuntimeError("shutdown reset failed")
    scheduler = _scheduler(decoder)

    with pytest.raises(RuntimeError, match="shutdown reset failed"):
        scheduler.stop()

    assert decoder.reset_all_calls == 1
    assert decoder.close_calls == 1


def test_stop_during_pump_drains_current_turn_only() -> None:
    decoder = _ScriptedAudioDecoder(capacity=4)
    decoder.block_stream_call = 1
    scheduler = _scheduler(decoder, initial=1, steady=2)
    request_id = "terminal-backlog"
    for chunk_id in range(4):
        scheduler.inbox.put(
            IncomingMessage(
                request_id,
                "stream_chunk",
                _stream_item(chunk_id, chunk_id, is_last=chunk_id == 3),
            )
        )
    scheduler.inbox.put(
        IncomingMessage(
            "must-not-start",
            "new_request",
            _payload(
                "must-not-start",
                stream=False,
                generated_latents=torch.ones((1, 2, 3)),
            ),
        )
    )
    thread = _start(scheduler)
    assert decoder.block_started.wait(_WAIT_TIMEOUT_S)

    scheduler.stop()
    decoder.block_release.set()
    thread.join(timeout=_WAIT_TIMEOUT_S)

    assert not thread.is_alive()
    assert [call.patch_values for call in decoder.stream_calls] == [
        ((0.0,),),
        ((1.0, 2.0),),
        ((3.0,),),
    ]
    assert [call.terminal_flags for call in decoder.stream_calls] == [
        (False,),
        (False,),
        (True,),
    ]
    assert decoder.full_calls == []
    assert decoder.reset_all_calls == 1
    assert decoder.close_calls == 1
