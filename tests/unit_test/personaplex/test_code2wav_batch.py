# SPDX-License-Identifier: Apache-2.0
"""Continuous cross-request batching in the PersonaPlex codec stage."""

import asyncio
from dataclasses import replace

import numpy as np
import pytest
import torch

from sglang_omni.models.personaplex.architecture import (
    MIMI,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)
from sglang_omni.models.personaplex.code2wav_stream import PersonaPlexCode2WavScheduler
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.models.personaplex.mimi_decode_graph import MimiDecodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage

# The default chunk ramp: the first frame alone, then four frames at a time.
RAMP = (1, 4)


def decode_waveform(payload: dict) -> torch.Tensor:
    assert payload["sample_rate"] == SAMPLE_RATE
    return torch.from_numpy(
        np.frombuffer(payload["audio_waveform"], dtype=np.float32).copy()
    )


def new_scheduler(
    codec: MimiCodec,
    *,
    num_slots: int,
    max_batch_size: int,
    max_graph_frames: int = 0,
    chunk_ramp_frames: tuple[int, ...] = RAMP,
) -> PersonaPlexCode2WavScheduler:
    return PersonaPlexCode2WavScheduler(
        MimiDecodeSlots(
            codec,
            num_slots=num_slots,
            max_batch_size=max_batch_size,
            max_graph_frames=max_graph_frames,
            max_chunk_frames=max(chunk_ramp_frames),
        ),
        compute_fn=lambda payload: payload,
        chunk_ramp_frames=chunk_ramp_frames,
    )


def start_stream(scheduler, request_id: str) -> None:
    payload = StagePayload(
        request_id, request=OmniRequest(inputs={}), data=PersonaPlexState().to_dict()
    )
    scheduler.stream_payloads[request_id] = payload
    scheduler.on_streaming_new_request(request_id, payload)


def item(codes: torch.Tensor, **metadata: int) -> StreamItem:
    return StreamItem(
        chunk_id=0, from_stage="lm", data=codes, metadata=metadata or None
    )


def deliver(scheduler, chunks: list[tuple[str, torch.Tensor]]) -> None:
    """One message batch from the LM: ingest every chunk, then catch up."""
    scheduler.on_stream_chunk_batch(
        [(request_id, item(codes)) for request_id, codes in chunks]
    )


def queue_frames(scheduler, chunks: list[tuple[str, torch.Tensor]]) -> None:
    """Ingest without decoding, as a backlog left behind by earlier batches."""
    with scheduler.state_lock:
        for request_id, codes in chunks:
            scheduler.ingest_stream_item(request_id, item(codes))


def drain_outbox(scheduler) -> list[OutgoingMessage]:
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


def run_until_idle(scheduler) -> list[OutgoingMessage]:
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    return drain_outbox(scheduler)


def streamed_audio(messages: list[OutgoingMessage], request_id: str) -> torch.Tensor:
    return torch.cat(
        [
            decode_waveform(message.data)
            for message in messages
            if message.request_id == request_id and message.type == "stream"
        ]
    )


def result_audio(messages: list[OutgoingMessage], request_id: str) -> torch.Tensor:
    (result,) = [
        message
        for message in messages
        if message.request_id == request_id and message.type == "result"
    ]
    return decode_waveform(result.data.data)


def random_codes(frames: int, seed: int) -> torch.Tensor:
    return torch.randint(
        0, 2048, (frames, 8), generator=torch.Generator().manual_seed(seed)
    )


def whole_decode(codec: MimiCodec, codes: torch.Tensor) -> torch.Tensor:
    return codec.decode(codes.T[None])[0, 0].float().cpu()


def ramp_decode(codec: MimiCodec, codes: torch.Tensor) -> torch.Tensor:
    """One stream decoded alone, in the chunks the ramp cuts it into."""
    state = codec.init_decode_state(batch_size=1, max_chunk_frames=max(RAMP))
    parts = []
    start = 0
    while start < codes.shape[0]:
        width = RAMP[min(len(parts), len(RAMP) - 1)]
        chunk = codes[start : start + width]
        parts.append(codec.decode_step(chunk.T[None], state)[0, 0].float().cpu())
        start += chunk.shape[0]
    return torch.cat(parts)


def record_shapes(scheduler) -> list[tuple[int, int]]:
    """(rows, frames) of every decode step the scheduler runs, in order."""
    shapes: list[tuple[int, int]] = []
    original_decode_step = scheduler.decode_slots.decode_step

    def recording_decode_step(codes_BKF, slot_indices):
        shapes.append((codes_BKF.shape[0], codes_BKF.shape[-1]))
        return original_decode_step(codes_BKF, slot_indices)

    scheduler.decode_slots.decode_step = recording_decode_step
    return shapes


def run_lm_steps(
    scheduler, schedule: dict[str, tuple[int, torch.Tensor]]
) -> tuple[list[OutgoingMessage], dict[str, set[int]]]:
    """Deliver one frame per running stream per LM step, as the LM emits them.

    schedule maps a request id to its first LM step and its codes; a stream
    ends on the step of its last frame. Returns every message and the decode
    slots each stream held.
    """
    last_step = max(start + len(codes) for start, codes in schedule.values())
    messages: list[OutgoingMessage] = []
    slots: dict[str, set[int]] = {request_id: set() for request_id in schedule}
    for step in range(last_step):
        deliver(
            scheduler,
            [
                (request_id, codes[step - start : step - start + 1])
                for request_id, (start, codes) in schedule.items()
                if 0 <= step - start < len(codes)
            ],
        )
        for request_id, state in scheduler.stream_states.items():
            if state.slot_index is not None:
                slots[request_id].add(state.slot_index)
            else:
                pass
        for request_id, (start, codes) in schedule.items():
            if step == start + len(codes) - 1:
                scheduler.handle_stream_done(request_id)
            else:
                pass
        messages.extend(run_until_idle(scheduler))
    return messages, slots


def small_context_codec() -> MimiCodec:
    """Random-weight Mimi whose attention ring wraps after a few frames."""
    torch.manual_seed(0)
    codec = MimiCodec(replace(MIMI, context=8)).eval()
    with torch.no_grad():
        for parameter in codec.parameters():
            parameter.normal_(std=0.05)
        for module in codec.modules():
            if hasattr(module, "embedding_sum"):
                module.embedding_sum.normal_()
                module.cluster_usage.fill_(1.0)
    return codec


@pytest.mark.parametrize("case", ["lockstep", "staggered_past_ring_with_slot_reuse"])
def test_batched_streams_match_their_own_decodes(
    random_codec: MimiCodec, case: str
) -> None:
    if case == "lockstep":
        codec = random_codec
        schedule = {
            request_id: (0, random_codes(10, seed))
            for request_id, seed in (("a", 1), ("b", 2), ("c", 3))
        }
        num_slots = 3
        # First frames, two steady chunks, then the one-frame tails at the end.
        expected_shapes = [(3, 1), (3, 4), (3, 4), (3, 1)]
    else:
        # a and b start four LM steps apart, so their steady chunks share steps
        # at different decode offsets, past the attention ring. c starts after
        # a ends and reuses a's slot, whose ring has wrapped.
        codec = small_context_codec()
        schedule = {
            "a": (0, random_codes(14, 4)),
            "b": (4, random_codes(18, 5)),
            "c": (16, random_codes(9, 6)),
        }
        num_slots = 2
        expected_shapes = [
            (1, 1),  # step 0: a's first frame
            (1, 4),  # step 4: a's steady chunk does not wait for b
            (1, 1),  # step 4: b's first frame
            (2, 4),  # step 8
            (2, 4),  # step 12
            (1, 1),  # step 13: a's tail
            (1, 4),  # step 16: b
            (1, 1),  # step 16: c's first frame
            (2, 4),  # step 20
            (1, 1),  # step 21: b's tail
            (1, 4),  # step 24: c's last chunk
        ]
    scheduler = new_scheduler(codec, num_slots=num_slots, max_batch_size=num_slots)
    for request_id in schedule:
        start_stream(scheduler, request_id)
    shapes = record_shapes(scheduler)

    messages, slots = run_lm_steps(scheduler, schedule)

    assert shapes == expected_shapes
    assert not any(message.type == "error" for message in messages)
    for request_id, (_, codes) in schedule.items():
        streamed = streamed_audio(messages, request_id)
        # Note (wilsonzheng0327): Matrix kernels may vary with the batch shape.
        torch.testing.assert_close(
            streamed, ramp_decode(codec, codes), rtol=1e-5, atol=1e-5
        )
        assert torch.equal(result_audio(messages, request_id), streamed)
    if case == "lockstep":
        assert len(set().union(*slots.values())) == 3
    else:
        assert len(slots["a"]) == len(slots["b"]) == 1
        assert slots["c"] == slots["a"] != slots["b"]
    assert scheduler.decode_slots.active_slot_count == 0


@pytest.mark.parametrize(
    ("num_slots", "max_batch_size", "started", "backlog", "expected_steps"),
    [
        # Backlogged streams take turns, the longest-waiting first.
        (4, 2, "abcd", dict.fromkeys("abcd", 8), ["ab", "cd", "ab", "cd"]),
        # A ready width never waits for another: a's steady chunk runs
        # alone, then the first frames of b and c share a step.
        (3, 2, "a", {"a": 4, "b": 1, "c": 1}, ["a", "bc"]),
        # The step size caps a batch, not how many streams hold slots.
        (16, 8, "", dict.fromkeys("0123456789abcdef", 1), ["01234567", "89abcdef"]),
    ],
    ids=["backlog_rotates", "widths_do_not_wait", "batch_size_caps_a_step"],
)
def test_ready_streams_share_steps_by_width_in_ready_order(
    random_codec: MimiCodec,
    num_slots: int,
    max_batch_size: int,
    started: str,
    backlog: dict[str, int],
    expected_steps: list[str],
) -> None:
    scheduler = new_scheduler(
        random_codec, num_slots=num_slots, max_batch_size=max_batch_size
    )
    for request_id in backlog:
        start_stream(scheduler, request_id)
    deliver(
        scheduler,
        [
            (request_id, random_codes(1, index))
            for index, request_id in enumerate(started)
        ],
    )
    run_until_idle(scheduler)
    queue_frames(
        scheduler,
        [
            (request_id, random_codes(frames, 10 + index))
            for index, (request_id, frames) in enumerate(backlog.items())
        ],
    )

    steps = []
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
        steps.append(
            "".join(sorted({message.request_id for message in drain_outbox(scheduler)}))
        )
    assert steps == expected_steps


def test_caller_length_is_trimmed_per_row(random_codec: MimiCodec) -> None:
    lengths = {"a": 1000, "b": 5 * SAMPLES_PER_FRAME - 100, "c": 0}
    codes = {
        request_id: random_codes(5, 16 + index)
        for index, request_id in enumerate(lengths)
    }
    scheduler = new_scheduler(random_codec, num_slots=3, max_batch_size=3)
    for request_id in lengths:
        start_stream(scheduler, request_id)
    shapes = record_shapes(scheduler)

    for frame in range(5):
        scheduler.on_stream_chunk_batch(
            [
                (
                    request_id,
                    item(codes[request_id][frame : frame + 1], num_samples=length),
                )
                for request_id, length in lengths.items()
            ]
        )
    messages = run_until_idle(scheduler)

    assert shapes == [(3, 1), (3, 4)]
    for request_id, length in lengths.items():
        expected = ramp_decode(random_codec, codes[request_id])[: length or None]
        torch.testing.assert_close(
            streamed_audio(messages, request_id), expected, rtol=1e-5, atol=1e-5
        )
        for part in scheduler.stream_states[request_id].audio_parts:
            # Each saved part owns its own samples, not the batch's host buffer.
            assert part.untyped_storage().nbytes() == part.numel() * part.element_size()


@pytest.mark.parametrize("message_frames", [1, 2, 16])
def test_sustained_multi_chunk_batches_do_not_grow_the_backlog(
    random_codec: MimiCodec, message_frames: int
) -> None:
    # Each call stands for one message batch while the inbox never drains, so
    # the ready-work branch never runs between calls. Every batch brings 16
    # frames per stream in messages of message_frames frames; one message can
    # hold several chunks, so catching up must count frames, not messages.
    codec = random_codec
    rounds = 3
    scheduler = new_scheduler(codec, num_slots=2, max_batch_size=8)
    codes = {"a": random_codes(16 * rounds, 71), "b": random_codes(16 * rounds, 72)}
    for request_id in codes:
        start_stream(scheduler, request_id)

    pending = []
    for round_index in range(rounds):
        scheduler.on_stream_chunk_batch(
            [
                (request_id, item(frames[start : start + message_frames]))
                for request_id, frames in codes.items()
                for start in range(
                    16 * round_index, 16 * (round_index + 1), message_frames
                )
            ]
        )
        assert not scheduler.has_ready_work()
        pending.append(
            {
                request_id: state.pending_frames
                for request_id, state in scheduler.stream_states.items()
            }
        )
    # Only the frames short of the next four-frame chunk wait.
    assert pending == [{"a": 3, "b": 3}] * rounds

    for request_id in codes:
        scheduler.handle_stream_done(request_id)
    messages = run_until_idle(scheduler)
    for request_id, frames in codes.items():
        expected = whole_decode(codec, frames)
        torch.testing.assert_close(
            streamed_audio(messages, request_id), expected, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            result_audio(messages, request_id), expected, rtol=1e-5, atol=1e-5
        )
    assert scheduler.decode_slots.active_slot_count == 0


def test_ramp_collects_queued_frames_before_selecting_a_decode_batch(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=8, max_batch_size=8)
    codes = {str(index): random_codes(5, index + 100) for index in range(8)}
    for request_id in codes:
        start_stream(scheduler, request_id)
    deliver(
        scheduler, [(request_id, frames[:1]) for request_id, frames in codes.items()]
    )
    shapes = record_shapes(scheduler)
    for request_id, frames in codes.items():
        for frame in range(1, 5):
            scheduler.inbox.put(
                IncomingMessage(
                    request_id=request_id,
                    type="stream_chunk",
                    data=StreamItem(
                        chunk_id=frame, from_stage="lm", data=frames[frame : frame + 1]
                    ),
                )
            )
    loop = asyncio.new_event_loop()
    try:
        scheduler.handle_message(scheduler.get_batch_message(), loop)
    finally:
        loop.close()
    assert shapes == [(8, 4)]
    assert scheduler.inbox.empty()
    assert {message.request_id for message in drain_outbox(scheduler)} == set(codes)


def test_start_loop_decodes_before_a_continuously_refilled_inbox_drains(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=2, max_batch_size=2)
    start_stream(scheduler, "a")
    frame_codes = random_codes(1, 45)
    processed_message_count = 0
    decode_message_counts: list[int] = []
    original_handle_message = scheduler.handle_message
    original_decode_step = scheduler.decode_slots.decode_step

    def make_message(message_index: int) -> IncomingMessage:
        return IncomingMessage(
            request_id="a",
            type="stream_chunk",
            data=StreamItem(chunk_id=message_index, data=frame_codes, from_stage="lm"),
        )

    def handle_message(
        message: IncomingMessage, loop: asyncio.AbstractEventLoop
    ) -> None:
        nonlocal processed_message_count
        original_handle_message(message, loop)
        processed_message_count += 1
        if processed_message_count < 30:
            scheduler.inbox.put(make_message(processed_message_count))
        else:
            scheduler.stop()

    def decode_step(codes_BKF: torch.Tensor, slot_indices: list[int]) -> torch.Tensor:
        decode_message_counts.append(processed_message_count)
        return original_decode_step(codes_BKF, slot_indices)

    scheduler.handle_message = handle_message
    scheduler.decode_slots.decode_step = decode_step
    scheduler.inbox.put(make_message(0))
    scheduler.start()

    # Each chunk decodes while the message that completes it is handled: the
    # first frame with message 0, then every four frames.
    assert decode_message_counts == list(range(0, 30, 4))
    assert processed_message_count == 30


def test_new_arrivals_cannot_starve_a_backlogged_stream(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=3, max_batch_size=2)
    long_codes = random_codes(21, 61)
    start_stream(scheduler, "long")
    deliver(scheduler, [("long", long_codes[:1])])
    drain_outbox(scheduler)
    queue_frames(scheduler, [("long", long_codes[1:])])

    long_decode_steps: list[int] = []
    short_waits: dict[str, int] = {}
    for step in range(10):
        # A fresh request arrives before every step. Its one-frame first chunk
        # never shares a step with the backlog's four-frame chunks, so each
        # step serves one side, and an arrival must not keep winning.
        short_id = f"short-{step}"
        start_stream(scheduler, short_id)
        queue_frames(scheduler, [(short_id, random_codes(1, 100 + step))])
        short_waits[short_id] = step
        scheduler.run_ready_step()
        for message in drain_outbox(scheduler):
            assert message.type == "stream"
            if message.request_id == "long":
                long_decode_steps.append(step)
            else:
                short_waits[message.request_id] = step - short_waits[message.request_id]
                scheduler.handle_stream_done(message.request_id)
                (result,) = drain_outbox(scheduler)
                assert result.type == "result"

    # The backlogged stream is served every other step despite the churn.
    assert long_decode_steps == [0, 2, 4, 6, 8]
    assert scheduler.stream_states["long"].pending_frames == 0
    # Arrivals still get their first step within one step of arriving.
    assert all(wait <= 1 for wait in short_waits.values())


def test_waiting_streams_take_freed_slots_in_ready_order(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=1, max_batch_size=2)
    codes = {request_id: random_codes(2, seed) for seed, request_id in enumerate("abc")}
    for request_id in ("a", "empty", "b", "c"):
        start_stream(scheduler, request_id)
    # c becomes ready before b, although its terminal payload arrived later.
    for request_id in ("a", "c", "b"):
        queue_frames(
            scheduler, [(request_id, chunk) for chunk in codes[request_id].split(1)]
        )
        scheduler.handle_stream_done(request_id)
    assert scheduler.stream_states["c"].slot_index is None
    assert scheduler.stream_states["b"].slot_index is None

    messages = run_until_idle(scheduler)
    assert [message.request_id for message in messages if message.type == "stream"] == [
        "a",
        "a",
        "c",
        "c",
        "b",
        "b",
    ]
    assert [message.request_id for message in messages if message.type == "result"] == [
        "a",
        "c",
        "b",
    ]
    for request_id, request_codes in codes.items():
        torch.testing.assert_close(
            result_audio(messages, request_id),
            whole_decode(random_codec, request_codes),
            rtol=1e-5,
            atol=1e-5,
        )
    assert scheduler.decode_slots.active_slot_count == 0
    assert scheduler.stream_states["empty"].slot_index is None
    # A previously empty waiter must become runnable when its first chunk arrives.
    deliver(scheduler, [("empty", codes["a"][:1])])
    assert drain_outbox(scheduler)
    scheduler.handle_stream_done("empty")
    assert scheduler.stream_states == {}
    assert scheduler.decode_slots.active_slot_count == 0


@pytest.mark.parametrize("release_reason", ["done", "abort", "error"])
def test_waiters_survive_slot_owner_exit_and_waiting_cancellation(
    random_codec: MimiCodec, release_reason: str, monkeypatch
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=1, max_batch_size=2)
    for request_id in ("owner", "cancelled", "survivor"):
        start_stream(scheduler, request_id)
    chunk = random_codes(1, 104)
    deliver(scheduler, [("owner", chunk)])
    drain_outbox(scheduler)
    deliver(scheduler, [("cancelled", chunk), ("survivor", chunk)])
    assert drain_outbox(scheduler) == []
    scheduler.abort("cancelled")
    scheduler.handle_stream_done("survivor")
    assert scheduler.decode_slots.active_slot_count == 1

    if release_reason == "done":
        scheduler.handle_stream_done("owner")
    elif release_reason == "abort":
        scheduler.abort("owner")
    else:
        original_decode_step = scheduler.decode_slots.decode_step

        def fail_once(codes_BKF, slot_indices):
            monkeypatch.setattr(
                scheduler.decode_slots, "decode_step", original_decode_step
            )
            raise RuntimeError("owner failed")

        monkeypatch.setattr(scheduler.decode_slots, "decode_step", fail_once)
        # A full steady chunk makes the owner ready, and its step fails.
        deliver(scheduler, [("owner", random_codes(4, 105))])
    messages = run_until_idle(scheduler)
    assert not any(message.request_id == "cancelled" for message in messages)
    assert [message.request_id for message in messages if message.type == "error"] == (
        ["owner"] if release_reason == "error" else []
    )
    torch.testing.assert_close(
        result_audio(messages, "survivor"),
        whole_decode(random_codec, chunk),
        rtol=1e-5,
        atol=1e-5,
    )
    assert scheduler.stream_states == {}
    assert scheduler.decode_slots.active_slot_count == 0


def test_waiting_stream_finishes_before_its_terminal_payload(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=1, max_batch_size=2)
    start_stream(scheduler, "owner")
    chunk = random_codes(1, 105)
    deliver(scheduler, [("owner", chunk)])
    drain_outbox(scheduler)
    deliver(scheduler, [("waiting", chunk)])
    scheduler.handle_stream_done("waiting")
    scheduler.handle_stream_done("owner")
    scheduler.run_ready_step()
    messages = drain_outbox(scheduler)
    assert [(message.request_id, message.type) for message in messages] == [
        ("owner", "result"),
        ("waiting", "stream"),
    ]
    assert scheduler.decode_slots.active_slot_count == 0
    payload = StagePayload(
        "waiting",
        request=OmniRequest(inputs={}),
        data=PersonaPlexState(codes=chunk).to_dict(),
    )
    scheduler.handle_streaming_new_request("waiting", payload)
    torch.testing.assert_close(
        result_audio(drain_outbox(scheduler), "waiting"),
        whole_decode(random_codec, chunk),
        rtol=1e-5,
        atol=1e-5,
    )
    assert scheduler.stream_states == {}
    assert scheduler.pending_done == set()
    assert scheduler.decode_slots.active_slot_count == 0


def test_a_failing_step_aborts_only_its_participants(random_codec: MimiCodec) -> None:
    scheduler = new_scheduler(random_codec, num_slots=3, max_batch_size=3)
    codes = random_codes(9, 33)
    for request_id in ("a", "b", "c"):
        start_stream(scheduler, request_id)
    deliver(scheduler, [("c", codes[:1])])
    original_decode_step = scheduler.decode_slots.decode_step

    def fail_two_rows(codes_BKF, slot_indices):
        if codes_BKF.shape[0] == 2:
            raise RuntimeError("boom")
        return original_decode_step(codes_BKF, slot_indices)

    scheduler.decode_slots.decode_step = fail_two_rows
    # a and b share a first-frame step, which fails; c's steady chunk is
    # another width and decodes in its own step.
    deliver(scheduler, [("a", codes[:1]), ("b", codes[:1]), ("c", codes[1:5])])
    messages = run_until_idle(scheduler)
    assert sorted(m.request_id for m in messages if m.type == "error") == ["a", "b"]
    assert {m.request_id for m in messages if m.type == "stream"} == {"c"}
    assert set(scheduler.stream_states) == {"c"}
    assert scheduler.decode_slots.active_slot_count == 1

    # The stage keeps serving: c finishes and a new stream takes a freed slot.
    scheduler.decode_slots.decode_step = original_decode_step
    start_stream(scheduler, "d")
    deliver(scheduler, [("c", codes[5:9]), ("d", codes[:1])])
    scheduler.handle_stream_done("c")
    messages += run_until_idle(scheduler)
    torch.testing.assert_close(
        result_audio(messages, "c"),
        whole_decode(random_codec, codes),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        streamed_audio(messages, "d"),
        whole_decode(random_codec, codes[:1]),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("bad_code", [-1, 2048])
def test_out_of_range_codes_fail_only_their_request(
    random_codec: MimiCodec, bad_code: int
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=2, max_batch_size=2)
    codes = random_codes(5, 35)
    bad_codes = codes[:1].clone()
    bad_codes[0, 3] = bad_code
    for request_id in ("good", "bad"):
        start_stream(scheduler, request_id)
    shapes = record_shapes(scheduler)

    # Both first frames arrive in one message batch, so they would share a step.
    deliver(scheduler, [("good", codes[:1]), ("bad", bad_codes)])
    deliver(scheduler, [("good", codes[1:5])])
    scheduler.handle_stream_done("good")
    messages = run_until_idle(scheduler)

    assert [m.request_id for m in messages if m.type == "error"] == ["bad"]
    assert shapes == [(1, 1), (1, 4)]
    torch.testing.assert_close(
        result_audio(messages, "good"),
        whole_decode(random_codec, codes),
        rtol=1e-5,
        atol=1e-5,
    )
    assert scheduler.stream_states == {}
    assert scheduler.decode_slots.active_slot_count == 0


def test_shutdown_and_late_chunks_release_every_slot(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=1, max_batch_size=2)
    codes = random_codes(4, 133)
    deliver(scheduler, [("a", codes), ("b", codes)])
    assert scheduler.decode_slots.active_slot_count == 1
    assert scheduler.stream_states["b"].slot_index is None
    scheduler.stop()
    assert scheduler.stream_states == {}
    assert scheduler.decode_slots.active_slot_count == 0

    scheduler = new_scheduler(random_codec, num_slots=1, max_batch_size=2)
    start_stream(scheduler, "done")
    deliver(scheduler, [("done", codes[:1])])
    scheduler.handle_stream_done("done")
    drain_outbox(scheduler)
    deliver(scheduler, [("done", codes[1:])])
    assert scheduler.stream_states == {}
    assert scheduler.decode_slots.active_slot_count == 0


class CountingGraph:
    def __init__(self, graph) -> None:
        self.graph = graph
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self.graph.replay()


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.enable_mimi_decode_graph(),
    reason="the platform does not capture Mimi decode graphs",
)
def test_graph_replay_matches_eager_decode_through_the_scheduler(
    random_codec: MimiCodec,
) -> None:
    device = torch.device(current_platform.device_type)
    codec = random_codec.to(device)
    # a and b run past the attention ring, eight frames apart, so their steady
    # chunks share steps at different offsets. A third stream arrives while
    # both slots are taken, ends before its payload, and decodes once b's
    # slot frees.
    codes = {
        "a": random_codes(160, 130),
        "b": random_codes(151, 131),
        "waiting": random_codes(10, 132),
    }
    starts = {"a": 0, "b": 8}
    runs = {}
    for max_graph_frames in (4, 0):
        scheduler = new_scheduler(
            codec, num_slots=2, max_batch_size=2, max_graph_frames=max_graph_frames
        )
        for captured_decode in scheduler.decode_slots.graphs.values():
            captured_decode.graph = CountingGraph(captured_decode.graph)
        shapes = record_shapes(scheduler)
        for request_id in starts:
            start_stream(scheduler, request_id)
        messages = []
        for step in range(160):
            chunks = [
                (request_id, codes[request_id][step - start : step - start + 1])
                for request_id, start in starts.items()
                if 0 <= step - start < len(codes[request_id])
            ]
            if step == 12:
                chunks.append(("waiting", codes["waiting"]))
            else:
                pass
            deliver(scheduler, chunks)
            if step == 12:
                assert scheduler.stream_states["waiting"].slot_index is None
                assert scheduler.stream_states["waiting"].pending_frames == 10
                scheduler.handle_stream_done("waiting")
            else:
                pass
            for request_id, start in starts.items():
                if step == start + len(codes[request_id]) - 1:
                    scheduler.handle_stream_done(request_id)
                else:
                    pass
            messages.extend(run_until_idle(scheduler))
        # The waiting stream's audio is out, but its result waits for the payload.
        assert not any(
            message.request_id == "waiting" and message.type == "result"
            for message in messages
        )
        scheduler.handle_streaming_new_request(
            "waiting",
            StagePayload(
                "waiting",
                request=OmniRequest(inputs={}),
                data=PersonaPlexState(codes=codes["waiting"]).to_dict(),
            ),
        )
        messages.extend(drain_outbox(scheduler))
        assert scheduler.stream_states == {}
        assert scheduler.decode_slots.active_slot_count == 0
        replays = {
            key: captured_decode.graph.replays
            for key, captured_decode in scheduler.decode_slots.graphs.items()
        }
        runs[max_graph_frames] = (messages, shapes, replays)

    (graphed, shapes, replays), (eager, eager_shapes, eager_replays) = (
        runs[4],
        runs[0],
    )
    assert set(replays) == {
        (rows, frames) for rows in (1, 2) for frames in (1, 2, 3, 4)
    }
    assert eager_replays == {}
    assert shapes == eager_shapes
    assert (2, 4) in shapes and {frames for _, frames in shapes} == {1, 2, 3, 4}
    # Every step, including tails and the waiting stream's chunks, replays a graph.
    assert sum(replays.values()) == len(shapes)
    assert [(m.request_id, m.type) for m in graphed] == [
        (m.request_id, m.type) for m in eager
    ]
    for request_id, frames in codes.items():
        streamed = streamed_audio(graphed, request_id)
        assert torch.equal(streamed, streamed_audio(eager, request_id))
        assert torch.equal(result_audio(graphed, request_id), streamed)
        # Compare the same fixed codes to a whole-utterance decode, independent of scheduling.
        reference = codec.decode(frames.T[None].to(device))[0, 0].float().cpu()
        assert streamed.shape == reference.shape
        assert (streamed - reference).abs().max() <= 0.05 * reference.abs().max()
