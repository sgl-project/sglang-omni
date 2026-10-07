# SPDX-License-Identifier: Apache-2.0
"""Continuous cross-request batching in the PersonaPlex codec stage."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.admission import QueueFullError
from sglang_omni.models.personaplex.architecture import (
    MIMI,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)
from sglang_omni.models.personaplex.code2wav_stream import PersonaPlexCode2WavScheduler
from sglang_omni.models.personaplex.components.mimi import MimiCodec, MimiDecodeState
from sglang_omni.models.personaplex.config import (
    CODE2WAV_STAGE,
    PersonaPlexPipelineConfig,
)
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage


def decode_waveform(payload: dict) -> torch.Tensor:
    assert payload["sample_rate"] == SAMPLE_RATE
    return torch.from_numpy(
        np.frombuffer(payload["audio_waveform"], dtype=np.float32).copy()
    )


def start_stream(scheduler, request_id: str) -> None:
    payload = StagePayload(
        request_id, request=OmniRequest(inputs={}), data=PersonaPlexState().to_dict()
    )
    scheduler.stream_payloads[request_id] = payload
    scheduler.on_streaming_new_request(request_id, payload)


def feed_serial(scheduler, request_id: str, chunk: torch.Tensor) -> torch.Tensor:
    (message,) = scheduler.on_stream_chunk(
        request_id, SimpleNamespace(data=chunk, metadata=None)
    )
    return decode_waveform(message.data)


def feed_native(
    codec: MimiCodec, state: MimiDecodeState, chunk: torch.Tensor
) -> torch.Tensor:
    return codec.decode_step(chunk.T[None], state)[0, 0].float().cpu()


def feed_batched(
    scheduler, chunks: list[tuple[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    scheduler.on_stream_chunk_batch(
        [
            (request_id, SimpleNamespace(data=chunk, metadata=None))
            for request_id, chunk in chunks
        ]
    )
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    waveforms: dict[str, torch.Tensor] = {}
    while not scheduler.outbox.empty():
        message = scheduler.outbox.get_nowait()
        assert message.type == "stream"
        waveforms[message.request_id] = decode_waveform(message.data)
    return waveforms


def drain_outbox(scheduler) -> list[OutgoingMessage]:
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


def random_codes(frames: int, seed: int) -> torch.Tensor:
    return torch.randint(
        0, 2048, (frames, 8), generator=torch.Generator().manual_seed(seed)
    )


def record_decode_steps(scheduler) -> list[int]:
    """Batch size of every arena step the scheduler fires, in order."""
    calls: list[int] = []
    original_decode_step = scheduler.decode_arena.decode_step

    def recording_decode_step(batched_codes, *, slot_indices):
        calls.append(batched_codes.shape[0])
        return original_decode_step(batched_codes, slot_indices=slot_indices)

    scheduler.decode_arena.decode_step = recording_decode_step
    return calls


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


def test_batched_lockstep_matches_serial(random_codec: MimiCodec) -> None:
    codec = random_codec
    frames = 6
    codes = {
        request_id: random_codes(frames, seed)
        for request_id, seed in (("a", 1), ("b", 2), ("c", 3))
    }
    native_states = {request_id: codec.init_decode_state() for request_id in codes}
    batched = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=4,
        stream_slots=4,
    )
    for request_id in codes:
        start_stream(batched, request_id)
    batch_sizes = record_decode_steps(batched)

    for frame in range(frames):
        expected = {
            request_id: feed_native(
                codec,
                native_states[request_id],
                codes[request_id][frame : frame + 1],
            )
            for request_id in codes
        }
        streamed = feed_batched(
            batched,
            [
                (request_id, codes[request_id][frame : frame + 1])
                for request_id in codes
            ],
        )
        assert set(streamed) == set(codes)
        for request_id in codes:
            # Note (wilsonzheng0327): Matrix kernels may vary with the batch shape.
            torch.testing.assert_close(
                streamed[request_id], expected[request_id], rtol=1e-5, atol=1e-5
            )

    # Every frame of the three lockstep requests decodes in a single step.
    assert batch_sizes == [3] * frames
    for request_id in codes:
        (result,) = batched.on_stream_done(request_id)
        whole = decode_waveform(result.data.data)
        torch.testing.assert_close(
            whole,
            codec.decode(codes[request_id].T[None])[0, 0].cpu(),
            atol=1e-5,
            rtol=1e-5,
        )


def test_survivors_stay_exact_and_batch_after_a_release(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec
    frames = 5
    codes = {
        request_id: random_codes(frames, seed)
        for request_id, seed in (("a", 4), ("b", 5), ("c", 6))
    }
    batched = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=3,
        stream_slots=3,
    )
    serial = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=1,
        stream_slots=8,
    )
    for request_id in codes:
        start_stream(batched, request_id)
    batch_sizes = record_decode_steps(batched)

    for frame in range(2):
        feed_batched(
            batched,
            [
                (request_id, codes[request_id][frame : frame + 1])
                for request_id in codes
            ],
        )
        for request_id in codes:
            feed_serial(serial, request_id, codes[request_id][frame : frame + 1])

    batched.clear_stream_state("b")
    streamed = feed_batched(
        batched,
        [(request_id, codes[request_id][2 : 2 + 1]) for request_id in ("a", "c")],
    )
    torch.testing.assert_close(
        streamed["a"], feed_serial(serial, "a", codes["a"][2:3]), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        streamed["c"], feed_serial(serial, "c", codes["c"][2:3]), rtol=1e-5, atol=1e-5
    )
    # Non-contiguous survivor slots still share one step after the release.
    assert batch_sizes == [3, 3, 2]

    # A late request joins the freed capacity and decodes alone at offset zero.
    start_stream(batched, "d")
    codes_d = random_codes(frames, 7)
    streamed = feed_batched(batched, [("d", codes_d[0:1])])
    torch.testing.assert_close(
        streamed["d"], feed_serial(serial, "d", codes_d[0:1]), rtol=1e-5, atol=1e-5
    )


def test_released_slot_is_reused_without_moving_survivors(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec
    scheduler = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=2,
        stream_slots=2,
    )
    for request_id in ("a", "b"):
        start_stream(scheduler, request_id)
    slot_a = scheduler.stream_states["a"].slot_index
    slot_b = scheduler.stream_states["b"].slot_index
    assert slot_a is not None and slot_b is not None

    scheduler.clear_stream_state("a")
    assert scheduler.stream_states["b"].slot_index == slot_b
    start_stream(scheduler, "c")
    assert scheduler.stream_states["c"].slot_index == slot_a


def test_ring_wraparound_then_slot_reuse_stays_exact() -> None:
    codec = small_context_codec()  # ring capacity 8 latent frames = 4 codec frames
    frames = 6
    batched = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=2,
        stream_slots=2,
    )
    serial = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=1,
        stream_slots=8,
    )
    codes_a = random_codes(frames, 23)
    start_stream(batched, "a")
    for frame in range(frames):
        feed_batched(batched, [("a", codes_a[frame : frame + 1])])
    batched.clear_stream_state("a")

    # The reused slot still holds the wrapped ring of its previous owner.
    codes_b = random_codes(frames, 24)
    start_stream(batched, "b")
    for frame in range(frames):
        streamed = feed_batched(batched, [("b", codes_b[frame : frame + 1])])
        torch.testing.assert_close(
            streamed["b"],
            feed_serial(serial, "b", codes_b[frame : frame + 1]),
            rtol=1e-5,
            atol=1e-5,
        )


def test_different_offsets_batch_after_attention_ring_wraparound() -> None:
    codec = small_context_codec()
    batched = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=2,
        stream_slots=2,
    )
    native_states = {request_id: codec.init_decode_state() for request_id in ("a", "b")}
    codes_a = random_codes(6, 25)
    codes_b = random_codes(2, 26)
    for request_id in ("a", "b"):
        start_stream(batched, request_id)
    batch_sizes = record_decode_steps(batched)

    for frame in range(5):
        streamed = feed_batched(batched, [("a", codes_a[frame : frame + 1])])
        torch.testing.assert_close(
            streamed["a"],
            feed_native(codec, native_states["a"], codes_a[frame : frame + 1]),
            rtol=1e-5,
            atol=1e-5,
        )
    streamed = feed_batched(batched, [("b", codes_b[0:1])])
    torch.testing.assert_close(
        streamed["b"],
        feed_native(codec, native_states["b"], codes_b[0:1]),
        rtol=1e-5,
        atol=1e-5,
    )

    streamed = feed_batched(
        batched,
        [("a", codes_a[5:6]), ("b", codes_b[1:2])],
    )
    for request_id, chunk in (("a", codes_a[5:6]), ("b", codes_b[1:2])):
        torch.testing.assert_close(
            streamed[request_id],
            feed_native(codec, native_states[request_id], chunk),
            rtol=1e-5,
            atol=1e-5,
        )
    assert batch_sizes == [1] * 6 + [2]


def test_stream_capacity_rejects_before_buffering_and_reuses_aborted_slot(
    random_codec: MimiCodec,
) -> None:
    scheduler = PersonaPlexCode2WavScheduler(
        random_codec,
        compute_fn=lambda payload: payload,
        max_batch_size=2,
        stream_slots=2,
    )
    start_stream(scheduler, "a")
    start_stream(scheduler, "b")

    with pytest.raises(QueueFullError):
        start_stream(scheduler, "c")
    assert "c" not in scheduler.stream_states
    assert scheduler.active_slot_count == 2

    released_slot_index = scheduler.stream_states["a"].slot_index
    scheduler.abort("a")
    start_stream(scheduler, "c")
    assert scheduler.stream_states["c"].slot_index == released_slot_index
    assert scheduler.active_slot_count == 2

    finished_slot_index = scheduler.stream_states["b"].slot_index
    assert scheduler.on_stream_done("b") is not None
    assert scheduler.active_slot_count == 1
    start_stream(scheduler, "d")
    assert scheduler.stream_states["d"].slot_index == finished_slot_index


def test_different_offsets_share_steps_but_different_widths_do_not(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec
    codes = {
        request_id: random_codes(8, seed) for request_id, seed in (("a", 11), ("b", 12))
    }
    batched = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=4,
        stream_slots=4,
    )
    serial = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=1,
        stream_slots=8,
    )
    for request_id in codes:
        start_stream(batched, request_id)
    batch_sizes = record_decode_steps(batched)

    rounds = [
        [("a", codes["a"][0:2])],
        [("a", codes["a"][2:4]), ("b", codes["b"][0:2])],
        [("a", codes["a"][4:5]), ("b", codes["b"][2:4])],
        [("a", codes["a"][5:6]), ("b", codes["b"][4:5])],
        [("a", codes["a"][6:7]), ("b", codes["b"][5:6])],
        [("a", codes["a"][7:8]), ("b", codes["b"][6:7])],
        [("b", codes["b"][7:8])],
    ]
    expected_batch_sizes = [1, 1, 1, 1, 1, 2, 2, 2, 1]
    for round_chunks in rounds:
        streamed = feed_batched(batched, round_chunks)
        for request_id, chunk in round_chunks:
            torch.testing.assert_close(
                streamed[request_id],
                feed_serial(serial, request_id, chunk),
                rtol=1e-5,
                atol=1e-5,
            )
    assert batch_sizes == expected_batch_sizes


def test_per_request_caller_length_trimmed_in_batch(random_codec: MimiCodec) -> None:
    codec = random_codec
    frames = 3
    samples = frames * SAMPLES_PER_FRAME
    scheduler = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=2,
        stream_slots=2,
    )
    for request_id in ("a", "b"):
        start_stream(scheduler, request_id)
    lengths = {"a": samples - 100, "b": samples - SAMPLES_PER_FRAME}
    codes = {
        request_id: random_codes(frames, seed)
        for request_id, seed in (("a", 16), ("b", 17))
    }

    totals = {"a": 0, "b": 0}
    for frame in range(frames):
        scheduler.on_stream_chunk_batch(
            [
                (
                    request_id,
                    SimpleNamespace(
                        data=codes[request_id][frame : frame + 1],
                        metadata={"num_samples": lengths[request_id]},
                    ),
                )
                for request_id in ("a", "b")
            ]
        )
        while scheduler.has_ready_work():
            scheduler.run_ready_step()
        while not scheduler.outbox.empty():
            message = scheduler.outbox.get_nowait()
            totals[message.request_id] += decode_waveform(message.data).shape[-1]
    assert totals == lengths


def test_a_failing_step_aborts_exactly_its_participants(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec
    codes = {
        request_id: random_codes(4, seed)
        for request_id, seed in (("a", 31), ("b", 32), ("c", 33))
    }
    batched = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=4,
        stream_slots=4,
    )
    serial = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=1,
        stream_slots=8,
    )
    for request_id in codes:
        start_stream(batched, request_id)
    # c runs one frame ahead and therefore belongs to the steady-state group.
    feed_batched(batched, [("c", codes["c"][0:1])])
    feed_serial(serial, "c", codes["c"][0:1])

    original_decode_step = batched.decode_arena.decode_step

    def fail_batch_of_two(batched_codes, *, slot_indices):
        if batched_codes.shape[0] == 2:
            raise RuntimeError("boom")
        else:
            pass
        return original_decode_step(batched_codes, slot_indices=slot_indices)

    batched.decode_arena.decode_step = fail_batch_of_two
    batched.on_stream_chunk_batch(
        [
            (request_id, SimpleNamespace(data=codes[request_id][1:2], metadata=None))
            for request_id in ("a", "b", "c")
        ]
    )
    while batched.has_ready_work():
        batched.run_ready_step()
    batched.decode_arena.decode_step = original_decode_step
    # c's chunk decoded fine inside the failed batch's sibling group.
    feed_serial(serial, "c", codes["c"][1:2])

    messages = drain_outbox(batched)
    errors = {m.request_id for m in messages if m.type == "error"}
    streams = {m.request_id for m in messages if m.type == "stream"}
    assert errors == {"a", "b"}
    assert streams == {"c"}
    assert "a" not in batched.stream_states
    assert "b" not in batched.stream_states
    assert batched.active_slot_count == 1

    # The survivor keeps decoding exactly; freed slots serve fresh requests.
    start_stream(batched, "d")
    codes_d = random_codes(4, 34)
    for frame in range(2, 4):
        streamed = feed_batched(
            batched,
            [
                ("c", codes["c"][frame : frame + 1]),
                ("d", codes_d[frame - 2 : frame - 1]),
            ],
        )
        torch.testing.assert_close(
            streamed["c"],
            feed_serial(serial, "c", codes["c"][frame : frame + 1]),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            streamed["d"],
            feed_serial(serial, "d", codes_d[frame - 2 : frame - 1]),
            rtol=1e-5,
            atol=1e-5,
        )


def test_a_failing_batch_aborts_every_participant(random_codec: MimiCodec) -> None:
    codec = random_codec
    codes = {
        request_id: random_codes(2, seed) for request_id, seed in (("a", 41), ("b", 42))
    }
    batched = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=4,
        stream_slots=4,
    )
    for request_id in codes:
        start_stream(batched, request_id)

    def always_fail(batched_codes, *, slot_indices):
        raise RuntimeError("boom")

    batched.decode_arena.decode_step = always_fail
    batched.on_stream_chunk_batch(
        [
            (request_id, SimpleNamespace(data=codes[request_id][0:1], metadata=None))
            for request_id in codes
        ]
    )
    batched.run_ready_step()
    messages = drain_outbox(batched)
    assert {m.request_id for m in messages if m.type == "error"} == {"a", "b"}
    assert batched.active_slot_count == 0
    assert batched.stream_states == {}

    # The scheduler keeps serving after the failure.
    start_stream(batched, "e")
    codes_e = random_codes(2, 43)
    del (
        batched.decode_arena.decode_step
    )  # drop the instance attribute, back to the class method
    streamed = feed_batched(batched, [("e", codes_e[0:1])])
    reference = PersonaPlexCode2WavScheduler(
        codec,
        compute_fn=lambda payload: payload,
        max_batch_size=1,
        stream_slots=8,
    )
    torch.testing.assert_close(
        streamed["e"], feed_serial(reference, "e", codes_e[0:1]), rtol=1e-5, atol=1e-5
    )


def test_multiple_chunks_for_one_request_are_drained_in_order(
    random_codec: MimiCodec,
) -> None:
    scheduler = PersonaPlexCode2WavScheduler(
        random_codec,
        compute_fn=lambda payload: payload,
        max_batch_size=2,
        stream_slots=2,
    )
    start_stream(scheduler, "a")
    codes = random_codes(2, 44)

    scheduler.on_stream_chunk_batch(
        [
            ("a", SimpleNamespace(data=codes[0:1], metadata=None)),
            ("a", SimpleNamespace(data=codes[1:2], metadata=None)),
        ]
    )
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    streamed = [decode_waveform(message.data) for message in drain_outbox(scheduler)]
    expected = random_codec.decode(codes.T[None])[0, 0].cpu()
    torch.testing.assert_close(torch.cat(streamed), expected, rtol=1e-5, atol=1e-5)


def test_backlogged_requests_rotate_between_decode_steps(
    random_codec: MimiCodec,
) -> None:
    scheduler = PersonaPlexCode2WavScheduler(
        random_codec,
        compute_fn=lambda payload: payload,
        max_batch_size=2,
        stream_slots=4,
    )
    request_ids = ["a", "b", "c", "d"]
    codes = {
        request_id: random_codes(3, seed)
        for request_id, seed in zip(request_ids, range(51, 55), strict=True)
    }
    for request_id in request_ids:
        start_stream(scheduler, request_id)
    feed_batched(
        scheduler,
        [(request_id, codes[request_id][0:1]) for request_id in request_ids],
    )

    with scheduler.state_lock:
        for request_id in request_ids:
            for frame in (1, 2):
                scheduler.ingest_chunk(
                    request_id,
                    SimpleNamespace(
                        data=codes[request_id][frame : frame + 1], metadata=None
                    ),
                )
    for expected_request_ids in ({"a", "b"}, {"c", "d"}, {"a", "b"}, {"c", "d"}):
        scheduler.run_ready_step()
        assert {
            message.request_id for message in drain_outbox(scheduler)
        } == expected_request_ids


def test_start_loop_decodes_before_a_continuously_refilled_inbox_drains(
    random_codec: MimiCodec,
) -> None:
    scheduler = PersonaPlexCode2WavScheduler(
        random_codec,
        compute_fn=lambda payload: payload,
        max_batch_size=2,
        stream_slots=2,
    )
    start_stream(scheduler, "a")
    frame_codes = random_codes(1, 45)
    processed_message_count = 0
    decoded_step_count = 0
    first_decode_message_count: list[int] = []
    original_handle_message = scheduler.handle_message
    original_decode_step = scheduler.decode_arena.decode_step

    def make_message(message_index: int) -> IncomingMessage:
        return IncomingMessage(
            request_id="a",
            type="stream_chunk",
            data=StreamItem(
                chunk_id=message_index,
                data=frame_codes,
                from_stage="lm",
            ),
        )

    def handle_message(
        message: IncomingMessage, loop: asyncio.AbstractEventLoop
    ) -> None:
        nonlocal processed_message_count
        original_handle_message(message, loop)
        processed_message_count += 1
        if processed_message_count < 30:
            scheduler.inbox.put(make_message(processed_message_count))
        elif decoded_step_count == 30:
            scheduler.stop()
        else:
            pass

    def decode_step(
        batched_codes: torch.Tensor, *, slot_indices: list[int]
    ) -> torch.Tensor:
        nonlocal decoded_step_count
        if not first_decode_message_count:
            first_decode_message_count.append(processed_message_count)
        else:
            pass
        waveform = original_decode_step(batched_codes, slot_indices=slot_indices)
        decoded_step_count += 1
        return waveform

    scheduler.handle_message = handle_message
    scheduler.decode_arena.decode_step = decode_step
    scheduler.inbox.put(make_message(0))
    scheduler.start()

    assert first_decode_message_count == [0]
    assert processed_message_count == 30
    assert decoded_step_count == 30


def test_ready_buckets_rotate_without_waiting(random_codec: MimiCodec) -> None:
    scheduler = PersonaPlexCode2WavScheduler(
        random_codec,
        compute_fn=lambda payload: payload,
        max_batch_size=2,
        stream_slots=3,
    )
    for request_id in ("a", "b", "c"):
        start_stream(scheduler, request_id)

    with scheduler.state_lock:
        scheduler.ingest_chunk(
            "a", SimpleNamespace(data=random_codes(2, 45), metadata=None)
        )
        for request_id, seed in (("b", 46), ("c", 47)):
            scheduler.ingest_chunk(
                request_id,
                SimpleNamespace(data=random_codes(1, seed), metadata=None),
            )
    assert scheduler.has_ready_work()
    scheduler.run_ready_step()
    assert {message.request_id for message in drain_outbox(scheduler)} == {"a"}

    assert scheduler.has_ready_work()
    scheduler.run_ready_step()
    assert {message.request_id for message in drain_outbox(scheduler)} == {"b", "c"}


def test_stream_capacity_is_independent_from_step_batch_size(
    random_codec: MimiCodec,
) -> None:
    scheduler = PersonaPlexCode2WavScheduler(
        random_codec,
        compute_fn=lambda payload: payload,
        max_batch_size=8,
        stream_slots=16,
    )
    request_ids = [f"request-{index}" for index in range(16)]
    for request_id in request_ids:
        start_stream(scheduler, request_id)
    batch_sizes = record_decode_steps(scheduler)

    streamed = feed_batched(
        scheduler,
        [
            (request_id, random_codes(1, index))
            for index, request_id in enumerate(request_ids)
        ],
    )
    assert set(streamed) == set(request_ids)
    assert batch_sizes == [8, 8]


def test_config_enables_batching_by_default(random_codec: MimiCodec) -> None:
    config = PersonaPlexPipelineConfig(model_path="unused")
    code2wav_config = config.stage_named(CODE2WAV_STAGE)
    factory = code2wav_config.factory
    assert factory.max_batch_size == 8
    assert factory.stream_slots == 16

    scheduler = PersonaPlexCode2WavScheduler(
        random_codec,
        compute_fn=lambda payload: payload,
        max_batch_size=factory.max_batch_size,
        stream_slots=factory.stream_slots,
    )
    assert scheduler.decode_arena.slot_count == 16
    assert scheduler.can_batch_stream_chunks
    assert scheduler.stream_chunk_batch_max == 8
