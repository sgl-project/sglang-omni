# SPDX-License-Identifier: Apache-2.0
"""The streaming codec stage renders per-request chunks into the whole-reply waveform."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.personaplex import mimi_decode_graph
from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.code2wav_stream import PersonaPlexCode2WavScheduler
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.models.personaplex.mimi_decode_graph import MimiDecodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import OutgoingMessage


def decode_waveform(payload: dict) -> torch.Tensor:
    assert payload["sample_rate"] == SAMPLE_RATE
    return torch.from_numpy(
        np.frombuffer(payload["audio_waveform"], dtype=np.float32).copy()
    )


def new_scheduler(
    codec: MimiCodec, chunk_ramp_frames: tuple[int, ...] = (1,)
) -> PersonaPlexCode2WavScheduler:
    return PersonaPlexCode2WavScheduler(
        MimiDecodeSlots(
            codec,
            num_slots=2,
            max_batch_size=1,
            max_graph_frames=1,
            max_chunk_frames=max(chunk_ramp_frames),
        ),
        compute_fn=lambda payload: payload,
        chunk_ramp_frames=chunk_ramp_frames,
    )


def deliver_chunk(
    scheduler, request_id: str, item: StreamItem
) -> list[OutgoingMessage]:
    scheduler.handle_stream_chunk(request_id, item)
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    return drain_messages(scheduler)


def finish_stream(scheduler, request_id: str) -> list[OutgoingMessage]:
    if request_id not in scheduler.stream_states:
        return []
    scheduler.handle_stream_done(request_id)
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    return drain_messages(scheduler)


def start_stream(
    scheduler: PersonaPlexCode2WavScheduler, request_id: str, *, stream: bool = False
) -> StagePayload:
    payload = StagePayload(
        request_id,
        request=OmniRequest(inputs={}, params={"stream": stream}),
        data=PersonaPlexState().to_dict(),
    )
    scheduler.stream_payloads[request_id] = payload
    scheduler.on_streaming_new_request(request_id, payload)
    return payload


def test_interleaved_requests_stream_their_own_waveforms(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec)
    codes = {
        "a": torch.randint(0, 2048, (4, 8), generator=torch.Generator().manual_seed(1)),
        "b": torch.randint(0, 2048, (4, 8), generator=torch.Generator().manual_seed(2)),
    }
    whole = {rid: codec.decode(c.T[None])[0, 0] for rid, c in codes.items()}
    payloads = {rid: start_stream(scheduler, rid) for rid in codes}

    streamed = {rid: [] for rid in codes}

    def push(rid: str, chunk: torch.Tensor) -> None:
        for message in deliver_chunk(
            scheduler,
            rid,
            StreamItem(chunk_id=0, from_stage="lm", data=chunk, metadata=None),
        ):
            assert message.type == "stream"
            streamed[rid].append(decode_waveform(message.data))

    for frame in range(4):
        push("a", codes["a"][frame : frame + 1])
        if frame % 2 == 1:
            push("b", codes["b"][frame - 1 : frame + 1])

    for rid in codes:
        torch.testing.assert_close(
            torch.cat(streamed[rid]), whole[rid], atol=1e-5, rtol=1e-5
        )
        (result,) = finish_stream(scheduler, rid)
        assert result.type == "result"
        assert result.data.request is payloads[rid].request
        torch.testing.assert_close(
            decode_waveform(result.data.data), whole[rid], atol=1e-5, rtol=1e-5
        )


def test_a_reply_with_frames_streams_whatever_arrives_first(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec)
    empty = StagePayload(
        "a", request=OmniRequest(inputs={}), data=PersonaPlexState().to_dict()
    )
    assert not scheduler.is_streaming_payload(empty)
    with_frames = StagePayload(
        "a",
        request=OmniRequest(inputs={}),
        data=PersonaPlexState(codes=torch.zeros(3, 8, dtype=torch.long)).to_dict(),
    )
    assert scheduler.is_streaming_payload(with_frames)


def test_abort_clears_stream_state(random_codec: MimiCodec) -> None:
    scheduler = new_scheduler(random_codec)
    start_stream(scheduler, "a")
    assert finish_stream(scheduler, "a") != []
    scheduler.clear_stream_state("a")
    assert finish_stream(scheduler, "a") == []
    assert finish_stream(scheduler, "never-started") == []


@pytest.mark.parametrize(
    "num_samples",
    [0, 4 * SAMPLES_PER_FRAME, 3 * SAMPLES_PER_FRAME + 100],
    ids=["unspecified", "whole", "partial"],
)
def test_reply_length_matches_the_caller_in_chunks_and_final_payload(
    random_codec: MimiCodec, num_samples: int
) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec)
    frames, samples_per_frame = 4, codec.samples_per_frame
    expected_samples = num_samples or frames * samples_per_frame
    codes = torch.randint(
        0, 2048, (frames, 8), generator=torch.Generator().manual_seed(3)
    )
    whole = codec.decode(codes.T[None])[0, 0]
    assert whole.shape[-1] == frames * samples_per_frame

    streamed = []
    for frame in range(frames):
        (message,) = deliver_chunk(
            scheduler,
            "a",
            StreamItem(
                chunk_id=0,
                from_stage="lm",
                data=codes[frame : frame + 1],
                metadata={"num_samples": num_samples},
            ),
        )
        streamed.append(decode_waveform(message.data))
    torch.testing.assert_close(
        torch.cat(streamed), whole[:expected_samples], atol=1e-5, rtol=1e-5
    )
    assert streamed[-1].shape[-1] == expected_samples - 3 * samples_per_frame

    # Note (wilsonzheng0327): The terminal payload lands after every chunk, as it does
    # when the LM stage finishes.
    start_stream(scheduler, "a")
    (result,) = finish_stream(scheduler, "a")
    reply = decode_waveform(result.data.data)
    torch.testing.assert_close(reply, whole[:expected_samples], atol=1e-5, rtol=1e-5)


def push_frames(
    scheduler: PersonaPlexCode2WavScheduler,
    request_id: str,
    chunk: torch.Tensor,
    metadata: dict[str, int] | None = None,
) -> list[torch.Tensor]:
    return [
        decode_waveform(message.data)
        for message in deliver_chunk(
            scheduler,
            request_id,
            StreamItem(chunk_id=0, from_stage="lm", data=chunk, metadata=metadata),
        )
    ]


def test_chunk_ramp_holds_frames_until_each_target_width(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec, (1, 3))
    frames = 8
    codes = torch.randint(
        0, 2048, (frames, 8), generator=torch.Generator().manual_seed(4)
    )
    whole = codec.decode(codes.T[None])[0, 0]
    samples_per_frame = codec.samples_per_frame
    start_stream(scheduler, "a")

    streamed = []
    for frame in range(frames):
        streamed.extend(push_frames(scheduler, "a", codes[frame : frame + 1]))
    chunk_widths_in_frames = [
        waveform.shape[-1] // samples_per_frame for waveform in streamed
    ]
    assert chunk_widths_in_frames == [1, 3, 3]

    # note (nightdecode): Stream completion releases an incomplete chunk.
    tail, result = finish_stream(scheduler, "a")
    assert decode_waveform(tail.data).shape[-1] == samples_per_frame
    assert result.type == "result"
    streamed.append(decode_waveform(tail.data))
    torch.testing.assert_close(torch.cat(streamed), whole, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        decode_waveform(result.data.data), whole, atol=1e-5, rtol=1e-5
    )


def test_chunk_ramp_decodes_burst_messages_frame_exactly(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec, (1, 4))
    codes = torch.randint(0, 2048, (6, 8), generator=torch.Generator().manual_seed(5))
    whole = codec.decode(codes.T[None])[0, 0]
    samples_per_frame = codec.samples_per_frame
    start_stream(scheduler, "a")

    streamed = push_frames(scheduler, "a", codes[0:1])
    assert [waveform.shape[-1] for waveform in streamed] == [samples_per_frame]
    # note (nightdecode): One frame remains after the four-frame chunk.
    streamed += push_frames(scheduler, "a", codes[1:6])
    assert [waveform.shape[-1] for waveform in streamed] == [
        samples_per_frame,
        4 * samples_per_frame,
    ]
    tail, result = finish_stream(scheduler, "a")
    assert decode_waveform(tail.data).shape[-1] == samples_per_frame
    torch.testing.assert_close(
        torch.cat(streamed + [decode_waveform(tail.data)]), whole, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        decode_waveform(result.data.data), whole, atol=1e-5, rtol=1e-5
    )


def test_a_streaming_client_gets_the_reply_once(random_codec: MimiCodec) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec, (1, 4))
    codes = torch.randint(0, 2048, (6, 8), generator=torch.Generator().manual_seed(12))
    whole = codec.decode(codes.T[None])[0, 0]
    start_stream(scheduler, "a", stream=True)

    streamed = push_frames(scheduler, "a", codes, {"stream": True})
    assert scheduler.stream_states["a"].audio_parts == []
    tail, result = finish_stream(scheduler, "a")
    streamed.append(decode_waveform(tail.data))
    torch.testing.assert_close(torch.cat(streamed), whole, atol=1e-5, rtol=1e-5)
    assert result.type == "result"
    assert result.data.data == {"modality": "audio", "sample_rate": SAMPLE_RATE}


def test_chunk_ramp_trims_aggregated_chunks_to_the_caller_length(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec, (1, 4))
    frames = 6
    samples_per_frame = codec.samples_per_frame
    num_samples = 3 * samples_per_frame + 100
    codes = torch.randint(
        0, 2048, (frames, 8), generator=torch.Generator().manual_seed(8)
    )
    whole = codec.decode(codes.T[None])[0, 0]
    start_stream(scheduler, "a")

    streamed = []
    for frame in range(frames):
        streamed.extend(
            push_frames(
                scheduler,
                "a",
                codes[frame : frame + 1],
                metadata={"num_samples": num_samples},
            )
        )
    assert sum(waveform.shape[-1] for waveform in streamed) == num_samples
    tail, result = finish_stream(scheduler, "a")
    assert decode_waveform(tail.data).shape[-1] == 0
    streamed.append(decode_waveform(tail.data))
    reply = torch.cat(streamed)
    torch.testing.assert_close(reply, whole[:num_samples], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        decode_waveform(result.data.data), whole[:num_samples], atol=1e-5, rtol=1e-5
    )


def test_chunk_ramp_keeps_interleaved_requests_apart(random_codec: MimiCodec) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec, (1, 3))
    codes = {
        "a": torch.randint(0, 2048, (6, 8), generator=torch.Generator().manual_seed(9)),
        "b": torch.randint(
            0, 2048, (4, 8), generator=torch.Generator().manual_seed(10)
        ),
    }
    whole = {rid: codec.decode(c.T[None])[0, 0] for rid, c in codes.items()}
    for rid in codes:
        start_stream(scheduler, rid)

    streamed = {rid: [] for rid in codes}
    for frame in range(4):
        streamed["a"].extend(push_frames(scheduler, "a", codes["a"][frame : frame + 1]))
        streamed["b"].extend(push_frames(scheduler, "b", codes["b"][frame : frame + 1]))
    streamed["a"].extend(push_frames(scheduler, "a", codes["a"][4:6]))

    for rid in codes:
        parts = streamed[rid]
        messages = finish_stream(scheduler, rid)
        parts.extend(decode_waveform(message.data) for message in messages[:-1])
        assert messages[-1].type == "result"
        torch.testing.assert_close(torch.cat(parts), whole[rid], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            decode_waveform(messages[-1].data.data), whole[rid], atol=1e-5, rtol=1e-5
        )


def test_chunk_ramp_pending_frames_die_with_a_cancelled_request(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec, (1, 4))
    codes = torch.randint(0, 2048, (3, 8), generator=torch.Generator().manual_seed(11))
    start_stream(scheduler, "a")
    assert push_frames(scheduler, "a", codes[0:1]) != []
    # note (nightdecode): Cancellation discards frames below the decode threshold.
    assert push_frames(scheduler, "a", codes[1:3]) == []
    scheduler.clear_stream_state("a")
    assert finish_stream(scheduler, "a") == []
    assert "a" not in scheduler.stream_states

    start_stream(scheduler, "b")
    (waveform,) = push_frames(scheduler, "b", codes[0:1])
    assert waveform.shape[-1] == codec.samples_per_frame


@pytest.mark.parametrize(
    "chunk_ramp_frames",
    [(1,) + (4,) * 31 + (8, 2)],
    ids=["largest-chunk-before-smaller-steady"],
)
def test_chunk_ramp_matches_whole_decode_past_the_attention_ring(
    random_codec: MimiCodec, chunk_ramp_frames: tuple[int, ...]
) -> None:
    frame_count = 160
    codes = torch.randint(
        0, 2048, (frame_count, 8), generator=torch.Generator().manual_seed(12)
    )
    whole = random_codec.decode(codes.T[None])[0, 0]
    scheduler = new_scheduler(random_codec, chunk_ramp_frames)
    start_stream(scheduler, "reply")
    streamed: list[torch.Tensor] = []
    for frame in range(frame_count):
        streamed.extend(push_frames(scheduler, "reply", codes[frame : frame + 1]))
    *tail, result = finish_stream(scheduler, "reply")
    streamed.extend(decode_waveform(message.data) for message in tail)
    torch.testing.assert_close(torch.cat(streamed), whole, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        decode_waveform(result.data.data), whole, atol=1e-5, rtol=1e-5
    )


def drain_messages(scheduler: PersonaPlexCode2WavScheduler) -> list[OutgoingMessage]:
    messages: list[OutgoingMessage] = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


@pytest.mark.parametrize("payload_first", [False, True])
@pytest.mark.parametrize("frame_count", [9, 10])
def test_stream_done_flushes_tail_in_either_payload_order(
    random_codec: MimiCodec, payload_first: bool, frame_count: int
) -> None:
    scheduler = new_scheduler(random_codec, (1, 4))
    codes = torch.randint(
        0, 2048, (frame_count, 8), generator=torch.Generator().manual_seed(13)
    )
    num_samples = frame_count * SAMPLES_PER_FRAME - 17
    payload = StagePayload(
        "reply",
        request=OmniRequest(inputs={}),
        data=PersonaPlexState(codes=codes, num_samples=num_samples).to_dict(),
    )
    if payload_first:
        scheduler.handle_streaming_new_request("reply", payload)
    for frame in range(frame_count):
        scheduler.handle_stream_chunk(
            "reply",
            StreamItem(
                chunk_id=frame,
                data=codes[frame : frame + 1],
                from_stage="lm",
                metadata={"num_samples": num_samples},
            ),
        )
    messages = drain_messages(scheduler)
    assert len(messages) == 3
    scheduler.handle_stream_done("reply")
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    tail = drain_messages(scheduler)
    expected_types = ["stream"] if frame_count > 9 else []
    assert [message.type for message in tail] == expected_types + (
        ["result"] if payload_first else []
    )
    messages.extend(tail)
    if not payload_first:
        scheduler.handle_stream_done("reply")
        assert drain_messages(scheduler) == []
        scheduler.handle_streaming_new_request("reply", payload)
        completion = drain_messages(scheduler)
        assert [message.type for message in completion] == ["result"]
        messages.extend(completion)
    whole = random_codec.decode(codes.T[None])[0, 0, :num_samples]
    streamed = torch.cat([decode_waveform(message.data) for message in messages[:-1]])
    torch.testing.assert_close(streamed, whole, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        decode_waveform(messages[-1].data.data), whole, atol=1e-5, rtol=1e-5
    )
    assert messages[-1].data.request is payload.request
    assert "reply" not in scheduler.stream_states


def test_platforms_without_codec_decode_graphs_batch_eagerly(random_codec, monkeypatch):
    monkeypatch.setattr(
        mimi_decode_graph,
        "current_platform",
        SimpleNamespace(
            enable_codec_decode_graph=lambda: False,
            get_device_graph_backend=lambda device: SimpleNamespace(),
        ),
    )
    slots = MimiDecodeSlots(
        random_codec,
        num_slots=8,
        max_batch_size=8,
        max_graph_frames=1,
        max_chunk_frames=1,
    )
    assert slots.graphs == {}
    assert [slots.acquire() for _ in range(8)] == list(range(8))
    assert slots.acquire() is None
    codes = torch.randint(
        0, 2048, (2, 8, 1), generator=torch.Generator().manual_seed(5)
    )
    assert slots.decode_step(codes, [3, 6]).shape == (2, 1, SAMPLES_PER_FRAME)


def decode_plan(
    slots: MimiDecodeSlots,
    codes: torch.Tensor,
    plan: list[tuple[tuple[int, ...], int]],
    slot_of_stream: list[int],
) -> list[torch.Tensor]:
    """Run each (streams, width) step of the plan on the streams' slots."""
    parts: list[list[torch.Tensor]] = [[] for _ in slot_of_stream]
    starts = [0] * len(slot_of_stream)
    for streams, width in plan:
        chunk = torch.stack(
            [
                codes[stream, :, starts[stream] : starts[stream] + width]
                for stream in streams
            ]
        )
        waveform = slots.decode_step(
            chunk, [slot_of_stream[s] for s in streams]
        ).clone()
        for row, stream in enumerate(streams):
            parts[stream].append(waveform[row])
            starts[stream] += width
    return [torch.cat(stream_parts, -1) for stream_parts in parts]


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.enable_codec_decode_graph(),
    reason="the platform does not capture codec decode graphs",
)
def test_batched_graph_replay_matches_eager_decode_across_slot_reuse(random_codec):
    codec = random_codec.cuda()
    graphed = MimiDecodeSlots(
        codec, num_slots=3, max_batch_size=2, max_graph_frames=2, max_chunk_frames=3
    )
    assert set(graphed.graphs) == {(1, 1), (1, 2), (2, 1), (2, 2)}
    eager = MimiDecodeSlots(
        codec, num_slots=3, max_batch_size=2, max_graph_frames=0, max_chunk_frames=3
    )
    assert eager.graphs == {}
    codes = torch.randint(
        0, 2048, (3, 8, 140), generator=torch.Generator().manual_seed(6)
    ).cuda()
    # Note (nightdecode): Streams advance at different rates, so rows in one step
    # sit at different offsets. Width 3 and the batch of three have no graph and
    # run eager on the same slots; stream 2 takes 140 frames, 280 steps, past the
    # transformer's 250-step ring.
    plan = [
        ((0, 1), 1),
        ((0,), 2),
        ((0, 1, 2), 1),
        ((1, 2), 3),
        ((2,), 1),
        ((0, 2), 2),
    ] * 20
    expected = []
    for stream in range(3):
        state = codec.init_decode_state(batch_size=1, max_chunk_frames=3)
        parts, start = [], 0
        for streams, width in plan:
            if stream in streams:
                chunk = codes[stream : stream + 1, :, start : start + width]
                parts.append(codec.decode_step(chunk, state)[0])
                start += width
            else:
                pass
        expected.append(torch.cat(parts, -1))

    for _ in range(2):
        outputs = {}
        for name, slots in (("graphed", graphed), ("eager", eager)):
            slot_of_stream = [slots.acquire() for _ in range(3)]
            outputs[name] = decode_plan(slots, codes, plan, slot_of_stream)
            # Released in stream order, so the next round maps the streams onto
            # the slots in reverse.
            for slot_index in slot_of_stream:
                slots.release(slot_index)
        for stream in range(3):
            # The same steps replayed and run eagerly agree bit for bit.
            assert torch.equal(outputs["graphed"][stream], outputs["eager"][stream])
            # A batch of one rounds differently (TF32 convolutions pick kernels
            # per shape), so against it only a row decoding the wrong stream,
            # an error the size of the signal, is ruled out.
            error = (outputs["graphed"][stream] - expected[stream]).abs().max()
            assert error <= 0.05 * expected[stream].abs().max()
