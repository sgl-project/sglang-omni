# SPDX-License-Identifier: Apache-2.0
"""The streaming codec stage renders per-request chunks into the whole-reply waveform."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.personaplex import code2wav_stream, mimi_decode_graph
from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.code2wav_stream import PersonaPlexCode2WavScheduler
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.models.personaplex.mimi_decode_graph import DecodeSlot, MimiDecodeSlots
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
            max_graph_frames=1,
            max_chunk_frames=max(chunk_ramp_frames),
        ),
        compute_fn=lambda payload: payload,
        chunk_ramp_frames=chunk_ramp_frames,
    )


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
        for message in scheduler.on_stream_chunk(
            rid, SimpleNamespace(data=chunk, metadata=None)
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
        (result,) = scheduler.on_stream_done(rid)
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
    assert scheduler.on_stream_done("a") != []
    scheduler.clear_stream_state("a")
    assert scheduler.on_stream_done("a") == []
    assert scheduler.on_stream_done("never-started") == []


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
        (message,) = scheduler.on_stream_chunk(
            "a",
            SimpleNamespace(
                data=codes[frame : frame + 1], metadata={"num_samples": num_samples}
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
    (result,) = scheduler.on_stream_done("a")
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
        for message in scheduler.on_stream_chunk(
            request_id, SimpleNamespace(data=chunk, metadata=metadata)
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
    tail, result = scheduler.on_stream_done("a")
    assert decode_waveform(tail.data).shape[-1] == samples_per_frame
    assert result.type == "result"
    streamed.append(decode_waveform(tail.data))
    torch.testing.assert_close(torch.cat(streamed), whole, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        decode_waveform(result.data.data), whole, atol=1e-5, rtol=1e-5
    )


def test_chunk_ramp_decodes_each_width_once(random_codec: MimiCodec) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec, (1, 3))
    decoded_widths: list[int] = []
    decode_step = scheduler.decode_slots.decode_step

    def counting_decode_step(codes_BKF: torch.Tensor, slot: DecodeSlot) -> torch.Tensor:
        decoded_widths.append(int(codes_BKF.shape[-1]))
        return decode_step(codes_BKF, slot)

    scheduler.decode_slots.decode_step = counting_decode_step
    codes = torch.randint(0, 2048, (5, 8), generator=torch.Generator().manual_seed(11))
    start_stream(scheduler, "a")
    for frame in range(codes.shape[0]):
        push_frames(scheduler, "a", codes[frame : frame + 1])
    assert decoded_widths == [1, 3]
    scheduler.on_stream_done("a")
    assert decoded_widths == [1, 3, 1]


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
    tail, result = scheduler.on_stream_done("a")
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

    streamed = push_frames(scheduler, "a", codes)
    tail, result = scheduler.on_stream_done("a")
    streamed.append(decode_waveform(tail.data))
    torch.testing.assert_close(torch.cat(streamed), whole, atol=1e-5, rtol=1e-5)
    assert result.type == "result"
    assert result.data.data == {"modality": "audio", "sample_rate": SAMPLE_RATE}


def test_one_frame_ramp_decodes_as_messages_arrive(random_codec: MimiCodec) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec, (1,))
    codes = torch.randint(0, 2048, (3, 8), generator=torch.Generator().manual_seed(6))
    start_stream(scheduler, "a")

    waveforms = push_frames(scheduler, "a", codes)
    assert len(waveforms) == 3
    torch.testing.assert_close(
        torch.cat(waveforms), codec.decode(codes.T[None])[0, 0], atol=1e-5, rtol=1e-5
    )


def test_chunk_ramp_single_frame_reply_decodes_immediately(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec, (1, 4))
    codes = torch.randint(0, 2048, (1, 8), generator=torch.Generator().manual_seed(7))
    start_stream(scheduler, "a")

    (waveform,) = push_frames(scheduler, "a", codes)
    torch.testing.assert_close(
        waveform, codec.decode(codes.T[None])[0, 0], atol=1e-5, rtol=1e-5
    )
    (result,) = scheduler.on_stream_done("a")
    assert result.type == "result"


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
    tail, result = scheduler.on_stream_done("a")
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
        messages = scheduler.on_stream_done(rid)
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
    assert scheduler.on_stream_done("a") == []
    assert "a" not in scheduler.stream_states

    start_stream(scheduler, "b")
    (waveform,) = push_frames(scheduler, "b", codes[0:1])
    assert waveform.shape[-1] == codec.samples_per_frame


@pytest.mark.parametrize(
    "chunk_ramp_frames",
    [(1,), (1, 4), (1,) + (4,) * 31 + (8, 2)],
    ids=["single-frame", "four-frames", "largest-chunk-before-smaller-steady"],
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
    *tail, result = scheduler.on_stream_done("reply")
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
@pytest.mark.parametrize("frame_count", [9, 10, 12])
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


class CodecStream:
    def __init__(self, names: list[str], waited_events: list[torch.Event]) -> None:
        self.names = names
        self.waited_events = waited_events

    def wait_event(self, event: torch.Event) -> None:
        self.names.append("wait_event")
        self.waited_events.append(event)


@dataclass(kw_only=True)
class CodecStreamTrace:
    names: list[str]
    waited_events: list[torch.Event]
    recorded_tensors: list[torch.Tensor]
    recorded_streams: list[CodecStream]
    codec_stream: CodecStream


def install_codec_stream_fakes(monkeypatch: pytest.MonkeyPatch) -> CodecStreamTrace:
    """Force device streams on and record wait, as_tensor, and record_stream order."""
    names: list[str] = []
    waited_events: list[torch.Event] = []
    recorded_tensors: list[torch.Tensor] = []
    recorded_streams: list[CodecStream] = []
    codec_stream = CodecStream(names, waited_events)
    trace = CodecStreamTrace(
        names=names,
        waited_events=waited_events,
        recorded_tensors=recorded_tensors,
        recorded_streams=recorded_streams,
        codec_stream=codec_stream,
    )
    original_as_tensor = code2wav_stream.torch.as_tensor

    def current_stream(device: torch.device) -> CodecStream:
        return trace.codec_stream

    def device_module(device: torch.device) -> SimpleNamespace:
        return SimpleNamespace(current_stream=current_stream, device=device)

    def logging_as_tensor(
        source: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        trace.names.append("as_tensor")
        adopted = original_as_tensor(source, dtype=dtype, device=device)
        trace.names.append("as_tensor_return")
        return adopted

    def logging_record_stream(tensor: torch.Tensor, stream: CodecStream) -> None:
        trace.names.append("record_stream")
        trace.recorded_tensors.append(tensor)
        trace.recorded_streams.append(stream)

    def device_streams_enabled(device: torch.device) -> bool:
        return True

    monkeypatch.setattr(
        code2wav_stream, "supports_device_streams", device_streams_enabled
    )
    monkeypatch.setattr(code2wav_stream.torch, "get_device_module", device_module)
    monkeypatch.setattr(code2wav_stream.torch, "as_tensor", logging_as_tensor)
    monkeypatch.setattr(torch.Tensor, "record_stream", logging_record_stream)
    return trace


def test_codes_ready_event_is_waited_before_the_chunk_is_adopted(
    monkeypatch: pytest.MonkeyPatch, random_codec: MimiCodec
) -> None:
    """The codec waits out the producer's event before adopting the chunk."""
    trace = install_codec_stream_fakes(monkeypatch)
    scheduler = new_scheduler(random_codec, (4,))
    chunk = torch.arange(8, dtype=torch.long).view(1, 8)
    event = torch.Event()
    start_stream(scheduler, "ready")
    scheduler.on_stream_chunk(
        "ready",
        SimpleNamespace(data=chunk, metadata={"codes_ready_event": event}),
    )
    assert trace.names.index("wait_event") < trace.names.index("as_tensor")
    assert trace.waited_events == [event]

    trace.names.clear()
    trace.waited_events.clear()
    start_stream(scheduler, "plain")
    scheduler.on_stream_chunk("plain", SimpleNamespace(data=chunk, metadata=None))
    assert "as_tensor" in trace.names
    assert "wait_event" not in trace.names


def test_resident_long_chunk_keeps_its_storage_and_a_cast_is_marked_after(
    monkeypatch: pytest.MonkeyPatch, random_codec: MimiCodec
) -> None:
    """A chunk already long on the codec is not marked; a cast is, after as_tensor."""
    trace = install_codec_stream_fakes(monkeypatch)
    scheduler = new_scheduler(random_codec, (4,))
    resident = torch.arange(8, dtype=torch.long).view(1, 8)
    start_stream(scheduler, "resident")
    scheduler.on_stream_chunk("resident", SimpleNamespace(data=resident, metadata=None))
    adopted = scheduler.stream_states["resident"].pending_codes[0]
    assert adopted.data_ptr() == resident.data_ptr()
    assert trace.recorded_tensors == []

    trace.names.clear()
    cast_source = torch.arange(8, dtype=torch.int32).view(1, 8)
    start_stream(scheduler, "cast")
    scheduler.on_stream_chunk("cast", SimpleNamespace(data=cast_source, metadata=None))
    assert trace.names.index("as_tensor_return") < trace.names.index("record_stream")
    assert len(trace.recorded_tensors) == 1
    assert trace.recorded_tensors[0] is cast_source
    assert trace.recorded_streams == [trace.codec_stream]


def test_decode_chunk_marks_the_read_storage_after_decode_step(
    monkeypatch: pytest.MonkeyPatch, random_codec: MimiCodec
) -> None:
    """One decode marks the storage it read, not the source of a contiguous chunk."""
    trace = install_codec_stream_fakes(monkeypatch)
    scheduler = new_scheduler(random_codec, (1,))
    decode_step = scheduler.decode_slots.decode_step
    read_codes: list[torch.Tensor] = []

    def logging_decode_step(codes_BKF: torch.Tensor, slot: DecodeSlot) -> torch.Tensor:
        read_codes.append(codes_BKF)
        trace.names.append("decode_step")
        waveform = decode_step(codes_BKF, slot)
        trace.names.append("decode_step_return")
        return waveform

    scheduler.decode_slots.decode_step = logging_decode_step
    chunk = torch.randint(0, 2048, (1, 8), generator=torch.Generator().manual_seed(1))
    start_stream(scheduler, "a")
    scheduler.on_stream_chunk("a", SimpleNamespace(data=chunk, metadata=None))
    assert len(read_codes) == 1
    assert read_codes[0].data_ptr() == chunk.data_ptr()
    assert len(trace.recorded_tensors) == 1
    assert trace.recorded_tensors[0] is read_codes[0]
    assert all(recorded is not chunk for recorded in trace.recorded_tensors)
    assert trace.recorded_streams == [trace.codec_stream]
    assert trace.names.index("decode_step_return") < trace.names.index("record_stream")


def test_platforms_without_codec_decode_graphs_keep_no_slots(random_codec, monkeypatch):
    monkeypatch.setattr(
        mimi_decode_graph,
        "current_platform",
        SimpleNamespace(
            enable_codec_decode_graph=lambda: False,
            get_device_graph_backend=lambda device: SimpleNamespace(),
        ),
    )
    slots = MimiDecodeSlots(
        random_codec, num_slots=8, max_graph_frames=1, max_chunk_frames=1
    )
    assert slots.graphs == {} and slots.decode_states == []
    assert slots.acquire().index is None


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.enable_codec_decode_graph(),
    reason="the platform does not capture codec decode graphs",
)
def test_graph_replay_matches_eager_decode_across_slot_reuse(random_codec):
    codec = random_codec.cuda()
    slots = MimiDecodeSlots(codec, num_slots=2, max_graph_frames=2, max_chunk_frames=3)
    assert len(slots.graphs) == 4
    codes = torch.randint(
        0, 2048, (2, 8, 140), generator=torch.Generator().manual_seed(6)
    ).cuda()
    # Note (wilsonzheng0327): Width 3 has no graph and runs eager on the same slot;
    # 140 frames are 280 steps, past the transformer's 250-step ring.
    widths = [1, 2, 1, 3] * 20
    expected = []
    for stream in range(2):
        state = codec.init_decode_state(batch_size=1, max_chunk_frames=3)
        parts, start = [], 0
        for width in widths:
            chunk = codes[stream : stream + 1, :, start : start + width]
            parts.append(codec.decode_step(chunk, state))
            start += width
        expected.append(torch.cat(parts, -1))

    for _ in range(2):
        pair = [slots.acquire(), slots.acquire()]
        assert {slot.index for slot in pair} == {0, 1}
        replayed = [[], []]
        start = 0
        for width in widths:
            for stream, slot in enumerate(pair):
                chunk = codes[stream : stream + 1, :, start : start + width]
                replayed[stream].append(slots.decode_step(chunk, slot).clone())
            start += width
        for stream, slot in enumerate(pair):
            assert torch.equal(torch.cat(replayed[stream], -1), expected[stream])
            slots.release(slot)
