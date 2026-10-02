# SPDX-License-Identifier: Apache-2.0
"""The streaming codec stage renders per-request chunks into the whole-reply waveform."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.code2wav_stream import PersonaPlexCode2WavScheduler
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest


def decode_waveform(payload: dict) -> torch.Tensor:
    assert payload["sample_rate"] == SAMPLE_RATE
    return torch.from_numpy(
        np.frombuffer(payload["audio_waveform"], dtype=np.float32).copy()
    )


def start_stream(scheduler, request_id: str) -> StagePayload:
    payload = StagePayload(
        request_id, request=OmniRequest(inputs={}), data=PersonaPlexState().to_dict()
    )
    scheduler.stream_payloads[request_id] = payload
    scheduler.on_streaming_new_request(request_id, payload)
    return payload


def test_interleaved_requests_stream_their_own_waveforms(random_codec):
    codec = random_codec
    scheduler = PersonaPlexCode2WavScheduler(
        codec, compute_fn=lambda payload: payload, chunk_ramp_frames=(1,)
    )
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


def test_a_reply_with_frames_streams_whatever_arrives_first(random_codec):
    scheduler = PersonaPlexCode2WavScheduler(
        random_codec, compute_fn=lambda payload: payload, chunk_ramp_frames=(1,)
    )
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


def test_abort_clears_stream_state(random_codec):
    scheduler = PersonaPlexCode2WavScheduler(
        random_codec, compute_fn=lambda payload: payload, chunk_ramp_frames=(1,)
    )
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
    scheduler = PersonaPlexCode2WavScheduler(
        codec, compute_fn=lambda payload: payload, chunk_ramp_frames=(1,)
    )
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


def test_chunk_ramp_holds_frames_until_each_target_width(random_codec):
    codec = random_codec
    scheduler = PersonaPlexCode2WavScheduler(
        codec, compute_fn=lambda payload: payload, chunk_ramp_frames=(1, 3)
    )
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
    widths = [waveform.shape[-1] // samples_per_frame for waveform in streamed]
    assert widths == [1, 3, 3]

    # note (nightdecode): the tail below the steady width flushes at stream_done
    # instead of waiting.
    tail, result = scheduler.on_stream_done("a")
    assert decode_waveform(tail.data).shape[-1] == samples_per_frame
    assert result.type == "result"
    streamed.append(decode_waveform(tail.data))
    torch.testing.assert_close(torch.cat(streamed), whole, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        decode_waveform(result.data.data), whole, atol=1e-5, rtol=1e-5
    )


def test_chunk_ramp_decodes_burst_messages_frame_exactly(random_codec):
    codec = random_codec
    scheduler = PersonaPlexCode2WavScheduler(
        codec, compute_fn=lambda payload: payload, chunk_ramp_frames=(1, 4)
    )
    codes = torch.randint(0, 2048, (6, 8), generator=torch.Generator().manual_seed(5))
    whole = codec.decode(codes.T[None])[0, 0]
    samples_per_frame = codec.samples_per_frame
    start_stream(scheduler, "a")

    streamed = push_frames(scheduler, "a", codes[0:1])
    assert [waveform.shape[-1] for waveform in streamed] == [samples_per_frame]
    # note (nightdecode): a 5-frame burst covers the steady width once and
    # holds the odd frame.
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


def test_one_frame_ramp_decodes_as_messages_arrive(random_codec):
    codec = random_codec
    scheduler = PersonaPlexCode2WavScheduler(
        codec, compute_fn=lambda payload: payload, chunk_ramp_frames=(1,)
    )
    codes = torch.randint(0, 2048, (3, 8), generator=torch.Generator().manual_seed(6))
    start_stream(scheduler, "a")

    waveforms = push_frames(scheduler, "a", codes)
    assert len(waveforms) == 3
    torch.testing.assert_close(
        torch.cat(waveforms), codec.decode(codes.T[None])[0, 0], atol=1e-5, rtol=1e-5
    )


def test_chunk_ramp_single_frame_reply_decodes_immediately(random_codec):
    codec = random_codec
    scheduler = PersonaPlexCode2WavScheduler(
        codec, compute_fn=lambda payload: payload, chunk_ramp_frames=(1, 4)
    )
    codes = torch.randint(0, 2048, (1, 8), generator=torch.Generator().manual_seed(7))
    start_stream(scheduler, "a")

    (waveform,) = push_frames(scheduler, "a", codes)
    torch.testing.assert_close(
        waveform, codec.decode(codes.T[None])[0, 0], atol=1e-5, rtol=1e-5
    )
    (result,) = scheduler.on_stream_done("a")
    assert result.type == "result"


def test_chunk_ramp_trims_aggregated_chunks_to_the_caller_length(random_codec):
    codec = random_codec
    scheduler = PersonaPlexCode2WavScheduler(
        codec, compute_fn=lambda payload: payload, chunk_ramp_frames=(1, 4)
    )
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


def test_chunk_ramp_keeps_interleaved_requests_apart(random_codec):
    codec = random_codec
    scheduler = PersonaPlexCode2WavScheduler(
        codec, compute_fn=lambda payload: payload, chunk_ramp_frames=(1, 3)
    )
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


def test_chunk_ramp_pending_frames_die_with_a_cancelled_request(random_codec):
    codec = random_codec
    scheduler = PersonaPlexCode2WavScheduler(
        codec, compute_fn=lambda payload: payload, chunk_ramp_frames=(1, 4)
    )
    codes = torch.randint(0, 2048, (3, 8), generator=torch.Generator().manual_seed(11))
    start_stream(scheduler, "a")
    assert push_frames(scheduler, "a", codes[0:1]) != []
    # note (nightdecode): two frames short of the steady width are held, then
    # the request goes away.
    assert push_frames(scheduler, "a", codes[1:3]) == []
    scheduler.clear_stream_state("a")
    assert scheduler.on_stream_done("a") == []
    assert "a" not in scheduler.stream_states

    start_stream(scheduler, "b")
    (waveform,) = push_frames(scheduler, "b", codes[0:1])
    assert waveform.shape[-1] == codec.samples_per_frame


@pytest.mark.parametrize(
    ("ramp_option", "expected_frames"),
    [
        ([1, 4], (1, 4)),
        ([4], (4,)),
        ((1, 4), (1, 4)),
    ],
)
def test_chunk_ramp_option_is_validated_at_construction(
    random_codec, ramp_option, expected_frames
):
    scheduler = PersonaPlexCode2WavScheduler(
        random_codec, compute_fn=lambda payload: payload, chunk_ramp_frames=ramp_option
    )
    assert scheduler.chunk_ramp_frames == expected_frames


@pytest.mark.parametrize(
    "ramp_option", [[], [0, 4], [1, -2], "1,4", "2,x,8", 4, None, True]
)
def test_invalid_chunk_ramp_option_rejects_construction(random_codec, ramp_option):
    with pytest.raises(ValueError, match="chunk_ramp_frames"):
        PersonaPlexCode2WavScheduler(
            random_codec,
            compute_fn=lambda payload: payload,
            chunk_ramp_frames=ramp_option,
        )
