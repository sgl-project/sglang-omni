# SPDX-License-Identifier: Apache-2.0
"""Drama 3 incremental stream decoding: window grouping on CPU, codec state on GPU."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang_omni.models.fishaudio_drama3.channels_last_decoder import (
    ChannelsLastDecoder,
)
from sglang_omni.models.fishaudio_drama3.config import (
    DEFAULT_CODEC_PATH,
    Drama3VocoderFactoryArgs,
)
from sglang_omni.models.fishaudio_drama3.incremental_decoder import (
    IncrementalDecoder,
    IncrementalWindowDecoder,
    incremental_window_lengths,
)
from sglang_omni.models.fishaudio_drama3.stages import resolve_codec_directory
from sglang_omni.models.fishaudio_drama3.vocoder import (
    Drama3IncrementalVocoderScheduler,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import DAC
from sglang_omni.models.fishaudio_s2_pro.stages import load_codec
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.profiler import event_recorder
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from tests.unit_test.fishaudio_drama3.test_vocoder import (
    NUM_CODEBOOKS,
    SAMPLES_PER_FRAME,
    FrameNumberCodec,
    drained,
    final_payload,
    numbered_frames,
)
from tests.unit_test.fishaudio_s2_pro.test_vocoder import tiny_codec

STRIDES = (4, 10, 12, 32, 64)


class RecordingWindowDecoder:
    """Decodes every frame to samples equal to its first code, as FrameNumberCodec does."""

    window_lengths = incremental_window_lengths(STRIDES)

    def __init__(self) -> None:
        self.decodes: list[dict[str, list[int]]] = []
        self.failure: Exception | None = None

    def window_length_for(self, frame_count: int) -> int:
        return next(length for length in self.window_lengths if length >= frame_count)

    def decode(
        self,
        windows: list[torch.Tensor],
        slots: list[int],
        positions: list[int],
        new_slots: list[int],
    ) -> torch.Tensor:
        if self.failure is not None:
            raise self.failure
        else:
            pass
        self.decodes.append(
            {
                "frames": [int(window.shape[1]) for window in windows],
                "first_codes": [int(window[0, 0]) for window in windows],
                "slots": list(slots),
                "positions": list(positions),
                "new_slots": list(new_slots),
            }
        )
        longest = max(int(window.shape[1]) for window in windows)
        audio = torch.zeros(len(windows), longest * SAMPLES_PER_FRAME)
        for row, window in enumerate(windows):
            samples = window[0].float().repeat_interleave(SAMPLES_PER_FRAME)
            audio[row, : samples.shape[0]] = samples
        return audio


def scheduler_with(
    window_decoder: RecordingWindowDecoder,
    *,
    max_stream_batch_size: int = 4,
    stream_slot_count: int = 8,
    max_stream_frames: int = 100000,
    stream_crossfade_samples: int = 0,
    decode_step_message_budget: int = 8,
) -> Drama3IncrementalVocoderScheduler:
    window_kwargs = Drama3VocoderFactoryArgs().model_dump(
        include={
            "stream_stride",
            "stream_followup_stride_ramp",
            "stream_followup_stride",
            "stream_overlap_tokens",
            "max_batch_size",
            "max_batch_wait_ms",
        }
    )
    return Drama3IncrementalVocoderScheduler(
        FrameNumberCodec(),
        window_decoder=window_decoder,
        max_stream_batch_size=max_stream_batch_size,
        stream_slot_count=stream_slot_count,
        max_stream_frames=max_stream_frames,
        device="cpu",
        decode_window=None,
        stream_crossfade_samples=stream_crossfade_samples,
        decode_step_message_budget=decode_step_message_budget,
        **window_kwargs,
    )


def send_frames(
    scheduler: Drama3IncrementalVocoderScheduler,
    request_id: str,
    frames: torch.Tensor,
    *,
    chunk_id: int = 0,
) -> None:
    scheduler.handle_stream_chunk(
        request_id,
        StreamItem(chunk_id=chunk_id, data=frames.T, from_stage="tts_engine"),
    )


def run_until_idle(scheduler: Drama3IncrementalVocoderScheduler) -> None:
    while scheduler.has_ready_work():
        scheduler.run_ready_step()


def finish(
    scheduler: Drama3IncrementalVocoderScheduler,
    request_id: str,
    frames: torch.Tensor,
) -> None:
    payload = final_payload(frames, is_streaming=True)
    payload.request_id = request_id
    scheduler.handle_streaming_new_request(request_id, payload)
    scheduler.handle_stream_done(request_id)


def audio_of(messages: list[OutgoingMessage], request_id: str) -> np.ndarray:
    return np.concatenate(
        [
            np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
            for message in messages
            if message.type == "stream" and message.request_id == request_id
        ]
    )


def test_window_lengths_add_midpoints_for_shorter_last_windows() -> None:
    assert incremental_window_lengths(STRIDES) == (4, 10, 12, 22, 32, 48, 64)


@pytest.mark.parametrize(
    ("chunk_frame_counts", "total_frame_count", "window_frames"),
    [
        ([1, 1, 1, 1, 10, 12, 32, 64], 120, [4, 10, 12, 32, 62]),
        ([120], 120, [4, 10, 12, 32, 62]),
        ([3, 50, 5], 200, [4, 10, 12, 32, 64, 64, 14]),
        ([4, 10], 14, [4, 10]),
        ([], 3, [3]),
    ],
)
def test_every_window_decodes_only_new_frames_however_frames_arrive(
    chunk_frame_counts: list[int], total_frame_count: int, window_frames: list[int]
) -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder)
    frames = numbered_frames(total_frame_count)
    start = 0
    for frame_count in chunk_frame_counts:
        send_frames(scheduler, "a", frames[start : start + frame_count])
        start += frame_count
    finish(scheduler, "a", frames)
    run_until_idle(scheduler)

    assert [decode["frames"][0] for decode in window_decoder.decodes] == window_frames
    assert [decode["positions"][0] for decode in window_decoder.decodes] == list(
        np.cumsum([0, *window_frames[:-1]])
    )
    assert [decode["new_slots"] for decode in window_decoder.decodes] == [
        [window_decoder.decodes[0]["slots"][0]],
        *[[] for _ in window_frames[1:]],
    ]
    messages = drained(scheduler)
    np.testing.assert_array_equal(
        audio_of(messages, "a"),
        np.repeat(np.arange(total_frame_count, dtype=np.float32), SAMPLES_PER_FRAME),
    )
    assert [message.type for message in messages][-1] == "result"
    assert sum(message.type == "result" for message in messages) == 1
    assert not scheduler.streams
    assert len(scheduler.free_slots) == 8


def test_frames_decode_on_a_step_not_on_arrival() -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder)
    send_frames(scheduler, "a", numbered_frames(4))

    assert window_decoder.decodes == []
    assert scheduler.has_ready_work()
    scheduler.run_ready_step()
    assert window_decoder.decodes[0]["frames"] == [4]
    assert not scheduler.has_ready_work()


def test_stream_frames_accepts_the_chunk_id_the_parent_scheduler_passes() -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder)

    scheduler.stream_frames("a", numbered_frames(3).T, chunk_id=7)

    assert scheduler.streams["a"].held_frame_count == 3
    assert window_decoder.decodes == []


def test_every_incremental_window_records_its_decode(tmp_path: Path) -> None:
    batches = (
        [("a", 1, 0, 4), ("b", 2, 0, 4)],
        [("a", 3, 0, 10), ("b", 4, 0, 10)],
        [("a", -1, 1, 3), ("b", -1, 1, 3)],
    )
    path = event_recorder.get_recorder().start("drama3", str(tmp_path), "vocoder")
    try:
        window_decoder = RecordingWindowDecoder()
        scheduler = scheduler_with(window_decoder, decode_step_message_budget=1000)
        frames = numbered_frames(17)
        send_frames(scheduler, "a", frames[:4], chunk_id=1)
        send_frames(scheduler, "b", frames[:4], chunk_id=2)
        scheduler.run_ready_step()
        assert window_decoder.decodes[0]["frames"] == [4, 4]
        send_frames(scheduler, "a", frames[4:14], chunk_id=3)
        send_frames(scheduler, "b", frames[4:14], chunk_id=4)
        finish(scheduler, "a", frames)
        finish(scheduler, "b", frames)
        run_until_idle(scheduler)
        drained(scheduler)
    finally:
        event_recorder.get_recorder().stop(wait=True)

    events = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert [
        (
            event["request_id"],
            event["event_name"].removeprefix("drama3_vocoder_window_decode_"),
            event["metadata"]["window_index"],
            event["metadata"]["chunk_id"],
            event["metadata"]["is_final"],
            event["metadata"]["emitted_frames"],
            event["metadata"]["decoded_frames"],
        )
        for event in events
    ] == [
        (request_id, phase, window_index, chunk_id, is_final, count, count)
        for window_index, rows in enumerate(batches)
        for phase in ("started", "finished")
        for request_id, chunk_id, is_final, count in rows
    ]
    assert [event["metadata"]["emitted_audio_s"] for event in events] == pytest.approx(
        [
            decoded * SAMPLES_PER_FRAME / scheduler.codec.sample_rate
            for decoded in (0, 0, 0, 0, 4, 4, 4, 4, 14, 14, 14, 14)
        ]
    )


def feed_chunks_without_draining(
    scheduler: Drama3IncrementalVocoderScheduler, count: int
) -> None:
    loop = asyncio.new_event_loop()
    try:
        for chunk_id in range(count):
            scheduler.handle_message(
                IncomingMessage(
                    request_id="a",
                    type="stream_chunk",
                    data=StreamItem(
                        chunk_id=chunk_id,
                        data=numbered_frames(4).T,
                        from_stage="tts_engine",
                    ),
                ),
                loop,
            )
    finally:
        loop.close()


def test_a_sustained_chunk_feed_still_decodes_ready_windows() -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder)

    feed_chunks_without_draining(scheduler, 32)

    assert len(window_decoder.decodes) >= 2


def test_a_sustained_chunk_feed_starves_without_a_message_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder)
    monkeypatch.setattr(scheduler, "DECODE_STEP_MESSAGE_BUDGET", 10**9)

    feed_chunks_without_draining(scheduler, 32)

    assert window_decoder.decodes == []


def test_a_stream_end_before_its_payload_waits_for_the_payload() -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder)
    frames = numbered_frames(6)
    send_frames(scheduler, "a", frames)
    scheduler.handle_stream_done("a")
    run_until_idle(scheduler)
    assert [m.type for m in drained(scheduler)] == ["stream"]

    payload = final_payload(frames, is_streaming=True)
    payload.request_id = "a"
    scheduler.handle_streaming_new_request("a", payload)
    run_until_idle(scheduler)

    assert [m.type for m in drained(scheduler)] == ["stream", "result"]


def test_windows_of_one_length_share_a_decode_and_the_least_audio_goes_first() -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder)
    send_frames(scheduler, "ahead", numbered_frames(4))
    scheduler.run_ready_step()
    send_frames(scheduler, "ahead", numbered_frames(14)[4:])
    send_frames(scheduler, "new-1", numbered_frames(4))
    send_frames(scheduler, "new-2", numbered_frames(4))
    window_decoder.decodes.clear()
    run_until_idle(scheduler)

    assert [decode["frames"] for decode in window_decoder.decodes] == [[4, 4], [10]]
    assert window_decoder.decodes[0]["positions"] == [0, 0]
    assert window_decoder.decodes[1]["positions"] == [4]


def test_a_shorter_last_window_joins_its_length_bucket() -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder)
    for request_id in ("a", "b"):
        send_frames(scheduler, request_id, numbered_frames(4))
    run_until_idle(scheduler)
    send_frames(scheduler, "a", numbered_frames(14)[4:])
    frames = numbered_frames(11)
    send_frames(scheduler, "b", frames[4:])
    finish(scheduler, "b", frames)
    window_decoder.decodes.clear()
    run_until_idle(scheduler)

    assert [decode["frames"] for decode in window_decoder.decodes] == [[10, 7]]


def test_a_group_holds_one_window_per_stream_and_at_most_the_batch_size() -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder, max_stream_batch_size=2)
    send_frames(scheduler, "backlog", numbered_frames(26))
    for request_id in ("b", "c", "d"):
        send_frames(scheduler, request_id, numbered_frames(4))
    run_until_idle(scheduler)

    assert [decode["frames"] for decode in window_decoder.decodes] == [
        [4, 4],
        [4, 4],
        [10],
        [12],
    ]


def test_a_stream_waits_for_a_free_slot_and_reuses_it_fresh() -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder, stream_slot_count=1)
    first = numbered_frames(4)
    send_frames(scheduler, "a", first)
    send_frames(scheduler, "b", numbered_frames(4))
    scheduler.run_ready_step()

    assert window_decoder.decodes[-1]["frames"] == [4]
    assert not scheduler.has_ready_work()

    finish(scheduler, "a", first)
    run_until_idle(scheduler)

    assert window_decoder.decodes[-1]["new_slots"] == [0]
    assert window_decoder.decodes[-1]["positions"] == [0]


def test_a_group_fills_only_as_many_new_streams_as_free_slots() -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder, stream_slot_count=2)
    for request_id in ("a", "b", "c"):
        send_frames(scheduler, request_id, numbered_frames(4))
    scheduler.run_ready_step()

    assert window_decoder.decodes[0]["frames"] == [4, 4]
    assert sorted(window_decoder.decodes[0]["new_slots"]) == [0, 1]
    assert not scheduler.has_ready_work()


def test_an_abort_frees_the_slot_and_drops_the_windows() -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder, stream_slot_count=1)
    send_frames(scheduler, "a", numbered_frames(4))
    scheduler.run_ready_step()
    send_frames(scheduler, "a", numbered_frames(14)[4:])
    send_frames(scheduler, "b", numbered_frames(4))
    scheduler.abort("a")
    run_until_idle(scheduler)

    assert window_decoder.decodes[-1]["frames"] == [4]
    assert window_decoder.decodes[-1]["new_slots"] == [0]


def test_a_failed_decode_errors_every_stream_of_the_group() -> None:
    window_decoder = RecordingWindowDecoder()
    scheduler = scheduler_with(window_decoder)
    for request_id in ("a", "b"):
        send_frames(scheduler, request_id, numbered_frames(4))
    window_decoder.failure = RuntimeError("decode failed")
    scheduler.run_ready_step()

    messages = drained(scheduler)
    assert sorted(message.request_id for message in messages) == ["a", "b"]
    assert all(message.type == "error" for message in messages)
    assert not scheduler.streams
    assert len(scheduler.free_slots) == 8


def test_a_stream_past_the_rotary_table_is_rejected() -> None:
    scheduler = scheduler_with(RecordingWindowDecoder(), max_stream_frames=10)
    with pytest.raises(ValueError, match="exceeds"):
        scheduler.stream_frames("a", numbered_frames(11).T)


def test_a_crossfade_is_rejected() -> None:
    with pytest.raises(ValueError, match="crossfade"):
        scheduler_with(RecordingWindowDecoder(), stream_crossfade_samples=8)


def windows_of(frame_count: int) -> list[tuple[int, int]]:
    spans = []
    start = 0
    index = 0
    while start < frame_count:
        stride = STRIDES[min(index, len(STRIDES) - 1)]
        spans.append((start, min(frame_count, start + stride)))
        start += stride
        index += 1
    return spans


def built_decoder() -> tuple[ChannelsLastDecoder, IncrementalDecoder]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    codec = tiny_codec().cuda()
    codec.fold_weight_norm()
    decoder = ChannelsLastDecoder(codec, dtype=torch.float16)
    return decoder, IncrementalDecoder(
        decoder,
        stream_slots=4,
        max_window_frames=max(STRIDES),
        device=torch.device("cuda"),
    )


def window_decoder_of(
    incremental: IncrementalDecoder, *, use_graphs: bool
) -> IncrementalWindowDecoder:
    return IncrementalWindowDecoder(
        incremental,
        num_codebooks=NUM_CODEBOOKS,
        frame_length=int(incremental.codec.frame_length),
        device=torch.device("cuda"),
        batch_sizes=(1, 2, 4),
        window_lengths=incremental_window_lengths(STRIDES),
        use_graphs=use_graphs,
    )


def random_codes(frame_count: int) -> torch.Tensor:
    return torch.randint(0, 16, (NUM_CODEBOOKS, frame_count))


def streamed_in_windows(
    window_decoder: IncrementalWindowDecoder, codes: torch.Tensor, slot: int
) -> torch.Tensor:
    pieces = []
    new_slots = [slot]
    frame_length = window_decoder.frame_length
    for first, last in windows_of(int(codes.shape[1])):
        audio = window_decoder.decode(
            [codes[:, first:last]], [slot], [first], new_slots
        )
        pieces.append(audio[0, : (last - first) * frame_length])
        new_slots = []
    return torch.cat(pieces)


@pytest.mark.accelerator
@pytest.mark.parametrize("frame_count", [3, 58, 100, 300])
def test_a_stream_decodes_like_its_whole_utterance(frame_count: int) -> None:
    decoder, incremental = built_decoder()
    window_decoder = window_decoder_of(incremental, use_graphs=False)
    codes = random_codes(frame_count)
    with torch.no_grad():
        whole = decoder(codes[None].cuda().clone())[0, 0].float().cpu()
    stream = streamed_in_windows(window_decoder, codes, slot=1)

    assert stream.shape == whole.shape
    assert (stream - whole).abs().max() < 2e-2


@pytest.mark.accelerator
def test_streams_at_different_positions_decode_in_one_batch() -> None:
    decoder, incremental = built_decoder()
    window_decoder = window_decoder_of(incremental, use_graphs=False)
    frame_length = window_decoder.frame_length
    early = random_codes(64)
    late = random_codes(140)
    late_prefix = streamed_in_windows(window_decoder, late[:, :76], slot=2)
    together = window_decoder.decode(
        [early, late[:, 76:]], [1, 2], [0, 76], new_slots=[1]
    )
    with torch.no_grad():
        whole_early = decoder(early[None].cuda().clone())[0, 0].float().cpu()
        whole_late = decoder(late[None].cuda().clone())[0, 0].float().cpu()

    assert (together[0] - whole_early).abs().max() < 2e-2
    assert (
        torch.cat([late_prefix, together[1, : 64 * frame_length]]) - whole_late
    ).abs().max() < 2e-2


@pytest.mark.accelerator
def test_rows_that_pad_a_batch_change_nothing() -> None:
    decoder, incremental = built_decoder()
    window_decoder = window_decoder_of(incremental, use_graphs=False)
    frame_length = window_decoder.frame_length
    codes = random_codes(12)
    three_rows = window_decoder.decode(
        [codes[:, :4], codes[:, :4], codes[:, :4]], [0, 1, 2], [0, 0, 0], [0, 1, 2]
    )
    following = window_decoder.decode(
        [codes[:, 4:12], codes[:, 4:12], codes[:, 4:12]],
        [0, 1, 2],
        [4, 4, 4],
        [],
    )
    with torch.no_grad():
        whole = decoder(codes[None].cuda().clone())[0, 0].float().cpu()

    assert three_rows.shape == (3, 4 * frame_length)
    for row in range(3):
        stream = torch.cat([three_rows[row], following[row, : 8 * frame_length]])
        assert (stream - whole).abs().max() < 2e-2


@pytest.mark.accelerator
def test_a_shorter_last_window_ignores_its_padding() -> None:
    decoder, incremental = built_decoder()
    window_decoder = window_decoder_of(incremental, use_graphs=False)
    frame_length = window_decoder.frame_length
    codes = random_codes(5)
    audio = window_decoder.decode([codes], [0], [0], [0])
    with torch.no_grad():
        whole = decoder(codes[None].cuda().clone())[0, 0].float().cpu()

    assert audio.shape == (1, window_decoder.window_length_for(5) * frame_length)
    assert (audio[0, : 5 * frame_length] - whole).abs().max() < 2e-2


@pytest.mark.accelerator
def test_a_reused_slot_starts_a_stream_afresh() -> None:
    _, incremental = built_decoder()
    window_decoder = window_decoder_of(incremental, use_graphs=False)
    codes = random_codes(30)
    first = streamed_in_windows(window_decoder, codes, slot=1)
    streamed_in_windows(window_decoder, random_codes(90), slot=1)
    again = streamed_in_windows(window_decoder, codes, slot=1)

    assert torch.equal(first, again)


@pytest.mark.accelerator
def test_graphs_decode_like_eager() -> None:
    _, incremental = built_decoder()
    eager = window_decoder_of(incremental, use_graphs=False)
    codes = random_codes(150)
    expected = streamed_in_windows(eager, codes, slot=1)
    graphs = window_decoder_of(incremental, use_graphs=True)
    assert len(graphs.graphs) == 3 * len(graphs.window_lengths)

    assert torch.equal(streamed_in_windows(graphs, codes, slot=1), expected)


@pytest.mark.accelerator
def test_frame_latents_sum_the_codebook_projections() -> None:
    _, incremental = built_decoder()
    quantizer = incremental.codec.quantizer
    indices = torch.randint(0, 16, (3, NUM_CODEBOOKS, 7), device="cuda")
    with torch.no_grad():
        expected = (
            quantizer.semantic_quantizer.from_codes(indices[:, :1])[0]
            + quantizer.quantizer.from_codes(indices[:, 1:])[0]
        ).transpose(1, 2)
        latents = incremental.frame_latents(indices)

    assert (latents.float() - expected.float()).abs().max() < 1e-2


@pytest.mark.accelerator
def test_frame_latents_clamp_codes_to_their_codebook() -> None:
    _, incremental = built_decoder()
    quantizer = incremental.codec.quantizer
    indices = torch.full((1, NUM_CODEBOOKS, 2), 100000, device="cuda")
    clamped = indices.clone()
    clamped[:, 0] = quantizer.semantic_quantizer.codebook_size - 1
    clamped[:, 1:] = quantizer.quantizer.codebook_size - 1
    with torch.no_grad():
        assert torch.equal(
            incremental.frame_latents(indices), incremental.frame_latents(clamped)
        )


@pytest.fixture(scope="module")
def checkpoint_codec() -> DAC:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    codec = load_codec(resolve_codec_directory(DEFAULT_CODEC_PATH), "cuda")
    codec.fold_weight_norm()
    return codec


@pytest.mark.accelerator
@pytest.mark.parametrize("use_graphs", [False, True])
def test_the_checkpoint_decodes_identically_across_chunk_boundaries(
    checkpoint_codec: DAC, use_graphs: bool
) -> None:
    codec = checkpoint_codec
    num_codebooks = (
        codec.quantizer.semantic_quantizer.n_codebooks
        + codec.quantizer.quantizer.n_codebooks
    )
    decoder = ChannelsLastDecoder(codec, dtype=torch.float16)
    incremental = IncrementalDecoder(
        decoder,
        stream_slots=2,
        max_window_frames=max(STRIDES),
        device=torch.device("cuda"),
    )
    window_decoder = IncrementalWindowDecoder(
        incremental,
        num_codebooks=num_codebooks,
        frame_length=int(codec.frame_length),
        device=torch.device("cuda"),
        batch_sizes=(1,),
        window_lengths=incremental_window_lengths(STRIDES),
        use_graphs=use_graphs,
    )
    frame_count = 120
    generator = torch.Generator().manual_seed(3630)
    codes = torch.randint(0, 64, (num_codebooks, frame_count), generator=generator)
    with torch.no_grad():
        whole = decoder(codes[None].cuda())[0, 0].float().cpu()
    stream = streamed_in_windows(window_decoder, codes, slot=1)

    assert stream.shape == whole.shape
    difference = (stream - whole).abs()
    max_difference = float(difference.max())
    assert max_difference < 5e-2, f"waveform delta {max_difference:.4f}"
    frame_length = window_decoder.frame_length
    for _, last in windows_of(frame_count)[:-1]:
        boundary = last * frame_length
        window_difference = difference[boundary : boundary + frame_length]
        peak = float(window_difference.max())
        assert peak < 5e-2, f"chunk boundary {last}: {peak:.4f}"
