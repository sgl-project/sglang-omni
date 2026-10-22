# SPDX-License-Identifier: Apache-2.0
"""Drama 3 vocoder windows on CPU with a codec that decodes a frame to its number."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.fishaudio_drama3.channels_last_decoder import (
    ChannelsLastDecoder,
    snake_channels_last_kernel,
)
from sglang_omni.models.fishaudio_drama3.config import Drama3VocoderFactoryArgs
from sglang_omni.models.fishaudio_drama3.request_builders import (
    Drama3FrameStreamer,
    Drama3GenerationResult,
    Drama3RequestData,
)
from sglang_omni.models.fishaudio_drama3.stream_windows import (
    frames_in_complete_windows,
    stream_window_lengths,
)
from sglang_omni.models.fishaudio_drama3.vocoder import (
    Drama3VocoderScheduler,
    WindowGraphDecoder,
)
from sglang_omni.models.fishaudio_s2_pro.streaming_vocoder import WindowDecoder
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.types import RequestOutput
from tests.unit_test.fishaudio_s2_pro.test_vocoder import tiny_codec

SAMPLES_PER_FRAME = 4
SAMPLE_RATE = 44100
NUM_CODEBOOKS = 10
PROMPT_ROW_COUNT = 37
# [first, end) frame numbers, one pair per chunk or decode.
FrameRanges = list[tuple[int, int]]


class FrameNumberCodec:
    """Decodes every frame to SAMPLES_PER_FRAME samples equal to its first code, which
    the tests set to the frame number, and records the codes of each decode."""

    sample_rate = SAMPLE_RATE
    frame_length = SAMPLES_PER_FRAME
    quantizer = SimpleNamespace(
        semantic_quantizer=SimpleNamespace(n_codebooks=1),
        quantizer=SimpleNamespace(n_codebooks=NUM_CODEBOOKS - 1),
    )

    def __init__(self) -> None:
        self.decoded_codes: list[torch.Tensor] = []
        self.is_weight_norm_folded = False

    def fold_weight_norm(self) -> None:
        self.is_weight_norm_folded = True

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        self.decoded_codes.append(indices.clone())
        return indices[:, :1, :].float().repeat_interleave(SAMPLES_PER_FRAME, dim=-1)

    def decoded_frame_ranges(self) -> FrameRanges:
        """The [first, end) frame numbers of each decode of one request."""
        return [
            (int(codes[0, 0, 0]), int(codes[0, 0, -1]) + 1)
            for codes in self.decoded_codes
        ]


def numbered_frames(frame_count: int) -> torch.Tensor:
    """[frames, 1 + num_codebooks] rows whose first code is the frame number."""
    frames = torch.zeros((frame_count, 1 + NUM_CODEBOOKS), dtype=torch.int64)
    frames[:, 1] = torch.arange(frame_count)
    frames[:, 0] = 5000 + frames[:, 1]
    return frames


def vocoder_with(
    codec: FrameNumberCodec,
    decode_window: WindowDecoder | None = None,
    *,
    factory_args: Drama3VocoderFactoryArgs | None = None,
    max_padded_frames_per_decode: int | None = None,
) -> Drama3VocoderScheduler:
    """The vocoder with the configured windows."""
    window_kwargs = (factory_args or Drama3VocoderFactoryArgs()).model_dump(
        include={
            "stream_stride",
            "stream_followup_stride_ramp",
            "stream_followup_stride",
            "stream_overlap_tokens",
            "stream_crossfade_samples",
            "max_batch_size",
            "max_batch_wait_ms",
        }
    )
    return Drama3VocoderScheduler(
        codec,
        device="cpu",
        decode_window=decode_window,
        max_padded_frames_per_decode=max_padded_frames_per_decode,
        **window_kwargs,
    )


def final_payload(
    frames: torch.Tensor, *, is_streaming: bool, finish_reason: str = "length"
) -> StagePayload:
    return StagePayload(
        request_id="request-a",
        request=OmniRequest(inputs="hello", params={"stream": is_streaming}),
        data=Drama3GenerationResult(
            frames=frames,
            prompt_row_count=PROMPT_ROW_COUNT,
            finish_reason=finish_reason,
        ),
    )


def drained(vocoder: Drama3VocoderScheduler) -> list[OutgoingMessage]:
    messages = []
    while not vocoder.outbox.empty():
        messages.append(vocoder.outbox.get_nowait())
    return messages


def chunk_frame_ranges(messages: list[OutgoingMessage]) -> FrameRanges:
    """The [first, end) frame numbers each streamed audio chunk covers, in order."""
    ranges = []
    for message in messages:
        if message.type == "stream":
            assert isinstance(message.data, dict)
            samples = np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
            assert samples.size % SAMPLES_PER_FRAME == 0
            ranges.append((int(samples[0]), int(samples[-1]) + 1))
        else:
            pass
    return ranges


def streamed_audio(messages: list[OutgoingMessage]) -> np.ndarray:
    return np.concatenate(
        [
            np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
            for message in messages
            if message.type == "stream" and isinstance(message.data, dict)
        ]
    )


def streamed(
    total_frame_count: int,
    chunk_frame_counts: list[int],
    *,
    is_done_first: bool,
    codec: FrameNumberCodec | None = None,
    decode_window: WindowDecoder | None = None,
) -> tuple[FrameNumberCodec, list[OutgoingMessage]]:
    """Stream chunks of the given sizes, then hand over the final payload with all
    total_frame_count frames and signal the end of the stream."""
    codec = FrameNumberCodec() if codec is None else codec
    vocoder = vocoder_with(codec, decode_window)
    frames = numbered_frames(total_frame_count)
    start = 0
    for chunk_id, frame_count in enumerate(chunk_frame_counts):
        vocoder.handle_stream_chunk(
            "request-a",
            StreamItem(
                chunk_id=chunk_id,
                data=frames[start : start + frame_count].T,
                from_stage="tts_engine",
            ),
        )
        start += frame_count
    payload = final_payload(frames, is_streaming=True)
    if is_done_first:
        vocoder.handle_stream_done("request-a")
        vocoder.handle_streaming_new_request("request-a", payload)
    else:
        vocoder.handle_streaming_new_request("request-a", payload)
        vocoder.handle_stream_done("request-a")
    return codec, drained(vocoder)


WINDOWS_TO_214 = [(0, 10), (10, 20), (20, 32), (32, 64), (64, 128), (128, 192)]
WINDOWS_TO_300 = WINDOWS_TO_214 + [(192, 256)]
FINAL_PAYLOAD_ONLY_CASE = pytest.param(
    300,
    [1] * 5,
    WINDOWS_TO_300 + [(256, 300)],
    [(0, 10), (0, 20), (0, 32), (0, 64), (32, 128), (96, 192), (160, 256)]
    + [(224, 300)],
    id="295-frames-only-in-the-final-payload",
)
STREAMING_CASES = [
    pytest.param(
        214,
        [1] * 214,
        WINDOWS_TO_214 + [(192, 214)],
        [(0, 10), (0, 20), (0, 32), (0, 64), (32, 128), (96, 192), (160, 214)],
        id="frame-by-frame",
    ),
    pytest.param(
        214,
        [1] * 5 + [150] + [1] * 59,
        WINDOWS_TO_214 + [(192, 214)],
        [(0, 10), (0, 20), (0, 32), (0, 64), (32, 128), (96, 192), (160, 214)],
        id="a-150-frame-chunk-midstream",
    ),
    pytest.param(
        300,
        [1] * 5 + [295],
        WINDOWS_TO_300 + [(256, 300)],
        [(0, 10), (0, 20), (0, 32), (0, 64), (32, 128), (96, 192), (160, 256)]
        + [(224, 300)],
        id="a-295-frame-last-chunk",
    ),
    FINAL_PAYLOAD_ONLY_CASE,
    pytest.param(
        128,
        [1] * 128,
        [(0, 10), (10, 20), (20, 32), (32, 64), (64, 128)],
        [(0, 10), (0, 20), (0, 32), (0, 64), (32, 128)],
        id="whole-windows-and-no-tail",
    ),
    pytest.param(7, [1] * 7, [(0, 7)], [(0, 7)], id="only-a-tail"),
]


@pytest.mark.parametrize(
    (
        "total_frame_count",
        "chunk_frame_counts",
        "chunk_ranges",
        "decoded_ranges",
        "is_done_first",
    ),
    [
        pytest.param(*case.values, False, id=f"payload-first-{case.id}")
        for case in STREAMING_CASES
    ]
    + [
        pytest.param(
            *FINAL_PAYLOAD_ONLY_CASE.values,
            True,
            id=f"done-first-{FINAL_PAYLOAD_ONLY_CASE.id}",
        )
    ],
)
def test_streaming_emits_the_fixed_windows_however_frames_arrive(
    total_frame_count: int,
    chunk_frame_counts: list[int],
    chunk_ranges: FrameRanges,
    decoded_ranges: FrameRanges,
    is_done_first: bool,
) -> None:
    codec, messages = streamed(
        total_frame_count, chunk_frame_counts, is_done_first=is_done_first
    )

    assert chunk_frame_ranges(messages) == chunk_ranges
    assert codec.decoded_frame_ranges() == decoded_ranges
    assert max(end - start for start, end in decoded_ranges) <= 96
    # No crossfade and no held-back tail: the chunks are each frame once, in order.
    np.testing.assert_array_equal(
        streamed_audio(messages),
        np.repeat(np.arange(total_frame_count, dtype=np.float32), SAMPLES_PER_FRAME),
    )
    assert [message.type for message in messages][-1] == "result"


@pytest.mark.parametrize(
    (
        "stream_stride",
        "stream_followup_stride_ramp",
        "stream_followup_stride",
        "stream_overlap_tokens",
        "window_frame_counts",
    ),
    [
        pytest.param(10, (10, 12, 32), 64, 32, (10, 20, 32, 64, 96), id="defaults"),
        pytest.param(40, (), 45, 20, (40, 65), id="no-ramp"),
        pytest.param(10, (), 16, 40, (10, 26, 42, 56), id="context-fills-late"),
        pytest.param(10, (), 10, 0, (10,), id="no-context"),
    ],
)
def test_the_stream_window_lengths_follow_the_window_config(
    stream_stride: int,
    stream_followup_stride_ramp: tuple[int, ...],
    stream_followup_stride: int,
    stream_overlap_tokens: int,
    window_frame_counts: tuple[int, ...],
) -> None:
    assert (
        stream_window_lengths(
            stream_stride=stream_stride,
            stream_followup_stride_ramp=stream_followup_stride_ramp,
            stream_followup_stride=stream_followup_stride,
            stream_overlap_tokens=stream_overlap_tokens,
        )
        == window_frame_counts
    )


def test_every_streamed_window_decodes_through_the_given_window_decoder() -> None:
    codec = FrameNumberCodec()
    decoded_frame_counts: list[int] = []

    def decode_window(indices: torch.Tensor) -> torch.Tensor:
        decoded_frame_counts.append(int(indices.shape[-1]))
        return codec.from_indices(indices)

    _, messages = streamed(
        214, [1] * 214, is_done_first=False, codec=codec, decode_window=decode_window
    )

    assert decoded_frame_counts == [10, 20, 32, 64, 96, 96, 54]
    np.testing.assert_array_equal(
        streamed_audio(messages),
        np.repeat(np.arange(214, dtype=np.float32), SAMPLES_PER_FRAME),
    )


@pytest.mark.accelerator
def test_window_graphs_decode_every_window_length_like_eager() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    device = torch.device("cuda")
    codebook_weights = torch.arange(
        1, NUM_CODEBOOKS + 1, dtype=torch.float32, device=device
    )

    def causal_decode(indices: torch.Tensor) -> torch.Tensor:
        frame_values = (indices.float() * codebook_weights[:, None]).sum(
            dim=1, keepdim=True
        )
        return frame_values.cumsum(dim=-1).repeat_interleave(SAMPLES_PER_FRAME, dim=-1)

    decoder = WindowGraphDecoder(
        causal_decode,
        num_codebooks=NUM_CODEBOOKS,
        frame_length=SAMPLES_PER_FRAME,
        device=device,
        window_lengths=(10, 20, 32, 64, 96),
    )
    codes = torch.randint(0, 100, (2, NUM_CODEBOOKS, 100), device=device)

    replay_results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for frame_count in (96, 7, 10, 11, 64, 95, 97):
        window = codes[:1, :, :frame_count]
        replay_results.append((decoder(window), causal_decode(window)))
    for decoded_audio, expected_audio in replay_results:
        assert torch.equal(decoded_audio, expected_audio)
    assert torch.equal(decoder(codes[:, :, :20]), causal_decode(codes[:, :, :20]))


def test_the_final_result_counts_prompt_rows_and_audio_frames() -> None:
    _, messages = streamed(20, [1] * 20, is_done_first=False)

    result = messages[-1].data
    assert isinstance(result, StagePayload)
    assert result.data == {
        "modality": "audio",
        "sample_rate": SAMPLE_RATE,
        "usage": {
            "prompt_tokens": PROMPT_ROW_COUNT,
            "completion_tokens": 20,
            "total_tokens": PROMPT_ROW_COUNT + 20,
        },
        "finish_reason": "length",
    }


def test_no_audio_frames_fail_without_a_codec_call() -> None:
    codec = FrameNumberCodec()
    vocoder = vocoder_with(codec)
    vocoder.handle_streaming_new_request(
        "request-a", final_payload(numbered_frames(0), is_streaming=True)
    )

    with pytest.raises(ValueError, match="generated no audio frames"):
        vocoder.handle_stream_done("request-a")
    with pytest.raises(ValueError, match="generated no audio frames"):
        vocoder.validate_non_streaming_payload(
            final_payload(numbered_frames(0), is_streaming=False)
        )
    assert codec.decoded_codes == []


def test_more_streamed_frames_than_the_result_holds_is_refused() -> None:
    vocoder = vocoder_with(FrameNumberCodec())
    vocoder.handle_stream_chunk(
        "request-a",
        StreamItem(chunk_id=0, data=numbered_frames(6).T, from_stage="tts_engine"),
    )
    vocoder.handle_streaming_new_request(
        "request-a", final_payload(numbered_frames(5), is_streaming=True)
    )

    with pytest.raises(RuntimeError, match="6 frames streamed but the result holds 5"):
        vocoder.handle_stream_done("request-a")


def test_a_non_streaming_batch_decodes_once() -> None:
    codec = FrameNumberCodec()
    vocoder = vocoder_with(codec)
    payloads = [
        final_payload(numbered_frames(5), is_streaming=False, finish_reason="stop"),
        final_payload(numbered_frames(7), is_streaming=False, finish_reason="length"),
    ]

    results = vocoder.vocode_payloads(payloads)

    assert len(codec.decoded_codes) == 1
    assert codec.decoded_codes[0].shape == (2, NUM_CODEBOOKS, 7)
    for result, frame_count, finish_reason in zip(
        results, [5, 7], ["stop", "length"], strict=True
    ):
        assert isinstance(result.data, dict)
        assert (
            result.data["audio_data"]
            == np.repeat(
                np.arange(frame_count, dtype=np.float32), SAMPLES_PER_FRAME
            ).tolist()
        )
        assert result.data["usage"] == {
            "prompt_tokens": PROMPT_ROW_COUNT,
            "completion_tokens": frame_count,
            "total_tokens": PROMPT_ROW_COUNT + frame_count,
        }
        assert result.data["finish_reason"] == finish_reason


def test_a_non_streaming_batch_decodes_through_the_given_window_decoder() -> None:
    codec = FrameNumberCodec()
    decoded_shapes: list[tuple[int, ...]] = []

    def decode_window(indices: torch.Tensor) -> torch.Tensor:
        decoded_shapes.append(tuple(indices.shape))
        return codec.from_indices(indices)

    vocoder = vocoder_with(codec, decode_window)

    results = vocoder.vocode_payloads(
        [
            final_payload(numbered_frames(5), is_streaming=False),
            final_payload(numbered_frames(7), is_streaming=False),
        ]
    )

    assert decoded_shapes == [(2, NUM_CODEBOOKS, 7)]
    assert [len(result.data["audio_data"]) for result in results] == [
        5 * SAMPLES_PER_FRAME,
        7 * SAMPLES_PER_FRAME,
    ]


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_the_snake_kernel_follows_the_formula_on_a_trimmed_activation(
    dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    generator = torch.Generator(device="cuda").manual_seed(0)
    channel_count = 96
    alpha = torch.rand(channel_count, generator=generator, device="cuda") * 4 + 0.05
    inverse_alpha = (alpha + 1e-9).reciprocal()
    untrimmed = torch.randn(
        (3, 130, channel_count), generator=generator, device="cuda"
    ).to(dtype)
    activation = untrimmed[:, :125]
    output = torch.empty(activation.shape, dtype=dtype, device="cuda")

    snake_channels_last_kernel[(12, 2)](
        activation,
        alpha,
        inverse_alpha,
        output,
        3 * 125,
        125,
        channel_count,
        activation.stride(0) // channel_count,
        BLOCK_ROWS=32,
        BLOCK_CHANNELS=64,
    )

    exact = activation.double() + (alpha.double() + 1e-9).reciprocal() * torch.sin(
        alpha.double() * activation.double()
    ).pow(2)
    torch.testing.assert_close(output, exact.to(dtype))


@pytest.mark.accelerator
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float32, 1e-4), (torch.float16, 2e-2)],
    ids=["float32", "float16"],
)
def test_the_channels_last_decoder_decodes_like_the_codec(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, tolerance: float
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    reference = tiny_codec().cuda()
    reference.fold_weight_norm()
    channels_last = tiny_codec().cuda()
    channels_last.fold_weight_norm()
    decoder = ChannelsLastDecoder(channels_last, dtype=dtype)
    codes = torch.randint(0, 16, (2, 10, 24), device="cuda")

    with torch.no_grad():
        expected = reference.from_indices(codes.clone())
        decoded = decoder(codes.clone())

    assert decoded.dtype == dtype
    assert decoded.shape == expected.shape
    torch.testing.assert_close(
        decoded.float(), expected, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize(
    "chunk_frame_counts",
    [
        pytest.param([1] * 128, id="frame-by-frame"),
        pytest.param([5, 123], id="crosses-multiple-windows"),
        pytest.param([128], id="all-frames-at-once"),
    ],
)
def test_streaming_crossfade_preserves_audio_across_chunk_sizes(
    chunk_frame_counts: list[int],
) -> None:
    codec = FrameNumberCodec()
    vocoder = Drama3VocoderScheduler(
        codec,
        device="cpu",
        stream_stride=10,
        stream_followup_stride_ramp=(10, 12, 32),
        stream_followup_stride=64,
        stream_overlap_tokens=32,
        stream_crossfade_samples=40,
        max_batch_size=8,
        max_batch_wait_ms=2,
    )
    frames = numbered_frames(128)
    first_frame = 0
    for chunk_id, chunk_frame_count in enumerate(chunk_frame_counts):
        vocoder.handle_stream_chunk(
            "request-a",
            StreamItem(
                chunk_id=chunk_id,
                data=frames[first_frame : first_frame + chunk_frame_count].T,
                from_stage="tts_engine",
            ),
        )
        first_frame += chunk_frame_count
    vocoder.handle_streaming_new_request(
        "request-a", final_payload(frames, is_streaming=True)
    )
    vocoder.handle_stream_done("request-a")

    assert [end for _, end in codec.decoded_frame_ranges()] == [10, 20, 32, 64, 128]
    np.testing.assert_allclose(
        streamed_audio(drained(vocoder)),
        np.repeat(np.arange(128, dtype=np.float32), SAMPLES_PER_FRAME),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "stream_followup_stride_ramp",
    [
        pytest.param((0,), id="zero"),
        pytest.param((2, 0), id="zero-after-positive"),
    ],
)
def test_a_ramp_with_a_non_positive_stride_is_refused(
    stream_followup_stride_ramp: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="stream_followup_stride_ramp"):
        Drama3VocoderScheduler(
            FrameNumberCodec(),
            device="cpu",
            stream_stride=10,
            stream_followup_stride_ramp=stream_followup_stride_ramp,
            stream_followup_stride=64,
            stream_overlap_tokens=32,
            stream_crossfade_samples=0,
            max_batch_size=8,
            max_batch_wait_ms=2,
        )


STREAMER_EOS_TOKEN_ID = 4999


@pytest.mark.parametrize(
    "stream_crossfade_samples",
    [
        pytest.param(
            Drama3VocoderFactoryArgs().stream_crossfade_samples, id="default-crossfade"
        ),
        pytest.param(10 * SAMPLES_PER_FRAME, id="crossfade-holds-back-a-whole-window"),
    ],
)
def test_the_frame_streamer_has_sent_every_window_when_the_vocoder_decodes_it(
    stream_crossfade_samples: int,
) -> None:
    factory_args = Drama3VocoderFactoryArgs(
        stream_crossfade_samples=stream_crossfade_samples
    )
    frame_streamer = Drama3FrameStreamer(
        eos_token_id=STREAMER_EOS_TOKEN_ID,
        stream_stride=factory_args.stream_stride,
        stream_followup_stride_ramp=factory_args.stream_followup_stride_ramp,
        stream_followup_stride=factory_args.stream_followup_stride,
    )
    codec = FrameNumberCodec()
    vocoder = vocoder_with(codec, factory_args=factory_args)
    frame_count = 200
    frames = numbered_frames(frame_count)
    request_data = Drama3RequestData(
        stage_payload=final_payload(frames, is_streaming=True),
        frame_rows=[[0] * (1 + NUM_CODEBOOKS)] * PROMPT_ROW_COUNT,
        prompt_row_count=PROMPT_ROW_COUNT,
    )
    streamed_counts: list[int] = []
    for chunk_id, frame_row in enumerate(frames.tolist()):
        request_data.frame_rows.append(frame_row)
        for message in frame_streamer(
            "request-a", request_data, RequestOutput(request_id="request-a")
        ):
            vocoder.handle_stream_chunk(
                "request-a",
                StreamItem(
                    chunk_id=chunk_id, data=message.data, from_stage="tts_engine"
                ),
            )
        streamed_counts.append(request_data.streamed_frame_count)

    window_ends = sorted(
        {
            frames_in_complete_windows(
                sampled_count,
                stream_stride=factory_args.stream_stride,
                stream_followup_stride_ramp=factory_args.stream_followup_stride_ramp,
                stream_followup_stride=factory_args.stream_followup_stride,
            )
            for sampled_count in range(1, frame_count + 1)
        }
        - {0}
    )
    assert [end for _, end in codec.decoded_frame_ranges()] == window_ends
    assert [streamed_counts[end - 1] for end in window_ends] == window_ends


def vocoded_through_the_engine(
    frames: torch.Tensor, *, ends_with_eos: bool, is_window_batched: bool
) -> list[OutgoingMessage]:
    factory_args = Drama3VocoderFactoryArgs()
    frame_streamer = Drama3FrameStreamer(
        eos_token_id=STREAMER_EOS_TOKEN_ID,
        stream_stride=factory_args.stream_stride,
        stream_followup_stride_ramp=factory_args.stream_followup_stride_ramp,
        stream_followup_stride=factory_args.stream_followup_stride,
    )
    vocoder = vocoder_with(FrameNumberCodec(), factory_args=factory_args)
    payload = final_payload(
        frames,
        is_streaming=True,
        finish_reason="stop" if ends_with_eos else "length",
    )
    request_data = Drama3RequestData(
        stage_payload=payload,
        frame_rows=[[0] * (1 + NUM_CODEBOOKS)] * PROMPT_ROW_COUNT,
        prompt_row_count=PROMPT_ROW_COUNT,
    )
    eos_rows = [[STREAMER_EOS_TOKEN_ID] + [0] * NUM_CODEBOOKS] if ends_with_eos else []
    chunk_id = 0
    for frame_row in frames.tolist() + eos_rows:
        request_data.frame_rows.append(frame_row)
        if is_window_batched:
            chunks = [
                message.data
                for message in frame_streamer(
                    "request-a", request_data, RequestOutput(request_id="request-a")
                )
            ]
        elif frame_row[0] == STREAMER_EOS_TOKEN_ID:
            chunks = []
        else:
            chunks = [torch.tensor([frame_row], dtype=torch.int64).T]
        for chunk in chunks:
            vocoder.handle_stream_chunk(
                "request-a",
                StreamItem(chunk_id=chunk_id, data=chunk, from_stage="tts_engine"),
            )
            chunk_id += 1
    vocoder.handle_streaming_new_request("request-a", payload)
    vocoder.handle_stream_done("request-a")
    return drained(vocoder)


@pytest.mark.parametrize(
    ("frame_count", "ends_with_eos"),
    [
        pytest.param(15, True, id="eos-inside-a-window"),
        pytest.param(15, False, id="max-new-tokens-inside-a-window"),
        pytest.param(20, False, id="max-new-tokens-on-a-window-end"),
        pytest.param(19, True, id="eos-on-the-window-end-step"),
        pytest.param(9, True, id="eos-before-the-first-window"),
    ],
)
def test_window_batched_frames_vocode_to_the_bytes_of_one_frame_per_message(
    frame_count: int, ends_with_eos: bool
) -> None:
    frames = numbered_frames(frame_count)

    batched = vocoded_through_the_engine(
        frames, ends_with_eos=ends_with_eos, is_window_batched=True
    )
    one_per_frame = vocoded_through_the_engine(
        frames, ends_with_eos=ends_with_eos, is_window_batched=False
    )

    assert [(message.type, message.data) for message in batched] == [
        (message.type, message.data) for message in one_per_frame
    ]
    assert [message.type for message in batched][-1] == "result"
    assert chunk_frame_ranges(batched)[-1][1] == frame_count


@pytest.mark.parametrize(
    ("frame_counts", "max_padded_frames_per_decode"),
    [
        pytest.param([3, 5, 2], 15, id="fits-one-batch"),
        pytest.param([4, 10, 7, 10, 2], 20, id="requires-splitting"),
        pytest.param([5, 7, 3], 12, id="uneven-batches"),
        pytest.param([30], 20, id="single-long-request"),
    ],
)
def test_non_streaming_batch_preserves_request_audio_with_bounded_padding(
    frame_counts: list[int], max_padded_frames_per_decode: int
) -> None:
    codec = FrameNumberCodec()
    vocoder = vocoder_with(
        codec, max_padded_frames_per_decode=max_padded_frames_per_decode
    )
    payloads: list[StagePayload] = []
    for request_index, frame_count in enumerate(frame_counts):
        frames = numbered_frames(frame_count)
        frames[:, 1] += request_index * 100
        payload = final_payload(frames, is_streaming=False)
        payload.request_id = f"request-{request_index}"
        payloads.append(payload)

    results = vocoder.vocode_payloads(payloads)

    assert [result.request_id for result in results] == [
        payload.request_id for payload in payloads
    ]
    for request_index, (result, frame_count) in enumerate(
        zip(results, frame_counts, strict=True)
    ):
        assert isinstance(result.data, dict)
        expected_samples = np.repeat(
            np.arange(frame_count, dtype=np.float32) + request_index * 100,
            SAMPLES_PER_FRAME,
        )
        np.testing.assert_array_equal(result.data["audio_data"], expected_samples)
    assert all(
        codes.shape[0] == 1
        or codes.shape[0] * codes.shape[-1] <= max_padded_frames_per_decode
        for codes in codec.decoded_codes
    )
