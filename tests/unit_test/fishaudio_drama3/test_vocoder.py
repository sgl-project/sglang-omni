# SPDX-License-Identifier: Apache-2.0
"""Drama 3 vocoder windows on CPU with a codec that decodes a frame to its number."""

from __future__ import annotations

import asyncio
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from dac.nn.layers import Snake1d

from sglang_omni.admission import QueueFullError
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
from sglang_omni.models.fishaudio_drama3.stateful_decoder import StatefulDACDecoder
from sglang_omni.models.fishaudio_drama3.stream_windows import (
    frames_in_complete_windows,
    stream_window_lengths,
)
from sglang_omni.models.fishaudio_drama3.vocoder import (
    Drama3VocoderScheduler,
    WindowGraphDecoder,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import (
    CausalConvNet,
    DecoderBlock,
    ResidualUnit,
)
from sglang_omni.models.fishaudio_s2_pro.streaming_vocoder import WindowDecoder
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.messages import CompleteMessage
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.types import RequestOutput
from tests.unit_test.fishaudio_s2_pro.test_vocoder import tiny_codec
from tests.unit_test.fixtures.fishaudio_codec import tiny_fishaudio_codec
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane

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


WINDOWS_TO_214 = [(0, 4), (4, 14), (14, 26), (26, 58), (58, 122), (122, 186)]
WINDOWS_TO_300 = WINDOWS_TO_214 + [(186, 250)]
DECODED_TO_214 = [
    (0, 4),
    (0, 14),
    (0, 26),
    (0, 58),
    (26, 122),
    (90, 186),
    (154, 214),
]
DECODED_TO_300 = [
    (0, 4),
    (0, 14),
    (0, 26),
    (0, 58),
    (26, 122),
    (90, 186),
    (154, 250),
    (218, 300),
]
FINAL_PAYLOAD_ONLY_CASE = pytest.param(
    300,
    [1] * 5,
    WINDOWS_TO_300 + [(250, 300)],
    DECODED_TO_300,
    id="295-frames-only-in-the-final-payload",
)
STREAMING_CASES = [
    pytest.param(
        214,
        [1] * 214,
        WINDOWS_TO_214 + [(186, 214)],
        DECODED_TO_214,
        id="frame-by-frame",
    ),
    pytest.param(
        214,
        [1] * 5 + [150] + [1] * 59,
        WINDOWS_TO_214 + [(186, 214)],
        DECODED_TO_214,
        id="a-150-frame-chunk-midstream",
    ),
    pytest.param(
        300,
        [1] * 5 + [295],
        WINDOWS_TO_300 + [(250, 300)],
        DECODED_TO_300,
        id="a-295-frame-last-chunk",
    ),
    FINAL_PAYLOAD_ONLY_CASE,
    pytest.param(
        122,
        [1] * 122,
        [(0, 4), (4, 14), (14, 26), (26, 58), (58, 122)],
        [(0, 4), (0, 14), (0, 26), (0, 58), (26, 122)],
        id="whole-windows-and-no-tail",
    ),
    pytest.param(3, [1] * 3, [(0, 3)], [(0, 3)], id="only-a-tail"),
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
        pytest.param(4, (10, 12, 32), 64, 32, (4, 14, 26, 58, 96), id="defaults"),
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

    assert decoded_frame_counts == [4, 14, 26, 58, 96, 96, 60]
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


class CpuSnakeDecoder(ChannelsLastDecoder):
    """Runs the real sequence and convolution paths with a CPU Snake boundary."""

    def snake(
        self,
        module: Snake1d,
        hidden_states: torch.Tensor,
        pad_left: int = 0,
        pad_right: int = 0,
    ) -> torch.Tensor:
        self.snake_padding.append((pad_left, pad_right))
        activated = module(hidden_states.transpose(1, 2)).transpose(1, 2).contiguous()
        return F.pad(activated, (0, 0, pad_left, pad_right))


@pytest.mark.parametrize("context", ["sequence", "residual", "block", "decoder"])
@pytest.mark.parametrize("frame_count", [11, 12])
def test_snake_padding_preserves_real_decoder_sequences(
    context: str, frame_count: int
) -> None:
    codec = tiny_codec()
    codec.fold_weight_norm()
    if context == "sequence":
        sequence = torch.nn.Sequential(
            Snake1d(8),
            Snake1d(8),
            CausalConvNet(8, 8, kernel_size=3, dilation=3),
            torch.nn.Tanh(),
            Snake1d(8),
        )
    elif context == "residual":
        sequence = torch.nn.Sequential(ResidualUnit(8, dilation=3, causal=True))
    elif context == "block":
        sequence = torch.nn.Sequential(
            DecoderBlock(input_dim=8, output_dim=4, stride=2, causal=True)
        )
    else:
        sequence = codec.decoder.model
    codec.decoder.model = sequence
    decoder = CpuSnakeDecoder(codec, dtype=torch.float32)
    decoder.snake_padding: list[tuple[int, int]] = []
    with torch.no_grad():
        if context == "decoder":
            codes = torch.randint(0, 16, (2, 10, frame_count))
            expected = codec.from_indices(codes)
            actual = decoder(codes)
        else:
            activation = torch.randn(2, frame_count + 5, 8)[:, :frame_count]
            expected = sequence(activation.transpose(1, 2)).transpose(1, 2)
            actual = decoder.apply_sequence(sequence, activation)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert any(left > 0 for left, _ in decoder.snake_padding)
    if context == "sequence":
        assert decoder.snake_padding == [(0, 0), (6, (-frame_count) % 3), (0, 0)]
    elif context == "block":
        assert decoder.snake_padding[0] == (0, 0)
        assert len(decoder.snake_padding) == 7
    else:
        assert len(decoder.snake_padding) >= 2


@pytest.mark.parametrize("dilation", [1, 3, 9])
@pytest.mark.parametrize("frame_count", [11, 12])
def test_prepadded_causal_conv_keeps_original_length(
    dilation: int, frame_count: int
) -> None:
    codec = tiny_codec()
    module = CausalConvNet(8, 4, kernel_size=3, dilation=dilation)
    codec.decoder.model = torch.nn.Sequential(module)
    decoder = ChannelsLastDecoder(codec, dtype=torch.float32)
    activation = torch.randn(2, frame_count + 5, 8)[:, :frame_count]
    alignment = (-(module.padding + frame_count)) % dilation
    padded = F.pad(activation, (0, 0, module.padding, alignment))
    with torch.no_grad():
        expected = module(activation.transpose(1, 2)).transpose(1, 2)
        actual = decoder.causal_conv(module, padded, original_frame_count=frame_count)
        fallback = decoder.causal_conv(module, activation)
    assert actual.shape == (2, frame_count, 4)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual, fallback)


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    ("frame_count", "channel_count", "pad_left", "dilation", "trim_frames"),
    [(11, 7, 0, 1, 0), (13, 65, 6, 3, 5), (17, 96, 54, 9, 3)],
)
def test_fused_snake_padding_is_bitwise_equal_to_snake_then_pad(
    dtype: torch.dtype,
    batch_size: int,
    frame_count: int,
    channel_count: int,
    pad_left: int,
    dilation: int,
    trim_frames: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    generator = torch.Generator(device="cuda").manual_seed(7)
    activation = torch.randn(
        batch_size,
        frame_count + trim_frames,
        channel_count,
        generator=generator,
        device="cuda",
        dtype=dtype,
    )[:, :frame_count]
    alpha = torch.rand(channel_count, generator=generator, device="cuda") * 4 + 0.05
    inverse_alpha = (alpha + 1e-9).reciprocal()
    pad_right = (-(pad_left + frame_count)) % dilation
    unpadded = torch.empty(activation.shape, device="cuda", dtype=dtype)
    output = torch.empty(
        (batch_size, frame_count + pad_left + pad_right, channel_count),
        device="cuda",
        dtype=dtype,
    )
    for target, left, right in [(unpadded, 0, 0), (output, pad_left, pad_right)]:
        rows = batch_size * target.shape[1]
        snake_channels_last_kernel[((rows + 31) // 32, (channel_count + 63) // 64)](
            activation,
            alpha,
            inverse_alpha,
            target,
            rows,
            frame_count,
            channel_count,
            activation.stride(0) // channel_count,
            BLOCK_ROWS=32,
            BLOCK_CHANNELS=64,
            PAD_LEFT=left,
            pad_right=right,
        )
    expected = F.pad(unpadded, (0, 0, pad_left, pad_right))
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_snake_new_lengths_reuse_the_compiled_kernel(dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    alpha = torch.ones(96, device="cuda")
    inverse_alpha = (alpha + 1e-9).reciprocal()
    compiled_hashes: set[str] = set()
    for frame_count in [1, 8, 9, 16, 17, 31, 32, 33, 64, 127]:
        pad_right = (-frame_count) % 9
        activation = torch.randn(2, frame_count + 3, 96, device="cuda", dtype=dtype)
        output = torch.empty(
            (2, frame_count + 54 + pad_right, 96), device="cuda", dtype=dtype
        )
        rows = 2 * output.shape[1]
        compiled = snake_channels_last_kernel[((rows + 31) // 32, 2)](
            activation,
            alpha,
            inverse_alpha,
            output,
            rows,
            frame_count,
            96,
            frame_count + 3,
            BLOCK_ROWS=32,
            BLOCK_CHANNELS=64,
            PAD_LEFT=54,
            pad_right=pad_right,
        )
        compiled_hashes.add(compiled.hash)
    assert len(compiled_hashes) == 1


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
    stream_followup_stride_ramp: tuple[int, ...],
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


def stateful_vocoder(capacity: int = 2) -> Drama3VocoderScheduler:
    codec = tiny_fishaudio_codec()
    factory = Drama3VocoderFactoryArgs()
    return Drama3VocoderScheduler(
        codec,
        device="cpu",
        stream_stride=factory.stream_stride,
        stream_followup_stride_ramp=factory.stream_followup_stride_ramp,
        stream_followup_stride=factory.stream_followup_stride,
        stream_overlap_tokens=factory.stream_overlap_tokens,
        stream_crossfade_samples=0,
        max_batch_size=8,
        max_batch_wait_ms=0,
        stateful_decoder=StatefulDACDecoder(codec, capacity=capacity),
        stateful_max_streams=capacity,
    )


def native_stream_chunk(
    vocoder: Drama3VocoderScheduler, request_id: str, frames: torch.Tensor
) -> None:
    vocoder.handle_stream_chunk(
        request_id, StreamItem(chunk_id=0, data=frames.T, from_stage="tts_engine")
    )


def random_codec_frames(frame_count: int) -> torch.Tensor:
    return torch.cat(
        (
            torch.full((frame_count, 1), 5000, dtype=torch.long),
            torch.randint(0, 16, (frame_count, 10)),
        ),
        dim=1,
    )


@pytest.mark.parametrize(
    "frame_count,chunk_counts,done_first",
    [
        (7, [1, 3], False),
        (64, [64], False),
        (128, [1] * 128, False),
        (214, [5, 150, 59], False),
        (31, [], True),
        (32, [7, 3], True),
    ],
)
def test_stateful_native_scheduler_matches_full_codec(
    frame_count: int, chunk_counts: list[int], done_first: bool
) -> None:
    vocoder = stateful_vocoder()
    frames = random_codec_frames(frame_count)
    with torch.no_grad():
        expected = vocoder.codec.from_indices(frames[:, 1:].T[None].clone())[0, 0]
    offset = 0
    for count in chunk_counts:
        native_stream_chunk(vocoder, "request-a", frames[offset : offset + count])
        offset += count
    if done_first:
        vocoder.handle_stream_done("request-a")
    else:
        pass
    vocoder.handle_streaming_new_request(
        "request-a", final_payload(frames, is_streaming=True)
    )
    if not done_first:
        vocoder.handle_stream_done("request-a")
    else:
        pass
    messages = drained(vocoder)
    assert messages[-1].type == "result"
    actual = streamed_audio(messages)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-4, rtol=1e-4)
    assert actual.size == frame_count * vocoder.codec.frame_length
    assert vocoder.request_slots == {}
    assert messages[-1].data.data["usage"]["completion_tokens"] == frame_count


@pytest.mark.parametrize(
    "first_frames,ramp_frames,followup_frames",
    [(65, (10,), 64), (10, (65,), 64), (10, (10,), 128)],
)
def test_stateful_large_windows_preserve_audio_and_chunk_boundaries(
    first_frames: int, ramp_frames: tuple[int, ...], followup_frames: int
) -> None:
    """Configured windows above 64 frames must not exceed the decoder delta limit."""
    vocoder = stateful_vocoder()
    vocoder.stream_stride = first_frames
    vocoder.stream_followup_stride_ramp = ramp_frames
    vocoder.stream_followup_stride = followup_frames
    frames = random_codec_frames(210)
    with torch.no_grad():
        expected = vocoder.codec.from_indices(frames[:, 1:].T[None].clone())[0, 0]
    native_stream_chunk(vocoder, "request-a", frames)
    vocoder.handle_streaming_new_request(
        "request-a", final_payload(frames, is_streaming=True)
    )
    vocoder.handle_stream_done("request-a")
    messages = drained(vocoder)
    np.testing.assert_allclose(
        streamed_audio(messages), expected.numpy(), atol=1e-4, rtol=1e-4
    )
    chunks = [message for message in messages if message.type == "stream"]
    assert (
        len(chunks[0].data["audio_waveform"])
        == first_frames * vocoder.codec.frame_length * 4
    )
    assert vocoder.request_slots == {}


def test_stateful_capacity_lag_abort_interleave_and_slot_reuse() -> None:
    vocoder = stateful_vocoder(capacity=1)
    first_window_frames = vocoder.stream_stride
    frames = random_codec_frames(20)
    native_stream_chunk(vocoder, "first", frames[:first_window_frames])
    first_messages = drained(vocoder)
    saved_bytes = first_messages[0].data["audio_waveform"]
    native_stream_chunk(vocoder, "waiting", frames[:first_window_frames])
    assert "waiting" in vocoder.window_fallback_requests
    assert set(vocoder.request_slots) == {"first"}
    fallback_messages = drained(vocoder)
    with torch.no_grad():
        expected = vocoder.codec.from_indices(
            frames[:first_window_frames, 1:].T[None].clone()
        )[0, 0]
    np.testing.assert_allclose(
        streamed_audio(fallback_messages), expected.numpy(), atol=1e-4, rtol=1e-4
    )
    vocoder.abort("waiting")
    assert "waiting" not in vocoder.window_fallback_requests
    assert "first" in vocoder.request_slots
    native_stream_chunk(vocoder, "first", frames[first_window_frames:])
    first_payload = final_payload(frames, is_streaming=True)
    first_payload.request_id = "first"
    vocoder.handle_streaming_new_request("first", first_payload)
    vocoder.handle_stream_done("first")
    assert not vocoder.request_slots
    native_stream_chunk(vocoder, "reused", frames[:first_window_frames])
    assert saved_bytes == first_messages[0].data["audio_waveform"]
    reused_messages = drained(vocoder)
    reused = [message for message in reused_messages if message.request_id == "reused"]
    assert reused[0].data["audio_waveform"] == saved_bytes
    vocoder.abort("reused")
    assert not vocoder.request_slots


def test_full_arena_fallback_stays_window_based_until_completion() -> None:
    vocoder = stateful_vocoder(capacity=1)
    frames = random_codec_frames(35)
    native_stream_chunk(vocoder, "owner", frames[:10])
    drained(vocoder)
    native_stream_chunk(vocoder, "overflow", frames[:10])
    first = drained(vocoder)
    assert "overflow" in vocoder.window_fallback_requests
    vocoder.abort("owner")
    native_stream_chunk(vocoder, "overflow", frames[10:])
    assert "overflow" not in vocoder.request_slots
    payload = final_payload(frames, is_streaming=True)
    payload.request_id = "overflow"
    vocoder.handle_streaming_new_request("overflow", payload)
    vocoder.handle_stream_done("overflow")
    messages = first + drained(vocoder)
    assert streamed_audio(messages).size == frames.shape[0] * vocoder.codec.frame_length
    assert len([message for message in messages if message.type == "result"]) == 1
    assert "overflow" not in vocoder.window_fallback_requests
    assert not vocoder.request_slots
    native_stream_chunk(vocoder, "replacement", frames[:10])
    assert "replacement" in vocoder.request_slots
    vocoder.abort("replacement")


def test_stateful_interleaved_requests_keep_independent_audio() -> None:
    vocoder = stateful_vocoder()
    requests = {name: random_codec_frames(33) for name in ("left", "right")}
    for start, end in ((0, 10), (10, 19), (19, 33)):
        for name, frames in requests.items():
            native_stream_chunk(vocoder, name, frames[start:end])
    for name, frames in requests.items():
        payload = final_payload(frames, is_streaming=True)
        payload.request_id = name
        vocoder.handle_streaming_new_request(name, payload)
        vocoder.handle_stream_done(name)
    messages = drained(vocoder)
    for name, frames in requests.items():
        with torch.no_grad():
            expected = vocoder.codec.from_indices(frames[:, 1:].T[None].clone())[0, 0]
        np.testing.assert_allclose(
            streamed_audio(
                [message for message in messages if message.request_id == name]
            ),
            expected.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )


def test_stateful_failed_decode_waits_before_reset_and_reuse() -> None:
    vocoder = stateful_vocoder(capacity=1)
    frames = random_codec_frames(10)
    assert vocoder.stateful_decoder is not None
    original_decode_slot = vocoder.stateful_decoder.decode_slot

    def execute_then_fail(
        codes: torch.Tensor, slot_index: torch.Tensor, frame_count: int
    ) -> torch.Tensor:
        original_decode_slot(codes, slot_index, frame_count)
        raise RuntimeError("decode failed")

    with patch.object(
        vocoder.stateful_decoder,
        "decode_slot",
        side_effect=execute_then_fail,
    ):
        with pytest.raises(RuntimeError, match="decode failed"):
            native_stream_chunk(vocoder, "failed", frames)
    entered = Event()
    completed = Event()

    def wait_for_completion() -> None:
        entered.set()
        assert completed.wait(timeout=10)

    with patch.object(
        vocoder, "wait_for_stream_completion", side_effect=wait_for_completion
    ):
        abort_thread = Thread(target=vocoder.abort, args=("failed",))
        abort_thread.start()
        assert entered.wait(timeout=10)
        assert "failed" in vocoder.request_slots
        assert vocoder.request_slots["failed"].slot_index == 0
        completed.set()
        abort_thread.join(timeout=10)
        assert not abort_thread.is_alive()
    assert not vocoder.request_slots
    native_stream_chunk(vocoder, "fresh", frames)
    assert drained(vocoder)[0].request_id == "fresh"


def test_failed_slot_release_keeps_owner_for_retry() -> None:
    """A failed release must not orphan the live slot before cleanup can retry."""
    vocoder = stateful_vocoder(capacity=1)
    frames = random_codec_frames(10)
    native_stream_chunk(vocoder, "failed", frames)
    decoder = vocoder.stateful_decoder
    assert decoder is not None
    state = vocoder.request_slots["failed"]
    with patch.object(
        decoder, "release_state", side_effect=RuntimeError("release failed")
    ):
        with pytest.raises(RuntimeError, match="release failed"):
            vocoder.abort("failed")
    assert vocoder.request_slots["failed"] is state
    with pytest.raises(ValueError, match="No free production state slot"):
        decoder.create_state()
    vocoder.abort("failed")
    assert not vocoder.request_slots
    native_stream_chunk(vocoder, "replacement", frames)
    assert "replacement" in vocoder.request_slots
    vocoder.abort("replacement")


def test_stateful_option_preserves_nonstream_full_decode() -> None:
    vocoder = stateful_vocoder()
    frames = random_codec_frames(23)
    with torch.no_grad():
        expected = vocoder.codec.from_indices(frames[:, 1:].T[None].clone())[0, 0]
    result = vocoder.vocode_payload(final_payload(frames, is_streaming=False))
    np.testing.assert_allclose(
        result.data["audio_data"], expected.numpy(), atol=0, rtol=0
    )
    assert not vocoder.request_slots


def test_cancelled_slot_cleanup_precedes_replacement_admission() -> None:
    """Delayed cancellation cannot admit a replacement into a still-owned slot."""

    async def run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["vocoder"],
            max_in_flight=1,
            abort_cleanup_stages=("vocoder",),
        )
        control = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control
        coordinator.register_stage("preprocess", "inproc://preprocess")
        vocoder = stateful_vocoder(capacity=1)
        frames = random_codec_frames(20)
        await coordinator.submit_request("old", "original")
        native_stream_chunk(vocoder, "old", frames[:10])
        assert len(vocoder.request_slots) == 1
        await coordinator.abort("old")
        confirmation_token = control.aborts[-1].cleanup_token
        assert confirmation_token is not None
        with pytest.raises(QueueFullError):
            await coordinator.submit_request("new", "replacement")
        assert "old" in vocoder.request_slots
        vocoder.abort("old")
        native_stream_chunk(vocoder, "old", frames[:10])
        assert vocoder.request_slots == {}
        confirmation = CompleteMessage(
            "old", "vocoder", True, cleanup_token=confirmation_token
        )
        await coordinator.handle_completion(confirmation)
        await coordinator.handle_completion(confirmation)
        await coordinator.submit_request("new", "replacement")
        native_stream_chunk(vocoder, "new", frames)
        vocoder.handle_streaming_new_request(
            "new", final_payload(frames, is_streaming=True)
        )
        vocoder.handle_stream_done("new")
        assert vocoder.request_slots == {}
        assert streamed_audio(drained(vocoder)).size > 0
        await coordinator.handle_completion(CompleteMessage("new", "vocoder", True))
        assert coordinator.requests == {}

    asyncio.run(run())


def test_stateful_terminal_output_follows_slot_release() -> None:
    """Terminal output must not become visible while its codec slot is owned."""
    vocoder = stateful_vocoder(capacity=1)
    frames = random_codec_frames(20)
    native_stream_chunk(vocoder, "old", frames)
    original_put = vocoder.outbox.put

    def record_output(message: OutgoingMessage) -> None:
        if message.type == "result":
            assert vocoder.request_slots == {}
        else:
            pass
        original_put(message)

    vocoder.outbox.put = record_output
    vocoder.handle_streaming_new_request(
        "old", final_payload(frames, is_streaming=True)
    )
    vocoder.handle_stream_done("old")
    assert any(message.type == "result" for message in drained(vocoder))
