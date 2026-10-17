# SPDX-License-Identifier: Apache-2.0
"""Incremental code2wav decoding must reproduce whole-utterance decoding."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from sglang_omni.models.nemotron_voicechat.code2wav_stream import (
    DECODE_WINDOW_FRAMES,
    TAIL_HOLDBACK_SAMPLES,
    NemotronCode2WavScheduler,
    StreamingCodec,
)
from sglang_omni.models.nemotron_voicechat.codec import RVQVAEDecoder
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage

NUM_QUANTIZERS = 4
SAMPLES_PER_FRAME = 512
CODEBOOK_SIZE = 32
# note (Xinhao Tan): the checkpoint codec's rates, kernel and STFT sizes with
# narrow channels, so every window length decodes at a fraction of the cost.
SMALL_CODEC_CONFIG = {
    "num_quantizers": NUM_QUANTIZERS,
    "codebook_size": CODEBOOK_SIZE,
    "latent_size": 8,
    "n_fft": 16,
    "hop_length": 4,
    "base_hidden_size": 8,
    "channel_mult": [1, 2, 4],
    "rates": [7, 7, 9],
    "num_blocks": 3,
    "kernel_size": 7,
    "groups": 1,
    "wav_to_token_ratio": 1764,
}
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.accelerator,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
        ],
    ),
]


class FrameLocalDecoder:
    """Decoder without cross-frame context: sample = frame id + intra-frame ramp.

    Frame-causal on the left, no lookahead, so the streaming path must match
    the whole-utterance render exactly once the holdback is accounted for.
    """

    samples_per_frame = SAMPLES_PER_FRAME

    def __init__(self):
        self.calls: list[int] = []
        self.batch_shapes: list[tuple[int, int]] = []

    def __call__(self, codes_TQ: torch.Tensor) -> torch.Tensor:
        self.calls.append(int(codes_TQ.shape[0]))
        frame_id = codes_TQ[:, 0].to(torch.float32)  # quantizer 0 carries the id
        ramp = torch.arange(SAMPLES_PER_FRAME, dtype=torch.float32) / SAMPLES_PER_FRAME
        return (frame_id[:, None] * 1000.0 + ramp[None, :]).reshape(-1)

    def decode_batch(self, codes_BTQ: torch.Tensor) -> torch.Tensor:
        self.batch_shapes.append((int(codes_BTQ.shape[0]), int(codes_BTQ.shape[1])))
        return torch.stack([self(codes_TQ) for codes_TQ in codes_BTQ])


def make_codes(num_frames: int) -> torch.Tensor:
    codes = torch.zeros(num_frames, NUM_QUANTIZERS, dtype=torch.long)
    codes[:, 0] = torch.arange(num_frames)
    return codes


def test_streaming_matches_whole_utterance_decode():
    num_frames = DECODE_WINDOW_FRAMES * 2 + 5
    codes = make_codes(num_frames)
    full = FrameLocalDecoder()(codes)

    decoder = FrameLocalDecoder()
    codec = StreamingCodec(decoder, "cpu")
    parts = [codec.push(row[None, :]) for row in codes]
    parts.append(codec.flush())
    streamed = torch.cat(parts)

    torch.testing.assert_close(streamed, full, rtol=0, atol=0)
    assert streamed.numel() == num_frames * SAMPLES_PER_FRAME
    assert max(decoder.calls) <= DECODE_WINDOW_FRAMES


def test_each_push_holds_back_the_tail_until_the_next_frame():
    codec = StreamingCodec(FrameLocalDecoder(), "cpu")
    first = codec.push(make_codes(1))
    assert first.numel() == SAMPLES_PER_FRAME - TAIL_HOLDBACK_SAMPLES
    second = codec.push(make_codes(2)[1:])
    # The second push releases the first frame's holdback plus its own share.
    assert second.numel() == SAMPLES_PER_FRAME
    assert codec.flush().numel() == TAIL_HOLDBACK_SAMPLES


def test_multi_row_push_and_empty_flush():
    codec = StreamingCodec(FrameLocalDecoder(), "cpu")
    assert codec.flush().numel() == 0
    out = codec.push(make_codes(3))
    assert out.numel() == 3 * SAMPLES_PER_FRAME - TAIL_HOLDBACK_SAMPLES
    assert codec.emitted_samples == out.numel()


def make_small_decoder(device: str) -> RVQVAEDecoder:
    torch.manual_seed(0)
    decoder = RVQVAEDecoder(SMALL_CODEC_CONFIG)
    for parameter in decoder.parameters():
        nn.init.normal_(parameter, std=0.3)
    decoder.control_codes.copy_(torch.arange(CODEBOOK_SIZE, CODEBOOK_SIZE + 3))
    decoder.silence_codes.zero_()
    return decoder.to(device).eval()


@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_batched_decode_matches_each_window_decoded_alone(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # note (Xinhao Tan): the batch size picks the convolution algorithm; in float32
    # that moves only the last bits, under TF32 about 1e-3, so TF32 stays off.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    decoder = make_small_decoder(device)
    for window_frames in range(1, DECODE_WINDOW_FRAMES + 1):
        codes_BTQ = torch.randint(
            0, CODEBOOK_SIZE + 3, (8, window_frames, NUM_QUANTIZERS), device=device
        )
        audio_BS = decoder.decode_batch(codes_BTQ)
        for codes_TQ, audio_S in zip(codes_BTQ, audio_BS, strict=True):
            alone_S = decoder(codes_TQ)
            torch.testing.assert_close(
                audio_S, alone_S, rtol=0, atol=1e-5 * alone_S.abs().max().item()
            )


def stream_chunk(
    request_id: str, chunk_id: int, codes_TQ: torch.Tensor
) -> IncomingMessage:
    return IncomingMessage(
        request_id, "stream_chunk", StreamItem(chunk_id, codes_TQ, "talker")
    )


def deliver(
    scheduler: NemotronCode2WavScheduler, messages: list[IncomingMessage]
) -> list[OutgoingMessage]:
    for message in messages:
        scheduler.inbox.put(message)
    while not scheduler.inbox.empty() or scheduler.pending_messages:
        scheduler.handle_message(scheduler.next_message(), None)
    outputs: list[OutgoingMessage] = []
    while not scheduler.outbox.empty():
        outputs.append(scheduler.outbox.get_nowait())
    return outputs


def waveform(payload: dict[str, bytes | list[int] | str | int]) -> torch.Tensor:
    return torch.from_numpy(
        np.frombuffer(payload["audio_waveform"], dtype=np.float32).copy()
    )


def test_one_talker_step_decodes_equal_windows_together_and_exactly() -> None:
    decoder = FrameLocalDecoder()
    scheduler = NemotronCode2WavScheduler(
        decoder, "cpu", compute_fn=None, max_batch_size=8
    )
    start_steps = {"early": 0, "late_a": 3, "late_b": 3}
    total_steps = DECODE_WINDOW_FRAMES + 6
    codes_by_request: dict[str, torch.Tensor] = {}
    for offset, (request_id, start_step) in enumerate(start_steps.items()):
        codes = make_codes(total_steps - start_step)
        codes[:, 0] += 100 * offset
        codes_by_request[request_id] = codes
    outputs: list[OutgoingMessage] = []
    for step in range(total_steps):
        outputs += deliver(
            scheduler,
            [
                stream_chunk(
                    request_id,
                    step - start_step,
                    codes_by_request[request_id][step - start_step][None],
                )
                for request_id, start_step in start_steps.items()
                if step >= start_step
            ],
        )
    for request_id in start_steps:
        outputs += deliver(
            scheduler,
            [
                IncomingMessage(request_id, "stream_done"),
                IncomingMessage(
                    request_id,
                    "new_request",
                    StagePayload(request_id, OmniRequest(None), {}),
                ),
            ],
        )

    for request_id, codes in codes_by_request.items():
        expected = FrameLocalDecoder()(codes)
        streamed = [
            waveform(output.data)
            for output in outputs
            if output.request_id == request_id and output.type == "stream"
        ]
        (result,) = [
            output
            for output in outputs
            if output.request_id == request_id and output.type == "result"
        ]
        torch.testing.assert_close(torch.cat(streamed), expected, rtol=0, atol=0)
        torch.testing.assert_close(waveform(result.data.data), expected, rtol=0, atol=0)
    assert (2, 1) in decoder.batch_shapes
    assert (3, DECODE_WINDOW_FRAMES) in decoder.batch_shapes


def test_failed_decode_errors_every_request_of_the_batch() -> None:
    decoder = FrameLocalDecoder()
    decoder.decode_batch = Mock(side_effect=RuntimeError("codec failed"))
    scheduler = NemotronCode2WavScheduler(
        decoder, "cpu", compute_fn=None, max_batch_size=8
    )
    outputs = deliver(
        scheduler,
        [stream_chunk("a", 0, make_codes(1)), stream_chunk("b", 0, make_codes(1))],
    )
    assert [(output.request_id, output.type) for output in outputs] == [
        ("a", "error"),
        ("b", "error"),
    ]
    assert scheduler.is_aborted("a") and scheduler.is_aborted("b")
