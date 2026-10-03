# SPDX-License-Identifier: Apache-2.0
"""Incremental code2wav decoding must reproduce whole-utterance decoding."""

from unittest.mock import MagicMock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.code2wav_stream import (
    DECODE_WINDOW_FRAMES,
    TAIL_HOLDBACK_SAMPLES,
    NemotronCode2WavScheduler,
    StreamingCodec,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload

NUM_QUANTIZERS = 4
SAMPLES_PER_FRAME = 512
CUDA_DELAY_CYCLES = 100_000_000


class FrameLocalDecoder:
    """Decoder without cross-frame context: sample = frame id + intra-frame ramp.

    Frame-causal on the left, no lookahead, so the streaming path must match
    the whole-utterance render exactly once the holdback is accounted for.
    """

    samples_per_frame = SAMPLES_PER_FRAME

    def __init__(self):
        self.calls: list[int] = []

    def __call__(self, codes_TQ: torch.Tensor) -> torch.Tensor:
        self.calls.append(int(codes_TQ.shape[0]))
        frame_id = codes_TQ[:, 0].to(torch.float32)  # quantizer 0 carries the id
        ramp = (
            torch.arange(SAMPLES_PER_FRAME, dtype=torch.float32, device=codes_TQ.device)
            / SAMPLES_PER_FRAME
        )
        return (frame_id[:, None] * 1000.0 + ramp[None, :]).reshape(-1)


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


def make_scheduler(
    device: str | torch.device, *, can_use_local_code_handoff: bool = False
) -> NemotronCode2WavScheduler:
    scheduler = NemotronCode2WavScheduler(
        FrameLocalDecoder(),
        device,
        compute_fn=None,
        can_use_local_code_handoff=can_use_local_code_handoff,
    )
    scheduler.stream_payloads["r"] = StagePayload("r", OmniRequest({}), {})
    return scheduler


@pytest.mark.parametrize(
    "device,can_use_local_code_handoff,expect_stream",
    [
        ("cpu", False, False),
        ("cpu", True, False),
        ("cuda:0", False, False),
        ("cuda:0", True, True),
    ],
)
def test_decode_stream_selection_and_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    can_use_local_code_handoff: bool,
    expect_stream: bool,
) -> None:
    create_stream, current_stream = MagicMock(), MagicMock()
    monkeypatch.setattr(torch.cuda, "Stream", create_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    scheduler = make_scheduler(
        device, can_use_local_code_handoff=can_use_local_code_handoff
    )
    if expect_stream:
        assert scheduler.decode_stream is create_stream.return_value
        create_stream.assert_called_once_with(device=torch.device(device))
        scheduler.decode_stream.wait_stream.assert_called_once_with(
            current_stream.return_value
        )
    else:
        assert scheduler.decode_stream is None
        create_stream.assert_not_called()
        current_stream.assert_not_called()
    if device == "cpu":
        scheduler.on_stream_chunk("r", StreamItem(0, make_codes(1), "talker"))
        result = scheduler.on_stream_done("r")[-1]
        assert result.data.data["audio_waveform_shape"] == [SAMPLES_PER_FRAME]
        current_stream.assert_not_called()
    else:
        pass


@pytest.mark.parametrize("metadata", [None, {}, {"codes_ready_event": "invalid"}])
def test_cuda_chunk_requires_ready_event(metadata: dict[str, str] | None) -> None:
    scheduler = make_scheduler("cuda:0")
    codes = MagicMock(spec=torch.Tensor, is_cuda=True, device=torch.device("cuda:0"))
    with pytest.raises(RuntimeError, match="request 'r'.*codes_ready_event"):
        scheduler.on_stream_chunk("r", StreamItem(0, codes, "talker", metadata))
    codes.record_stream.assert_not_called()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_identical_codes_replay_waits_for_producer_and_matches_cpu_hop() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    warmup_codes = make_codes(1).to(device)
    warmup_codes.fill_(1)
    warmup_codes.clone()
    torch.cuda._sleep(1)  # noqa: leading-underscore - CUDA delay has no public API
    warmup_codec = StreamingCodec(FrameLocalDecoder(), device)
    warmup_codec.push(warmup_codes)
    warmup_codec.flush()
    torch.cuda.synchronize(device)
    codes = make_codes(DECODE_WINDOW_FRAMES * 2 + 5)
    device_codes = codes.to(device)
    producer_stream = torch.cuda.Stream(device=device)
    replays = []
    for direct_handoff in (False, True):
        scheduler = make_scheduler(device, can_use_local_code_handoff=direct_handoff)
        messages = []
        for chunk_id, frame_codes in enumerate(device_codes.split(3)):
            with torch.cuda.stream(producer_stream):
                chunk = torch.zeros_like(frame_codes)
                torch.cuda._sleep(
                    CUDA_DELAY_CYCLES
                )  # noqa: leading-underscore - CUDA delay has no public API
                chunk.copy_(frame_codes)
                if direct_handoff:
                    event = torch.cuda.Event()
                    event.record()
                    metadata = {"codes_ready_event": event}
                else:
                    chunk = chunk.cpu()
                    metadata = None
            messages += scheduler.on_stream_chunk(
                "r", StreamItem(chunk_id, chunk, "talker", metadata)
            )
        messages += scheduler.on_stream_done("r")
        replays.append(messages)
    baseline, direct = replays
    assert [message.type for message in direct] == [
        message.type for message in baseline
    ]
    assert [message.data for message in direct[:-1]] == [
        message.data for message in baseline[:-1]
    ]
    assert direct[-2].data["audio_waveform_shape"] == [TAIL_HOLDBACK_SAMPLES]
    waveform = direct[-1].data.data
    assert waveform == baseline[-1].data.data
    assert waveform["audio_waveform_shape"] == [len(codes) * SAMPLES_PER_FRAME]
    assert (
        b"".join(message.data["audio_waveform"] for message in direct[:-1])
        == waveform["audio_waveform"]
    )
    assert waveform["audio_waveform"] == FrameLocalDecoder()(codes).numpy().tobytes()
