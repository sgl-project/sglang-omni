# SPDX-License-Identifier: Apache-2.0
"""Incremental code2wav decoding must reproduce whole-utterance decoding."""

import threading

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
from sglang_omni.models.nemotron_voicechat.codec_cuda_graph import (
    CodecDecodeGraphRunner,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms.device_graph import CudaDeviceGraphBackend
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage

NUM_QUANTIZERS = 4
SAMPLES_PER_FRAME = 512
CODEBOOK_SIZE = 32
# note (Xinhao Tan): the checkpoint codec's rates, kernel and STFT sizes with
# narrow channels, so every window shape is captured at a fraction of the cost.
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
    num_quantizers = NUM_QUANTIZERS

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


@pytest.mark.parametrize("rows_per_push", [1, DECODE_WINDOW_FRAMES + 1])
def test_retained_codes_stay_within_the_decode_window(rows_per_push):
    codec = StreamingCodec(FrameLocalDecoder(), "cpu")
    codes = make_codes(DECODE_WINDOW_FRAMES * 3)
    for start in range(0, codes.shape[0], rows_per_push):
        codec.push(codes[start : start + rows_per_push])
    assert codec.window_codes_TQ.shape[0] == DECODE_WINDOW_FRAMES


def make_small_decoder(device: str) -> RVQVAEDecoder:
    torch.manual_seed(0)
    decoder = RVQVAEDecoder(SMALL_CODEC_CONFIG)
    for parameter in decoder.parameters():
        nn.init.normal_(parameter, std=0.3)
    decoder.control_codes.copy_(torch.arange(CODEBOOK_SIZE, CODEBOOK_SIZE + 3))
    decoder.silence_codes.zero_()
    return decoder.to(device).eval()


def make_graph_runner() -> tuple[RVQVAEDecoder, CodecDecodeGraphRunner]:
    device = torch.device("cuda")
    decoder = make_small_decoder("cuda")
    runner = CodecDecodeGraphRunner(
        decoder, CudaDeviceGraphBackend(), device, DECODE_WINDOW_FRAMES
    )
    return decoder, runner


def make_random_codes(window_frames: int) -> torch.Tensor:
    return torch.randint(
        0, CODEBOOK_SIZE + 3, (window_frames, NUM_QUANTIZERS), device="cuda"
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires a GPU")
def test_graph_replay_matches_eager_decode_for_every_window_size():
    decoder, runner = make_graph_runner()
    with torch.inference_mode():
        for window_frames in range(1, DECODE_WINDOW_FRAMES + 1):
            codes_TQ = make_random_codes(window_frames)
            torch.testing.assert_close(
                runner(codes_TQ), decoder(codes_TQ), rtol=0, atol=0
            )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires a GPU")
def test_graph_replay_result_survives_later_replays():
    _, runner = make_graph_runner()
    with torch.inference_mode():
        held = runner(make_random_codes(DECODE_WINDOW_FRAMES))
        expected = held.clone()
        runner(make_random_codes(3))
        runner(make_random_codes(DECODE_WINDOW_FRAMES))
    torch.testing.assert_close(held, expected, rtol=0, atol=0)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires a GPU")
def test_graph_runner_decodes_a_window_without_a_graph_eagerly():
    decoder, runner = make_graph_runner()
    with torch.inference_mode():
        codes_TQ = make_random_codes(DECODE_WINDOW_FRAMES + 1)
        torch.testing.assert_close(runner(codes_TQ), decoder(codes_TQ), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_batched_decode_matches_each_window_decoded_alone(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # note (Xinhao Tan): the batch size picks the convolution algorithm. In float32
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


def stream_interleaved(
    scheduler: NemotronCode2WavScheduler,
    codes_by_request: dict[str, torch.Tensor],
    start_steps: dict[str, int],
) -> list[OutgoingMessage]:
    outputs: list[OutgoingMessage] = []
    total_steps = max(
        start_steps[request_id] + codes.shape[0]
        for request_id, codes in codes_by_request.items()
    )
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
                if 0 <= step - start_step < codes_by_request[request_id].shape[0]
            ],
        )
    return outputs


def streamed_audio(outputs: list[OutgoingMessage], request_id: str) -> torch.Tensor:
    return torch.cat(
        [
            waveform(output.data)
            for output in outputs
            if output.request_id == request_id and output.type == "stream"
        ]
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
    outputs = stream_interleaved(scheduler, codes_by_request, start_steps)
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
        (result,) = [
            output
            for output in outputs
            if output.request_id == request_id and output.type == "result"
        ]
        torch.testing.assert_close(
            streamed_audio(outputs, request_id), expected, rtol=0, atol=0
        )
        torch.testing.assert_close(waveform(result.data.data), expected, rtol=0, atol=0)
    assert (2, 1) in decoder.batch_shapes
    # note (Xinhao Tan): a batch of 3 windows is padded to a power of two.
    assert (4, DECODE_WINDOW_FRAMES) in decoder.batch_shapes


BAD_FRAME_ID = 999


class FailingFrameDecoder(FrameLocalDecoder):
    def decode_batch(self, codes_BTQ: torch.Tensor) -> torch.Tensor:
        if (codes_BTQ[:, :, 0] == BAD_FRAME_ID).any():
            raise RuntimeError("codec failed")
        else:
            pass
        return super().decode_batch(codes_BTQ)


@pytest.mark.parametrize("failing_phase", ["plan", "decode"])
def test_a_failure_errors_only_its_own_request(failing_phase: str) -> None:
    scheduler = NemotronCode2WavScheduler(
        FailingFrameDecoder(), "cpu", compute_fn=None, max_batch_size=8
    )
    if failing_phase == "plan":
        bad_codes = torch.zeros(1, NUM_QUANTIZERS + 1, dtype=torch.long)
    else:
        bad_codes = make_codes(1) + BAD_FRAME_ID
    outputs = deliver(
        scheduler,
        [
            stream_chunk("a", 0, make_codes(1)),
            stream_chunk("b", 0, bad_codes),
            stream_chunk("c", 0, make_codes(1)),
        ],
    )
    assert sorted((output.request_id, output.type) for output in outputs) == [
        ("a", "stream"),
        ("b", "error"),
        ("c", "stream"),
    ]
    assert scheduler.is_aborted("b")
    assert not scheduler.is_aborted("a") and not scheduler.is_aborted("c")


def test_decode_runs_without_the_state_lock() -> None:
    lock_free_during_decode: list[bool] = []

    def try_lock_from_another_thread() -> None:
        acquired = scheduler.state_lock.acquire(blocking=False)
        lock_free_during_decode.append(acquired)
        if acquired:
            scheduler.state_lock.release()
        else:
            pass

    class LockProbeDecoder(FrameLocalDecoder):
        def decode_batch(self, codes_BTQ: torch.Tensor) -> torch.Tensor:
            probe = threading.Thread(target=try_lock_from_another_thread)
            probe.start()
            probe.join()
            return super().decode_batch(codes_BTQ)

    scheduler = NemotronCode2WavScheduler(
        LockProbeDecoder(), "cpu", compute_fn=None, max_batch_size=8
    )
    deliver(
        scheduler,
        [stream_chunk("a", 0, make_codes(1)), stream_chunk("b", 0, make_codes(1))],
    )
    assert lock_free_during_decode == [True]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires a GPU")
def test_graph_runner_batches_match_each_request_decoded_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # note (Xinhao Tan): steps mix one-window batches (graph replay) and padded
    # batches of several windows (eager).
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    decoder, runner = make_graph_runner()
    scheduler = NemotronCode2WavScheduler(
        runner, "cuda", compute_fn=None, max_batch_size=8
    )
    start_steps = {"early": 0, "late_a": 3, "late_b": 3}
    codes_by_request = {
        request_id: make_random_codes(DECODE_WINDOW_FRAMES + 6 - start_step)
        for request_id, start_step in start_steps.items()
    }
    outputs = stream_interleaved(scheduler, codes_by_request, start_steps)
    for request_id, codes in codes_by_request.items():
        alone = StreamingCodec(decoder, "cuda")
        expected = torch.cat([alone.push(row[None]) for row in codes])
        torch.testing.assert_close(
            streamed_audio(outputs, request_id),
            expected,
            rtol=0,
            atol=1e-5 * expected.abs().max().item(),
        )
