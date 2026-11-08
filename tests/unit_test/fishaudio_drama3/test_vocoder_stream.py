# SPDX-License-Identifier: Apache-2.0
"""The Drama 3 vocoder serves on its own CUDA stream in the engine's process.

Its codes arrive as CPU tensors and its audio leaves as CPU bytes, so the only GPU
state it shares with another stream is its own: the codec weights and the window
graphs' static buffers, which construction writes on the thread's default stream
before serving starts on the decode stream."""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from sglang_omni.models.fishaudio_drama3.channels_last_decoder import (
    ChannelsLastDecoder,
)
from sglang_omni.models.fishaudio_drama3.config import Drama3VocoderFactoryArgs
from sglang_omni.models.fishaudio_drama3.request_builders import Drama3GenerationResult
from sglang_omni.models.fishaudio_drama3.stream_windows import (
    stream_window_lengths,
    utterance_graph_lengths,
)
from sglang_omni.models.fishaudio_drama3.vocoder import (
    Drama3VocoderScheduler,
    WindowGraphDecoder,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from tests.unit_test.fishaudio_s2_pro.test_vocoder import tiny_codec

NUM_CODEBOOKS = 10
CODEBOOK_SIZE = 16
PROMPT_ROW_COUNT = 37
UTTERANCE_GRAPH_MAX_FRAMES = 128
WHOLE_UTTERANCE_FRAMES = 120
# Replays queued on the default stream ahead of serving, standing in for
# construction work still in flight when the serving thread starts.
IN_FLIGHT_REPLAYS = 200
HANDOFF_ATTEMPTS = 5
# (request id, frames, streaming)
REQUESTS = (
    ("stream-a", 150, True),
    ("stream-b", 97, True),
    ("stream-c", 61, True),
    ("whole-d", WHOLE_UTTERANCE_FRAMES, False),
    ("whole-e", 9, False),
)
Outputs = dict[str, list[tuple[str, object]]]


def cuda_vocoder(
    codec: torch.nn.Module, factory_args: Drama3VocoderFactoryArgs
) -> Drama3VocoderScheduler:
    """The vocoder as the stage factory assembles it, with the given codec."""
    decode = ChannelsLastDecoder(codec, dtype=torch.float16)
    window_graph_decoder = WindowGraphDecoder(
        decode,
        num_codebooks=NUM_CODEBOOKS,
        frame_length=int(codec.frame_length),
        device=torch.device("cuda"),
        window_lengths=stream_window_lengths(
            stream_stride=factory_args.stream_stride,
            stream_followup_stride_ramp=factory_args.stream_followup_stride_ramp,
            stream_followup_stride=factory_args.stream_followup_stride,
            stream_overlap_tokens=factory_args.stream_overlap_tokens,
        ),
        utterance_lengths=utterance_graph_lengths(
            UTTERANCE_GRAPH_MAX_FRAMES, frame_step=8
        ),
        max_batch_size=factory_args.stream_window_batch_size,
    )
    return Drama3VocoderScheduler(
        codec,
        device="cuda",
        stream_stride=factory_args.stream_stride,
        stream_followup_stride_ramp=factory_args.stream_followup_stride_ramp,
        stream_followup_stride=factory_args.stream_followup_stride,
        stream_overlap_tokens=factory_args.stream_overlap_tokens,
        stream_crossfade_samples=factory_args.stream_crossfade_samples,
        max_batch_size=factory_args.max_batch_size,
        max_batch_wait_ms=factory_args.max_batch_wait_ms,
        decode_window=window_graph_decoder,
        max_padded_frames_per_decode=None,
        window_batch_decoder=window_graph_decoder,
    )


def random_frames(frame_count: int, seed: int) -> torch.Tensor:
    """[frames, 1 + num_codebooks] CPU rows: a text token, then a code per codebook."""
    generator = torch.Generator().manual_seed(seed)
    frames = torch.randint(
        0, CODEBOOK_SIZE, (frame_count, 1 + NUM_CODEBOOKS), generator=generator
    )
    frames[:, 0] += 5000
    return frames


def request_messages(
    request_id: str, frames: torch.Tensor, *, is_streaming: bool
) -> list[IncomingMessage]:
    """The engine's messages for one request: chunks of a few frames each, the
    terminal payload, then stream done; a non-streaming request is its payload."""
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": is_streaming}),
        data=Drama3GenerationResult(
            frames=frames, prompt_row_count=PROMPT_ROW_COUNT, finish_reason="length"
        ),
    )
    new_request = IncomingMessage(
        request_id=request_id, type="new_request", data=payload
    )
    if not is_streaming:
        return [new_request]
    else:
        pass
    generator = torch.Generator().manual_seed(frames.shape[0])
    messages: list[IncomingMessage] = []
    first_frame = 0
    while first_frame < frames.shape[0]:
        end_frame = min(
            frames.shape[0],
            first_frame + int(torch.randint(1, 8, (1,), generator=generator)),
        )
        messages.append(
            IncomingMessage(
                request_id=request_id,
                type="stream_chunk",
                data=StreamItem(
                    chunk_id=len(messages),
                    data=frames[first_frame:end_frame].T,
                    from_stage="tts_engine",
                ),
            )
        )
        first_frame = end_frame
    return [
        *messages,
        new_request,
        IncomingMessage(request_id=request_id, type="stream_done"),
    ]


def messages_for(run: str) -> list[IncomingMessage]:
    """The same requests under ids unique to one serving run."""
    return [
        message
        for seed, (name, frame_count, is_streaming) in enumerate(REQUESTS, start=1)
        for message in request_messages(
            f"{name}/{run}", random_frames(frame_count, seed), is_streaming=is_streaming
        )
    ]


def served(vocoder: Drama3VocoderScheduler, run: str) -> Outputs:
    """Each request's output messages, as (type, data), from one serving thread."""
    messages = messages_for(run)
    outputs: Outputs = {name: [] for name, _, _ in REQUESTS}
    thread = threading.Thread(target=vocoder.start, daemon=True)
    thread.start()
    try:
        for message in messages:
            vocoder.inbox.put(message)
        unfinished = set(outputs)
        while unfinished:
            out: OutgoingMessage = vocoder.outbox.get(timeout=30.0)
            assert out.type != "error", out.data
            name = out.request_id.rsplit("/", 1)[0]
            data = out.data.data if isinstance(out.data, StagePayload) else out.data
            outputs[name].append((out.type, data))
            if out.type == "result":
                unfinished.discard(name)
            else:
                pass
    finally:
        vocoder.stop()
        thread.join(timeout=5.0)
    return outputs


def samples_of(outputs: list[tuple[str, object]]) -> np.ndarray:
    return np.concatenate(
        [
            np.frombuffer(data["audio_waveform"], dtype=np.float32)
            for _, data in outputs
            if isinstance(data, dict) and "audio_waveform" in data
        ]
    )


class DefaultStreamLoad:
    """A thread that keeps the default stream busy the way the engine does:
    allocating, computing and freeing its own tensors."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.rounds = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(3)
        while not self.stop.is_set():
            size = 512 * (1 + self.rounds % 7)
            left = torch.randn((size, size), device="cuda", generator=generator)
            right = torch.randn((size, 1024), device="cuda", generator=generator)
            (left @ right).sum().item()
            self.rounds += 1

    def __enter__(self) -> DefaultStreamLoad:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=10.0)


@pytest.fixture
def vocoder() -> Drama3VocoderScheduler:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    codec = tiny_codec().cuda()
    codec.fold_weight_norm()
    return cuda_vocoder(codec, Drama3VocoderFactoryArgs())


@pytest.mark.accelerator
def test_serving_on_the_decode_stream_under_default_stream_load_matches_the_default_stream(
    vocoder: Drama3VocoderScheduler,
) -> None:
    decode_stream = vocoder.decode_stream
    assert decode_stream is not None
    vocoder.decode_stream = None
    on_default_stream = served(vocoder, "default")

    vocoder.decode_stream = decode_stream
    with DefaultStreamLoad() as load:
        on_decode_stream = served(vocoder, "decode")

    assert load.rounds > 0
    assert on_decode_stream == on_default_stream
    assert samples_of(on_decode_stream["stream-a"]).size == 150 * int(
        vocoder.codec.frame_length
    )
    assert samples_of(on_decode_stream["whole-d"]).size == WHOLE_UTTERANCE_FRAMES * int(
        vocoder.codec.frame_length
    )


@pytest.mark.accelerator
def test_serving_starts_after_the_default_streams_in_flight_work(
    vocoder: Drama3VocoderScheduler,
) -> None:
    """Construction writes the window graphs' static buffers and their shared pool
    on the default stream. Serving starts on the decode stream after that work, so
    replays still queued on the default stream do not change what serving returns."""
    expected = served(vocoder, "expected")
    decoder = vocoder.window_batch_decoder
    assert isinstance(decoder, WindowGraphDecoder)
    other_codes = torch.randint(
        0, CODEBOOK_SIZE, (1, NUM_CODEBOOKS, WHOLE_UTTERANCE_FRAMES), device="cuda"
    )

    mismatches = 0
    for attempt in range(HANDOFF_ATTEMPTS):
        with torch.no_grad():
            for _ in range(IN_FLIGHT_REPLAYS):
                decoder(other_codes)
        mismatches += served(vocoder, f"attempt-{attempt}") != expected
    torch.cuda.synchronize()

    assert mismatches == 0
