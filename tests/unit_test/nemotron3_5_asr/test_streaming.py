# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.nemotron3_5_asr.model_runner import (
    Nemotron3_5ASRDecodeState,
    Nemotron3_5ASRStreamingBatchResult,
)
from sglang_omni.models.nemotron3_5_asr.streaming import (
    Nemotron3_5ASRAudioWindow,
    Nemotron3_5ASRStreamingChunkSpec,
    Nemotron3_5ASRStreamingScheduler,
    Nemotron3_5ASRStreamState,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload

LOOKAHEAD_3 = Nemotron3_5ASRStreamingChunkSpec(
    sample_rate=16000,
    first_samples=4040,
    subsequent_samples=5520,
    first_frames=25,
    subsequent_frames=32,
    hop_length=160,
    n_fft=512,
    streaming_latency_ms=80,
)


def _payload(request_id: str, *, language: str = "en-US") -> StagePayload:
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None, params={"language": language}),
        data=None,
    )
    payload.external_input_stream = True
    return payload


def _item(request_id: str, samples: np.ndarray) -> tuple[str, StreamItem]:
    return request_id, StreamItem(
        chunk_id=0,
        data=torch.from_numpy(samples.astype(np.int16, copy=False)),
        from_stage="test",
        metadata={"sample_rate": 16000, "modality": "pcm16"},
    )


class _FakeRunner:
    prompt_dictionary = {"auto": 101, "en-US": 0, "zh-CN": 4}
    streaming_chunk_spec = {
        "sample_rate": LOOKAHEAD_3.sample_rate,
        "first_samples": LOOKAHEAD_3.first_samples,
        "subsequent_samples": LOOKAHEAD_3.subsequent_samples,
        "first_frames": LOOKAHEAD_3.first_frames,
        "subsequent_frames": LOOKAHEAD_3.subsequent_frames,
        "hop_length": LOOKAHEAD_3.hop_length,
        "n_fft": LOOKAHEAD_3.n_fft,
        "streaming_latency_ms": LOOKAHEAD_3.streaming_latency_ms,
    }

    def __init__(self) -> None:
        self.batches: list[list[Nemotron3_5ASRDecodeState]] = []
        self.closed = False

    def new_streaming_decode_state(self) -> Nemotron3_5ASRDecodeState:
        return Nemotron3_5ASRDecodeState(tokens=[99], durations=[0])

    def prepare_streaming_chunk(self, waveform, *, language, is_first):
        return SimpleNamespace(waveform=waveform, language=language, is_first=is_first)

    def run_streaming_batch(
        self,
        states,
        chunks,
        *,
        requested_languages,
        max_new_tokens=None,
    ) -> Nemotron3_5ASRStreamingBatchResult:
        del chunks, requested_languages
        self.batches.append(list(states))
        raw_texts = []
        clean_texts = []
        for state in states:
            state.tokens.append(len(state.tokens))
            state.durations.append(1)
            state.decoder_steps += 1
            state.encoder_frames += 2
            text = "word" if len(state.tokens) == 2 else "word more"
            raw_texts.append(f"<en-US> {text}")
            clean_texts.append(text)
        return Nemotron3_5ASRStreamingBatchResult(
            elapsed_s=0.001,
            raw_texts=raw_texts,
            clean_texts=clean_texts,
            languages=["en-US"] * len(states),
            emitted_token_counts=[1] * len(states),
            encoder_frame_counts=[2] * len(states),
        )

    def close(self) -> None:
        self.closed = True


def _scheduler(runner: _FakeRunner) -> Nemotron3_5ASRStreamingScheduler:
    return Nemotron3_5ASRStreamingScheduler(
        runner,
        lambda payload: payload,
        batch_compute_fn=lambda payloads: payloads,
        prompt_dictionary=runner.prompt_dictionary,
        max_batch_size=4,
        max_batch_wait_ms=0,
        max_pending_messages=8,
    )


def test_pcm16_fragmentation_and_final_padding_geometry() -> None:
    state = Nemotron3_5ASRStreamState(
        request_id="r",
        payload=_payload("r"),
        language="en-US",
        spec=LOOKAHEAD_3,
        decode=Nemotron3_5ASRDecodeState(tokens=[99], durations=[0]),
    )
    waveform = np.arange(5000, dtype=np.int16)
    raw_bytes = waveform.astype("<i2", copy=False).view(np.uint8)
    boundaries = (0, 1025, 3074, 7171, raw_bytes.size)
    for start, end in zip(boundaries, boundaries[1:]):
        state.append_pcm16(
            torch.from_numpy(raw_bytes[start:end]),
            {"sample_rate": 16000, "modality": "pcm16"},
        )

    first = state.pop_ready_window()
    assert isinstance(first, Nemotron3_5ASRAudioWindow)
    assert first.raw_start_sample == 0
    assert first.real_samples == 4040
    assert first.right_padding_samples == 0
    assert np.array_equal((first.waveform[:5] * 32768).astype(np.int16), waveform[:5])

    state.mark_done()
    final = state.pop_ready_window(finalizing=True)
    assert final.raw_start_sample == 3744
    assert final.real_samples == 1256
    assert final.right_padding_samples == 4264
    assert state.total_samples / LOOKAHEAD_3.sample_rate == pytest.approx(0.3125)


def test_lookahead_zero_preserves_negative_stft_start() -> None:
    spec = Nemotron3_5ASRStreamingChunkSpec(
        sample_rate=16000,
        first_samples=200,
        subsequent_samples=1680,
        first_frames=1,
        subsequent_frames=8,
        hop_length=160,
        n_fft=512,
        streaming_latency_ms=20,
    )
    state = Nemotron3_5ASRStreamState(
        request_id="r",
        payload=_payload("r"),
        language="en-US",
        spec=spec,
        decode=Nemotron3_5ASRDecodeState(tokens=[99], durations=[0]),
    )
    state.append_pcm16(torch.arange(300, dtype=torch.int16), {"sample_rate": 16000})
    state.pop_ready_window()
    state.mark_done()
    final = state.pop_ready_window(finalizing=True)
    assert final.raw_start_sample == -96
    assert final.left_padding_samples == 96
    assert final.real_samples == 300


def test_scheduler_batches_one_window_per_request_and_cleans_state() -> None:
    runner = _FakeRunner()
    scheduler = _scheduler(runner)
    scheduler._on_streaming_new_request("a", _payload("a"))
    scheduler._on_streaming_new_request("b", _payload("b"))

    first = np.arange(4040, dtype=np.int16)
    scheduler.on_stream_chunk_batch([_item("a", first), _item("b", first)])
    assert [len(batch) for batch in runner.batches] == [2]
    assert (
        scheduler._stream_states["a"].decode is not scheduler._stream_states["b"].decode
    )

    continuation = np.arange(5224, dtype=np.int16)
    scheduler.on_stream_chunk_batch(
        [_item("a", continuation), _item("b", continuation)]
    )
    assert [len(batch) for batch in runner.batches] == [2, 2]
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            break
    assert [message.request_id for message in messages] == ["a", "b", "a", "b"]

    scheduler._on_done("a")
    scheduler._on_done("b")
    results = [scheduler.outbox.get_nowait(), scheduler.outbox.get_nowait()]
    assert all(message.type == "result" for message in results)
    assert scheduler.stats()["active_streams"] == 0


def test_scheduler_processes_at_most_one_window_per_request_per_input_batch() -> None:
    runner = _FakeRunner()
    scheduler = _scheduler(runner)
    scheduler._on_streaming_new_request("r", _payload("r"))

    scheduler.on_stream_chunk_batch([_item("r", np.arange(20000, dtype=np.int16))])

    assert [len(batch) for batch in runner.batches] == [1]
    assert scheduler._stream_states["r"].has_ready_window()


def test_stream_done_rejects_incomplete_pcm16_sample() -> None:
    state = Nemotron3_5ASRStreamState(
        request_id="r",
        payload=_payload("r"),
        language="en-US",
        spec=LOOKAHEAD_3,
        decode=Nemotron3_5ASRDecodeState(tokens=[99], durations=[0]),
    )
    state.append_pcm16(torch.tensor([1, 2, 3], dtype=torch.uint8), {})

    with pytest.raises(ValueError, match="incomplete PCM16"):
        state.mark_done()


def test_scheduler_rejects_non_pcm16_or_wrong_rate() -> None:
    runner = _FakeRunner()
    scheduler = _scheduler(runner)
    scheduler._on_streaming_new_request("r", _payload("r"))
    bad = StreamItem(
        chunk_id=0,
        data=torch.zeros(2, dtype=torch.float32),
        from_stage="test",
        metadata={"sample_rate": 8000},
    )
    scheduler.on_stream_chunk_batch([("r", bad)])
    message = scheduler.outbox.get_nowait()
    assert message.type == "error"
    assert "PCM16" in str(message.data)
    assert scheduler.stats()["active_streams"] == 0
