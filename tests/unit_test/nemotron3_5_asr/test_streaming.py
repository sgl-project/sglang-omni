# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pytest
import torch

from sglang_omni.models.nemotron3_5_asr.model_runner import (
    Nemotron3_5ASRDecodeState,
    Nemotron3_5ASRModelRunner,
    Nemotron3_5ASRPreparedChunk,
    Nemotron3_5ASRStreamingBatchResult,
)
from sglang_omni.models.nemotron3_5_asr.streaming import (
    Nemotron3_5ASRStreamingChunkSpec,
    Nemotron3_5ASRStreamingScheduler,
    Nemotron3_5ASRStreamState,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload

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


def make_payload(request_id: str, *, language: str = "en-US") -> StagePayload:
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None, params={"language": language}),
        data=None,
    )
    payload.external_input_stream = True
    return payload


def make_pcm_item(request_id: str, samples: np.ndarray) -> tuple[str, StreamItem]:
    return request_id, StreamItem(
        chunk_id=0,
        data=torch.from_numpy(samples.astype(np.int16, copy=False)),
        from_stage="test",
        metadata={"sample_rate": 16000, "modality": "pcm16"},
    )


class FakeRunner(Nemotron3_5ASRModelRunner):
    def __init__(self) -> None:
        self.batches: list[list[Nemotron3_5ASRDecodeState]] = []
        self.is_closed = False

    @property
    def prompt_dictionary(self) -> dict[str, int]:
        return {"auto": 101, "en-US": 0, "zh-CN": 4}

    @property
    def streaming_chunk_spec(self) -> dict[str, int]:
        return asdict(LOOKAHEAD_3)

    def new_streaming_decode_state(self) -> Nemotron3_5ASRDecodeState:
        return Nemotron3_5ASRDecodeState(tokens=[99], durations=[0])

    def prepare_streaming_chunk(
        self, waveform: np.ndarray, *, language: str, is_first: bool
    ) -> Nemotron3_5ASRPreparedChunk:
        return Nemotron3_5ASRPreparedChunk(
            input_features=torch.from_numpy(waveform),
            prompt_ids=torch.tensor([self.prompt_dictionary[language]]),
        )

    def run_streaming_batch(
        self,
        states: Sequence[Nemotron3_5ASRDecodeState],
        chunks: Sequence[Nemotron3_5ASRPreparedChunk],
        *,
        requested_languages: Sequence[str],
        max_new_tokens: Sequence[int | None] | None = None,
    ) -> Nemotron3_5ASRStreamingBatchResult:
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
        )

    def close(self) -> None:
        self.is_closed = True


def make_scheduler(runner: FakeRunner) -> Nemotron3_5ASRStreamingScheduler:
    return Nemotron3_5ASRStreamingScheduler(
        runner,
        lambda payload: payload,
        batch_compute_fn=lambda payloads: list(payloads),
        prompt_dictionary=runner.prompt_dictionary,
        max_batch_size=4,
        max_batch_wait_ms=0,
        max_pending_messages=8,
    )


def test_pcm16_fragmentation_and_final_padding_geometry() -> None:
    state = Nemotron3_5ASRStreamState(
        request_id="r",
        payload=make_payload("r"),
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
    np.testing.assert_array_equal(first.waveform, waveform[:4040] / 32768.0)

    state.mark_done()
    final = state.pop_ready_window(finalizing=True)
    np.testing.assert_array_equal(
        final.waveform, np.pad(waveform[3744:] / 32768.0, (0, 4264))
    )
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
        payload=make_payload("r"),
        language="en-US",
        spec=spec,
        decode=Nemotron3_5ASRDecodeState(tokens=[99], durations=[0]),
    )
    state.append_pcm16(torch.arange(300, dtype=torch.int16), {"sample_rate": 16000})
    state.pop_ready_window()
    state.mark_done()
    final = state.pop_ready_window(finalizing=True)
    np.testing.assert_array_equal(
        final.waveform, np.pad(np.arange(300) / 32768.0, (96, 1284))
    )


def test_scheduler_batches_one_window_per_request_and_cleans_state() -> None:
    runner = FakeRunner()
    scheduler = make_scheduler(runner)
    scheduler.handle_streaming_new_request("a", make_payload("a"))
    scheduler.handle_streaming_new_request("b", make_payload("b"))

    first = np.arange(4040, dtype=np.int16)
    scheduler.on_stream_chunk_batch(
        [make_pcm_item("a", first), make_pcm_item("b", first)]
    )
    assert [len(batch) for batch in runner.batches] == [2]
    assert runner.batches[0][0] is not runner.batches[0][1]

    continuation = np.arange(5224, dtype=np.int16)
    scheduler.on_stream_chunk_batch(
        [make_pcm_item("a", continuation), make_pcm_item("b", continuation)]
    )
    assert [len(batch) for batch in runner.batches] == [2, 2]
    messages = [scheduler.outbox.get_nowait() for _ in range(4)]
    assert scheduler.outbox.empty()
    assert [message.request_id for message in messages] == ["a", "b", "a", "b"]
    assert all("metrics" not in message.data for message in messages)

    scheduler.handle_stream_done("a")
    scheduler.handle_stream_done("b")
    results = [scheduler.outbox.get_nowait(), scheduler.outbox.get_nowait()]
    assert all(message.type == "result" for message in results)
    final_payloads = [message.data for message in results]
    assert all(isinstance(payload, StagePayload) for payload in final_payloads)
    assert all(payload.data["text"] == "word more" for payload in final_payloads)
    assert all(
        payload.data["raw_text"] == "<en-US> word more" for payload in final_payloads
    )
    assert all(payload.data["language"] == "en-US" for payload in final_payloads)
    assert all("asr_latency_s" in payload.data for payload in final_payloads)
    assert all(
        payload.data["model_latency_s"] == pytest.approx(0.001)
        for payload in final_payloads
    )
    assert all(
        payload.data["usage"]["engine_time_s"] == pytest.approx(0.001)
        for payload in final_payloads
    )
    assert all("metrics" not in payload.data for payload in final_payloads)
    assert scheduler.stats()["completed_streams"] == 2
    assert scheduler.stats()["active_streams"] == 0


def test_scheduler_processes_at_most_one_window_per_request_per_input_batch() -> None:
    runner = FakeRunner()
    scheduler = make_scheduler(runner)
    scheduler.handle_streaming_new_request("r", make_payload("r"))

    scheduler.on_stream_chunk_batch(
        [make_pcm_item("r", np.arange(20000, dtype=np.int16))]
    )

    assert [len(batch) for batch in runner.batches] == [1]
    scheduler.handle_stream_done("r")
    assert [len(batch) for batch in runner.batches] == [1, 1, 1, 1, 1]
    messages = [scheduler.outbox.get_nowait() for _ in range(scheduler.outbox.qsize())]
    assert messages[-1].type == "result"
    result = messages[-1].data.data
    assert result["model_latency_s"] == pytest.approx(0.005)
    assert result["usage"]["engine_time_s"] == pytest.approx(0.005)


def test_stream_done_rejects_incomplete_pcm16_sample() -> None:
    state = Nemotron3_5ASRStreamState(
        request_id="r",
        payload=make_payload("r"),
        language="en-US",
        spec=LOOKAHEAD_3,
        decode=Nemotron3_5ASRDecodeState(tokens=[99], durations=[0]),
    )
    state.append_pcm16(torch.tensor([1, 2, 3], dtype=torch.uint8), {})

    with pytest.raises(ValueError, match="incomplete PCM16"):
        state.mark_done()


@pytest.mark.parametrize(
    "dtype,sample_rate,error",
    [
        (torch.float32, 16000, "PCM16"),
        (torch.int16, 16000.5, "sample_rate"),
    ],
)
def test_scheduler_rejects_non_pcm16_or_wrong_rate(
    dtype: torch.dtype, sample_rate: int | float, error: str
) -> None:
    runner = FakeRunner()
    scheduler = make_scheduler(runner)
    scheduler.handle_streaming_new_request("r", make_payload("r"))
    bad = StreamItem(
        chunk_id=0,
        data=torch.zeros(2, dtype=dtype),
        from_stage="test",
        metadata={"sample_rate": sample_rate},
    )
    scheduler.on_stream_chunk_batch([("r", bad)])
    message = scheduler.outbox.get_nowait()
    assert message.type == "error"
    assert error in str(message.data)
    assert scheduler.stats()["active_streams"] == 0
