# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from threading import Thread

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
from sglang_omni.scheduling.messages import IncomingMessage

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
            text = "word" + " more" * (state.decoder_steps - 1)
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


def make_scheduler(
    runner: FakeRunner, *, max_batch_size: int = 4
) -> Nemotron3_5ASRStreamingScheduler:
    return Nemotron3_5ASRStreamingScheduler(
        runner,
        lambda payload: payload,
        batch_compute_fn=lambda payloads: list(payloads),
        prompt_dictionary=runner.prompt_dictionary,
        max_batch_size=max_batch_size,
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
    assert not runner.batches
    scheduler.run_ready_step()
    assert [len(batch) for batch in runner.batches] == [2]
    assert runner.batches[0][0] is not runner.batches[0][1]

    continuation = np.arange(5224, dtype=np.int16)
    scheduler.on_stream_chunk_batch(
        [make_pcm_item("a", continuation), make_pcm_item("b", continuation)]
    )
    scheduler.run_ready_step()
    assert [len(batch) for batch in runner.batches] == [2, 2]
    messages = [scheduler.outbox.get_nowait() for _ in range(4)]
    assert scheduler.outbox.empty()
    assert [message.request_id for message in messages] == ["a", "b", "a", "b"]
    assert all("metrics" not in message.data for message in messages)

    scheduler.handle_stream_done("a")
    scheduler.handle_stream_done("b")
    assert scheduler.outbox.empty()
    assert scheduler.has_ready_work()
    scheduler.run_ready_step()
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
    assert not scheduler.has_ready_work()


def test_scheduler_drains_buffered_windows_without_new_input() -> None:
    runner = FakeRunner()
    scheduler = make_scheduler(runner)
    scheduler.inbox.put(IncomingMessage("r", "new_request", make_payload("r")))
    _, item = make_pcm_item("r", np.arange(20000, dtype=np.int16))
    scheduler.inbox.put(IncomingMessage("r", "stream_chunk", item))
    thread = Thread(target=scheduler.start)
    thread.start()
    try:
        messages = [scheduler.outbox.get(timeout=5) for _ in range(4)]
        assert [message.data["full_text"] for message in messages] == [
            "word",
            "word more",
            "word more more",
            "word more more more",
        ]
        assert all(message.type == "stream" for message in messages)
        assert not scheduler.has_ready_work()
        assert [len(batch) for batch in runner.batches] == [1, 1, 1, 1]

        scheduler.inbox.put(IncomingMessage("r", "stream_done"))
        tail = scheduler.outbox.get(timeout=5)
        result = scheduler.outbox.get(timeout=5)
        assert tail.type == "stream"
        assert tail.data["text"] == " more"
        assert result.type == "result"
        assert result.data.data["text"] == "word more more more more"
        assert result.data.data["model_latency_s"] == pytest.approx(0.005)
        assert result.data.data["usage"]["engine_time_s"] == pytest.approx(0.005)
        assert scheduler.stats()["active_streams"] == 0
    finally:
        scheduler.stop()
        thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.mark.parametrize("done", [False, True])
def test_ready_steps_rotate_requests_and_batch_compatible_windows(done: bool) -> None:
    runner = FakeRunner()
    scheduler = make_scheduler(runner, max_batch_size=2)
    for request_id in ("a", "b", "c"):
        scheduler.handle_streaming_new_request(request_id, make_payload(request_id))
    request_ids = {
        id(state.decode): request_id
        for request_id, state in scheduler.stream_states.items()
    }
    scheduler.on_stream_chunk_batch(
        [
            make_pcm_item(r, np.arange(10000, dtype=np.int16))
            for r in request_ids.values()
        ]
    )
    if done:
        for request_id in request_ids.values():
            scheduler.handle_stream_done(request_id)
    assert not runner.batches

    expected_batches = [["a", "b"], ["c"]] * (3 if done else 2)
    for expected in expected_batches:
        assert scheduler.has_ready_work()
        previous_calls = len(runner.batches)
        scheduler.run_ready_step()
        assert len(runner.batches) == previous_calls + 1
        assert [request_ids[id(state)] for state in runner.batches[-1]] == expected
        assert len({state.decoder_steps for state in runner.batches[-1]}) == 1
    if done:
        scheduler.run_ready_step()
        messages = [
            scheduler.outbox.get_nowait() for _ in range(scheduler.outbox.qsize())
        ]
        assert sorted(m.request_id for m in messages if m.type == "result") == [
            "a",
            "b",
            "c",
        ]
        assert scheduler.stats()["active_streams"] == 0
    assert not scheduler.has_ready_work()


def test_ready_steps_skip_aborted_requests_and_wait_for_eos_at_decode_limit() -> None:
    runner = FakeRunner()
    scheduler = make_scheduler(runner, max_batch_size=1)
    for request_id in ("a", "b"):
        payload = make_payload(request_id)
        payload.request.params["max_new_tokens"] = 1
        scheduler.handle_streaming_new_request(request_id, payload)
    state_b = scheduler.stream_states["b"].decode
    scheduler.on_stream_chunk_batch(
        [make_pcm_item(r, np.arange(20000, dtype=np.int16)) for r in ("a", "b")]
    )
    scheduler.run_ready_step()
    scheduler.abort("b")
    assert not scheduler.has_ready_work()
    scheduler.run_ready_step()
    assert len(runner.batches) == 1
    assert all(state is not state_b for state in runner.batches[0])
    scheduler.handle_stream_done("a")
    assert scheduler.has_ready_work()
    scheduler.run_ready_step()
    assert len(runner.batches) == 1
    assert scheduler.outbox.get_nowait().type == "stream"
    assert scheduler.outbox.get_nowait().type == "result"
    assert scheduler.outbox.empty()
    assert not scheduler.stream_states
    assert not scheduler.has_ready_work()


def test_ready_step_model_failure_cleans_request_and_allows_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner()
    scheduler = make_scheduler(runner, max_batch_size=1)
    aborted = []
    scheduler.abort_callback = aborted.append
    for request_id in ("a", "b"):
        scheduler.handle_streaming_new_request(request_id, make_payload(request_id))
    scheduler.on_stream_chunk_batch(
        [make_pcm_item(r, np.arange(4040, dtype=np.int16)) for r in ("a", "b")]
    )

    def _fail(*args, **kwargs):
        raise RuntimeError("model failure")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "run_streaming_batch", _fail)
        scheduler.run_ready_step()
    error = scheduler.outbox.get_nowait()
    assert (error.request_id, error.type) == ("a", "error")
    assert str(error.data) == "model failure"
    assert aborted == ["a"]
    assert "a" not in scheduler.stream_states
    assert "a" not in scheduler.stream_payloads
    assert scheduler.has_ready_work()
    scheduler.run_ready_step()
    message = scheduler.outbox.get_nowait()
    assert (message.request_id, message.type) == ("b", "stream")
    assert not scheduler.has_ready_work()


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
