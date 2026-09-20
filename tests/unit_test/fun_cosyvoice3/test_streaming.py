# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from queue import Empty, Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import EOS_ID, VOCAB_SIZE
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MEL_RATIO,
    first_ar_flush_tokens,
    next_stream_hop_len,
    pad_flow_prompt_to_hop,
    prompt_token_pad,
    stream_hop_len,
    tokens_needed_for_causal_chunk,
)
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
    FunCosyVoice3StreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from tests.unit_test.fun_cosyvoice3.test_flow_batch import _FakeFlow as _PackedFlow

AR_INITIAL_FLUSH_TOKENS = TOKEN_HOP_LEN + PRE_LOOKAHEAD_LEN
AR_FOLLOWUP_FLUSH_TOKENS = TOKEN_HOP_LEN


def test_stream_hop_math_matches_cosyvoice3() -> None:
    assert prompt_token_pad(0) == 0
    assert prompt_token_pad(10) == 15
    assert prompt_token_pad(25) == 0
    assert prompt_token_pad(26) == 24
    assert stream_hop_len(0, hop_len=25, prompt_pad=15) == 40
    assert stream_hop_len(40, hop_len=25, prompt_pad=15) == 25
    assert next_stream_hop_len(25) == 50
    assert next_stream_hop_len(50) == 100
    assert next_stream_hop_len(100) == 100
    assert next_stream_hop_len(25, max_hop_len=50) == 50
    assert next_stream_hop_len(50, max_hop_len=50) == 50
    assert next_stream_hop_len(25, disable_growth=True) == 25
    assert tokens_needed_for_causal_chunk(0, hop_len=25, prompt_pad=0) == 28
    assert tokens_needed_for_causal_chunk(0, hop_len=25, prompt_pad=15) == 43
    assert first_ar_flush_tokens(0) == AR_INITIAL_FLUSH_TOKENS
    assert first_ar_flush_tokens(10) == AR_INITIAL_FLUSH_TOKENS
    assert first_ar_flush_tokens(25) == AR_INITIAL_FLUSH_TOKENS
    assert first_ar_flush_tokens(0, hop_len=15) == 18


def test_pad_flow_prompt_repeats_last_frame_to_hop_multiple() -> None:
    token = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.int32)
    feat = (
        torch.arange(10 * TOKEN_MEL_RATIO, dtype=torch.float32)
        .reshape(1, 10 * TOKEN_MEL_RATIO, 1)
        .repeat(1, 1, 80)
    )
    padded_token, padded_feat = pad_flow_prompt_to_hop(token, feat)
    assert tuple(padded_token.shape) == (1, 25)
    assert torch.equal(padded_token[:, :10], token)
    assert torch.equal(padded_token[:, 10:], torch.full((1, 15), 10, dtype=torch.int32))
    assert tuple(padded_feat.shape) == (1, 50, 80)
    assert torch.equal(padded_feat[:, :20], feat)
    assert torch.equal(padded_feat[:, 20:], feat[:, -1:, :].repeat(1, 30, 1))
    aligned_token, aligned_feat = pad_flow_prompt_to_hop(padded_token, padded_feat)
    assert torch.equal(aligned_token, padded_token)
    assert torch.equal(aligned_feat, padded_feat)


class _FakeFlow(_PackedFlow):
    def __init__(self) -> None:
        super().__init__(channels=80, max_frames=512)
        self.spk_embed_affine_layer = torch.nn.Linear(192, 80, bias=False)
        self.input_embedding = torch.nn.Embedding(VOCAB_SIZE, 80)


class _FakeHiFT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[tuple] = []
        self.upsample_rates = [8, 5, 3]
        self.istft_params = {"n_fft": 16, "hop_len": 4}

    def inference(self, *, speech_feat, finalize):
        self.calls.append((speech_feat, finalize))
        return torch.arange(speech_feat.shape[-1]).reshape(1, -1).float(), None


def _drain(scheduler: FunCosyVoice3StreamingVocoderScheduler) -> list[OutgoingMessage]:
    messages: list[OutgoingMessage] = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except Empty:
            return messages


def _serve(scheduler: FunCosyVoice3StreamingVocoderScheduler) -> int:
    steps = 0
    while True:
        try:
            msg = scheduler.get_batch_message()
        except Empty:
            if not scheduler.has_ready_work():
                return steps
            scheduler.run_ready_step()
            steps += 1
            continue
        scheduler.handle_message(msg, None)


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _waveform(data: dict) -> np.ndarray:
    return np.frombuffer(data["audio_waveform"], dtype=np.float32).reshape(
        data["audio_waveform_shape"]
    )


def _scheduler(
    **scheduler_kwargs,
) -> tuple[_FakeFlow, FunCosyVoice3StreamingVocoderScheduler]:
    flow = _FakeFlow()
    return flow, FunCosyVoice3StreamingVocoderScheduler(
        stages.CosyVoice3Vocoder(
            stages.FunCosyVoice3Flow(flow, packed_estimator=flow.packed_estimator),
            _FakeHiFT(),
        ),
        **scheduler_kwargs,
    )


def _estimator_calls(flow: _FakeFlow) -> list[dict]:
    return flow.packed_estimator.calls


def _stream_ids(messages: list[OutgoingMessage]) -> list[str]:
    return [m.request_id for m in messages if m.type == "stream"]


def _hop_frames(flow: _FakeFlow) -> set[int]:
    return {
        length
        for call in _estimator_calls(flow)
        if call["streaming"]
        for length in call["lengths"]
    }


def _window_frames(tokens: int) -> int:
    return (TOKEN_HOP_LEN + tokens - PRE_LOOKAHEAD_LEN) * TOKEN_MEL_RATIO


def _model_runner() -> FunCosyVoice3ModelRunner:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner._token_hop_len = TOKEN_HOP_LEN
    runner._ar_followup_flush_tokens = AR_FOLLOWUP_FLUSH_TOKENS
    runner._outbox = Queue()
    runner._vocoder_target = "vocoder"
    return runner


def _stream_payload(
    request_id: str = "req-stream",
    *,
    codes: list[int] | None = None,
    prompt_feat_frames: int = TOKEN_HOP_LEN * TOKEN_MEL_RATIO,
    prompt_token_len: int = TOKEN_HOP_LEN,
) -> StagePayload:
    state = FunCosyVoice3State(
        text="hello",
        stream=True,
        audio_codes=None if codes is None else torch.tensor(codes, dtype=torch.long),
        flow_prompt_speech_token=torch.zeros(1, prompt_token_len, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, prompt_feat_frames, 80),
        flow_embedding=torch.ones(1, 192),
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": True}),
        data=state.to_dict(),
    )


def _item(tokens: list[int]) -> StreamItem:
    return StreamItem(
        chunk_id=0,
        data=torch.tensor(tokens, dtype=torch.long),
        from_stage="tts_engine",
        metadata={"modality": "audio_codes", "stream": True},
    )


def test_streaming_vocoder_emits_causal_chunk_then_finalizes_remainder() -> None:
    flow, scheduler = _scheduler()
    scheduler.handle_streaming_new_request("req-stream", _stream_payload())
    scheduler.handle_stream_chunk("req-stream", _item(list(range(28))))
    assert _drain(scheduler) == []
    assert _serve(scheduler) == 1
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream"]
    assert _hop_frames(flow) == {_window_frames(28)}
    assert _waveform(messages[0].data).shape == (50,)

    scheduler.handle_stream_done("req-stream")
    assert _drain(scheduler) == []
    assert scheduler.stream_states["req-stream"].done is True
    assert _serve(scheduler) == 1
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "result"]
    assert _estimator_calls(flow)[-1]["streaming"] is False
    assert _waveform(messages[0].data).shape == (6,)
    assert messages[1].data.data["modality"] == "audio"
    assert messages[1].data.data["sample_rate"] == 24000
    assert "req-stream" not in scheduler.stream_states


def test_streaming_vocoder_does_not_decode_before_lookahead_tokens_arrive() -> None:
    flow, scheduler = _scheduler()
    scheduler.handle_streaming_new_request("req-stream", _stream_payload())
    scheduler.handle_stream_chunk("req-stream", _item(list(range(27))))
    assert not scheduler.has_ready_work()
    assert _serve(scheduler) == 0
    assert _drain(scheduler) == []
    assert _estimator_calls(flow) == []


def test_streaming_vocoder_pads_prompt_and_decodes_first_hop_at_28() -> None:
    flow, scheduler = _scheduler()
    prompt_len = 10
    scheduler.handle_streaming_new_request(
        "req-pad",
        _stream_payload(
            "req-pad",
            prompt_token_len=prompt_len,
            prompt_feat_frames=prompt_len * TOKEN_MEL_RATIO,
        ),
    )
    scheduler.handle_stream_chunk("req-pad", _item(list(range(27))))
    assert _serve(scheduler) == 0
    assert _drain(scheduler) == []
    assert _estimator_calls(flow) == []

    scheduler.handle_stream_chunk("req-pad", _item([27]))
    assert _serve(scheduler) == 1
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream"]
    assert scheduler.stream_states["req-pad"].prompt_token.shape == (1, 25)
    assert scheduler.stream_states["req-pad"].prompt_feat.shape == (1, 50, 80)
    assert _hop_frames(flow) == {_window_frames(28)}
    assert _waveform(messages[0].data).shape == (TOKEN_HOP_LEN * TOKEN_MEL_RATIO,)


def test_model_runner_flushes_speech_tokens_and_skips_control_ids() -> None:
    runner = _model_runner()
    data = CosyVoice3SGLangRequestData(
        stream_metadata={"modality": "audio_codes", "stream": True},
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 0, 80),
        flow_embedding=torch.ones(1, 192),
    )
    request = SimpleNamespace(request_id="req-ar", data=data)

    for token_id in range(AR_INITIAL_FLUSH_TOKENS - 1):
        runner.collect_tokens(
            SimpleNamespace(next_token_ids=torch.tensor([token_id])),
            None,
            None,
            [request],
        )
    assert runner._outbox.empty()

    runner.collect_tokens(
        SimpleNamespace(next_token_ids=torch.tensor([EOS_ID])),
        None,
        None,
        [request],
    )
    assert runner._outbox.empty()
    assert all(code.item() < VOCAB_SIZE for code in data.output_codes)

    runner.collect_tokens(
        SimpleNamespace(next_token_ids=torch.tensor([7])),
        None,
        None,
        [request],
    )
    message = runner._outbox.get_nowait()
    assert message.type == "stream"
    assert message.target == "vocoder"
    assert message.metadata["stream"] is True
    assert message.metadata["flow_embedding"].shape == (1, 192)
    assert tuple(message.data.tolist()) == tuple(range(AR_INITIAL_FLUSH_TOKENS - 1)) + (
        7,
    )
    assert data.stream_code_next_flush == (
        AR_INITIAL_FLUSH_TOKENS + AR_FOLLOWUP_FLUSH_TOKENS
    )

    runner.on_request_finished("req-ar", data)
    assert runner._outbox.empty()


def test_model_runner_first_flush_ignores_prompt_pad() -> None:
    runner = _model_runner()
    prompt_len = 10
    data = CosyVoice3SGLangRequestData(
        stream_metadata={"modality": "audio_codes", "stream": True},
        flow_prompt_speech_token=torch.zeros(1, prompt_len, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 1, 80),
        flow_embedding=torch.ones(1, 192),
    )
    request = SimpleNamespace(request_id="req-pad", data=data)
    first_flush = first_ar_flush_tokens(prompt_len)
    assert first_flush == AR_INITIAL_FLUSH_TOKENS

    early = _feed_tokens(runner, request, list(range(first_flush - 1)))
    assert early == []
    assert data.stream_code_next_flush == first_flush

    ready = _feed_tokens(runner, request, [7])
    assert len(ready) == 1
    assert tuple(ready[0].data.tolist()) == tuple(range(first_flush - 1)) + (7,)
    assert data.stream_code_next_flush == first_flush + AR_FOLLOWUP_FLUSH_TOKENS


def _feed_tokens(
    runner: FunCosyVoice3ModelRunner,
    request: SimpleNamespace,
    token_ids: list[int],
) -> list[OutgoingMessage]:
    for token_id in token_ids:
        runner.collect_tokens(
            SimpleNamespace(next_token_ids=torch.tensor([token_id])),
            None,
            None,
            [request],
        )
    messages: list[OutgoingMessage] = []
    while True:
        try:
            messages.append(runner._outbox.get_nowait())
        except Empty:
            return messages


def _to_stream_chunk(outgoing: OutgoingMessage, chunk_id: int) -> IncomingMessage:
    return IncomingMessage(
        request_id=outgoing.request_id,
        type="stream_chunk",
        data=StreamItem(
            chunk_id=chunk_id,
            data=outgoing.data,
            from_stage="tts_engine",
            metadata=outgoing.metadata,
        ),
    )


def test_ar_to_vocoder_grows_hops_then_finalizes_remainder() -> None:
    request_id = "req-hops"
    runner = _model_runner()
    data = CosyVoice3SGLangRequestData(
        stream_metadata={"modality": "audio_codes", "stream": True},
        flow_prompt_speech_token=torch.zeros(1, TOKEN_HOP_LEN, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, TOKEN_HOP_LEN * TOKEN_MEL_RATIO, 80),
        flow_embedding=torch.ones(1, 192),
    )
    request = SimpleNamespace(request_id=request_id, data=data)
    flow, scheduler = _scheduler()

    generated = list(range(AR_INITIAL_FLUSH_TOKENS + 2 * AR_FOLLOWUP_FLUSH_TOKENS))
    ar_messages = _feed_tokens(runner, request, generated)
    assert len(ar_messages) == 3
    assert "flow_embedding" in ar_messages[0].metadata
    assert "flow_embedding" not in ar_messages[1].metadata

    pcm_chunks: list[np.ndarray] = []
    for chunk_id, outgoing in enumerate(ar_messages):
        scheduler.handle_message(_to_stream_chunk(outgoing, chunk_id), None)
        _serve(scheduler)
        for message in _drain(scheduler):
            assert message.type == "stream"
            pcm_chunks.append(_waveform(message.data))

    assert _hop_frames(flow) == {_window_frames(28), _window_frames(78)}
    assert all(call["streaming"] for call in _estimator_calls(flow))
    assert [chunk.shape[0] for chunk in pcm_chunks] == [
        TOKEN_HOP_LEN * TOKEN_MEL_RATIO,
        2 * TOKEN_HOP_LEN * TOKEN_MEL_RATIO,
    ]

    scheduler.handle_message(
        IncomingMessage(request_id=request_id, type="stream_done"), None
    )
    assert _serve(scheduler) == 0
    assert _drain(scheduler) == []

    scheduler.handle_message(
        IncomingMessage(
            request_id=request_id,
            type="new_request",
            data=_stream_payload(request_id, codes=generated),
        ),
        None,
    )
    assert _serve(scheduler) == 1
    final_messages = _drain(scheduler)
    assert [message.type for message in final_messages] == ["stream", "result"]
    remainder = _waveform(final_messages[0].data)
    assert remainder.shape == (PRE_LOOKAHEAD_LEN * TOKEN_MEL_RATIO,)
    assert _estimator_calls(flow)[-1]["streaming"] is False
    total = np.concatenate(pcm_chunks + [remainder])
    assert total.shape == (len(generated) * TOKEN_MEL_RATIO,)


@pytest.mark.parametrize("codes", [None, []])
def test_streaming_vocoder_fallback_errors_on_empty_audio_codes(codes) -> None:
    _, scheduler = _scheduler()
    scheduler.handle_streaming_new_request("req-empty", _stream_payload(codes=codes))
    scheduler.handle_stream_done("req-empty")
    assert _serve(scheduler) == 1
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["error"]
    assert "no usable speech tokens" in str(messages[0].data)
    assert scheduler.is_aborted("req-empty")
    assert "req-empty" not in scheduler.stream_states


def test_equal_first_hops_share_one_causal_flow_batch() -> None:
    flow, scheduler = _scheduler(max_batch_size=8)
    for request_id in ("req-a", "req-b"):
        scheduler.handle_streaming_new_request(request_id, _stream_payload(request_id))
    for request_id in ("req-a", "req-b"):
        scheduler.ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler.state_lock:
        failed = scheduler.pump_streams()
    assert failed == []
    assert _estimator_calls(flow)[0]["streaming"] is True
    assert len(_estimator_calls(flow)[0]["lengths"]) == 4
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "stream"]
    assert {_waveform(message.data).shape[0] for message in messages} == {
        TOKEN_HOP_LEN * TOKEN_MEL_RATIO
    }


def test_late_payloads_share_one_causal_flow_batch() -> None:
    flow, scheduler = _scheduler(max_batch_size=8)
    for request_id in ("req-a", "req-b"):
        scheduler.ingest_stream_item(request_id, _item(list(range(28))))
        scheduler.inbox.put(
            IncomingMessage(
                request_id=request_id,
                type="new_request",
                data=_stream_payload(request_id),
            )
        )
    assert _serve(scheduler) == 1
    assert _estimator_calls(flow)[0]["streaming"] is True
    assert len(_estimator_calls(flow)[0]["lengths"]) == 4
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "stream"]


def test_queued_peer_chunk_is_ingested_before_the_first_hop_step() -> None:
    flow, scheduler = _scheduler(max_batch_size=8)
    for request_id in ("req-a", "req-b"):
        scheduler.handle_streaming_new_request(request_id, _stream_payload(request_id))
    scheduler.ingest_stream_item("req-a", _item(list(range(28))))
    scheduler.inbox.put(
        IncomingMessage(
            request_id="req-b",
            type="stream_chunk",
            data=_item(list(range(28))),
        )
    )
    assert _serve(scheduler) == 1
    assert len(_estimator_calls(flow)[0]["lengths"]) == 4


def test_equal_follow_up_hops_share_one_causal_flow_batch() -> None:
    flow, scheduler = _scheduler()
    for request_id in ("req-a", "req-b"):
        scheduler.handle_streaming_new_request(request_id, _stream_payload(request_id))
        scheduler.ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler.state_lock:
        failed = scheduler.pump_streams()
    assert failed == []
    first_calls = len(flow.packed_estimator.calls)
    assert len(flow.packed_estimator.calls[0]["lengths"]) == 4

    for request_id in ("req-a", "req-b"):
        scheduler.ingest_stream_item(request_id, _item([i % 31 for i in range(28, 78)]))
    with scheduler.state_lock:
        failed = scheduler.pump_streams()
    assert failed == []
    follow_calls = flow.packed_estimator.calls[first_calls:]
    assert follow_calls
    assert follow_calls[0]["streaming"] is True
    assert len(follow_calls[0]["lengths"]) == 4
    messages = [m for m in _drain(scheduler) if m.type == "stream"]
    shapes = {_waveform(m.data).shape[0] for m in messages}
    assert 100 in shapes


def test_mixed_prompt_follow_ups_share_one_causal_flow_batch() -> None:
    flow, scheduler = _scheduler()
    payloads = {
        "req-a": _stream_payload(
            "req-a",
            prompt_token_len=TOKEN_HOP_LEN,
            prompt_feat_frames=TOKEN_HOP_LEN * TOKEN_MEL_RATIO,
        ),
        "req-b": _stream_payload(
            "req-b",
            prompt_token_len=TOKEN_HOP_LEN * 2,
            prompt_feat_frames=TOKEN_HOP_LEN * 2 * TOKEN_MEL_RATIO,
        ),
    }
    for request_id, payload in payloads.items():
        scheduler.handle_streaming_new_request(request_id, payload)
        scheduler.ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler.state_lock:
        failed = scheduler.pump_streams()
    assert failed == []
    first_calls = len(flow.packed_estimator.calls)
    assert len(flow.packed_estimator.calls[0]["lengths"]) == 4

    for request_id in payloads:
        scheduler.ingest_stream_item(request_id, _item([i % 31 for i in range(28, 78)]))
    with scheduler.state_lock:
        failed = scheduler.pump_streams()
    assert failed == []
    follow_calls = flow.packed_estimator.calls[first_calls:]
    assert follow_calls
    assert follow_calls[0]["streaming"] is True
    assert len(follow_calls[0]["lengths"]) == 4


def test_backlogged_request_runs_one_hop_per_step() -> None:
    # note (guozhihao-224): 178 tokens cover first hop + two follow-ups
    # (28 / 78 / 178 windows). One step must advance only one hop.
    flow, scheduler = _scheduler()
    scheduler.handle_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler.ingest_stream_item("req-a", _item(list(range(178))))
    with scheduler.state_lock:
        assert scheduler.pump_one_step() == []
    assert _hop_frames(flow) == {_window_frames(28)}
    state = scheduler.stream_states["req-a"]
    assert state.token_offset == TOKEN_HOP_LEN
    assert state.hop_len == next_stream_hop_len(TOKEN_HOP_LEN)

    with scheduler.state_lock:
        assert scheduler.pump_one_step() == []
    assert _hop_frames(flow) == {_window_frames(28), _window_frames(78)}
    assert state.token_offset == TOKEN_HOP_LEN + next_stream_hop_len(TOKEN_HOP_LEN)


def test_pump_drains_backlog_across_one_hop_steps() -> None:
    flow, scheduler = _scheduler()
    scheduler.handle_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler.ingest_stream_item("req-a", _item(list(range(178))))
    with scheduler.state_lock:
        failed = scheduler.pump_streams()
    assert failed == []
    assert _hop_frames(flow) == {
        _window_frames(28),
        _window_frames(78),
        _window_frames(178),
    }
    messages = [m for m in _drain(scheduler) if m.type == "stream"]
    assert len(messages) == 3


def test_hops_of_different_token_windows_share_one_causal_flow_batch() -> None:
    flow, scheduler = _scheduler()
    for request_id in ("req-a", "req-b"):
        scheduler.handle_streaming_new_request(request_id, _stream_payload(request_id))
        scheduler.ingest_stream_item(request_id, _item(list(range(28))))
    assert _serve(scheduler) == 1
    _drain(scheduler)
    scheduler.ingest_stream_item("req-a", _item([i % 31 for i in range(28, 78)]))
    scheduler.handle_streaming_new_request("req-c", _stream_payload("req-c"))
    scheduler.ingest_stream_item("req-c", _item(list(range(28))))

    assert _serve(scheduler) == 1
    assert len(_estimator_calls(flow)[-1]["lengths"]) == 4
    samples = {
        m.request_id: _waveform(m.data).shape[0]
        for m in _drain(scheduler)
        if m.type == "stream"
    }
    assert samples == {
        "req-a": 2 * TOKEN_HOP_LEN * TOKEN_MEL_RATIO,
        "req-c": TOKEN_HOP_LEN * TOKEN_MEL_RATIO,
    }
    assert scheduler.stream_states["req-b"].token_offset == TOKEN_HOP_LEN


def test_step_takes_started_streams_before_new_ones_up_to_the_batch_size() -> None:
    flow, scheduler = _scheduler(max_batch_size=2)
    scheduler.clock = _Clock()
    for request_id in ("req-a", "req-b"):
        scheduler.handle_streaming_new_request(request_id, _stream_payload(request_id))
        scheduler.ingest_stream_item(request_id, _item(list(range(28))))
    assert _serve(scheduler) == 1
    for request_id in ("req-a", "req-b"):
        scheduler.ingest_stream_item(request_id, _item([i % 31 for i in range(28, 78)]))
    for request_id in ("req-c", "req-d"):
        scheduler.handle_streaming_new_request(request_id, _stream_payload(request_id))
        scheduler.ingest_stream_item(request_id, _item(list(range(28))))

    assert _serve(scheduler) == 2
    assert _stream_ids(_drain(scheduler)) == [
        "req-a",
        "req-b",
        "req-a",
        "req-b",
        "req-c",
        "req-d",
    ]
    assert {len(call["lengths"]) for call in _estimator_calls(flow)} == {4}


def test_first_hop_runs_when_no_started_stream_is_runnable() -> None:
    flow, scheduler = _scheduler()
    clock = _Clock()
    scheduler.clock = clock
    for request_id in ("req-a", "req-b"):
        scheduler.handle_streaming_new_request(request_id, _stream_payload(request_id))
    scheduler.ingest_stream_item("req-a", _item(list(range(28))))
    assert _serve(scheduler) == 1
    clock.now += 1.0
    scheduler.ingest_stream_item("req-b", _item(list(range(28))))
    assert _serve(scheduler) == 1
    assert _stream_ids(_drain(scheduler)) == ["req-a", "req-b"]
    assert _hop_frames(flow) == {_window_frames(28)}


def test_finals_rank_by_slack_with_the_other_started_streams() -> None:
    flow, scheduler = _scheduler()
    clock = _Clock()
    scheduler.clock = clock
    for request_id in ("req-a", "req-c"):
        scheduler.handle_streaming_new_request(request_id, _stream_payload(request_id))
    scheduler.ingest_stream_item("req-a", _item(list(range(28))))
    assert _serve(scheduler) == 1
    scheduler.ingest_stream_item("req-c", _item(list(range(78))))
    assert _serve(scheduler) == 2
    a_samples = scheduler.stream_states["req-a"].speech_offset
    c_samples = scheduler.stream_states["req-c"].speech_offset
    assert 0 < a_samples < c_samples
    _drain(scheduler)

    scheduler.ingest_stream_item("req-a", _item(list(range(28, 78))))
    scheduler.handle_stream_done("req-c")
    clock.now += (a_samples + c_samples) / 2 / scheduler.sample_rate
    assert _serve(scheduler) == 2
    assert _stream_ids(_drain(scheduler)) == ["req-a", "req-c"]
    assert _estimator_calls(flow)[-1]["streaming"] is False
    assert "req-c" not in scheduler.stream_states


def test_stream_done_defers_the_final_to_a_step() -> None:
    flow, scheduler = _scheduler()
    scheduler.handle_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler.ingest_stream_item("req-a", _item(list(range(30))))
    assert _serve(scheduler) == 1
    assert [m.type for m in _drain(scheduler)] == ["stream"]

    scheduler.handle_stream_done("req-a")
    assert _drain(scheduler) == []
    assert scheduler.has_ready_work()
    scheduler.run_ready_step()
    assert _estimator_calls(flow)[-1]["streaming"] is False
    assert [m.type for m in _drain(scheduler)] == ["stream", "result"]
    assert "req-a" not in scheduler.stream_states
    assert not scheduler.has_ready_work()


def test_finals_share_one_non_streaming_flow_batch() -> None:
    flow, scheduler = _scheduler()
    for request_id in ("req-a", "req-b"):
        scheduler.handle_streaming_new_request(request_id, _stream_payload(request_id))
        scheduler.ingest_stream_item(request_id, _item(list(range(30))))
    assert _serve(scheduler) == 1
    _drain(scheduler)
    for request_id in ("req-a", "req-b"):
        scheduler.handle_stream_done(request_id)

    assert _serve(scheduler) == 1
    messages = _drain(scheduler)
    assert [(m.request_id, m.type) for m in messages] == [
        ("req-a", "stream"),
        ("req-a", "result"),
        ("req-b", "stream"),
        ("req-b", "result"),
    ]
    assert _estimator_calls(flow)[-1]["streaming"] is False
    assert len(_estimator_calls(flow)[-1]["lengths"]) == 4
    assert {_waveform(m.data).shape[0] for m in messages if m.type == "stream"} == {
        (30 - TOKEN_HOP_LEN) * TOKEN_MEL_RATIO
    }
    assert scheduler.stream_states == {}


def test_a_wide_row_shares_one_step_with_short_rows_at_their_own_lengths() -> None:
    flow, scheduler = _scheduler()
    scheduler.clock = _Clock()
    scheduler.handle_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler.ingest_stream_item("req-a", _item([i % 31 for i in range(78)]))
    assert _serve(scheduler) == 2
    _drain(scheduler)
    scheduler.ingest_stream_item("req-a", _item([i % 31 for i in range(78, 178)]))
    for request_id in ("req-b", "req-c"):
        scheduler.handle_streaming_new_request(request_id, _stream_payload(request_id))
        scheduler.ingest_stream_item(request_id, _item(list(range(28))))
    _estimator_calls(flow).clear()

    assert _serve(scheduler) == 1
    assert _stream_ids(_drain(scheduler)) == ["req-a", "req-b", "req-c"]
    assert _estimator_calls(flow)[0]["lengths"] == (400, 100, 100, 400, 100, 100)


def test_finals_of_different_widths_share_one_step_at_their_own_lengths() -> None:
    flow, scheduler = _scheduler()
    scheduler.clock = _Clock()
    scheduler.handle_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler.ingest_stream_item("req-a", _item(list(range(30))))
    scheduler.handle_streaming_new_request("req-b", _stream_payload("req-b"))
    scheduler.ingest_stream_item("req-b", _item([i % 31 for i in range(200)]))
    assert _serve(scheduler) == 3
    _drain(scheduler)
    scheduler.handle_stream_done("req-a")
    scheduler.handle_stream_done("req-b")
    _estimator_calls(flow).clear()

    assert _serve(scheduler) == 1
    assert [(m.request_id, m.type) for m in _drain(scheduler)] == [
        ("req-a", "stream"),
        ("req-a", "result"),
        ("req-b", "stream"),
        ("req-b", "result"),
    ]
    assert _estimator_calls(flow)[0]["lengths"] == (110, 450, 110, 450)
    assert _estimator_calls(flow)[0]["streaming"] is False


def test_stream_without_tokens_fails_in_its_own_step() -> None:
    flow, scheduler = _scheduler()
    scheduler.handle_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler.ingest_stream_item("req-a", _item(list(range(30))))
    assert _serve(scheduler) == 1
    _drain(scheduler)
    scheduler.handle_streaming_new_request(
        "req-empty", _stream_payload("req-empty", codes=[])
    )
    scheduler.handle_stream_done("req-empty")
    scheduler.handle_stream_done("req-a")

    assert _serve(scheduler) == 2
    assert [(m.request_id, m.type) for m in _drain(scheduler)] == [
        ("req-a", "stream"),
        ("req-a", "result"),
        ("req-empty", "error"),
    ]
    assert scheduler.is_aborted("req-empty")
    assert scheduler.stream_states == {}


def test_backlogged_chunks_stay_ordered_before_stream_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow, scheduler = _scheduler(max_batch_size=8)
    tokens = list(range(78))
    scheduler.handle_streaming_new_request(
        "req-a", _stream_payload("req-a", codes=tokens)
    )
    for codes in (tokens[:28], tokens[28:53]):
        scheduler.inbox.put(
            IncomingMessage(request_id="req-a", type="stream_chunk", data=_item(codes))
        )
    vocoder = scheduler.vocoder
    original_hop_batch = vocoder.hop_batch
    hop_batches: list[int] = []

    def hop_batch(items):
        hop_batches.append(len(items))
        if len(hop_batches) == 1:
            scheduler.inbox.put(
                IncomingMessage(
                    request_id="req-a", type="stream_chunk", data=_item(tokens[53:])
                )
            )
            scheduler.inbox.put(IncomingMessage(request_id="req-a", type="stream_done"))
        return original_hop_batch(items)

    monkeypatch.setattr(vocoder, "hop_batch", hop_batch)

    _serve(scheduler)

    assert _estimator_calls(flow)[-1]["streaming"] is False
    assert set(_estimator_calls(flow)[-1]["lengths"]) == {
        (TOKEN_HOP_LEN + len(tokens)) * TOKEN_MEL_RATIO
    }
    messages = _drain(scheduler)
    assert [m.type for m in messages].count("result") == 1
    assert messages[-1].type == "result"
    audio = np.concatenate([_waveform(m.data) for m in messages if m.type == "stream"])
    np.testing.assert_array_equal(audio, np.arange(len(tokens) * TOKEN_MEL_RATIO))
    assert "req-a" not in scheduler.stream_states


@pytest.mark.parametrize("streaming", [False, True])
def test_new_request_collection_stops_at_pending_chunk(streaming: bool) -> None:
    _, scheduler = _scheduler(max_batch_size=8)
    payload = _stream_payload("req-a")
    payload.request.params["stream"] = streaming
    first = IncomingMessage(request_id="req-a", type="new_request", data=payload)
    pending = IncomingMessage(request_id="req-a", type="stream_chunk", data=_item([2]))
    newer = IncomingMessage(request_id="req-a", type="stream_chunk", data=_item([3]))
    scheduler.pending_messages.append(pending)
    scheduler.inbox.put(newer)

    assert scheduler.collect_new_request_batch(first) == [first]
    assert list(scheduler.pending_messages) == [pending]
    assert scheduler.inbox.get_nowait() is newer


def test_chunk_collection_keeps_arrival_order_and_stops_at_done() -> None:
    _, scheduler = _scheduler(max_batch_size=8)
    first = IncomingMessage("a", "stream_chunk", _item([1]))
    second = IncomingMessage("a", "stream_chunk", _item([2]))
    third = IncomingMessage("a", "stream_chunk", _item([3]))
    peer_b = IncomingMessage("b", "stream_chunk", _item([4]))
    peer_c = IncomingMessage("c", "stream_chunk", _item([5]))
    done = IncomingMessage("a", "stream_done")
    later = IncomingMessage("d", "stream_chunk", _item([6]))
    scheduler.pending_messages.extend([second, peer_b])
    for msg in (third, peer_c, done, later):
        scheduler.inbox.put(msg)

    batch = scheduler.collect_stream_chunk_batch(first)

    assert batch == [first, second, peer_b, third, peer_c]
    assert list(scheduler.pending_messages) == [done]
    assert scheduler.inbox.get_nowait() is later


@pytest.mark.parametrize("coalescing", [False, True])
def test_non_streaming_fallback_batches_past_pending_done_with_cost_limit(
    coalescing: bool,
) -> None:
    _, scheduler = _scheduler(
        max_batch_size=8, request_cost_fn=lambda payload: 1, max_batch_cost=2
    )
    scheduler.can_batch_stream_chunks = coalescing
    messages = []
    for rid in ("a", "b", "c", "d"):
        payload = _stream_payload(rid)
        payload.request.params["stream"] = False
        messages.append(IncomingMessage(rid, "new_request", payload))
    done = IncomingMessage("b", "stream_done")
    scheduler.pending_messages.extend([done, messages[1], messages[2]])
    scheduler.inbox.put(messages[3])

    assert scheduler.collect_new_request_batch(messages[0]) == messages[:2]
    assert list(scheduler.pending_messages) == [done, messages[2]]
    assert scheduler.inbox.get_nowait() is messages[3]


def test_payloads_from_pending_and_inbox_share_one_first_hop_batch() -> None:
    flow, scheduler = _scheduler()
    for rid in ("a", "b"):
        scheduler.ingest_stream_item(rid, _item(list(range(28))))
    scheduler.pending_messages.append(
        IncomingMessage("b", "new_request", _stream_payload("b"))
    )
    scheduler.inbox.put(IncomingMessage("a", "new_request", _stream_payload("a")))

    assert _serve(scheduler) == 1
    assert len(_estimator_calls(flow)[0]["lengths"]) == 4
    assert len([msg for msg in _drain(scheduler) if msg.type == "stream"]) == 2


def test_disable_hop_growth_keeps_fixed_follow_up_windows() -> None:
    flow, scheduler = _scheduler(disable_hop_growth=True)
    scheduler.handle_streaming_new_request("req-a", _stream_payload("req-a"))
    # First hop 28, then two fixed 25-token hops -> need 28+25+25 = 78 tokens
    # for three windows of 28 / 53 / 78 (lookahead included in prefix).
    scheduler.ingest_stream_item("req-a", _item(list(range(78))))
    with scheduler.state_lock:
        assert scheduler.pump_streams() == []
    assert _hop_frames(flow) == {
        _window_frames(28),
        _window_frames(53),
        _window_frames(78),
    }
    state = scheduler.stream_states["req-a"]
    assert state.hop_len == TOKEN_HOP_LEN


def test_token_max_hop_len_caps_growth() -> None:
    flow, scheduler = _scheduler(token_max_hop_len=50)
    scheduler.handle_streaming_new_request("req-a", _stream_payload("req-a"))
    # With max 50: hops 25 -> 50 -> 50. Windows 28 / 78 / 128.
    scheduler.ingest_stream_item("req-a", _item(list(range(128))))
    with scheduler.state_lock:
        assert scheduler.pump_streams() == []
    assert _hop_frames(flow) == {
        _window_frames(28),
        _window_frames(78),
        _window_frames(128),
    }
    state = scheduler.stream_states["req-a"]
    assert state.hop_len == 50
