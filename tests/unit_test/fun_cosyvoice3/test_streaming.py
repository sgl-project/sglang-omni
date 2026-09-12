# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
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
    AR_FOLLOWUP_FLUSH_TOKENS,
    AR_INITIAL_FLUSH_TOKENS,
    LEFTOVER_FLOW_STREAMING,
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


class _FakeEstimator(torch.nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("causal hops must use CosyVoice Flow.inference")


class _FakeFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[dict] = []
        self.decoder = SimpleNamespace(estimator=_FakeEstimator())

    def inference(self, **kwargs):
        self.calls.append(kwargs)
        token_count = int(kwargs["token"].shape[1])
        if not kwargs.get("finalize", True):
            token_count = max(token_count - PRE_LOOKAHEAD_LEN, 0)
        return torch.ones(1, 80, token_count * TOKEN_MEL_RATIO), None


class _FakeHiFT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[tuple] = []

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


def _waveform(data: dict) -> np.ndarray:
    return np.frombuffer(data["audio_waveform"], dtype=np.float32).reshape(
        data["audio_waveform_shape"]
    )


def _scheduler(
    **scheduler_kwargs,
) -> tuple[_FakeFlow, FunCosyVoice3StreamingVocoderScheduler]:
    flow = _FakeFlow()
    return flow, FunCosyVoice3StreamingVocoderScheduler(
        stages._CosyVoice3Vocoder(flow, _FakeHiFT()),
        **scheduler_kwargs,
    )


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
    prompt_feat_frames: int = 0,
    prompt_token_len: int = 0,
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
    scheduler._on_streaming_new_request("req-stream", _stream_payload())
    scheduler._on_chunk("req-stream", _item(list(range(28))))
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream"]
    assert flow.calls[0]["streaming"] is True
    assert flow.calls[0]["finalize"] is False
    assert int(flow.calls[0]["token"].shape[1]) == 28
    assert _waveform(messages[0].data).shape == (50,)

    scheduler._on_done("req-stream")
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "result"]
    assert LEFTOVER_FLOW_STREAMING is False
    assert flow.calls[1]["streaming"] is False
    assert flow.calls[1]["finalize"] is True
    assert _waveform(messages[0].data).shape == (6,)
    assert messages[1].data.data["modality"] == "audio"
    assert messages[1].data.data["sample_rate"] == 24000


def test_streaming_vocoder_does_not_decode_before_lookahead_tokens_arrive() -> None:
    flow, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-stream", _stream_payload())
    scheduler._on_chunk("req-stream", _item(list(range(27))))
    assert _drain(scheduler) == []
    assert flow.calls == []


def test_streaming_vocoder_pads_prompt_and_decodes_first_hop_at_28() -> None:
    flow, scheduler = _scheduler()
    prompt_len = 10
    scheduler._on_streaming_new_request(
        "req-pad",
        _stream_payload(
            "req-pad",
            prompt_token_len=prompt_len,
            prompt_feat_frames=prompt_len * TOKEN_MEL_RATIO,
        ),
    )
    scheduler._on_chunk("req-pad", _item(list(range(27))))
    assert _drain(scheduler) == []
    assert flow.calls == []

    scheduler._on_chunk("req-pad", _item([27]))
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream"]
    assert int(flow.calls[0]["prompt_token"].shape[1]) == 25
    assert int(flow.calls[0]["prompt_feat"].shape[1]) == 50
    assert int(flow.calls[0]["token"].shape[1]) == 28
    assert flow.calls[0]["streaming"] is True
    assert flow.calls[0]["finalize"] is False
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
        runner._collect_tokens(
            SimpleNamespace(next_token_ids=torch.tensor([token_id])),
            None,
            None,
            [request],
        )
    assert runner._outbox.empty()

    runner._collect_tokens(
        SimpleNamespace(next_token_ids=torch.tensor([EOS_ID])),
        None,
        None,
        [request],
    )
    assert runner._outbox.empty()
    assert all(code.item() < VOCAB_SIZE for code in data.output_codes)

    runner._collect_tokens(
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
        runner._collect_tokens(
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
    # note (guozhihao-224): feat uses a non-empty time axis because (1, 0, 80)
    # does not round-trip through the tensor_list wire codec.
    data = CosyVoice3SGLangRequestData(
        stream_metadata={"modality": "audio_codes", "stream": True},
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 1, 80),
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
        scheduler._handle_message(_to_stream_chunk(outgoing, chunk_id), None)
        for message in _drain(scheduler):
            assert message.type == "stream"
            pcm_chunks.append(_waveform(message.data))

    assert [int(call["token"].shape[1]) for call in flow.calls] == [28, 78]
    assert all(
        call["streaming"] is True and call["finalize"] is False for call in flow.calls
    )
    assert [chunk.shape[0] for chunk in pcm_chunks] == [
        TOKEN_HOP_LEN * TOKEN_MEL_RATIO,
        2 * TOKEN_HOP_LEN * TOKEN_MEL_RATIO,
    ]

    scheduler._handle_message(
        IncomingMessage(request_id=request_id, type="stream_done"), None
    )
    assert _drain(scheduler) == []

    scheduler._handle_message(
        IncomingMessage(
            request_id=request_id,
            type="new_request",
            data=_stream_payload(request_id, codes=generated, prompt_feat_frames=1),
        ),
        None,
    )
    final_messages = _drain(scheduler)
    assert [message.type for message in final_messages] == ["stream", "result"]
    remainder = _waveform(final_messages[0].data)
    assert remainder.shape == (PRE_LOOKAHEAD_LEN * TOKEN_MEL_RATIO,)
    assert flow.calls[-1]["streaming"] is False
    assert flow.calls[-1]["finalize"] is True
    total = np.concatenate(pcm_chunks + [remainder])
    assert total.shape == (len(generated) * TOKEN_MEL_RATIO,)


def test_streaming_vocoder_fallback_raises_on_empty_audio_codes() -> None:
    _, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-empty", _stream_payload(codes=None))
    with pytest.raises(RuntimeError, match="no usable speech tokens"):
        scheduler._on_done("req-empty")

    _, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-empty-list", _stream_payload(codes=[]))
    with pytest.raises(RuntimeError, match="no usable speech tokens"):
        scheduler._on_done("req-empty-list")


def test_streaming_vocoder_enables_first_hop_coalescing_by_default() -> None:
    _, scheduler = _scheduler()
    assert scheduler._can_batch_stream_chunks is True
    assert scheduler._stream_chunk_batch_distinct_requests is True
    assert scheduler._first_hop_peer_wait_ms == 30
    assert scheduler._can_batch_follow_up_hops is True


def test_equal_first_hops_share_one_causal_flow_batch() -> None:
    from sglang_omni.models.fun_cosyvoice3.stages import FunCosyVoice3Flow
    from tests.unit_test.fun_cosyvoice3.test_flow_batch import _FakeFlow as _PackedFlow

    flow = _PackedFlow(channels=80, max_frames=128)
    flow.spk_embed_affine_layer = torch.nn.Linear(192, 80, bias=False)
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        stages._CosyVoice3Vocoder(FunCosyVoice3Flow(flow), _FakeHiFT()),
        max_batch_size=8,
    )
    scheduler._can_batch_stream_chunks = True
    # note (guozhihao-224): empty (1, 0, 80) prompt_feat round-trips through
    # tensor_list as [[]] and loses the channel dim; use a hop-aligned prompt.
    prompt_len = TOKEN_HOP_LEN
    payload_kwargs = {
        "prompt_token_len": prompt_len,
        "prompt_feat_frames": prompt_len * TOKEN_MEL_RATIO,
    }
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(
            request_id, _stream_payload(request_id, **payload_kwargs)
        )
    for request_id in ("req-a", "req-b"):
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert flow.decoder.estimator.calls
    assert flow.decoder.estimator.calls[0]["streaming"] is True
    assert flow.decoder.estimator.calls[0]["x"].shape[0] == 4
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "stream"]
    assert {_waveform(message.data).shape[0] for message in messages} == {
        TOKEN_HOP_LEN * TOKEN_MEL_RATIO
    }


def test_disabled_coalescing_keeps_equal_first_hops_serial() -> None:
    flow, scheduler = _scheduler()
    scheduler._can_batch_stream_chunks = False
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(request_id, _stream_payload(request_id))
    for request_id in ("req-a", "req-b"):
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [28, 28]
    assert all(
        call["streaming"] is True and call["finalize"] is False for call in flow.calls
    )


def test_late_payloads_share_one_causal_flow_batch() -> None:
    from sglang_omni.models.fun_cosyvoice3.stages import FunCosyVoice3Flow
    from tests.unit_test.fun_cosyvoice3.test_flow_batch import _FakeFlow as _PackedFlow

    flow = _PackedFlow(channels=80, max_frames=128)
    flow.spk_embed_affine_layer = torch.nn.Linear(192, 80, bias=False)
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        stages._CosyVoice3Vocoder(FunCosyVoice3Flow(flow), _FakeHiFT()),
        max_batch_size=8,
    )
    scheduler._can_batch_stream_chunks = True
    prompt_len = TOKEN_HOP_LEN
    payload_kwargs = {
        "prompt_token_len": prompt_len,
        "prompt_feat_frames": prompt_len * TOKEN_MEL_RATIO,
    }
    for request_id in ("req-a", "req-b"):
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
        scheduler.inbox.put(
            IncomingMessage(
                request_id=request_id,
                type="new_request",
                data=_stream_payload(request_id, **payload_kwargs),
            )
        )
    first = scheduler.inbox.get()
    scheduler._handle_message(first, None)
    assert flow.decoder.estimator.calls
    assert flow.decoder.estimator.calls[0]["streaming"] is True
    assert flow.decoder.estimator.calls[0]["x"].shape[0] == 4
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "stream"]


def test_c1_first_hop_does_not_wait_for_peers() -> None:
    flow, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler._ingest_stream_item("req-a", _item(list(range(28))))
    started = time.monotonic()
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert time.monotonic() - started < 0.05
    assert len(flow.calls) == 1


def test_queued_peer_chunk_joins_first_hop_batch_during_wait() -> None:
    from sglang_omni.models.fun_cosyvoice3.stages import FunCosyVoice3Flow
    from tests.unit_test.fun_cosyvoice3.test_flow_batch import _FakeFlow as _PackedFlow

    flow = _PackedFlow(channels=80, max_frames=128)
    flow.spk_embed_affine_layer = torch.nn.Linear(192, 80, bias=False)
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        stages._CosyVoice3Vocoder(FunCosyVoice3Flow(flow), _FakeHiFT()),
        max_batch_size=8,
    )
    prompt_len = TOKEN_HOP_LEN
    payload_kwargs = {
        "prompt_token_len": prompt_len,
        "prompt_feat_frames": prompt_len * TOKEN_MEL_RATIO,
    }
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(
            request_id, _stream_payload(request_id, **payload_kwargs)
        )
    scheduler._ingest_stream_item("req-a", _item(list(range(28))))
    scheduler.inbox.put(
        IncomingMessage(
            request_id="req-b",
            type="stream_chunk",
            data=_item(list(range(28))),
        )
    )
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert flow.decoder.estimator.calls[0]["x"].shape[0] == 4


def _packed_scheduler(*, max_batch_size: int = 8):
    from sglang_omni.models.fun_cosyvoice3.stages import FunCosyVoice3Flow
    from tests.unit_test.fun_cosyvoice3.test_flow_batch import _FakeFlow as _PackedFlow

    flow = _PackedFlow(channels=80, max_frames=512)
    flow.spk_embed_affine_layer = torch.nn.Linear(192, 80, bias=False)
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        stages._CosyVoice3Vocoder(FunCosyVoice3Flow(flow), _FakeHiFT()),
        max_batch_size=max_batch_size,
    )
    return flow, scheduler


def _aligned_payload(request_id: str) -> StagePayload:
    prompt_len = TOKEN_HOP_LEN
    return _stream_payload(
        request_id,
        prompt_token_len=prompt_len,
        prompt_feat_frames=prompt_len * TOKEN_MEL_RATIO,
    )


def test_equal_follow_up_hops_share_one_causal_flow_batch() -> None:
    flow, scheduler = _packed_scheduler()
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(request_id, _aligned_payload(request_id))
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    first_calls = len(flow.decoder.estimator.calls)
    assert flow.decoder.estimator.calls[0]["x"].shape[0] == 4

    for request_id in ("req-a", "req-b"):
        scheduler._ingest_stream_item(
            request_id, _item([i % 31 for i in range(28, 78)])
        )
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    follow_calls = flow.decoder.estimator.calls[first_calls:]
    assert follow_calls
    assert follow_calls[0]["streaming"] is True
    assert follow_calls[0]["x"].shape[0] == 4
    messages = [m for m in _drain(scheduler) if m.type == "stream"]
    shapes = {_waveform(m.data).shape[0] for m in messages}
    assert 100 in shapes


def test_mixed_prompt_follow_ups_share_one_causal_flow_batch() -> None:
    flow, scheduler = _packed_scheduler()
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
        scheduler._on_streaming_new_request(request_id, payload)
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    first_calls = len(flow.decoder.estimator.calls)
    assert flow.decoder.estimator.calls[0]["x"].shape[0] == 4

    for request_id in payloads:
        scheduler._ingest_stream_item(
            request_id, _item([i % 31 for i in range(28, 78)])
        )
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    follow_calls = flow.decoder.estimator.calls[first_calls:]
    assert follow_calls
    assert follow_calls[0]["streaming"] is True
    assert follow_calls[0]["x"].shape[0] == 4


def test_c1_follow_up_stays_native_and_does_not_wait() -> None:
    flow, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler._ingest_stream_item("req-a", _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    scheduler._ingest_stream_item("req-a", _item(list(range(28, 78))))
    started = time.monotonic()
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert time.monotonic() - started < 0.05
    assert len(flow.calls) == 2
    assert int(flow.calls[1]["token"].shape[1]) == 78


def test_queued_peer_chunk_joins_follow_up_batch_during_wait() -> None:
    flow, scheduler = _packed_scheduler()
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(request_id, _aligned_payload(request_id))
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    first_calls = len(flow.decoder.estimator.calls)

    scheduler._ingest_stream_item("req-a", _item([i % 31 for i in range(28, 78)]))
    scheduler.inbox.put(
        IncomingMessage(
            request_id="req-b",
            type="stream_chunk",
            data=_item([i % 31 for i in range(28, 78)]),
        )
    )
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    follow_calls = flow.decoder.estimator.calls[first_calls:]
    assert follow_calls[0]["x"].shape[0] == 4


def test_backlogged_request_runs_one_hop_per_step() -> None:
    # note (guozhihao-224): 178 tokens cover first hop + two follow-ups
    # (28 / 78 / 178 windows). One step must advance only one hop.
    flow, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler._ingest_stream_item("req-a", _item(list(range(178))))
    with scheduler._state_lock:
        assert scheduler._pump_one_step() is None
    assert len(flow.calls) == 1
    assert int(flow.calls[0]["token"].shape[1]) == 28
    state = scheduler._stream_states["req-a"]
    assert state.token_offset == TOKEN_HOP_LEN
    assert state.hop_len == next_stream_hop_len(TOKEN_HOP_LEN)

    with scheduler._state_lock:
        assert scheduler._pump_one_step() is None
    assert len(flow.calls) == 2
    assert int(flow.calls[1]["token"].shape[1]) == 78
    assert state.token_offset == TOKEN_HOP_LEN + next_stream_hop_len(TOKEN_HOP_LEN)


def test_pump_drains_backlog_across_one_hop_steps() -> None:
    flow, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler._ingest_stream_item("req-a", _item(list(range(178))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [28, 78, 178]
    messages = [m for m in _drain(scheduler) if m.type == "stream"]
    assert len(messages) == 3


def test_inbox_first_hop_preempts_follow_up_backlog() -> None:
    # note (guozhihao-224): after A's first hop, leave two follow-ups ready
    # and park B's first hop in the inbox. Between steps the pump must
    # ingest B and prefer that first hop over draining A's backlog.
    flow, scheduler = _scheduler()
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(request_id, _stream_payload(request_id))
    scheduler._ingest_stream_item("req-a", _item(list(range(28))))
    with scheduler._state_lock:
        assert scheduler._pump_streams() == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [28]

    scheduler._ingest_stream_item("req-a", _item(list(range(28, 178))))
    scheduler.inbox.put(
        IncomingMessage(
            request_id="req-b",
            type="stream_chunk",
            data=_item(list(range(28))),
        )
    )
    with scheduler._state_lock:
        assert scheduler._pump_streams() == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [
        28,
        78,
        28,
        178,
    ]
    messages = [m for m in _drain(scheduler) if m.type == "stream"]
    assert len(messages) == 4


def test_disable_hop_growth_keeps_fixed_follow_up_windows() -> None:
    flow, scheduler = _scheduler(disable_hop_growth=True)
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    # First hop 28, then two fixed 25-token hops -> need 28+25+25 = 78 tokens
    # for three windows of 28 / 53 / 78 (lookahead included in prefix).
    scheduler._ingest_stream_item("req-a", _item(list(range(78))))
    with scheduler._state_lock:
        assert scheduler._pump_streams() == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [28, 53, 78]
    state = scheduler._stream_states["req-a"]
    assert state.hop_len == TOKEN_HOP_LEN


def test_token_max_hop_len_caps_growth() -> None:
    flow, scheduler = _scheduler(token_max_hop_len=50)
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    # With max 50: hops 25 -> 50 -> 50. Windows 28 / 78 / 128.
    scheduler._ingest_stream_item("req-a", _item(list(range(128))))
    with scheduler._state_lock:
        assert scheduler._pump_streams() == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [28, 78, 128]
    state = scheduler._stream_states["req-a"]
    assert state.hop_len == 50
