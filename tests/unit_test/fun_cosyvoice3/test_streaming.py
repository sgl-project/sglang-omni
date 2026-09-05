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
    AR_FOLLOWUP_FLUSH_TOKENS,
    AR_INITIAL_FLUSH_TOKENS,
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MEL_RATIO,
    first_ar_flush_tokens,
    next_stream_hop_len,
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
    assert tokens_needed_for_causal_chunk(0, hop_len=25, prompt_pad=0) == 28
    assert tokens_needed_for_causal_chunk(0, hop_len=25, prompt_pad=15) == 43
    assert first_ar_flush_tokens(0) == AR_INITIAL_FLUSH_TOKENS
    assert first_ar_flush_tokens(10) == 43
    assert first_ar_flush_tokens(25) == AR_INITIAL_FLUSH_TOKENS


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


def _scheduler() -> tuple[_FakeFlow, FunCosyVoice3StreamingVocoderScheduler]:
    flow = _FakeFlow()
    return flow, FunCosyVoice3StreamingVocoderScheduler(
        stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    )


def _stream_payload(
    request_id: str = "req-stream",
    *,
    codes: list[int] | None = None,
    prompt_feat_frames: int = 0,
) -> StagePayload:
    state = FunCosyVoice3State(
        text="hello",
        stream=True,
        audio_codes=None if codes is None else torch.tensor(codes, dtype=torch.long),
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
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


def test_model_runner_flushes_speech_tokens_and_skips_control_ids() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner._outbox = Queue()
    runner._vocoder_target = "vocoder"
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


def test_model_runner_first_flush_includes_prompt_pad() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner._outbox = Queue()
    runner._vocoder_target = "vocoder"
    prompt_len = 10
    data = CosyVoice3SGLangRequestData(
        stream_metadata={"modality": "audio_codes", "stream": True},
        flow_prompt_speech_token=torch.zeros(1, prompt_len, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 1, 80),
        flow_embedding=torch.ones(1, 192),
    )
    request = SimpleNamespace(request_id="req-pad", data=data)
    first_flush = first_ar_flush_tokens(prompt_len)
    assert first_flush == 43

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
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner._outbox = Queue()
    runner._vocoder_target = "vocoder"
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
