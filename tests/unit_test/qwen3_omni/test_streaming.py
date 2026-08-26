# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Qwen3-Omni real-streaming path."""

from __future__ import annotations

import asyncio
import base64
import struct
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner._hidden_capture import StaticAuxHiddenCapture
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.models.qwen3_omni.components.streaming_detokenizer import (
    StreamingDetokenizeScheduler,
)
from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_stream_output_builder,
    resolve_encoder_next_stages,
    resolve_terminal_stages,
    resolve_thinker_next_stages,
    resolve_thinker_stream_done_targets,
    should_generate_audio_output,
)
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest


class _ByteTokenizer:
    """Token id → fixed bytes mapping; UTF-8 decode with errors='replace'."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        special_token_ids: set[int] | None = None,
        eos_token_id: int | None = None,
    ):
        self._vocab = vocab
        self._special = special_token_ids or set()
        self.eos_token_id = eos_token_id

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        chunks: list[bytes] = []
        for tid in ids:
            if skip_special_tokens and tid in self._special:
                continue
            chunks.append(self._vocab[tid])
        return b"".join(chunks).decode("utf-8", errors="replace")


@dataclass
class _StreamItem:
    """Mimics StreamItem.data shape passed to the scheduler inbox."""

    data: object
    metadata: dict | None = None


def _make_payload(stream: bool) -> StagePayload:
    """Build a StagePayload with the streaming flag plumbed through params."""
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": stream}),
        data={
            # Minimal Qwen3OmniPipelineState dict shape (decode_events will produce []).
            "engine_outputs": {
                "thinker": {
                    "output_ids": [],
                    "step": 0,
                    "is_final": True,
                    "extra_model_outputs": {},
                    "finish_reason": "stop",
                }
            },
            "thinker_out": None,
            "prompt": {"input_ids": []},
        },
    )


def _drain_outbox(scheduler: StreamingDetokenizeScheduler) -> list[OutgoingMessage]:
    out: list[OutgoingMessage] = []
    while not scheduler.outbox.empty():
        out.append(scheduler.outbox.get_nowait())
    return out


def _thinker_stage_payload(output_modalities: list[str] | None) -> StagePayload:
    metadata = {}
    if output_modalities is not None:
        metadata["output_modalities"] = output_modalities
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": True}, metadata=metadata),
        data={},
    )


def test_qwen_text_output_uses_text_only_active_subgraph():
    payload = _thinker_stage_payload(["text"])

    assert resolve_encoder_next_stages("req-1", payload) == "thinker"
    assert resolve_thinker_next_stages("req-1", payload) == "decode"
    assert resolve_thinker_stream_done_targets("req-1", payload) == ["decode"]
    assert resolve_terminal_stages(payload.request) == ["decode"]


def test_qwen_audio_output_uses_speech_active_subgraph():
    payload = _thinker_stage_payload(["text", "audio"])

    assert resolve_encoder_next_stages("req-1", payload) == [
        "thinker",
        "talker_ar",
    ]
    assert resolve_thinker_next_stages("req-1", payload) == "decode"
    assert resolve_thinker_stream_done_targets("req-1", payload) == [
        "talker_ar",
        "decode",
    ]
    assert resolve_terminal_stages(payload.request) == ["decode", "code2wav"]


def test_qwen_missing_output_modalities_uses_speech_active_subgraph():
    payload = _thinker_stage_payload(None)

    assert resolve_encoder_next_stages("req-1", payload) == [
        "thinker",
        "talker_ar",
    ]
    assert resolve_thinker_next_stages("req-1", payload) == "decode"
    assert resolve_thinker_stream_done_targets("req-1", payload) == [
        "talker_ar",
        "decode",
    ]
    assert resolve_terminal_stages(payload.request) == ["decode", "code2wav"]


def test_qwen_thinker_stream_builder_suppresses_talker_for_text_output():
    builder = make_thinker_stream_output_builder()
    req_data = SimpleNamespace(
        req=SimpleNamespace(inflight_middle_chunks=0),
        stage_payload=_thinker_stage_payload(["text"]),
    )
    req_output = SimpleNamespace(
        data=11,
        extra={"hidden_states": torch.tensor([[1.0, 2.0]])},
    )

    messages = builder("req-1", req_data, req_output)

    assert [msg.target for msg in messages] == ["decode"]


def test_qwen_thinker_stream_builder_keeps_talker_for_audio_output():
    builder = make_thinker_stream_output_builder()
    req_data = SimpleNamespace(
        req=SimpleNamespace(inflight_middle_chunks=0),
        stage_payload=_thinker_stage_payload(["audio"]),
    )
    req_output = SimpleNamespace(
        data=11,
        extra={"hidden_states": torch.tensor([[1.0, 2.0]])},
    )

    messages = builder("req-1", req_data, req_output)

    assert [msg.target for msg in messages] == ["decode", "talker_ar"]


def test_qwen_thinker_stream_builder_keeps_talker_when_modalities_missing():
    builder = make_thinker_stream_output_builder()
    req_data = SimpleNamespace(
        req=SimpleNamespace(inflight_middle_chunks=0),
        stage_payload=_thinker_stage_payload(None),
    )
    req_output = SimpleNamespace(
        data=11,
        extra={"hidden_states": torch.tensor([[1.0, 2.0]])},
    )

    messages = builder("req-1", req_data, req_output)

    assert [msg.target for msg in messages] == ["decode", "talker_ar"]


def test_qwen_thinker_stream_builder_prefers_embed_over_layer_hidden():
    builder = make_thinker_stream_output_builder()
    req_data = SimpleNamespace(
        req=SimpleNamespace(inflight_middle_chunks=0),
        stage_payload=_thinker_stage_payload(["audio"]),
    )
    embed = torch.tensor([[1.0, 2.0]])
    layer_hidden = torch.tensor([[3.0, 4.0]])
    req_output = SimpleNamespace(
        data=11,
        extra={"hidden_states": {"embed": embed, 24: layer_hidden}},
    )

    messages = builder("req-1", req_data, req_output)

    talker_message = next(msg for msg in messages if msg.target == "talker_ar")
    assert torch.equal(talker_message.data, embed[0])
    assert talker_message.metadata == {"token_id": 11}


def test_qwen_thinker_stream_builder_falls_back_to_layer_hidden():
    builder = make_thinker_stream_output_builder()
    req_data = SimpleNamespace(
        req=SimpleNamespace(inflight_middle_chunks=0),
        stage_payload=_thinker_stage_payload(["audio"]),
    )
    layer_hidden = torch.tensor([[3.0, 4.0]])
    req_output = SimpleNamespace(
        data=11,
        extra={"hidden_states": {24: layer_hidden}},
    )

    messages = builder("req-1", req_data, req_output)

    talker_message = next(msg for msg in messages if msg.target == "talker_ar")
    assert torch.equal(talker_message.data, layer_hidden[0])
    assert talker_message.metadata == {"token_id": 11}


def test_qwen_thinker_stream_embed_preserves_talker_prefill_contract():
    builder = make_thinker_stream_output_builder()
    req_data = SimpleNamespace(
        req=SimpleNamespace(inflight_middle_chunks=0),
        stage_payload=_thinker_stage_payload(["audio"]),
    )
    embed = torch.tensor([[7.0, 8.0]])
    layer_hidden = torch.tensor([[70.0, 80.0]])
    req_output = SimpleNamespace(
        data=11,
        extra={"hidden_states": {"embed": embed, 24: layer_hidden}},
    )

    messages = builder("req-1", req_data, req_output)
    talker_chunk = next(msg for msg in messages if msg.target == "talker_ar")

    class _TokenMetadataOnlyChunk:
        metadata = talker_chunk.metadata

        @property
        def data(self):
            raise AssertionError("prompt prefill must reconstruct assistant rows")

    token_metadata_only_chunk = _TokenMetadataOnlyChunk()
    legacy_chunk = SimpleNamespace(
        data=embed[0],
        metadata={"token_id": 11, "layer_hidden": layer_hidden[0]},
    )

    hidden_projection_calls: list[torch.Tensor] = []

    def hidden_projection(tensor: torch.Tensor) -> torch.Tensor:
        hidden_projection_calls.append(tensor.detach().clone())
        return tensor + 100.0

    prefill_builder = object.__new__(TalkerPrefillBuilder)
    prefill_builder._model = SimpleNamespace(
        text_projection=lambda tensor: tensor,
        hidden_projection=hidden_projection,
        get_input_embeddings=lambda: (
            lambda token_ids: torch.zeros((token_ids.numel(), 2))
        ),
    )
    prefill_builder._device = torch.device("cpu")
    prefill_builder._dtype = torch.float32
    prefill_builder._audio_token_id = 30
    prefill_builder._image_token_id = None
    prefill_builder._video_token_id = None
    prefill_builder._im_start_token_id = 10
    prefill_builder._im_end_token_id = 99
    prefill_builder._system_token_id = 19
    prefill_builder._user_token_id = 20
    prefill_builder._assistant_token_id = 40
    prefill_builder._codec_nothink_id = 1
    prefill_builder._codec_think_bos_id = 2
    prefill_builder._codec_think_eos_id = 3
    prefill_builder._codec_pad_id = 4
    prefill_builder._codec_bos_id = 5
    prefill_builder._tts_pad_token_id = 6
    prefill_builder._speaker_map = {}

    prompt_ids = torch.tensor([10, 20, 30, 10, 40], dtype=torch.long)
    prompt_embed = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    prompt_hidden = prompt_embed.clone()
    prompt_hidden[2] = torch.tensor([5.0, 6.0])
    prefill_builder._reconstruct_prompt_states = lambda _state: (
        prompt_ids,
        prompt_embed,
        prompt_hidden,
        {},
    )
    prefill_builder._load_prompt_token_embeddings = lambda _token_ids: embed
    zero_special = torch.zeros((1, 2), dtype=torch.float32)
    prefill_builder.get_tts_special_embeds = lambda: (
        zero_special,
        zero_special,
        zero_special,
    )

    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={}),
        data={},
    )
    current = prefill_builder.build_prompt_prefill(
        payload,
        [token_metadata_only_chunk],
        thinker_done=True,
    )
    current_projection_calls = list(hidden_projection_calls)
    hidden_projection_calls.clear()
    legacy = prefill_builder.build_prompt_prefill(
        payload,
        [legacy_chunk],
        thinker_done=True,
    )

    assert torch.equal(talker_chunk.data, embed[0])
    assert talker_chunk.metadata == {"token_id": 11}
    assert torch.equal(current["input_embeds"], legacy["input_embeds"])
    assert torch.equal(
        current["pending_text_queue"].rows,
        legacy["pending_text_queue"].rows,
    )
    assert len(current_projection_calls) == 1
    assert len(hidden_projection_calls) == 1
    assert torch.equal(current_projection_calls[0], prompt_hidden[2:3])
    assert torch.equal(hidden_projection_calls[0], prompt_hidden[2:3])


def test_qwen_hidden_states_skip_only_explicit_text_output_requests():
    output_processor = SGLangOutputProcessor(
        capture_hidden=True,
        should_emit_hidden=lambda request: should_generate_audio_output(
            request.data.stage_payload
        ),
    )
    text_request = SchedulerRequest(
        request_id="text",
        data=SGLangARRequestData(stage_payload=_thinker_stage_payload(["text"])),
    )
    audio_request = SchedulerRequest(
        request_id="audio",
        data=SGLangARRequestData(stage_payload=_thinker_stage_payload(["audio"])),
    )
    default_request = SchedulerRequest(
        request_id="default",
        data=SGLangARRequestData(stage_payload=_thinker_stage_payload(None)),
    )
    model_output = SimpleNamespace(
        next_token_ids=torch.tensor([11, 22, 33]),
        logits_output=SimpleNamespace(
            hidden_states=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ),
    )
    scheduler_output = SchedulerOutput(
        requests=[text_request, audio_request, default_request],
        batch_data=SimpleNamespace(
            reqs=[
                SimpleNamespace(extend_range=SimpleNamespace(length=1)),
                SimpleNamespace(extend_range=SimpleNamespace(length=1)),
                SimpleNamespace(extend_range=SimpleNamespace(length=1)),
            ]
        ),
    )

    outputs = output_processor.process(model_output, scheduler_output)

    assert outputs["text"].extra is None
    assert torch.equal(
        outputs["audio"].extra["hidden_states"],
        torch.tensor([3.0, 4.0]),
    )
    assert torch.equal(
        outputs["default"].extra["hidden_states"],
        torch.tensor([5.0, 6.0]),
    )


def _static_aux_scheduler_output(
    *lengths: int,
    is_extend: bool,
) -> SchedulerOutput:
    reqs = [
        (
            SimpleNamespace(extend_range=SimpleNamespace(length=length))
            if is_extend
            else SimpleNamespace()
        )
        for length in lengths
    ]
    return SchedulerOutput(
        requests=[SchedulerRequest(request_id=str(i)) for i in range(len(reqs))],
        batch_data=SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: is_extend),
            reqs=reqs,
        ),
    )


def test_qwen_static_aux_hidden_prefill_slices_only_logical_token_rows():
    static_embed = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    static_layer = torch.arange(100, 112, dtype=torch.float32).reshape(6, 2)
    capture = StaticAuxHiddenCapture(
        buffers=[static_embed, static_layer],
        hook_handles=[],
        max_tokens=6,
    )
    model = SimpleNamespace(_omni_aux_hidden_capture=capture)
    output_processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 24],
        model=model,
        should_emit_hidden=lambda request: request.request_id == "audio",
    )
    scheduler_output = SchedulerOutput(
        requests=[
            SchedulerRequest(request_id="text"),
            SchedulerRequest(request_id="audio"),
        ],
        batch_data=SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            reqs=[
                SimpleNamespace(extend_range=SimpleNamespace(length=1)),
                SimpleNamespace(extend_range=SimpleNamespace(length=2)),
            ],
        ),
    )
    model_output = SimpleNamespace(
        next_token_ids=torch.tensor([11, 22]),
        logits_output=SimpleNamespace(
            hidden_states=torch.arange(6, dtype=torch.float32).reshape(3, 2)
        ),
    )

    outputs = output_processor.process(model_output, scheduler_output)

    audio_hidden = outputs["audio"].extra["hidden_states"]
    assert outputs["text"].extra is None
    assert audio_hidden["embed"].shape == (2, 2)
    torch.testing.assert_close(audio_hidden["embed"], static_embed[1:3])
    torch.testing.assert_close(audio_hidden[24], static_layer[1:3])
    assert (
        audio_hidden["embed"].untyped_storage().nbytes()
        == audio_hidden["embed"].numel() * audio_hidden["embed"].element_size()
    )


def test_qwen_static_aux_hidden_prefill_keeps_token_major_ambiguous_shape():
    scheduler_output = _static_aux_scheduler_output(2, 0, is_extend=True)
    tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    first = SGLangOutputProcessor._slice_static_aux_hidden_tensor(
        tensor,
        request_index=0,
        scheduler_output=scheduler_output,
    )
    second = SGLangOutputProcessor._slice_static_aux_hidden_tensor(
        tensor,
        request_index=1,
        scheduler_output=scheduler_output,
    )

    torch.testing.assert_close(first, tensor)
    assert second.shape == (0, 2)


def test_qwen_static_aux_hidden_decode_slices_only_request_rows():
    static_embed = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    static_layer = torch.arange(100, 112, dtype=torch.float32).reshape(6, 2)
    capture = StaticAuxHiddenCapture(
        buffers=[static_embed, static_layer],
        hook_handles=[],
        max_tokens=6,
    )
    model = SimpleNamespace(_omni_aux_hidden_capture=capture)
    output_processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 24],
        model=model,
        should_emit_hidden=lambda request: request.request_id == "audio",
    )
    scheduler_output = SchedulerOutput(
        requests=[
            SchedulerRequest(request_id="text"),
            SchedulerRequest(request_id="audio"),
        ],
        batch_data=SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            reqs=[SimpleNamespace(), SimpleNamespace()],
        ),
    )
    model_output = SimpleNamespace(
        next_token_ids=torch.tensor([11, 22]),
        logits_output=SimpleNamespace(
            hidden_states=torch.arange(4, dtype=torch.float32).reshape(2, 2)
        ),
    )

    outputs = output_processor.process(model_output, scheduler_output)

    audio_hidden = outputs["audio"].extra["hidden_states"]
    assert outputs["text"].extra is None
    torch.testing.assert_close(audio_hidden["embed"], static_embed[1])
    torch.testing.assert_close(audio_hidden[24], static_layer[1])


@pytest.mark.parametrize(
    ("is_extend", "expected_rows", "actual_rows", "layout"),
    [
        (True, 2, 1, "token-major prefill"),
        (False, 1, 2, "request-major decode"),
    ],
)
def test_qwen_static_aux_hidden_rejects_wrong_row_count(
    is_extend: bool,
    expected_rows: int,
    actual_rows: int,
    layout: str,
):
    scheduler_output = _static_aux_scheduler_output(
        expected_rows,
        is_extend=is_extend,
    )

    with pytest.raises(
        RuntimeError,
        match=rf"{layout}.*expected {expected_rows} rows, got {actual_rows}",
    ):
        SGLangOutputProcessor._slice_static_aux_hidden_tensor(
            torch.zeros(actual_rows, 2),
            request_index=0,
            scheduler_output=scheduler_output,
        )


def test_qwen_static_aux_hidden_skips_capture_for_text_only_decode():
    class _UnexpectedCaptureRead:
        def __init__(self) -> None:
            self.views_calls = 0

        def views(self, _num_rows: int) -> None:
            self.views_calls += 1
            raise AssertionError("text-only decode must not read hidden capture")

    capture = _UnexpectedCaptureRead()
    output_processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 24],
        model=SimpleNamespace(_omni_aux_hidden_capture=capture),
        should_emit_hidden=lambda _request: False,
    )
    scheduler_output = SchedulerOutput(
        requests=[
            SchedulerRequest(request_id="text-1"),
            SchedulerRequest(request_id="text-2"),
        ],
        batch_data=SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            reqs=[SimpleNamespace(), SimpleNamespace()],
        ),
    )
    model_output = SimpleNamespace(
        next_token_ids=torch.tensor([11, 22]),
        logits_output=None,
    )

    outputs = output_processor.process(model_output, scheduler_output)

    assert capture.views_calls == 0
    assert outputs["text-1"].extra is None
    assert outputs["text-2"].extra is None


def test_utf8_multibyte_hold_then_emit():
    """A 3-byte CJK char split across 3 tokens must hold until complete."""
    # "你" is U+4F60 → b'\xe4\xbd\xa0'. Split byte-per-token.
    tok = _ByteTokenizer(
        vocab={1: b"\xe4", 2: b"\xbd", 3: b"\xa0", 99: b"hello"},
    )
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_chunk("req-1", _StreamItem(data=2))
    out = _drain_outbox(sched)
    assert out == [], "should hold until UTF-8 char completes"

    sched._on_stream_chunk("req-1", _StreamItem(data=3))
    out = _drain_outbox(sched)
    assert len(out) == 1
    assert out[0].type == "stream"
    assert out[0].target is None  # → Coordinator
    assert out[0].data["text"] == "你"

    sched._on_stream_chunk("req-1", _StreamItem(data=99))
    out = _drain_outbox(sched)
    assert len(out) == 1
    assert out[0].data["text"] == "hello"


def test_special_tokens_emit_no_delta():
    """A token in the special set must not produce a stream chunk."""
    tok = _ByteTokenizer(
        vocab={1: b"hi", 2: b"<eos>"},
        special_token_ids={2},
    )
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=2)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_chunk("req-1", _StreamItem(data=2))
    out = _drain_outbox(sched)
    assert len(out) == 1
    assert out[0].data["text"] == "hi"


def test_zero_token_stream_done_does_not_deadlock():
    """``stream_done`` arriving before any chunk and before ``new_request``
    must still let ``new_request`` finalize the streaming request."""
    tok = _ByteTokenizer(vocab={})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_done("req-1")
    sched._on_new_request("req-1", _make_payload(stream=True))
    out = _drain_outbox(sched)
    result_msgs = [m for m in out if m.type == "result"]
    assert len(result_msgs) == 1, "finalize must run even with zero-token output"


def test_non_streaming_finalizes_on_new_request():
    """``stream=False`` must finalize immediately on ``new_request``."""
    tok = _ByteTokenizer(vocab={})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_new_request("req-1", _make_payload(stream=False))
    out = _drain_outbox(sched)
    result_msgs = [m for m in out if m.type == "result"]
    assert len(result_msgs) == 1


def test_streaming_finalize_after_chunks_then_done_then_new_request():
    """Normal streaming order: chunks → done → new_request → finalize."""
    tok = _ByteTokenizer(vocab={1: b"hi"})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_done("req-1")
    sched._on_new_request("req-1", _make_payload(stream=True))
    out = _drain_outbox(sched)
    types = [m.type for m in out]
    assert types.count("stream") >= 1
    assert types.count("result") == 1


def _payload_with_output_ids(stream: bool, output_ids: list[int]) -> StagePayload:
    """Variant of _make_payload that injects a non-empty output_ids list so
    decode_events produces a text_final event with the full reconstructed
    text in its payload — the case the slim-final invariant guards against.
    """
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": stream}),
        data={
            "engine_outputs": {
                "thinker": {
                    "output_ids": list(output_ids),
                    "step": len(output_ids),
                    "is_final": True,
                    "extra_model_outputs": {},
                    "finish_reason": "stop",
                }
            },
            "thinker_out": None,
            "prompt": {"input_ids": []},
            "stream_state": {},
        },
    )


def test_streaming_final_result_drops_full_text_to_avoid_duplication():
    """When stream=True, the terminal result must NOT carry the full
    reconstructed text — text deltas were already streamed via
    OutgoingMessage(type='stream'). A direct client that appends every
    chunk's text would otherwise emit the whole response twice.
    """
    tok = _ByteTokenizer(vocab={1: b"hi", 2: b" there"})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_chunk("req-1", _StreamItem(data=2))
    sched._on_stream_done("req-1")
    sched._on_new_request(
        "req-1", _payload_with_output_ids(stream=True, output_ids=[1, 2])
    )

    out = _drain_outbox(sched)
    stream_msgs = [m for m in out if m.type == "stream"]
    result_msgs = [m for m in out if m.type == "result"]
    assert stream_msgs, "deltas must reach the client before the final result"
    assert len(result_msgs) == 1

    final_data = result_msgs[0].data.data
    assert (
        "text" not in final_data
    ), "streaming final must not duplicate text already emitted as deltas"
    assert "events" in final_data
    assert "usage" in final_data
    assert final_data.get("finish_reason") == "stop"


def test_non_streaming_final_result_keeps_full_text():
    """Non-streaming clients receive a single terminal result and must
    still see the full reconstructed text (regression guard for the
    slim-final branch in _build_result).
    """
    tok = _ByteTokenizer(vocab={1: b"hi", 2: b" there"})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_new_request(
        "req-1", _payload_with_output_ids(stream=False, output_ids=[1, 2])
    )
    result_msgs = [m for m in _drain_outbox(sched) if m.type == "result"]
    assert len(result_msgs) == 1
    final_data = result_msgs[0].data.data
    assert final_data.get("text") == "hi there"
    assert final_data.get("finish_reason") == "stop"


def test_abort_clears_state():
    tok = _ByteTokenizer(vocab={1: b"hi"})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    assert "req-1" in sched._state
    sched.abort("req-1")
    assert "req-1" not in sched._state


class _FakeCode2Wav:
    """Stand-in for the real vocoder; produces 4 audio samples per code frame."""

    total_upsample = 4

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: (1, codebooks, num_frames). Output shape (1, frames * upsample).
        n_frames = codes.shape[-1]
        return torch.zeros(1, n_frames * self.total_upsample)


def _make_code_chunk(metadata: dict | None) -> StreamItem:
    """One frame per chunk, single codebook, non-EOS code id."""
    return StreamItem(
        chunk_id=0,
        data=torch.tensor([7], dtype=torch.long),
        from_stage="talker",
        metadata=metadata,
    )


def test_code2wav_chunk_without_stream_metadata_raises():
    """Missing metadata is rejected; the serving loop turns the raise into an
    outbox 'error' + abort (StreamingSimpleScheduler.start)."""
    sched = Code2WavScheduler(
        model=_FakeCode2Wav(),
        device="cpu",
        stream_chunk_size=10,
        left_context_size=0,
    )
    with pytest.raises(RuntimeError, match="missing metadata"):
        sched._on_chunk("req-1", _make_code_chunk(metadata=None))

    sched.abort("req-1")
    assert "req-1" not in sched._stream_states


def test_code2wav_streaming_emits_per_window_and_slim_final():
    sched = Code2WavScheduler(
        model=_FakeCode2Wav(),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=0,
    )
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": True}),
        data={},
    )
    sched._stream_payloads["req-1"] = payload

    # Two chunks trigger the first decode step (stream_chunk_size=2).
    sched._on_chunk("req-1", _make_code_chunk(metadata={"stream": True}))
    sched._on_chunk("req-1", _make_code_chunk(metadata={"stream": True}))

    out: list[OutgoingMessage] = []
    while not sched.outbox.empty():
        out.append(sched.outbox.get_nowait())
    assert any(
        m.type == "stream" and m.target is None for m in out
    ), "streaming clients should receive per-window audio"

    # Done → slim final.
    sched._on_done("req-1")
    final = [
        m
        for m in (sched.outbox.get_nowait() for _ in range(sched.outbox.qsize()))
        if m.type == "result"
    ]
    assert len(final) == 1
    fdata = final[0].data.data
    assert fdata.get("modality") == "audio"
    assert "audio_waveform" not in fdata, "streaming final must be slim"


def test_code2wav_non_streaming_returns_full_pcm():
    sched = Code2WavScheduler(
        model=_FakeCode2Wav(),
        device="cpu",
        stream_chunk_size=10,  # never trips during chunk feed
        left_context_size=0,
    )
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": False}),
        data={},
    )
    sched._stream_payloads["req-1"] = payload

    sched._on_chunk("req-1", _make_code_chunk(metadata={"stream": False}))
    sched._on_done("req-1")

    msgs: list[OutgoingMessage] = []
    while not sched.outbox.empty():
        msgs.append(sched.outbox.get_nowait())
    final = [m for m in msgs if m.type == "result"]
    assert len(final) == 1
    fdata = final[0].data.data
    assert "audio_waveform" in fdata, "non-streaming final must carry full PCM"
    assert fdata["modality"] == "audio"
    # No per-window stream emit on non-streaming.
    assert not any(m.type == "stream" for m in msgs)


def test_code2wav_done_without_audio_raises():
    """The serving loop turns the raise into an outbox 'error' + abort."""
    sched = Code2WavScheduler(
        model=_FakeCode2Wav(),
        device="cpu",
        stream_chunk_size=10,
        left_context_size=0,
    )
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": False}),
        data={},
    )
    sched._stream_payloads["req-1"] = payload
    state = sched._get_or_create_stream_state("req-1")
    state.stream_enabled = False

    with pytest.raises(RuntimeError, match="produced no audio"):
        sched._on_done("req-1")
    assert not any(m.type == "stream" for m in list(sched.outbox.queue))

    sched.abort("req-1")
    assert "req-1" not in sched._stream_states
    assert "req-1" not in sched._stream_payloads


def _bare_stage(*, is_terminal: bool, owns_io: bool = True) -> Stage:
    """Construct a Stage shell that bypasses __init__ for unit-level checks."""
    s = Stage.__new__(Stage)
    s.name = "decode" if is_terminal else "thinker"
    s._is_terminal = is_terminal
    s._owns_external_io = owns_io
    s._aborted = set()
    s._active_requests = set()
    s._replica_bindings = {}
    s._stream_queue = None
    s._stream_chunk_counters = {}
    s._first_stream_chunk_seen = set()
    s._local_stream_targets = {}
    s._nonlocal_stream_targets = {}
    s.relay = SimpleNamespace()
    s.input_handler = SimpleNamespace(cancel=lambda request_id: None)
    s.scheduler = SimpleNamespace(abort=lambda request_id: None)
    s.control_plane = SimpleNamespace(completions=[])

    async def _send_complete(msg):
        s.control_plane.completions.append(msg)

    s.control_plane.send_complete = _send_complete
    return s


def test_send_stream_to_coordinator_raises_on_non_terminal():
    s = _bare_stage(is_terminal=False)
    with pytest.raises(RuntimeError, match="terminal"):
        asyncio.run(
            s._send_stream_to_coordinator(
                request_id="req-1",
                data={"text": "hi"},
                metadata={"modality": "text"},
            )
        )


def test_send_stream_to_coordinator_short_circuits_for_followers():
    """TP follower (owns_external_io=False) must drop silently, not raise."""
    s = _bare_stage(is_terminal=True, owns_io=False)
    asyncio.run(
        s._send_stream_to_coordinator(
            request_id="req-1",
            data={"text": "hi"},
            metadata={"modality": "text"},
        )
    )


def test_queue_stream_error_fast_fails_when_no_queue():
    """When _stream_queue is None, _queue_stream_error must surface a
    coordinator failure rather than silently dropping the error."""
    s = _bare_stage(is_terminal=True)
    asyncio.run(
        s._queue_stream_error("req-1", from_stage="thinker", error=RuntimeError("boom"))
    )
    assert len(s.control_plane.completions) == 1
    assert s.control_plane.completions[0].request_id == "req-1"
    assert s.control_plane.completions[0].error == "boom"
    assert "req-1" in s._aborted


def test_queue_stream_error_aborted_request_no_op():
    """An aborted request must not surface another failure to the coordinator."""
    s = _bare_stage(is_terminal=True)
    s._aborted.add("req-1")
    asyncio.run(
        s._queue_stream_error("req-1", from_stage="thinker", error=RuntimeError("late"))
    )
    assert s.control_plane.completions == []


def test_queue_stream_error_repeated_calls_are_idempotent_at_handler():
    """The first failure marks the request aborted; repeated errors are local no-ops."""
    s = _bare_stage(is_terminal=True)

    async def _drive():
        await s._queue_stream_error("req-1", "thinker", RuntimeError("first"))
        await s._queue_stream_error("req-1", "thinker", RuntimeError("second"))

    asyncio.run(_drive())
    assert len(s.control_plane.completions) == 1
    assert s.control_plane.completions[0].error == "first"


def test_late_stream_done_after_finalize_does_not_re_create_state():
    """Invariant: a late duplicate done does not allocate a new _RequestState row."""
    tok = _ByteTokenizer(vocab={1: b"hi"})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_done("req-1")
    sched._on_new_request("req-1", _make_payload(stream=True))
    _drain_outbox(sched)
    assert "req-1" not in sched._state
    assert "req-1" not in sched._done_seen

    sched._on_stream_done("req-1")  # duplicate / late
    assert "req-1" not in sched._state, "late done must not re-create state"


def test_done_seen_cleared_on_abort():
    """_done_seen latches must be cleared on abort to bound memory."""
    tok = _ByteTokenizer(vocab={})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_done("req-1")
    assert "req-1" in sched._done_seen
    sched.abort("req-1")
    assert "req-1" not in sched._done_seen


class _RaisingTokenizer:
    """Decode raises on a specific marker token; succeeds otherwise.

    Used to force ``_on_stream_chunk`` to raise from inside the scheduler
    loop without monkey-patching private methods.
    """

    def __init__(self, *, eos_token_id: int | None = None) -> None:
        self.eos_token_id = eos_token_id

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if any(tid == 999 for tid in ids):
            raise RuntimeError("tokenizer-decode-boom")
        # Map every other id to a single ASCII letter; keeps deltas non-empty.
        return "".join(chr(ord("a") + (int(tid) % 26)) for tid in ids)


def test_scheduler_isolates_per_request_chunk_failure():
    """An exception inside ``_on_stream_chunk`` must surface as an
    ``OutgoingMessage(type="error")`` for that request only, and the
    scheduler thread must stay alive to serve later requests.
    """
    sched = StreamingDetokenizeScheduler(
        tokenizer=_RaisingTokenizer(),
        eos_token_id=None,
    )
    thread = threading.Thread(target=sched.start, daemon=True)
    thread.start()
    try:
        # req-bad: chunk carries the poison token id (999) → decode raises.
        sched.inbox.put(
            IncomingMessage(
                request_id="req-bad",
                type="stream_chunk",
                data=_StreamItem(data=999),
            )
        )
        err = sched.outbox.get(timeout=2.0)
        assert err.type == "error"
        assert err.request_id == "req-bad"
        assert isinstance(err.data, RuntimeError)
        assert "tokenizer-decode-boom" in str(err.data)
        # State for the failed request must be cleared.
        assert "req-bad" not in sched._state
        assert "req-bad" not in sched._done_seen

        # Scheduler thread must still be alive and processing.
        assert thread.is_alive()

        # req-good: a healthy non-streaming request finalizes normally.
        sched.inbox.put(
            IncomingMessage(
                request_id="req-good",
                type="new_request",
                data=_make_payload(stream=False),
            )
        )
        ok = sched.outbox.get(timeout=2.0)
        assert ok.type == "result"
        assert ok.request_id == "req-good"
    finally:
        sched.stop()
        thread.join(timeout=2.0)


def test_scheduler_isolates_per_request_finalize_failure():
    """An exception inside ``_finalize`` (e.g., via Qwen3OmniPipelineState.from_dict
    on a malformed payload) must isolate to that request without taking
    down the scheduler thread.
    """
    sched = StreamingDetokenizeScheduler(
        tokenizer=_RaisingTokenizer(),
        eos_token_id=None,
    )
    thread = threading.Thread(target=sched.start, daemon=True)
    thread.start()
    try:
        # Force _finalize to raise: poison token 999 in output_ids makes
        # _build_result call tokenizer.decode([999], ...) → RuntimeError.
        bad_payload = StagePayload(
            request_id="req-bad",
            request=OmniRequest(inputs=[], params={"stream": False}),
            data={
                "engine_outputs": {
                    "thinker": {
                        "output_ids": [999],
                        "step": 1,
                        "is_final": True,
                        "extra_model_outputs": {},
                        "finish_reason": "stop",
                    }
                },
                "thinker_out": None,
                "prompt": {"input_ids": []},
            },
        )
        sched.inbox.put(
            IncomingMessage(
                request_id="req-bad",
                type="new_request",
                data=bad_payload,
            )
        )
        err = sched.outbox.get(timeout=2.0)
        assert err.type == "error"
        assert err.request_id == "req-bad"
        assert isinstance(err.data, Exception)
        assert "req-bad" not in sched._state
        assert thread.is_alive()

        # Scheduler is still healthy.
        sched.inbox.put(
            IncomingMessage(
                request_id="req-good",
                type="new_request",
                data=_make_payload(stream=False),
            )
        )
        ok = sched.outbox.get(timeout=2.0)
        assert ok.type == "result"
        assert ok.request_id == "req-good"
    finally:
        sched.stop()
        thread.join(timeout=2.0)


def test_code2wav_abort_clears_all_per_request_state():
    sched = Code2WavScheduler(
        model=_FakeCode2Wav(),
        device="cpu",
        stream_chunk_size=10,
        left_context_size=0,
    )
    sched._on_chunk("req-1", _make_code_chunk(metadata={"stream": True}))
    assert "req-1" in sched._stream_states
    assert sched._stream_states["req-1"].stream_enabled is True

    sched.abort("req-1")
    assert "req-1" not in sched._stream_states
    assert "req-1" not in sched._stream_payloads
    assert "req-1" not in sched._pending_done


class _FakeCoordinatorForClient:
    """Async-iterates a pre-seeded message list as a Coordinator.stream() stand-in."""

    def __init__(self, messages, *, submit_result=None):
        self._messages = list(messages)
        self._submit_result = submit_result
        self.submitted_params: list[dict] = []

    async def stream(self, request_id, omni_request):
        del request_id, omni_request
        for m in self._messages:
            yield m

    async def submit(self, request_id, omni_request):
        del request_id
        self.submitted_params.append(dict(omni_request.params))
        return self._submit_result


def test_client_completion_stream_does_not_duplicate_full_text():
    """Two text deltas + a slim completion must yield a chunk sequence whose
    concatenated text equals the response once, not twice. Covers the
    `_default_stream_builder` / `_default_result_builder` translation path
    that scheduler-level tests can't reach.
    """
    from sglang_omni.client.client import Client
    from sglang_omni.client.types import GenerateRequest
    from sglang_omni.proto import CompleteMessage, StreamMessage

    messages = [
        StreamMessage(
            request_id="req-1",
            from_stage="decode",
            chunk={"text": "hi", "modality": "text", "stage_name": "decode"},
            stage_name="decode",
            modality="text",
        ),
        StreamMessage(
            request_id="req-1",
            from_stage="decode",
            chunk={"text": " there", "modality": "text", "stage_name": "decode"},
            stage_name="decode",
            modality="text",
        ),
        CompleteMessage(
            request_id="req-1",
            from_stage="decode",
            success=True,
            # Slim shape produced by StreamingDetokenizeScheduler when
            # stream=True: no top-level "text".
            result={
                "events": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 2,
                    "total_tokens": 2,
                },
                "finish_reason": "stop",
                "modality": "text",
            },
        ),
    ]
    client = Client(coordinator=_FakeCoordinatorForClient(messages))

    request = GenerateRequest(prompt="ignored-in-fake", stream=True)

    async def _collect():
        out = []
        async for chunk in client.completion_stream(request, request_id="req-1"):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())

    text_parts = [c.text for c in chunks if c.text]
    assert text_parts == ["hi", " there"], (
        f"streaming consumer must see each delta exactly once and no "
        f"reconstructed full text, got {text_parts!r}"
    )

    final = chunks[-1]
    assert final.finish_reason == "stop"
    # The terminal chunk must not re-emit the full response text.
    assert final.text in (None, "", "hi", " there"), (
        f"final chunk text must not be the full reconstructed response, "
        f"got {final.text!r}"
    )

    full = "".join(c.text or "" for c in chunks)
    assert (
        full == "hi there"
    ), f"concatenated stream must equal the response once, got {full!r}"


def test_client_completion_stream_non_streaming_keeps_full_text():
    """Regression guard: when the scheduler does NOT slim (non-streaming
    path), `Client.completion_stream()` must surface the full text on
    the terminal chunk so callers using the unified API still receive it.
    """
    from sglang_omni.client.client import Client
    from sglang_omni.client.types import GenerateRequest
    from sglang_omni.proto import CompleteMessage

    messages = [
        CompleteMessage(
            request_id="req-1",
            from_stage="decode",
            success=True,
            result={
                "events": [],
                "text": "hi there",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 2,
                    "total_tokens": 2,
                },
                "finish_reason": "stop",
                "modality": "text",
            },
        ),
    ]
    client = Client(coordinator=_FakeCoordinatorForClient(messages))

    # stream=True at the client surface still drives the streaming path;
    # what matters is that the coordinator hands us a non-slim result.
    request = GenerateRequest(prompt="ignored", stream=True)

    async def _collect():
        out = []
        async for chunk in client.completion_stream(request, request_id="req-1"):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())
    assert len(chunks) == 1
    assert chunks[0].text == "hi there"
    assert chunks[0].finish_reason == "stop"


def test_client_completion_audio_uses_chunk_sample_rate():
    from sglang_omni.client.client import Client
    from sglang_omni.client.types import GenerateRequest

    coordinator = _FakeCoordinatorForClient(
        [],
        submit_result={
            "audio_data": [0.0, 0.1, -0.1, 0.0],
            "sample_rate": 48000,
            "modality": "audio",
        },
    )
    client = Client(coordinator=coordinator)

    result = asyncio.run(
        client.completion(
            GenerateRequest(prompt="ignored", stream=False),
            request_id="req-1",
            audio_format="wav",
        )
    )

    assert result.audio is not None
    wav = base64.b64decode(result.audio.data)
    assert struct.unpack("<I", wav[24:28])[0] == 48000


def test_client_completion_stream_audio_uses_chunk_sample_rate():
    from sglang_omni.client.client import Client
    from sglang_omni.client.types import GenerateRequest
    from sglang_omni.proto import StreamMessage

    messages = [
        StreamMessage(
            request_id="req-1",
            from_stage="vocoder",
            chunk={
                "audio_data": [0.0, 0.1, -0.1, 0.0],
                "sample_rate": 48000,
                "modality": "audio",
            },
            stage_name="vocoder",
            modality="audio",
        )
    ]
    client = Client(coordinator=_FakeCoordinatorForClient(messages))

    async def _collect():
        out = []
        async for chunk in client.completion_stream(
            GenerateRequest(prompt="ignored", stream=True),
            request_id="req-1",
            audio_format="wav",
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())

    assert chunks[0].audio_b64 is not None
    wav = base64.b64decode(chunks[0].audio_b64)
    assert struct.unpack("<I", wav[24:28])[0] == 48000


def test_client_speech_forces_non_streaming_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.client import client as client_module
    from sglang_omni.client.client import Client
    from sglang_omni.client.types import GenerateRequest

    monkeypatch.setattr(
        client_module,
        "encode_audio",
        lambda audio_data, **kwargs: (b"encoded-audio", "audio/wav"),
    )
    coordinator = _FakeCoordinatorForClient(
        [],
        submit_result={
            "audio_data": [0.0, 0.1],
            "sample_rate": 16000,
            "modality": "audio",
        },
    )
    client = Client(coordinator=coordinator)

    result = asyncio.run(
        client.speech(
            GenerateRequest(
                prompt="ignored", stream=True, extra_params={"stream": True}
            ),
            request_id="req-1",
        )
    )

    assert result.audio_bytes == b"encoded-audio"
    assert coordinator.submitted_params == [
        {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "stop": [],
            "stop_token_ids": [],
            "seed": None,
            "stream": False,
        }
    ]
