# SPDX-License-Identifier: Apache-2.0
"""Stream-aware text decoder for LLaDA2-Uni accepted token blocks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sglang_omni.models.llada2_uni.config import THINKER_STAGE
from sglang_omni.models.llada2_uni.merge import decode_events, decode_text_output
from sglang_omni.models.llada2_uni.payload_types import (
    LLaDA2UniEvent,
    LLaDA2UniPipelineState,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler

logger = logging.getLogger(__name__)


@dataclass
class _TextStreamState:
    token_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""


def _event_to_dict(event: LLaDA2UniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


def _coerce_token_ids(data: Any) -> list[int]:
    if isinstance(data, dict):
        data = data.get("token_ids")
    if data is None:
        return []
    if isinstance(data, (list, tuple)):
        values = data
    elif hasattr(data, "detach") and hasattr(data, "reshape"):
        values = data.detach().cpu().reshape(-1).tolist()
    elif hasattr(data, "tolist"):
        values = data.tolist()
    else:
        values = [data]
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [int(token_id) for token_id in values]


class LLaDA2StreamingDetokenizeScheduler(StreamingSimpleScheduler):
    """Decode accepted dLLM blocks and emit append-only text deltas."""

    def __init__(
        self,
        tokenizer: Any,
        eos_token_id: int | None,
        *,
        stage_name: str = "decode",
    ) -> None:
        self._tokenizer = tokenizer
        self._eos_token_id = eos_token_id
        self._stage_name = stage_name
        self._text_states: dict[str, _TextStreamState] = {}
        super().__init__(self._decode_non_streaming)

    def is_streaming_payload(self, payload: Any) -> bool:
        return isinstance(payload, StagePayload) and bool(
            (payload.request.params or {}).get("stream", False)
        )

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        del payload
        self._text_states.setdefault(request_id, _TextStreamState())

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        token_ids = _coerce_token_ids(item.data)
        if not token_ids:
            return []

        state = self._text_states.setdefault(request_id, _TextStreamState())
        state.token_ids.extend(token_ids)
        text = self._decode_text(state.token_ids)
        if text.endswith("\ufffd"):
            return []
        delta = self._append_only_delta(state, text)
        return self._stream_messages(request_id, delta)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        payload = self._stream_payloads.get(request_id)
        if not isinstance(payload, StagePayload):
            raise TypeError(
                f"LLaDA2 decode expected StagePayload for {request_id!r}, "
                f"got {type(payload).__name__}"
            )

        state = self._text_states.setdefault(request_id, _TextStreamState())
        pipeline_state, thinker_out = self._load_thinker_output(payload)
        del pipeline_state
        final_text = decode_text_output(
            thinker_out=thinker_out,
            tokenizer=self._tokenizer,
        )
        delta = self._append_only_delta(state, final_text)
        messages = self._stream_messages(request_id, delta)
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=self._build_result(payload, is_streaming=True),
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        self._text_states.pop(request_id, None)

    def _decode_text(self, token_ids: list[int]) -> str:
        visible_ids = [
            int(token_id)
            for token_id in token_ids
            if self._eos_token_id is None or int(token_id) != int(self._eos_token_id)
        ]
        if not visible_ids:
            return ""
        return self._tokenizer.decode(visible_ids, skip_special_tokens=True)

    @staticmethod
    def _append_only_delta(state: _TextStreamState, text: str) -> str:
        if not text.startswith(state.emitted_text):
            raise ValueError(
                "LLaDA2 incremental decode is not prefix-stable: "
                f"emitted={state.emitted_text!r}, decoded={text!r}"
            )
        delta = text[len(state.emitted_text) :]
        state.emitted_text = text
        return delta

    def _stream_messages(self, request_id: str, delta: str) -> list[OutgoingMessage]:
        if not delta:
            return []
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,
                data={
                    "text": delta,
                    "modality": "text",
                    "stage_name": self._stage_name,
                },
                metadata={"modality": "text"},
            )
        ]

    def _decode_non_streaming(self, payload: StagePayload) -> StagePayload:
        return self._build_result(payload, is_streaming=False)

    def _load_thinker_output(
        self, payload: StagePayload
    ) -> tuple[LLaDA2UniPipelineState, dict[str, Any]]:
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            logger.warning(
                "request %s: thinker produced no output (got %s), returning empty text",
                payload.request_id,
                type(thinker_out).__name__,
            )
            thinker_out = {"output_ids": [], "is_final": True}
        return state, thinker_out

    def _build_result(
        self, payload: StagePayload, *, is_streaming: bool
    ) -> StagePayload:
        state, thinker_out = self._load_thinker_output(payload)
        events = decode_events(thinker_out=thinker_out, tokenizer=self._tokenizer)
        result: dict[str, Any] = {"events": [_event_to_dict(event) for event in events]}
        if events:
            result.update(events[-1].payload)
            result.setdefault("modality", events[-1].modality)

        if is_streaming:
            result.pop("text", None)

        finish_reason = thinker_out.get("finish_reason")
        if finish_reason is not None:
            result["finish_reason"] = finish_reason

        input_ids = (
            state.prompt.get("input_ids") if isinstance(state.prompt, dict) else None
        )
        if input_ids is None:
            prompt_tokens = 0
        elif hasattr(input_ids, "numel"):
            prompt_tokens = int(input_ids.numel())
        else:
            prompt_tokens = len(input_ids)

        completion_tokens = len(thinker_out.get("output_ids") or [])
        result["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        payload.data = result
        return payload


def create_streaming_detokenize_scheduler(
    model_path: str,
    *,
    stage_name: str = "decode",
) -> LLaDA2StreamingDetokenizeScheduler:
    from sglang_omni.models.llada2_uni.components.common import load_llada2_tokenizer

    tokenizer = load_llada2_tokenizer(model_path)
    return LLaDA2StreamingDetokenizeScheduler(
        tokenizer=tokenizer,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        stage_name=stage_name,
    )


__all__ = [
    "LLaDA2StreamingDetokenizeScheduler",
    "create_streaming_detokenize_scheduler",
]
