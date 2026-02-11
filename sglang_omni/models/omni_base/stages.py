# SPDX-License-Identifier: Apache-2.0
"""Generic stage executor factories for omni pipelines.

These factories encapsulate common orchestration logic (engine creation,
request/result building, step counting, streaming, decoding) so that concrete
model packages only need to instantiate their own components and pass them in.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

from sglang_omni.engines.omni import create_ar_engine, create_encoder_engine
from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.omni_base.engine_io import (
    apply_encoder_result,
    apply_thinker_result,
    build_encoder_request,
    build_thinker_request,
)
from sglang_omni.models.omni_base.types import OmniEvent, ThinkerOutput
from sglang_omni.proto import StagePayload

# Type alias for the model-specific decode callback
DecodeEventsFn = Callable[..., Iterable[OmniEvent]]


def event_to_dict(event: OmniEvent) -> dict[str, Any]:
    """Serialize an OmniEvent to a plain dict."""
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


# ---------------------------------------------------------------------------
# Aggregate (identity)
# ---------------------------------------------------------------------------


def create_aggregate_executor() -> PreprocessingExecutor:
    """Create an identity aggregate executor.

    The aggregate stage normally receives the merged payload from upstream;
    the executor itself just passes it through.
    """

    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return PreprocessingExecutor(_identity)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def create_encoder_executor(
    *,
    stage_name: str,
    model: torch.nn.Module,
    device: str,
) -> EngineExecutor:
    """Create a generic encoder executor.

    Parameters
    ----------
    stage_name:
        Pipeline stage name (e.g. "image_encoder", "audio_encoder").
    model:
        The encoder torch.nn.Module.
    device:
        Device string passed to the engine.
    """

    def _request_builder(payload: StagePayload):
        return build_encoder_request(payload, stage_name=stage_name)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        apply_encoder_result(payload, stage_name=stage_name, result=result)
        return payload

    engine = create_encoder_engine(model, device=device)
    return EngineExecutor(
        engine=engine, request_builder=_request_builder, result_builder=_result_builder
    )


# ---------------------------------------------------------------------------
# Thinker (autoregressive LLM)
# ---------------------------------------------------------------------------


def create_thinker_executor(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    decode_events_fn: DecodeEventsFn,
    stage_name: str,
    max_seq_len: int = 8192,
    device: str = "cuda",
) -> EngineExecutor:
    """Create a generic thinker executor.

    Parameters
    ----------
    model:
        The thinker torch.nn.Module.
    tokenizer:
        HuggingFace-compatible tokenizer.
    decode_events_fn:
        Model-specific callback that converts a partial ThinkerOutput
        into OmniEvent instances (used for both streaming and final decode).
    stage_name:
        Pipeline stage name (e.g. "thinker").
    max_seq_len:
        Maximum sequence length for the AR engine.
    device:
        Device string.
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    step_counters: dict[str, int] = {}

    def _request_builder(payload: StagePayload):
        step_counters.pop(payload.request_id, None)
        return build_thinker_request(payload, params=payload.request.params)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        apply_thinker_result(payload, stage_name=stage_name, result=result)
        step_counters.pop(payload.request_id, None)
        return payload

    def _stream_builder(payload: StagePayload | None, item: Any) -> Any:
        if payload is None:
            return None
        request_id = payload.request_id
        step = step_counters.get(request_id, 0) + 1
        step_counters[request_id] = step

        try:
            token_id = int(item)
        except Exception:
            return {"token_id": item, "step": step}

        thinker_out: ThinkerOutput = {
            "output_ids": [token_id],
            "step": step,
            "is_final": False,
        }
        events = list(
            decode_events_fn(
                thinker_out=thinker_out,
                payload=payload,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                step=step,
            )
        )
        if eos_token_id is not None and token_id == eos_token_id and not events:
            events = [
                OmniEvent(type="text_final", modality="text", payload={}, is_final=True)
            ]
        return {
            "events": [event_to_dict(event) for event in events],
            "token_id": token_id,
            "step": step,
            "stage": stage_name,
        }

    engine = create_ar_engine(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        device=device,
    )

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
        stream_builder=_stream_builder,
    )


# ---------------------------------------------------------------------------
# Decode (post-thinker event extraction)
# ---------------------------------------------------------------------------


def create_decode_executor(
    *,
    tokenizer: Any,
    decode_events_fn: DecodeEventsFn,
    thinker_stage_name: str,
) -> PreprocessingExecutor:
    """Create a generic decode executor.

    Parameters
    ----------
    tokenizer:
        HuggingFace-compatible tokenizer.
    decode_events_fn:
        Model-specific callback that converts a ThinkerOutput
        into OmniEvent instances.
    thinker_stage_name:
        The stage name used by the thinker (needed to look up results in
        payload.engine_outputs).
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def _decode(payload: StagePayload) -> StagePayload:
        engine_outputs = payload.engine_outputs or {}
        thinker_out = payload.thinker_out or engine_outputs.get(thinker_stage_name)
        if not isinstance(thinker_out, dict):
            thinker_out = {
                "output_ids": [],
                "step": 0,
                "is_final": True,
                "extra_model_outputs": {},
            }

        step = int(thinker_out.get("step") or len(thinker_out.get("output_ids", [])))
        events = list(
            decode_events_fn(
                thinker_out=thinker_out,
                payload=payload,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                step=step,
            )
        )
        event_dicts = [event_to_dict(event) for event in events]

        result: dict[str, Any] = {"events": event_dicts}
        final_event = next(
            (
                event
                for event in reversed(events)
                if event.is_final or event.type in {"text_final", "final"}
            ),
            None,
        )
        if final_event is not None:
            result.update(final_event.payload)
            result.setdefault("modality", final_event.modality)

        if "text" not in result:
            output_ids = thinker_out.get("output_ids")
            if (
                callable(getattr(tokenizer, "decode", None))
                and isinstance(output_ids, list)
                and output_ids
            ):
                result["text"] = tokenizer.decode(output_ids, skip_special_tokens=True)
                result.setdefault("modality", "text")

        payload.data = result
        return payload

    return PreprocessingExecutor(_decode)
