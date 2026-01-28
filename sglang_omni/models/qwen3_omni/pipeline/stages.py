# SPDX-License-Identifier: Apache-2.0
"""Stage executors for Qwen3-Omni pipelines."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer

from sglang_omni.engines.omni import create_ar_engine, create_encoder_engine
from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.executors import EngineExecutor, FrontendExecutor
from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.frontend import Qwen3OmniFrontend
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.thinker import Qwen3OmniSplitThinker
from sglang_omni.models.qwen3_omni.io import OmniEvent, ThinkerOutput
from sglang_omni.models.qwen3_omni.pipeline.merge import decode_events
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    AUDIO_STAGE,
    IMAGE_STAGE,
    THINKER_STAGE,
)
from sglang_omni.proto import StagePayload


def _ensure_data(payload: StagePayload) -> dict[str, Any]:
    if not isinstance(payload.data, dict):
        payload.data = {}
    return payload.data


def _event_to_dict(event: OmniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


def create_frontend_executor(model_id: str) -> FrontendExecutor:
    frontend = Qwen3OmniFrontend(model_id=model_id)

    def _frontend(payload: StagePayload) -> StagePayload:
        return frontend(payload)

    return FrontendExecutor(_frontend)


def create_aggregate_executor() -> FrontendExecutor:
    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return FrontendExecutor(_identity)


def _create_encoder_executor(
    *,
    stage_name: str,
    model: torch.nn.Module,
    device: str,
) -> EngineExecutor:
    def _request_builder(payload: StagePayload) -> EncoderRequestData:
        data = _ensure_data(payload)
        encoder_inputs = data.get("encoder_inputs")
        if not isinstance(encoder_inputs, dict):
            return EncoderRequestData(input_dict={"_skip": True, "_result": {}})
        inputs = encoder_inputs.get(stage_name)
        if not isinstance(inputs, dict) or not inputs:
            return EncoderRequestData(input_dict={"_skip": True, "_result": {}})
        if inputs.get("_skip"):
            skip_result = inputs.get("_result")
            return EncoderRequestData(
                input_dict=inputs,
                output_dict=skip_result if isinstance(skip_result, dict) else {},
            )
        cache_key = inputs.get("cache_key")
        return EncoderRequestData(
            input_dict=inputs,
            cache_key=str(cache_key) if cache_key is not None else None,
        )

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        data = _ensure_data(payload)
        encoder_outs = data.setdefault("encoder_outs", {})
        engine_outputs = data.setdefault("engine_outputs", {})
        if isinstance(result, EncoderRequestData):
            if result.output_dict is not None:
                encoder_out = result.output_dict
            elif result.embeddings is not None:
                encoder_out = result.embeddings
            else:
                encoder_out = {}
        else:
            encoder_out = result if isinstance(result, dict) else {"result": result}
        encoder_outs[stage_name] = encoder_out
        engine_outputs[stage_name] = encoder_out
        return payload

    engine = create_encoder_engine(model, device=device)
    return EngineExecutor(
        engine=engine, request_builder=_request_builder, result_builder=_result_builder
    )


def create_image_encoder_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniImageEncoder(model_id=model_id, device=device, dtype=dtype)
    return _create_encoder_executor(stage_name=IMAGE_STAGE, model=model, device=device)


def create_audio_encoder_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniAudioEncoder(model_id=model_id, device=device, dtype=dtype)
    return _create_encoder_executor(stage_name=AUDIO_STAGE, model=model, device=device)


def create_thinker_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    max_seq_len: int = 8192,
) -> EngineExecutor:
    model = Qwen3OmniSplitThinker(model_id=model_id, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    step_counters: dict[str, int] = {}

    def _request_builder(payload: StagePayload) -> ARRequestData:
        data = _ensure_data(payload)
        prompt = data.get("prompt")
        if not isinstance(prompt, dict):
            raise TypeError("prompt missing for thinker request")

        input_ids = prompt.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("prompt.input_ids must be a torch.Tensor")

        attention_mask = prompt.get("attention_mask")
        thinker_inputs = data.get("thinker_inputs")
        if not isinstance(thinker_inputs, dict):
            thinker_inputs = data.get("engine_inputs", {}).get(THINKER_STAGE, {})
        if not isinstance(thinker_inputs, dict):
            thinker_inputs = {}

        model_inputs = dict(thinker_inputs.get("model_inputs", {}))
        if not model_inputs:
            model_inputs = {
                k: v
                for k, v in thinker_inputs.items()
                if k != "capture_model_output_keys"
            }
        capture_keys = thinker_inputs.get("capture_model_output_keys", ())
        if "attention_mask" in model_inputs:
            model_inputs.pop("attention_mask", None)

        step_counters.pop(payload.request_id, None)

        return ARRequestData(
            input_ids=input_ids.to(dtype=torch.long),
            attention_mask=(
                attention_mask if isinstance(attention_mask, torch.Tensor) else None
            ),
            model_inputs=model_inputs,
            capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
            max_new_tokens=payload.request.params.get("max_new_tokens"),
            temperature=payload.request.params.get("temperature", 0.0),
        )

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        data = _ensure_data(payload)
        thinker_out: ThinkerOutput
        if isinstance(result, ARRequestData):
            output_ids = list(result.output_ids)
            thinker_out = {
                "output_ids": output_ids,
                "step": len(output_ids),
                "is_final": True,
                "extra_model_outputs": dict(result.extra_model_outputs),
            }
        else:
            thinker_out = {
                "output_ids": [],
                "step": 0,
                "is_final": True,
                "extra_model_outputs": {"result": result},
            }

        data["thinker_out"] = thinker_out
        engine_outputs = data.setdefault("engine_outputs", {})
        engine_outputs[THINKER_STAGE] = thinker_out
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

        data = _ensure_data(payload)
        thinker_out: ThinkerOutput = {
            "output_ids": [token_id],
            "step": step,
            "is_final": False,
        }
        events = list(
            decode_events(
                thinker_out=thinker_out,
                data=data,
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
            "events": [_event_to_dict(event) for event in events],
            "token_id": token_id,
            "step": step,
            "stage": THINKER_STAGE,
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


def create_decode_executor(model_id: str) -> FrontendExecutor:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def _decode(payload: StagePayload) -> StagePayload:
        data = _ensure_data(payload)
        thinker_out = data.get("thinker_out")
        if not isinstance(thinker_out, dict):
            thinker_out = data.get("engine_outputs", {}).get(THINKER_STAGE, {})
        if not isinstance(thinker_out, dict):
            thinker_out = {
                "output_ids": [],
                "step": 0,
                "is_final": True,
                "extra_model_outputs": {},
            }

        step = int(thinker_out.get("step") or len(thinker_out.get("output_ids", [])))
        events = list(
            decode_events(
                thinker_out=thinker_out,
                data=data,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                step=step,
            )
        )
        event_dicts = [_event_to_dict(event) for event in events]

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

    return FrontendExecutor(_decode)
