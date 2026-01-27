# SPDX-License-Identifier: Apache-2.0
"""Generic glue for omni adapters and staged pipelines."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.engines.async_module import AsyncModuleEngine
from sglang_omni.engines.omni import create_ar_engine
from sglang_omni.engines.omni.runtime import ARRequestData
from sglang_omni.executors import EngineExecutor, FrontendExecutor
from sglang_omni.models.adapter_registry import get_adapter, get_adapter_from_payload
from sglang_omni.models.omni_adapter import FrontendOutput, OmniEvent, ThinkerOutput
from sglang_omni.proto import OmniRequest, StagePayload

AGGREGATE_STAGE_NAME = "mm_aggregate"
THINKER_STAGE_NAME = "thinker"
DECODE_STAGE_NAME = "decode"


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


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    return value


def _update_compat_keys(data: dict[str, Any], frontend_out: FrontendOutput) -> None:
    prompt = frontend_out.get("prompt")
    if isinstance(prompt, dict):
        data["prompt"] = prompt
    mm_inputs = frontend_out.get("mm_inputs")
    if isinstance(mm_inputs, dict):
        data["mm_inputs"] = mm_inputs
    adapter_state = frontend_out.get("adapter_state")
    if isinstance(adapter_state, dict):
        data["adapter_state"] = adapter_state


def create_adapter_frontend_executor(adapter_name: str) -> FrontendExecutor:
    adapter = get_adapter(adapter_name)

    def _frontend(payload: StagePayload) -> StagePayload:
        data = _ensure_data(payload)
        frontend_out = adapter.build_frontend(
            request_inputs=payload.request.inputs,
            request_params=dict(payload.request.params),
        )
        encoder_inputs = adapter.build_encoder_inputs(frontend_out=frontend_out)

        data["adapter_name"] = adapter_name
        data["frontend_out"] = frontend_out
        data["encoder_inputs"] = encoder_inputs
        _update_compat_keys(data, frontend_out)
        return payload

    return FrontendExecutor(_frontend)


def create_adapter_encoder_executor(
    adapter_name: str,
    *,
    stage_name: str,
    model: torch.nn.Module,
) -> EngineExecutor:
    adapter = get_adapter(adapter_name)
    del adapter  # Adapter presence is validated during construction.

    def _request_builder(payload: StagePayload) -> dict[str, Any]:
        data = _ensure_data(payload)
        encoder_inputs = data.get("encoder_inputs")
        if not isinstance(encoder_inputs, dict):
            return {"_skip": True, "_result": {}}
        inputs = encoder_inputs.get(stage_name)
        if not isinstance(inputs, dict) or not inputs:
            return {"_skip": True, "_result": {}}
        if inputs.get("_skip"):
            return inputs
        return inputs

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        data = _ensure_data(payload)
        encoder_outs = data.setdefault("encoder_outs", {})
        engine_outputs = data.setdefault("engine_outputs", {})
        encoder_out = _to_cpu(
            result if isinstance(result, dict) else {"result": result}
        )
        encoder_outs[stage_name] = encoder_out
        engine_outputs[stage_name] = encoder_out
        return payload

    engine = AsyncModuleEngine(model)
    return EngineExecutor(
        engine=engine, request_builder=_request_builder, result_builder=_result_builder
    )


def merge_for_adapter(payloads: dict[str, StagePayload]) -> StagePayload:
    base = payloads.get("frontend") or next(iter(payloads.values()))
    data = _ensure_data(base)

    adapter = get_adapter_from_payload(base)
    frontend_out = data.get("frontend_out")
    if not isinstance(frontend_out, dict):
        raise TypeError("frontend_out missing for aggregated merge")

    encoder_outs: dict[str, Any] = {}
    existing = data.get("encoder_outs")
    if isinstance(existing, dict):
        encoder_outs.update(existing)

    for stage_name, payload in payloads.items():
        stage_data = payload.data if isinstance(payload.data, dict) else {}
        stage_encoder_outs = stage_data.get("encoder_outs")
        if isinstance(stage_encoder_outs, dict) and stage_name in stage_encoder_outs:
            encoder_outs[stage_name] = stage_encoder_outs[stage_name]
            continue
        stage_engine_outputs = stage_data.get("engine_outputs")
        if (
            isinstance(stage_engine_outputs, dict)
            and stage_name in stage_engine_outputs
        ):
            encoder_outs[stage_name] = stage_engine_outputs[stage_name]

    thinker_inputs = adapter.merge_for_thinker(
        frontend_out=frontend_out, encoder_outs=encoder_outs
    )

    data["encoder_outs"] = encoder_outs
    data["thinker_inputs"] = thinker_inputs
    engine_inputs = data.setdefault("engine_inputs", {})
    engine_inputs[THINKER_STAGE_NAME] = thinker_inputs
    # Encoder inputs can be very large (e.g., pixel/audio tensors); drop them.
    data["encoder_inputs"] = {}

    prune_fn = getattr(adapter, "prune_frontend_for_thinker", None)
    if callable(prune_fn):
        pruned = prune_fn(frontend_out=frontend_out, encoder_outs=encoder_outs)
        if isinstance(pruned, dict):
            frontend_out = pruned
            data["frontend_out"] = frontend_out
            _update_compat_keys(data, frontend_out)

    return base


def create_adapter_aggregate_executor(adapter_name: str) -> FrontendExecutor:
    adapter = get_adapter(adapter_name)
    del adapter

    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return FrontendExecutor(_identity)


def create_adapter_thinker_executor(
    adapter_name: str,
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    max_seq_len: int,
    device: str,
) -> EngineExecutor:
    adapter = get_adapter(adapter_name)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    step_counters: dict[str, int] = {}

    def _request_builder(payload: StagePayload) -> ARRequestData:
        data = _ensure_data(payload)
        frontend_out = data.get("frontend_out")
        if not isinstance(frontend_out, dict):
            raise TypeError("frontend_out missing for thinker request")
        prompt = frontend_out.get("prompt")
        if not isinstance(prompt, dict):
            raise TypeError("frontend_out.prompt missing for thinker request")

        input_ids = prompt.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("prompt.input_ids must be a torch.Tensor")

        attention_mask = prompt.get("attention_mask")
        thinker_inputs = data.get("thinker_inputs")
        if not isinstance(thinker_inputs, dict):
            thinker_inputs = data.get("engine_inputs", {}).get(THINKER_STAGE_NAME, {})
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
        engine_outputs[THINKER_STAGE_NAME] = thinker_out
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
        frontend_out = data.get("frontend_out")
        if not isinstance(frontend_out, dict):
            return {"token_id": token_id, "step": step}

        thinker_out: ThinkerOutput = {
            "output_ids": [token_id],
            "step": step,
            "is_final": False,
        }
        events = list(
            adapter.decode_events(
                thinker_out=thinker_out, frontend_out=frontend_out, step=step
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
            "stage": THINKER_STAGE_NAME,
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


def create_adapter_decode_executor(adapter_name: str) -> FrontendExecutor:
    adapter = get_adapter(adapter_name)

    def _decode(payload: StagePayload) -> StagePayload:
        data = _ensure_data(payload)
        frontend_out = data.get("frontend_out")
        if not isinstance(frontend_out, dict):
            raise TypeError("frontend_out missing for decode")

        thinker_out = data.get("thinker_out")
        if not isinstance(thinker_out, dict):
            thinker_out = data.get("engine_outputs", {}).get(THINKER_STAGE_NAME, {})
        if not isinstance(thinker_out, dict):
            thinker_out = {
                "output_ids": [],
                "step": 0,
                "is_final": True,
                "extra_model_outputs": {},
            }

        step = int(thinker_out.get("step") or len(thinker_out.get("output_ids", [])))
        events = list(
            adapter.decode_events(
                thinker_out=thinker_out, frontend_out=frontend_out, step=step
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
            tokenizer = getattr(adapter, "tokenizer", None)
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


def frontend_next(request_id: str, output: Any) -> list[str]:
    del request_id
    if not isinstance(output, StagePayload):
        return [AGGREGATE_STAGE_NAME]
    data = output.data if isinstance(output.data, dict) else {}
    encoder_inputs = data.get("encoder_inputs")
    if not isinstance(encoder_inputs, dict):
        return [AGGREGATE_STAGE_NAME]
    stages = [stage for stage in encoder_inputs.keys() if stage != AGGREGATE_STAGE_NAME]
    stages = sorted(stages)
    stages.append(AGGREGATE_STAGE_NAME)
    return stages


def encoder_next(request_id: str, output: Any) -> str:
    del request_id, output
    return AGGREGATE_STAGE_NAME


def aggregate_next(request_id: str, output: Any) -> str:
    del request_id, output
    return THINKER_STAGE_NAME


def thinker_next(request_id: str, output: Any) -> str:
    del request_id, output
    return DECODE_STAGE_NAME


def decode_next(request_id: str, output: Any) -> None:
    del request_id, output
    return None


def build_frontend_payload(
    request_id: str, request: OmniRequest, *, adapter_name: str
) -> StagePayload:
    """Helper for tests and debugging; not used by the pipeline directly."""

    payload = StagePayload(
        request_id=request_id, request=request, data={"adapter_name": adapter_name}
    )
    return payload
