# SPDX-License-Identifier: Apache-2.0
"""Helpers to build engine request data from StagePayloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.proto import StagePayload

if TYPE_CHECKING:
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData


def _as_long_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.long)
    return torch.as_tensor(value, dtype=torch.long)


def _select_engine_inputs(payload: StagePayload) -> dict[str, Any] | None:
    data = payload.data if isinstance(payload.data, dict) else None
    if not data:
        return None

    engine_inputs = data.get("engine_inputs")
    if not isinstance(engine_inputs, dict):
        return engine_inputs if isinstance(engine_inputs, dict) else None

    engine_key = payload.request.metadata.get(
        "engine_input_key"
    ) or payload.request.params.get("engine_input_key")
    if engine_key and engine_key in engine_inputs:
        selected = engine_inputs.get(engine_key)
        return selected if isinstance(selected, dict) else None

    if len(engine_inputs) == 1:
        only_value = next(iter(engine_inputs.values()))
        return only_value if isinstance(only_value, dict) else None

    return None


def _extract_input_ids(payload: StagePayload) -> tuple[Any, dict[str, Any]]:
    params = payload.request.params
    data = payload.data
    engine_inputs = _select_engine_inputs(payload)
    if engine_inputs and "input_ids" in engine_inputs:
        return engine_inputs["input_ids"], params
    if isinstance(data, dict):
        input_ids = data.get("input_ids")
        if input_ids is None:
            input_ids = data.get("raw_inputs", data)
    else:
        input_ids = data

    return input_ids, params


def build_encoder_request(payload: StagePayload) -> EncoderRequestData:
    """Build EncoderRequestData from StagePayload."""
    input_ids, _ = _extract_input_ids(payload)
    return EncoderRequestData(input_ids=input_ids)


def build_ar_request(payload: StagePayload) -> ARRequestData:
    """Build ARRequestData from StagePayload."""
    input_ids, params = _extract_input_ids(payload)
    engine_inputs = _select_engine_inputs(payload) or {}
    model_inputs = engine_inputs.get("model_inputs", {})
    capture_keys = engine_inputs.get("capture_model_output_keys", ())
    return ARRequestData(
        input_ids=_as_long_tensor(input_ids),
        model_inputs=model_inputs if isinstance(model_inputs, dict) else {},
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=params.get("max_new_tokens"),
        temperature=params.get("temperature", 0.0),
    )


def build_sglang_ar_request(
    payload: StagePayload,
    *,
    tokenizer: Any = None,
    vocab_size: int = 32000,
) -> "SGLangARRequestData":
    """Build SGLangARRequestData from StagePayload.

    Creates a SGLang Req with normalized SamplingParams, wrapped in
    SGLangARRequestData for use with the SGLang backend engine.
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData

    input_ids, params = _extract_input_ids(payload)
    input_ids_tensor = _as_long_tensor(input_ids)
    input_ids_list = input_ids_tensor.tolist()

    engine_inputs = _select_engine_inputs(payload) or {}
    model_inputs = engine_inputs.get("model_inputs", {})
    capture_keys = engine_inputs.get("capture_model_output_keys", ())

    max_new_tokens = params.get("max_new_tokens", 2048)
    temperature = params.get("temperature", 0.0)

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )

    return SGLangARRequestData(
        input_ids=input_ids_tensor,
        model_inputs=model_inputs if isinstance(model_inputs, dict) else {},
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
    )
