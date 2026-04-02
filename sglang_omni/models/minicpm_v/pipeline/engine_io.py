# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers for MiniCPM-V stages.

This module handles the translation between PipelineState and engine-specific
request/response formats for both encoder and LLM stages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.models.minicpm_v.io import LLMOutput, PipelineState

if TYPE_CHECKING:
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData


def build_encoder_request(
    state: PipelineState, *, stage_name: str
) -> EncoderRequestData:
    """Build encoder request from pipeline state.

    Handles cache key extraction for encoder result caching.
    """
    inputs = state.encoder_inputs.get(stage_name)
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


def apply_encoder_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> None:
    """Apply encoder result to pipeline state."""
    if isinstance(result, EncoderRequestData):
        if result.output_dict is not None:
            encoder_out = result.output_dict
        elif result.embeddings is not None:
            encoder_out = result.embeddings
        else:
            encoder_out = {}
    else:
        encoder_out = result if isinstance(result, dict) else {"result": result}

    state.encoder_outs[stage_name] = encoder_out
    state.engine_outputs[stage_name] = encoder_out


def build_llm_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
) -> ARRequestData:
    """Build HF AR Engine request from pipeline state.

    This is used for Phase 1 with HuggingFace generate() backend.
    """
    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for LLM request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    attention_mask = prompt.get("attention_mask")
    llm_inputs = state.llm_inputs or {}

    model_inputs = dict(llm_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in llm_inputs.items() if k != "capture_model_output_keys"
        }

    capture_keys = llm_inputs.get("capture_model_output_keys", ())
    if "attention_mask" in model_inputs:
        model_inputs.pop("attention_mask", None)

    return ARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=params.get("max_new_tokens"),
        temperature=params.get("temperature", 0.0),
    )


def build_sglang_llm_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
    tokenizer: Any,
    vocab_size: int,
    request_id: str | None = None,
    llm_config: Any = None,
) -> "SGLangARRequestData":
    """Build SGLangARRequestData from pipeline state.

    Constructs a SGLang Req with normalized SamplingParams, then wraps it
    in SGLangARRequestData (which inherits ARRequestData).

    For MiniCPM-V, we don't need M-RoPE computation (that's Qwen3-Omni specific).
    The 2D position encoding is handled internally by the LLM using tgt_sizes.
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData

    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for LLM request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    input_ids_list = input_ids.to(dtype=torch.long).tolist()

    attention_mask = prompt.get("attention_mask")
    llm_inputs = state.llm_inputs or {}

    model_inputs = dict(llm_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in llm_inputs.items() if k != "capture_model_output_keys"
        }
    capture_keys = llm_inputs.get("capture_model_output_keys", ())
    if "attention_mask" in model_inputs:
        model_inputs.pop("attention_mask", None)

    max_new_tokens = params.get("max_new_tokens", 2048)
    temperature = params.get("temperature", 0.0)

    # Build SGLang SamplingParams and normalize
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    # Build SGLang Req
    rid = request_id or "req-0"
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )

    # Attach model_inputs to Req for image embedding merge in SGLangModelRunner.
    # MiniCPM-V doesn't use M-RoPE, so no multimodal_inputs setup needed here.
    req.omni_model_inputs = model_inputs if model_inputs else None
    req._omni_consumed = None

    # Build SGLangARRequestData — output_ids points to req.output_ids
    data = SGLangARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
    )
    return data


def apply_llm_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> LLMOutput:
    """Apply LLM result to pipeline state."""
    if isinstance(result, ARRequestData):
        output_ids = list(result.output_ids)
        llm_out: LLMOutput = {
            "output_ids": output_ids,
            "step": len(output_ids),
            "is_final": True,
            "extra_model_outputs": dict(result.extra_model_outputs),
        }
    else:
        llm_out = {
            "output_ids": [],
            "step": 0,
            "is_final": True,
            "extra_model_outputs": {"result": result},
        }

    state.llm_out = llm_out
    state.engine_outputs[stage_name] = llm_out
    return llm_out
