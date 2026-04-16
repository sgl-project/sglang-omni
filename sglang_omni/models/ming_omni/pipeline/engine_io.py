# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers for Ming-Omni stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.models.ming_omni.io import PipelineState, ThinkerOutput

if TYPE_CHECKING:
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData


def build_encoder_request(
    state: PipelineState, *, stage_name: str
) -> EncoderRequestData:
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


def build_thinker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
) -> ARRequestData:
    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in thinker_inputs.items() if k != "capture_model_output_keys"
        }

    capture_keys = thinker_inputs.get("capture_model_output_keys", ())
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


def build_sglang_thinker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
    tokenizer: Any,
    vocab_size: int,
    request_id: str | None = None,
) -> "SGLangARRequestData":
    """Build SGLangARRequestData for the Ming thinker.

    Ming uses standard 1D RoPE (no M-RoPE), so no special position computation needed.
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData

    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    input_ids_list = input_ids.to(dtype=torch.long).flatten().tolist()

    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in thinker_inputs.items() if k != "capture_model_output_keys"
        }
    capture_keys = list(thinker_inputs.get("capture_model_output_keys", ()))
    model_inputs.pop("attention_mask", None)

    max_new_tokens = params.get("max_new_tokens", 2048)
    temperature = params.get("temperature", 0.0)

    # --- Image generation: prefill-only mode + query token injection ---
    image_gen = (state.mm_inputs or {}).get("image_gen", {})
    if image_gen.get("prefill_only"):
        # Prefill-only: no autoregressive decoding, just run the forward pass
        max_new_tokens = 0

        # Convert serialised query_tokens list back to a tensor and inject
        # as image_embeds so that _inject_multimodal_embeds() replaces
        # image_patch_token positions with query embeddings.
        query_tokens_list = image_gen.get("query_tokens")
        if query_tokens_list is not None:
            query_tokens = torch.tensor(query_tokens_list, dtype=torch.bfloat16)
            model_inputs["image_embeds"] = query_tokens

        # Ensure hidden states are captured for downstream image generation
        if "hidden_states" not in capture_keys:
            capture_keys.append("hidden_states")

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token_ids = {eos_token_id} if eos_token_id is not None else None

    rid = request_id or "req-0"
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
        eos_token_ids=eos_token_ids,
    )

    # Attach multimodal model inputs (audio embeddings, placeholder locations)
    req.omni_model_inputs = model_inputs if model_inputs else None
    req._omni_consumed = None

    data = SGLangARRequestData(
        input_ids=input_ids.to(dtype=torch.long).flatten(),
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


def apply_thinker_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> ThinkerOutput:
    if isinstance(result, ARRequestData):
        output_ids = list(result.output_ids)
        thinker_out: ThinkerOutput = {
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

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out
