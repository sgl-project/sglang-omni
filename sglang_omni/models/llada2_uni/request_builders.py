# SPDX-License-Identifier: Apache-2.0
"""Request/result builders for LLaDA2-Uni pipeline stages."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.llada2_uni.components.preprocessor import DUMMY_IMAGE_TOKEN_ID
from sglang_omni.models.llada2_uni.payload_types import PipelineState, ThinkerOutput
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangDLLMRequestData


def build_encoder_request(
    state: PipelineState,
    *,
    stage_name: str,
) -> dict[str, Any]:
    """Build encoder request dict from pipeline state."""
    inputs = state.encoder_inputs.get(stage_name)
    if not isinstance(inputs, dict) or not inputs:
        return {"_skip": True, "_result": {}}
    if inputs.get("_skip"):
        return {"_skip": True, "_result": inputs.get("_result", {})}
    return dict(inputs)


def apply_encoder_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> None:
    """Apply encoder result to pipeline state."""
    encoder_out = result
    state.encoder_outs[stage_name] = encoder_out
    state.engine_outputs[stage_name] = encoder_out


def merge_image_tokens_for_thinker(state: PipelineState) -> None:
    """Merge VQ token IDs from image encoder output into prompt input_ids.

    Replaces DUMMY_IMAGE_TOKEN_ID placeholders with actual VQ token IDs
    offset by image_token_offset.
    """
    image_out = state.encoder_outs.get("image_encoder")
    if not image_out:
        return

    image_token_ids_list = image_out.get("image_token_ids")
    if not image_token_ids_list:
        return

    prompt = state.prompt
    if not isinstance(prompt, dict) or "input_ids" not in prompt:
        return

    input_ids = prompt["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.flatten().tolist()

    image_token_offset = image_out.get("image_token_offset", 157184)

    all_vq_tokens = []
    for token_ids in image_token_ids_list:
        all_vq_tokens.extend(tid + image_token_offset for tid in token_ids)

    if not all_vq_tokens:
        return

    new_ids = []
    vq_idx = 0
    for tid in input_ids:
        if tid == DUMMY_IMAGE_TOKEN_ID and vq_idx < len(all_vq_tokens):
            new_ids.append(all_vq_tokens[vq_idx])
            vq_idx += 1
        else:
            new_ids.append(tid)

    if vq_idx < len(all_vq_tokens):
        new_ids.extend(all_vq_tokens[vq_idx:])

    prompt["input_ids"] = torch.tensor([new_ids], dtype=torch.long)


def build_dllm_thinker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
    tokenizer: Any,
    vocab_size: int,
    dllm_config: Any,
    request_id: str | None = None,
) -> SGLangDLLMRequestData:
    """Build SGLangDLLMRequestData for the LLaDA2-Uni thinker."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    input_ids_list = input_ids.to(dtype=torch.long).flatten().tolist()

    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in thinker_inputs.items() if k != "capture_model_output_keys"
        }

    max_new_tokens = params.get("max_new_tokens", 2048)
    temperature = params.get("temperature", 0.0)

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
        dllm_config=dllm_config,
    )

    req.omni_model_inputs = model_inputs if model_inputs else None
    req._omni_consumed = None

    data = SGLangDLLMRequestData(
        input_ids=input_ids.to(dtype=torch.long).flatten(),
        model_inputs=model_inputs,
        output_ids=req.output_ids,
        req=req,
    )
    return data


def apply_dllm_thinker_result(
    state: PipelineState,
    *,
    stage_name: str,
    output_ids: list[int],
    finish_reason: str | None = None,
) -> ThinkerOutput:
    """Apply DLLM thinker result to pipeline state."""
    thinker_out: ThinkerOutput = {
        "output_ids": output_ids,
        "is_final": True,
    }
    if finish_reason is not None:
        thinker_out["finish_reason"] = finish_reason

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out


def make_dllm_thinker_scheduler_adapters(
    *,
    tokenizer: Any,
    vocab_size: int,
    dllm_config: Any,
    stage_name: str = "thinker",
):
    """Build StagePayload <-> scheduler adapters for the dLLM thinker."""

    def request_builder(payload: StagePayload) -> SGLangDLLMRequestData:
        state = PipelineState.from_dict(payload.data)
        data = build_dllm_thinker_request(
            state,
            params=payload.request.params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            dllm_config=dllm_config,
            request_id=payload.request_id,
        )
        data.stage_payload = payload
        return data

    def result_adapter(data: SGLangDLLMRequestData) -> StagePayload:
        payload = data.stage_payload
        state = PipelineState.from_dict(payload.data)
        apply_dllm_thinker_result(
            state,
            stage_name=stage_name,
            output_ids=data.output_ids,
            finish_reason=data.finish_reason,
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter
