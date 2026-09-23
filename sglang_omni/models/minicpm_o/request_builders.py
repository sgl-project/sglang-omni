# SPDX-License-Identifier: Apache-2.0
"""Encoder and thinker request adapters for MiniCPM-o."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import xxhash

from sglang_omni.models.minicpm_o.payload_types import (
    MiniCPMOPipelineState,
    ThinkerOutput,
)
from sglang_omni.models.minicpm_o.routing import (
    DECODE_STAGE,
    THINKER_STAGE,
    payload_with_state,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.types import RequestOutput

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


def resolve_sampling_seed(params: dict[str, Any]) -> int | None:
    for key in ("seed", "sampling_seed"):
        value = params.get(key)
        if value is not None:
            return int(value)
    return None


@dataclass(kw_only=True)
class EncoderRequestData:
    """Prepared inputs for one encoder stage forward."""

    model_inputs: dict[str, Any]
    cache_key: str | None = None
    skip_result: dict[str, Any] | None = None


def build_encoder_request(
    state: MiniCPMOPipelineState, *, stage_name: str
) -> EncoderRequestData:
    inputs = state.encoder_inputs.get(stage_name)
    if not isinstance(inputs, dict) or not inputs:
        return EncoderRequestData(model_inputs={}, skip_result={})
    cache_key = inputs.get("cache_key")
    model_inputs = {
        k: v for k, v in inputs.items() if k not in ("cache_key", "_active")
    }
    return EncoderRequestData(
        model_inputs=model_inputs,
        cache_key=str(cache_key) if cache_key is not None else None,
    )


def apply_mm_pad_values(
    input_ids: torch.Tensor,
    *,
    mm_inputs: dict[str, Any],
    model_inputs: dict[str, Any],
    vocab_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """Replace shared placeholder ids with cache-key-derived modality ids."""
    pad_values: dict[str, int] = {}
    empty = torch.empty(0, dtype=torch.long)
    # note (MayDomine): the shared runner reads positions for all three modalities.
    mm_positions: dict[str, torch.Tensor] = {
        "image": empty,
        "video": empty,
        "audio": empty,
    }
    has_any = False
    input_ids = input_ids.clone()
    for modality in ("image", "audio"):
        info = mm_inputs.get(modality)
        if not isinstance(info, dict):
            continue
        bounds = info.get("bounds")
        if bounds is None or bounds.numel() == 0:
            continue
        positions = torch.cat(
            [torch.arange(int(start), int(end)) for start, end in bounds]
        )
        has_any = True
        cache_key = str(info.get("cache_key") or modality)
        h = xxhash.xxh3_64(cache_key.encode()).intdigest()
        pad_val = vocab_size + h % (1 << 62)
        pad_values[modality] = pad_val
        input_ids[positions] = pad_val
        mm_positions[modality] = positions
    if not has_any:
        return input_ids, None
    model_inputs["pad_values"] = pad_values
    return input_ids, mm_positions


def build_sglang_thinker_request(
    state: MiniCPMOPipelineState,
    *,
    params: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    request_id: str | None = None,
) -> SGLangARRequestData:
    """Build a thinker request with sampling parameters and media embeddings."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

    prompt = state.prompt
    input_ids = prompt["input_ids"]
    attention_mask = prompt.get("attention_mask")

    thinker_inputs = state.thinker_inputs or {}
    model_inputs = thinker_inputs.get("model_inputs")
    if model_inputs is None:
        model_inputs = {}
    elif not isinstance(model_inputs, dict):
        raise TypeError("MiniCPM-o thinker model_inputs must be a dict when provided")
    else:
        model_inputs = dict(model_inputs)

    max_new_tokens = params.get("max_new_tokens", 2048)
    temperature = params.get("temperature", 0.0)

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=params.get("top_p", 1.0),
        top_k=params.get("top_k", -1),
        min_p=params.get("min_p", 0.0),
        repetition_penalty=params.get("repetition_penalty", 1.0),
        stop=params.get("stop") or [],
        stop_token_ids=params.get("stop_token_ids") or [],
        sampling_seed=resolve_sampling_seed(params),
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    input_ids = input_ids.to(dtype=torch.long)
    mm_positions = None
    if model_inputs:
        input_ids, mm_positions = apply_mm_pad_values(
            input_ids,
            mm_inputs=state.mm_inputs,
            model_inputs=model_inputs,
            vocab_size=vocab_size,
        )
    req = Req(
        rid=request_id or "req-0",
        origin_input_text="",
        origin_input_ids=input_ids.tolist(),
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )
    req.tokenizer = tokenizer

    req.omni_model_inputs = model_inputs if model_inputs else None
    req._omni_consumed = None
    req._codec_suppress_tokens = None
    req._omni_mm_positions = mm_positions

    data = SGLangARRequestData(
        input_ids=input_ids,
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
    )
    data.return_logprob = bool(params.get("return_logprob"))
    return data


def apply_thinker_result(
    state: MiniCPMOPipelineState,
    *,
    stage_name: str,
    result: SGLangARRequestData,
) -> ThinkerOutput:
    output_ids = list(result.output_ids)
    thinker_out: ThinkerOutput = {
        "output_ids": output_ids,
        "step": len(output_ids),
        "is_final": True,
        "extra_model_outputs": dict(result.extra_model_outputs),
    }

    if result.finish_reason is not None:
        thinker_out["finish_reason"] = result.finish_reason
    if result.weight_version is not None:
        thinker_out["weight_version"] = result.weight_version
    if result.output_token_logprobs is not None:
        thinker_out["output_token_logprobs"] = result.output_token_logprobs

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out


def make_thinker_scheduler_adapters(
    *,
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    stage_name: str = THINKER_STAGE,
) -> tuple[
    Callable[[StagePayload], SGLangARRequestData],
    Callable[[SGLangARRequestData], StagePayload],
]:
    """Build model-specific StagePayload <-> scheduler adapters for thinker."""

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        state = MiniCPMOPipelineState.from_dict(payload.data)
        req_data = build_sglang_thinker_request(
            state,
            params=payload.request.params or {},
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            request_id=payload.request_id,
        )
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = data.stage_payload
        state = MiniCPMOPipelineState.from_dict(payload.data)
        apply_thinker_result(state, stage_name=stage_name, result=data)
        return payload_with_state(payload, state)

    return request_builder, result_adapter


def build_thinker_stream_output(
    request_id: str, req_data: SGLangARRequestData, req_output: RequestOutput
) -> list[OutgoingMessage]:
    """Emit one token for streaming requests after a complete prefill or decode."""
    if req_data.req.inflight_middle_chunks > 0 or req_output.data is None:
        return []
    if not req_data.stage_payload.request.params.get("stream", False):
        return []

    token_id = int(req_output.data)
    # note (MayDomine): stream transport accepts tensors, not scalar ids.
    return [
        OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=torch.tensor([token_id], dtype=torch.long),
            target=DECODE_STAGE,
            metadata={"token_id": token_id},
        )
    ]
