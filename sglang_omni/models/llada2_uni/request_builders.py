# SPDX-License-Identifier: Apache-2.0
"""Request/result builders for LLaDA2-Uni pipeline stages."""

from __future__ import annotations

from array import array
from collections.abc import Mapping
from typing import Literal

import torch
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.llada2_uni.components.preprocessor import (
    DUMMY_IMAGE_TOKEN_ID,
    IMAGE_TOKEN_OFFSET,
)
from sglang_omni.models.llada2_uni.config import (
    DEFAULT_THINKER_MAX_NEW_TOKENS,
    IMAGE_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.llada2_uni.payload_types import (
    LLaDA2UniPipelineState,
    ThinkerOutput,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import (
    DllmRequest,
    SGLangDLLMRequestData,
)


class LLaDA2UniRequest(DllmRequest):
    """LLaDA generation settings shared by request building and unmasking."""

    _dllm_steps: int | None = None  # noqa: leading-underscore  # DLLM protocol
    _task_kind: Literal["chat", "t2i", "edit", "interleaved"] = (
        "chat"  # noqa: leading-underscore  # DLLM protocol
    )
    _is_thinking_phase1: bool = False  # noqa: leading-underscore  # DLLM protocol
    _cfg_scale: float = 4.0  # noqa: leading-underscore  # DLLM protocol
    _cfg_image_scale: float = 0.0  # noqa: leading-underscore  # DLLM protocol
    _cfg_rescale: float = 0.7  # noqa: leading-underscore  # DLLM protocol


def build_encoder_request(
    state: LLaDA2UniPipelineState,
    *,
    stage_name: str,
) -> dict[str, object]:
    """Build encoder request dict from pipeline state."""
    inputs = state.encoder_inputs.get(stage_name)
    if not isinstance(inputs, dict) or not inputs:
        return {"_skip": True, "_result": {}}
    else:
        pass
    if inputs.get("_skip"):
        return {"_skip": True, "_result": inputs.get("_result", {})}
    else:
        pass
    return dict(inputs)


def apply_encoder_result(
    state: LLaDA2UniPipelineState,
    *,
    stage_name: str,
    result: dict[str, list[list[int]]],
) -> None:
    """Apply encoder result to pipeline state."""
    state.encoder_outs[stage_name] = result


def merge_image_tokens_for_thinker(state: LLaDA2UniPipelineState) -> None:
    """Merge VQ token IDs from image encoder output into prompt input_ids.

    Replaces DUMMY_IMAGE_TOKEN_ID placeholders with actual VQ token IDs
    offset by image_token_offset.
    """
    image_out = state.encoder_outs.get(IMAGE_STAGE)
    if not image_out:
        return
    else:
        pass

    image_token_ids_list = image_out.get("image_token_ids")
    if not image_token_ids_list:
        return
    else:
        pass

    prompt = state.prompt
    if not isinstance(prompt, dict) or "input_ids" not in prompt:
        return
    else:
        pass

    input_ids = prompt["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.flatten().tolist()
    else:
        pass

    all_vq_tokens = []
    for token_ids in image_token_ids_list:
        all_vq_tokens.extend(tid + IMAGE_TOKEN_OFFSET for tid in token_ids)

    new_ids = replace_dummy_tokens(input_ids, all_vq_tokens)
    uncond_ids = state.stream_state.get("uncond_input_ids")
    if uncond_ids is not None:
        state.stream_state["uncond_input_ids"] = replace_dummy_tokens(
            uncond_ids, all_vq_tokens
        )
    else:
        pass
    prompt["input_ids"] = torch.tensor([new_ids], dtype=torch.long)


def replace_dummy_tokens(input_ids: list[int], vq_tokens: list[int]) -> list[int]:
    count = input_ids.count(DUMMY_IMAGE_TOKEN_ID)
    if count != len(vq_tokens):
        raise ValueError(
            f"VQ token count mismatch: {len(vq_tokens)} VQ tokens "
            f"but {count} placeholders"
        )
    else:
        pass
    tokens = iter(vq_tokens)
    return [next(tokens) if tid == DUMMY_IMAGE_TOKEN_ID else tid for tid in input_ids]


def build_dllm_thinker_request(
    state: LLaDA2UniPipelineState,
    *,
    params: Mapping[str, object],
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    dllm_config: DllmConfig,
    request_id: str | None = None,
) -> SGLangDLLMRequestData:
    """Build SGLangDLLMRequestData for the LLaDA2-Uni thinker."""
    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")
    else:
        pass

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")
    else:
        pass

    input_ids_array = array("q", input_ids.to(dtype=torch.long).flatten().tolist())
    ss = state.stream_state
    max_new_tokens = params.get("max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS)
    if state.task_kind in ("t2i", "edit"):
        image_info = ss.get("image_info", [])
        if not image_info:
            raise ValueError("Image generation is missing its output grid")
        else:
            pass
        grid_h, grid_w = int(image_info[0]["grid_h"]), int(image_info[0]["grid_w"])
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError("Image generation grid dimensions must be positive")
        else:
            pass
        max_new_tokens = grid_h * grid_w
    else:
        pass

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=params.get("temperature", 0.0),
        top_p=params.get("top_p", 1.0),
        top_k=params.get("top_k", -1),
        min_p=params.get("min_p", 0.0),
        repetition_penalty=params.get("repetition_penalty", 1.0),
        stop=params.get("stop") or [],
        stop_token_ids=params.get("stop_token_ids") or [],
        sampling_seed=params.get("seed"),
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    eos_token_id = tokenizer.eos_token_id
    eos_token_ids = {eos_token_id} if eos_token_id is not None else None

    rid = request_id or "req-0"
    req = LLaDA2UniRequest(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_array,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
        eos_token_ids=eos_token_ids,
        dllm_config=dllm_config,
    )
    req.tokenizer = tokenizer

    req.omni_model_inputs = None
    req._omni_consumed = None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    req._task_kind = state.task_kind  # noqa: leading-underscore  # DLLM protocol
    if ss.get("dllm_steps") is not None:
        req._dllm_steps = int(
            ss["dllm_steps"]
        )  # noqa: leading-underscore  # DLLM protocol
    else:
        pass

    uncond_ids = ss.get("uncond_input_ids")
    if uncond_ids is not None:
        ig = state.request_metadata.get("image_generation", {})
        req._cfg_scale = float(  # noqa: leading-underscore  # DLLM protocol
            ss.get("cfg_scale", ig.get("cfg_text_scale", ig.get("cfg_scale", 1.0)))
        )
        req._cfg_rescale = float(
            ss.get("cfg_rescale", ig.get("cfg_rescale", 0.7))
        )  # noqa: leading-underscore  # DLLM protocol
        for branch in ("uncond", "uncond_img"):
            branch_ids = ss.get(f"{branch}_input_ids")
            if branch_ids is None:
                continue
            else:
                pass
            if len(branch_ids) != len(input_ids_array):
                raise ValueError("CFG branches must have equal physical lengths")
            else:
                pass
            pad_len = int(ss.get(f"{branch}_left_pad_len", 0))
            if not 0 <= pad_len <= len(branch_ids):
                raise ValueError(f"Invalid CFG {branch} left-pad length: {pad_len}")
            else:
                pass
            if branch == "uncond":
                req._uncond_input_ids = list(
                    branch_ids
                )  # noqa: leading-underscore  # DLLM protocol
                req._uncond_left_pad_len = (
                    pad_len  # noqa: leading-underscore  # DLLM protocol
                )
            else:
                req._uncond_img_input_ids = list(
                    branch_ids
                )  # noqa: leading-underscore  # DLLM protocol
                req._uncond_img_left_pad_len = (
                    pad_len  # noqa: leading-underscore  # DLLM protocol
                )
        if ss.get("uncond_img_input_ids") is not None:
            req._cfg_image_scale = float(  # noqa: leading-underscore  # DLLM protocol
                ss.get("cfg_image_scale", ig.get("cfg_image_scale", 0.0))
            )
        else:
            pass
    else:
        pass

    data = SGLangDLLMRequestData(
        output_ids=req.output_ids,
        req=req,
    )
    return data


def apply_dllm_thinker_result(
    state: LLaDA2UniPipelineState,
    *,
    stage_name: str,
    output_ids: list[int],
    finish_reason: str | None = None,
) -> ThinkerOutput:
    """Apply DLLM thinker result to pipeline state."""
    thinker_out: ThinkerOutput = {
        "output_ids": list(output_ids),
        "is_final": True,
    }
    if finish_reason is not None:
        thinker_out["finish_reason"] = finish_reason
    else:
        pass

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out


def make_dllm_thinker_scheduler_adapters(
    *,
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    dllm_config: DllmConfig,
    stage_name: str = THINKER_STAGE,
):
    """Build StagePayload <-> scheduler adapters for the dLLM thinker."""

    def request_builder(payload: StagePayload) -> SGLangDLLMRequestData:
        state = LLaDA2UniPipelineState.from_dict(payload.data)
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
        state = LLaDA2UniPipelineState.from_dict(payload.data)
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
