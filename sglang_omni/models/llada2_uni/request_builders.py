# SPDX-License-Identifier: Apache-2.0
"""Request/result builders for LLaDA2-Uni pipeline stages."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any

import torch

from sglang_omni.models.llada2_uni.components.preprocessor import (
    BOI_TOKEN,
    DUMMY_IMAGE_TOKEN_ID,
    ROLE_ASSISTANT,
    ROLE_HUMAN,
    ROLE_SYSTEM,
    SOI_TOKEN,
    SYSTEM_PROMPT_T2I_THINKING,
    UNCOND_TEXT,
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
from sglang_omni.scheduling.dllm_group import (
    DllmRequestGroupSpec,
    align_cfg_request_group,
)

if TYPE_CHECKING:
    from sglang_omni.scheduling.sglang_backend import SGLangDLLMRequestData


def _is_thinking_phase(state: LLaDA2UniPipelineState, phase: int) -> bool:
    thinking = state.generation_state.get("thinking")
    return isinstance(thinking, dict) and thinking.get("phase") == phase


def _token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        raise ValueError(f"tokenizer does not define {token}")
    return token_id


def _boi_token_id(tokenizer: Any) -> int:
    token_id = getattr(tokenizer, "boi_token_id", None)
    if token_id is None:
        return _token_id(tokenizer, BOI_TOKEN)
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        raise ValueError(f"tokenizer does not define {BOI_TOKEN}")
    return token_id


def prepare_dllm_input_group(
    state: LLaDA2UniPipelineState,
    *,
    mask_token_id: int | None,
) -> tuple[tuple[int, ...], DllmRequestGroupSpec | None]:
    """Finalize physical CFG rows immediately before creating SGLang Reqs."""
    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")
    input_ids = prompt.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        conditional = tuple(
            int(token_id)
            for token_id in input_ids.to(dtype=torch.long).flatten().tolist()
        )
    elif isinstance(input_ids, (list, tuple)):
        conditional = tuple(int(token_id) for token_id in input_ids)
    else:
        raise TypeError("prompt.input_ids must be a tensor or token sequence")

    cfg = state.generation_state.get("cfg")
    if not isinstance(cfg, dict):
        return conditional, None
    unconditional = cfg.get("unconditional_input_ids")
    if unconditional is None:
        return conditional, None
    if state.image_token_offset is None:
        raise ValueError("native image CFG requires checkpoint image_token_offset")

    no_image = cfg.get("no_image_input_ids")
    algorithm_args = {
        "cfg_scale": float(cfg.get("text_scale", cfg.get("scale", 1.0))),
        "cfg_rescale": float(cfg.get("rescale", 0.0)),
        "force_image_only": True,
        "image_token_offset": state.image_token_offset,
    }
    if no_image is not None:
        algorithm_args["cfg_image_scale"] = float(cfg.get("image_scale", 0.0))

    return align_cfg_request_group(
        mask_token_id=mask_token_id,
        conditional_input_ids=conditional,
        unconditional_input_ids=tuple(int(token_id) for token_id in unconditional),
        no_image_input_ids=(
            tuple(int(token_id) for token_id in no_image)
            if no_image is not None
            else None
        ),
        algorithm_args=algorithm_args,
    )


def resolve_native_image_token_offset(
    state: LLaDA2UniPipelineState,
    *,
    vocab_size: int,
) -> int | None:
    """Return the checkpoint image-vocabulary boundary for native image tasks."""
    if state.task_kind not in {"t2i", "edit"}:
        return None
    if _is_thinking_phase(state, 1):
        return None
    image_token_offset = state.image_token_offset
    if (
        not isinstance(image_token_offset, int)
        or isinstance(image_token_offset, bool)
        or not 0 < image_token_offset < vocab_size
    ):
        raise ValueError(
            "native image generation requires a checkpoint-provided "
            "image_token_offset inside the model vocabulary"
        )
    return image_token_offset


def resolve_thinker_max_new_tokens(
    state: LLaDA2UniPipelineState,
    params: dict[str, Any],
) -> int:
    """Use a fixed thinking budget or one token per native image grid cell."""
    if _is_thinking_phase(state, 1):
        return DEFAULT_THINKER_MAX_NEW_TOKENS
    if state.task_kind not in {"t2i", "edit"}:
        return int(params.get("max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS))
    grid = state.generation_state.get("image_grid")
    if not isinstance(grid, dict):
        raise ValueError("native image request is missing image_grid")
    height = grid.get("height")
    width = grid.get("width")
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("native image grid dimensions must be integers")
    if height < 1 or width < 1:
        raise ValueError("native image grid dimensions must be positive")
    return height * width


def build_encoder_request(
    state: LLaDA2UniPipelineState,
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
    state: LLaDA2UniPipelineState,
    *,
    stage_name: str,
    result: Any,
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

    image_token_ids_list = image_out.get("image_token_ids")
    if not image_token_ids_list:
        return

    prompt = state.prompt
    if not isinstance(prompt, dict) or "input_ids" not in prompt:
        return

    input_ids = prompt["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.flatten().tolist()

    image_token_offset = state.image_token_offset
    if image_token_offset is None:
        raise ValueError("image_token_offset missing from LLaDA2 pipeline state")

    all_vq_tokens = []
    for token_ids in image_token_ids_list:
        all_vq_tokens.extend(tid + image_token_offset for tid in token_ids)

    if not all_vq_tokens:
        return

    new_ids = []
    vq_idx = 0
    for tid in input_ids:
        if tid == DUMMY_IMAGE_TOKEN_ID:
            if vq_idx >= len(all_vq_tokens):
                raise ValueError(
                    f"More placeholders than VQ tokens ({len(all_vq_tokens)})"
                )
            new_ids.append(all_vq_tokens[vq_idx])
            vq_idx += 1
        else:
            new_ids.append(tid)

    if vq_idx != len(all_vq_tokens):
        raise ValueError(
            f"VQ token count mismatch: {len(all_vq_tokens)} VQ tokens "
            f"but only {vq_idx} placeholders"
        )

    prompt["input_ids"] = torch.tensor([new_ids], dtype=torch.long)

    cfg = state.generation_state.get("cfg")
    if isinstance(cfg, dict):
        for branch_name in ("unconditional_input_ids", "no_image_input_ids"):
            branch = cfg.get(branch_name)
            if isinstance(branch, list) and DUMMY_IMAGE_TOKEN_ID in branch:
                cfg[branch_name] = _replace_image_placeholders(branch, all_vq_tokens)


def _replace_image_placeholders(
    input_ids: list[int], image_tokens: list[int]
) -> list[int]:
    placeholder_count = input_ids.count(DUMMY_IMAGE_TOKEN_ID)
    if placeholder_count != len(image_tokens):
        raise ValueError(
            "CFG image placeholder count does not match encoded VQ token count: "
            f"{placeholder_count} != {len(image_tokens)}"
        )
    token_iter = iter(image_tokens)
    return [
        next(token_iter) if token_id == DUMMY_IMAGE_TOKEN_ID else token_id
        for token_id in input_ids
    ]


def build_dllm_thinker_request(
    state: LLaDA2UniPipelineState,
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

    from sglang_omni.scheduling.sglang_backend.request_data import (
        SGLangDLLMRequestData,
    )

    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    conditional_input_ids, group_spec = prepare_dllm_input_group(
        state,
        mask_token_id=getattr(tokenizer, "mask_token_id", None),
    )
    input_ids_array = array("q", conditional_input_ids)

    sampling_params = SamplingParams(
        max_new_tokens=resolve_thinker_max_new_tokens(state, params),
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

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token_ids = {eos_token_id} if eos_token_id is not None else set()
    if _is_thinking_phase(state, 1):
        eos_token_ids.add(_boi_token_id(tokenizer))

    rid = request_id or "req-0"
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_array,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
        eos_token_ids=eos_token_ids or None,
        dllm_config=dllm_config,
    )
    req.tokenizer = tokenizer

    req.omni_model_inputs = None
    req._omni_consumed = None
    image_token_offset = resolve_native_image_token_offset(
        state,
        vocab_size=vocab_size,
    )
    if image_token_offset is not None:
        req.omni_dllm_image_token_offset = image_token_offset
    if group_spec is not None:
        req.omni_dllm_group_spec = group_spec

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
        "output_ids": output_ids,
        "is_final": True,
    }
    if finish_reason is not None:
        thinker_out["finish_reason"] = finish_reason

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out


def transition_thinking_phase1_to_phase2(
    state: LLaDA2UniPipelineState,
    *,
    tokenizer: Any,
    output_ids: list[int],
) -> None:
    """Atomically prepare a completed thinking request for image generation."""
    if not _is_thinking_phase(state, 1):
        raise ValueError("thinking transition requires a Phase 1 request")

    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinking transition")
    input_ids = prompt.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        original_ids = [
            int(token_id)
            for token_id in input_ids.to(dtype=torch.long).flatten().tolist()
        ]
    elif isinstance(input_ids, (list, tuple)):
        original_ids = [int(token_id) for token_id in input_ids]
    else:
        raise TypeError("prompt.input_ids must be a tensor or token sequence")

    boi_token_id = _boi_token_id(tokenizer)
    boi_position = next(
        (
            index
            for index, token_id in enumerate(output_ids)
            if int(token_id) == boi_token_id
        ),
        None,
    )
    if boi_position is None:
        raise RuntimeError(
            "Thinking Phase 1 did not produce <boi> after generating "
            f"{len(output_ids)} token(s)"
        )

    image_grid = state.generation_state.get("image_grid")
    if not isinstance(image_grid, dict):
        raise ValueError("thinking image request is missing image_grid")
    grid_height = image_grid.get("height")
    grid_width = image_grid.get("width")
    if (
        not isinstance(grid_height, int)
        or isinstance(grid_height, bool)
        or not isinstance(grid_width, int)
        or isinstance(grid_width, bool)
        or grid_height < 1
        or grid_width < 1
    ):
        raise ValueError("thinking image grid dimensions must be positive integers")

    phase2_prompt_ids = original_ids + [
        int(token_id) for token_id in output_ids[: boi_position + 1]
    ]
    trace = tokenizer.decode(
        output_ids[:boi_position],
        skip_special_tokens=True,
    )

    image_generation = state.request_metadata.get("image_generation")
    image_generation = image_generation if isinstance(image_generation, dict) else {}
    cfg_scale = float(image_generation.get("cfg_scale", 4.0))
    if cfg_scale <= 0.0:
        raise ValueError("image_generation.cfg_scale must be positive")

    next_generation_state = dict(state.generation_state)
    next_generation_state["thinking"] = {
        "phase": 2,
        "needs_reentry": True,
        "trace": trace,
    }
    if cfg_scale > 1.0:
        unconditional_prompt = (
            f"{ROLE_SYSTEM} {SYSTEM_PROMPT_T2I_THINKING} "
            f"{ROLE_HUMAN}{UNCOND_TEXT}{ROLE_ASSISTANT}"
        )
        unconditional_ids = tokenizer.encode(
            unconditional_prompt,
            add_special_tokens=False,
        )
        unconditional_ids.extend(
            [
                _token_id(tokenizer, SOI_TOKEN),
                *tokenizer.encode(
                    f"<|reserved_token_{grid_height}|>",
                    add_special_tokens=False,
                ),
                *tokenizer.encode(
                    f"<|reserved_token_{grid_width}|>",
                    add_special_tokens=False,
                ),
                boi_token_id,
            ]
        )
        next_generation_state["cfg"] = {
            "unconditional_input_ids": unconditional_ids,
            "scale": cfg_scale,
            "rescale": float(image_generation.get("cfg_rescale", 0.7)),
        }
    else:
        next_generation_state.pop("cfg", None)

    next_engine_outputs = dict(state.engine_outputs)
    next_engine_outputs.pop(THINKER_STAGE, None)

    state.prompt = {"input_ids": torch.tensor([phase2_prompt_ids], dtype=torch.long)}
    state.generation_state = next_generation_state
    state.thinker_out = None
    state.engine_outputs = next_engine_outputs


def make_dllm_thinker_scheduler_adapters(
    *,
    tokenizer: Any,
    vocab_size: int,
    dllm_config: Any,
    stage_name: str = THINKER_STAGE,
):
    """Build StagePayload <-> scheduler adapters for the dLLM thinker."""

    def request_builder(payload: StagePayload) -> SGLangDLLMRequestData:
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        thinking = state.generation_state.get("thinking")
        if (
            isinstance(thinking, dict)
            and thinking.get("phase") == 2
            and thinking.get("needs_reentry")
        ):
            next_generation_state = dict(state.generation_state)
            next_thinking = dict(thinking)
            next_thinking.pop("needs_reentry", None)
            next_generation_state["thinking"] = next_thinking
            state.generation_state = next_generation_state
            payload = StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data=state.to_dict(),
            )
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
        is_thinking_phase1 = _is_thinking_phase(state, 1)
        if is_thinking_phase1:
            transition_thinking_phase1_to_phase2(
                state,
                tokenizer=tokenizer,
                output_ids=data.output_ids,
            )
        else:
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
