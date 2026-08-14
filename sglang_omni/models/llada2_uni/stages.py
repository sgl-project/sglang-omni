# SPDX-License-Identifier: Apache-2.0
"""Stage factories for LLaDA2-Uni pipeline."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.llada2_uni.config import IMAGE_STAGE, THINKER_STAGE

logger = logging.getLogger(__name__)


def attach_thinking_trace(result: dict[str, Any], state: Any) -> None:
    """Expose Phase 1 reasoning only for combined text-and-image requests."""
    modalities = state.request_metadata.get("output_modalities")
    if isinstance(modalities, str):
        modalities = (modalities,)
    if not isinstance(modalities, (list, tuple, set)):
        return
    normalized_modalities = {str(modality).lower() for modality in modalities}
    if not {"text", "image"}.issubset(normalized_modalities):
        return
    thinking = state.generation_state.get("thinking")
    if not isinstance(thinking, dict):
        return
    trace = thinking.get("trace")
    if isinstance(trace, str):
        result["text"] = trace


def _event_to_dict(event) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


def create_preprocessing_executor(
    model_path: str,
    *,
    thinker_max_seq_len: int | None = None,
):
    from sglang_omni.models.llada2_uni.components.preprocessor import LLaDA2Preprocessor
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    preprocessor = LLaDA2Preprocessor(
        model_path=model_path,
        max_seq_len=thinker_max_seq_len,
    )
    return SimpleScheduler(preprocessor)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: Any = None,
):
    import torch

    from sglang_omni.models.llada2_uni.components.image_encoder import (
        LLaDA2ImageEncoder,
    )
    from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
    from sglang_omni.models.llada2_uni.request_builders import (
        apply_encoder_result,
        build_encoder_request,
        merge_image_tokens_for_thinker,
    )
    from sglang_omni.models.weight_loader import resolve_dtype
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    dtype = resolve_dtype(dtype)

    model = LLaDA2ImageEncoder(model_path=model_path, device=device, dtype=dtype)

    def _encode(payload):
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name=IMAGE_STAGE)

        if request.get("_skip"):
            result = request.get("_result", {})
        else:
            with torch.no_grad():
                result = model(**request)

        apply_encoder_result(state, stage_name=IMAGE_STAGE, result=result)
        merge_image_tokens_for_thinker(state)
        state.encoder_inputs.clear()
        state.encoder_outs.clear()
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(_encode)


def create_sglang_dllm_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    thinker_max_seq_len: int = 8192,
    dllm_algorithm: str = "LowConfidence",
    dllm_algorithm_config: str | None = None,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Create an DllmScheduler for the LLaDA2-Uni thinker."""
    overrides: dict[str, Any] = {
        "attention_backend": "flashinfer",
        "disable_cuda_graph": True,
        "sampling_backend": "pytorch",
    }
    overrides.update(server_args_overrides or {})

    if (
        dllm_algorithm == "LowConfidenceCFG"
        and overrides["attention_backend"] != "flashinfer"
    ):
        raise ValueError(
            "LLaDA2-Uni native image CFG requires attention_backend='flashinfer'"
        )

    from sglang_omni.models.llada2_uni.bootstrap import create_dllm_thinker_scheduler
    from sglang_omni.scheduling.sglang_backend import build_sglang_server_args

    if dllm_algorithm == "LowConfidenceCFG":
        from sglang.srt.dllm.algorithm import algo_name_to_cls

        from sglang_omni.models.llada2_uni.algorithm.low_confidence_cfg import (
            LowConfidenceCFG,
        )
        from sglang_omni.models.llada2_uni.cfg_attention_backend import (
            register_llada2_cfg_flashinfer_backend,
        )

        algo_name_to_cls["LowConfidenceCFG"] = LowConfidenceCFG
        register_llada2_cfg_flashinfer_backend()

    server_args = build_sglang_server_args(
        model_path,
        context_length=thinker_max_seq_len,
        dllm_algorithm=dllm_algorithm,
        dllm_algorithm_config=dllm_algorithm_config,
        **overrides,
    )
    logger.info(
        "create_sglang_dllm_thinker_executor_from_config: "
        "dllm_algorithm=%s, mem_fraction_static=%s",
        server_args.dllm_algorithm,
        server_args.mem_fraction_static,
    )
    return create_dllm_thinker_scheduler(server_args, gpu_id)


def create_image_decode_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: Any = None,
    decode_mode: str = "normal",
    num_steps: int = 50,
    resolution_multiplier: int = 2,
    interleaved_nonterminal: bool = False,
):
    """Create the native VQ-to-image terminal stage."""
    import base64

    from sglang_omni.models.llada2_uni.components.image_decoder import (
        LLaDA2ImageDecoder,
    )
    from sglang_omni.models.llada2_uni.merge import extract_image_vq_tokens
    from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
    from sglang_omni.models.weight_loader import resolve_dtype
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    decoder = LLaDA2ImageDecoder(
        model_path=model_path,
        device=device,
        dtype=resolve_dtype(dtype),
        decode_mode=decode_mode,
        num_steps=num_steps,
        resolution_multiplier=resolution_multiplier,
    )

    def _decode_image(payload):
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        extracted = extract_image_vq_tokens(state)
        if extracted is None:
            raise ValueError("image decoder received a non-image request")
        token_ids, height, width, params = extracted
        call_kwargs: dict[str, Any] = {}
        if isinstance(params.get("decode_mode"), str):
            call_kwargs["decode_mode"] = params["decode_mode"]
        if isinstance(params.get("decoder_steps"), int):
            call_kwargs["num_steps"] = params["decoder_steps"]
        if isinstance(params.get("resolution_multiplier"), int):
            call_kwargs["resolution_multiplier"] = params["resolution_multiplier"]
        if isinstance(params.get("seed"), int):
            call_kwargs["seed"] = params["seed"]

        image_bytes, image_width, image_height = decoder.decode_to_bytes(
            token_ids, height, width, **call_kwargs
        )
        frame_index = 0
        if state.task_kind == "interleaved":
            interleaved = state.generation_state.get("interleaved")
            if not isinstance(interleaved, dict):
                raise ValueError(
                    "interleaved image decoder is missing generation state"
                )
            frame_index = int(interleaved.get("frame_index", 0))
            if frame_index <= 0:
                raise ValueError("interleaved image decoder is missing a frame index")
        image_index = frame_index - 1 if frame_index else 0
        result = {
            "modality": "image",
            "images": [
                {
                    "id": f"image-{payload.request_id}-{image_index}",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                    "format": "png",
                    "width": image_width,
                    "height": image_height,
                }
            ],
            "finish_reason": "stop",
        }
        if state.task_kind == "interleaved":
            payload.data = {
                "kind": "interleaved_frame",
                "frame": {
                    "index": frame_index,
                    "image": result["images"][0],
                },
            }
            return payload
        attach_thinking_trace(result, state)
        payload.data = result
        return payload

    return SimpleScheduler(
        _decode_image,
        allow_multiple_inflight_per_request=interleaved_nonterminal,
    )


def create_decode_executor(model_path: str):
    from sglang_omni.models.llada2_uni.components.common import load_llada2_tokenizer
    from sglang_omni.models.llada2_uni.merge import decode_events
    from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    tokenizer = load_llada2_tokenizer(model_path)

    def _decode(payload):
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            logger.warning(
                "request %s: thinker produced no output (got %s), returning empty text",
                payload.request_id,
                type(thinker_out).__name__,
            )
            thinker_out = {
                "output_ids": [],
                "is_final": True,
            }

        events = decode_events(
            thinker_out=thinker_out,
            tokenizer=tokenizer,
        )
        event_dicts = [_event_to_dict(event) for event in events]

        result: dict[str, Any] = {"events": event_dicts}
        if events:
            result.update(events[0].payload)
            result.setdefault("modality", events[0].modality)

        finish_reason = thinker_out.get("finish_reason")
        if finish_reason is not None:
            result.setdefault("finish_reason", finish_reason)

        input_ids = (
            state.prompt.get("input_ids") if isinstance(state.prompt, dict) else None
        )
        if input_ids is None:
            prompt_tokens = 0
        elif hasattr(input_ids, "numel"):
            prompt_tokens = int(input_ids.numel())
        else:
            prompt_tokens = len(input_ids)

        completion_ids = thinker_out.get("output_ids") or []
        completion_tokens = len(completion_ids)

        result.setdefault(
            "usage",
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

        payload.data = result
        return payload

    return SimpleScheduler(_decode)
