# SPDX-License-Identifier: Apache-2.0
"""Stage factories for LLaDA2-Uni pipeline."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.llada2_uni.config import (
    IMAGE_DECODE_STAGE,
    IMAGE_STAGE,
    THINKER_STAGE,
)

logger = logging.getLogger(__name__)


def _event_to_dict(event) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


def _usage_from_state(
    state,
    thinker_out: dict[str, Any],
) -> dict[str, int]:
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
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
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
    """Create the LLaDA2-Uni thinker scheduler.

    Text requests use the existing SGLang dLLM scheduler. Image-output requests
    use the checkpoint's HF ``generate_image`` path so VQ token generation stays
    aligned with upstream LLaDA2-Uni.
    """
    from sglang_omni.models.llada2_uni.bootstrap import create_dllm_thinker_scheduler
    from sglang_omni.models.llada2_uni.components.image_token_generator import (
        LLaDA2ImageTokenGenerator,
    )
    from sglang_omni.models.llada2_uni.hybrid_scheduler import (
        LLaDA2HybridThinkerScheduler,
    )
    from sglang_omni.scheduling.sglang_backend import build_sglang_server_args

    def _create_text_scheduler():
        overrides: dict[str, Any] = {
            "attention_backend": "flashinfer",
            "disable_cuda_graph": True,
            "sampling_backend": "pytorch",
        }
        overrides.update(server_args_overrides or {})

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

    image_generator = LLaDA2ImageTokenGenerator(
        model_path=model_path,
        device=f"cuda:{gpu_id}",
        dtype=None,
    )
    return LLaDA2HybridThinkerScheduler(
        text_scheduler_factory=_create_text_scheduler,
        image_compute_fn=image_generator,
    )


def create_image_decode_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: Any = None,
):
    from sglang_omni.models.llada2_uni.components.image_decoder import (
        LLaDA2ImageDecoder,
    )
    from sglang_omni.models.llada2_uni.merge import decode_events
    from sglang_omni.models.llada2_uni.payload_types import (
        LLaDA2UniEvent,
        LLaDA2UniPipelineState,
    )
    from sglang_omni.models.weight_loader import resolve_dtype
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    decoder: LLaDA2ImageDecoder | None = None

    def _get_decoder() -> LLaDA2ImageDecoder:
        nonlocal decoder
        if decoder is None:
            decoder = LLaDA2ImageDecoder(
                model_path=model_path,
                device=device,
                dtype=resolve_dtype(dtype),
            )
        return decoder

    def _image_decode(payload):
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        if state.generation.get("type") != "image":
            return payload

        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict) or not thinker_out.get("output_ids"):
            raise ValueError("LLaDA2-Uni image decode requires generated VQ tokens.")

        events = decode_events(
            thinker_out=thinker_out,
            tokenizer=object(),
            generation=state.generation,
        )
        if not events:
            raise ValueError("LLaDA2-Uni image decode produced no image token event.")

        image_token_event = events[0]
        token_payload = image_token_event.payload
        image_token_ids = token_payload.get("image_token_ids") or []
        token_grid_h = int(token_payload["token_grid_h"])
        token_grid_w = int(token_payload["token_grid_w"])
        decode_mode = str(token_payload.get("decode_mode") or "decoder-turbo")
        decoded = _get_decoder().decode(
            list(image_token_ids),
            token_grid_h=token_grid_h,
            token_grid_w=token_grid_w,
            resolution_multiplier=int(token_payload.get("resolution_multiplier") or 2),
            num_steps=int(token_payload.get("decoder_steps") or 8),
            decode_mode=decode_mode,
            image_format=str(state.generation.get("format") or "png"),
            seed=state.generation.get("seed"),
        )
        image = decoded.to_payload()
        image_final_event = LLaDA2UniEvent(
            type="image_final",
            modality="image",
            payload={"images": [image]},
            is_final=True,
        )
        result: dict[str, Any] = {
            "events": [
                _event_to_dict(image_token_event),
                _event_to_dict(image_final_event),
            ],
            "images": [image],
            "modality": "image",
            "format": decoded.format,
            "mime_type": decoded.mime_type,
            "width": decoded.width,
            "height": decoded.height,
            "image_token_count": len(image_token_ids),
        }
        state.engine_outputs[IMAGE_DECODE_STAGE] = result
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(_image_decode)


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

        decoded_image_out = state.engine_outputs.get(IMAGE_DECODE_STAGE)
        if state.generation.get("type") == "image" and isinstance(
            decoded_image_out, dict
        ):
            result = dict(decoded_image_out)
            finish_reason = thinker_out.get("finish_reason")
            if finish_reason is not None:
                result.setdefault("finish_reason", finish_reason)
            result.setdefault("usage", _usage_from_state(state, thinker_out))
            payload.data = result
            return payload

        events = decode_events(
            thinker_out=thinker_out,
            tokenizer=tokenizer,
            generation=state.generation,
        )
        event_dicts = [_event_to_dict(event) for event in events]

        result: dict[str, Any] = {"events": event_dicts}
        if events:
            result.update(events[0].payload)
            result.setdefault("modality", events[0].modality)

        finish_reason = thinker_out.get("finish_reason")
        if finish_reason is not None:
            result.setdefault("finish_reason", finish_reason)

        result.setdefault("usage", _usage_from_state(state, thinker_out))

        payload.data = result
        return payload

    return SimpleScheduler(_decode)
