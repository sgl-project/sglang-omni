# SPDX-License-Identifier: Apache-2.0
"""Stage factories for LLaDA2-Uni pipeline."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.llada2_uni.config import IMAGE_STAGE, THINKER_STAGE
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniEvent
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def event_to_dict(event: LLaDA2UniEvent) -> dict[str, object]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": event.is_final,
    }


def create_preprocessing_executor(
    model_path: str,
    *,
    max_seq_len: int | None = None,
):
    from sglang_omni.models.llada2_uni.components.preprocessor import LLaDA2Preprocessor
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    preprocessor = LLaDA2Preprocessor(
        model_path=model_path,
        max_seq_len=max_seq_len,
    )
    return SimpleScheduler(preprocessor)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
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
    from sglang_omni.utils.device import resolve_concrete_device

    dtype = resolve_dtype(dtype)
    device = str(resolve_concrete_device(device, gpu_id))

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
    device: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    max_seq_len: int = 8192,
    dllm_algorithm: str = "LowConfidence",
    dllm_algorithm_config: str | None = None,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Create an DllmScheduler for the LLaDA2-Uni thinker."""
    from sglang_omni.models.llada2_uni.bootstrap import create_dllm_thinker_scheduler
    from sglang_omni.scheduling.sglang_backend import (
        build_sglang_server_args,
        pin_resolved_device_type,
    )
    from sglang_omni.utils.device import resolve_concrete_device

    concrete_device = resolve_concrete_device(device, gpu_id)
    resolved_gpu_id = concrete_device.index or 0

    overrides: dict[str, Any] = {
        "attention_backend": "flashinfer",
        "disable_cuda_graph": True,
        "sampling_backend": "pytorch",
    }
    if dllm_algorithm == "LowConfidenceCFG":
        from sglang_omni.models.llada2_uni.bootstrap import register_llada2_uni_cfg
        from sglang_omni.models.llada2_uni.cfg_attention_backend import (
            CFG_ATTENTION_BACKEND,
        )

        register_llada2_uni_cfg()
        overrides["attention_backend"] = CFG_ATTENTION_BACKEND
        overrides["dllm_fdfo"] = False
    overrides.update(server_args_overrides or {})
    overrides["tp_size"] = tp_size
    pin_resolved_device_type(overrides, concrete_device.type)

    server_args = build_sglang_server_args(
        model_path,
        context_length=max_seq_len,
        dllm_algorithm=dllm_algorithm,
        dllm_algorithm_config=dllm_algorithm_config,
        **overrides,
    )
    if dllm_algorithm == "LowConfidenceCFG":
        from sglang_omni.vendor.sglang.server_args import override_server_args

        # note (Anmuliar): DLLM graph defaults replace custom backends with FlashInfer.
        override_server_args(
            server_args,
            "sglang_omni.llada2_uni.cfg_attention",
            attention_backend=overrides["attention_backend"],
        )
    from sglang.srt.arg_groups.model_override_base import resolved_view

    cfg = resolved_view(server_args)
    logger.info(
        "create_sglang_dllm_thinker_executor_from_config: "
        "dllm_algorithm=%s, mem_fraction_static=%s",
        cfg.dllm_algorithm,
        cfg.mem_fraction_static,
    )
    return create_dllm_thinker_scheduler(
        server_args, resolved_gpu_id, tp_rank=tp_rank, nccl_port=nccl_port
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

        if state.thinking_phase == "image":
            events = [
                LLaDA2UniEvent(
                    type="text_final",
                    modality="text",
                    payload={"text": state.thinking_text},
                    is_final=True,
                )
            ]
        else:
            events = decode_events(
                thinker_out=thinker_out,
                tokenizer=tokenizer,
            )
        event_dicts = [event_to_dict(event) for event in events]

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


def create_image_decode_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: Any = None,
    decode_mode: str = "normal",
    num_steps: int = 50,
    resolution_multiplier: int = 2,
    backend: str = "diffusers",
    attention_backend: str = "torch_sdpa",
    interleaved_nonterminal: bool = False,
):
    import base64
    import io
    from contextlib import nullcontext

    from sglang_omni.models.llada2_uni.components.image_decoder import (
        LLaDA2ImageDecoder,
    )
    from sglang_omni.models.llada2_uni.merge import extract_image_vq_tokens
    from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
    from sglang_omni.models.weight_loader import resolve_dtype
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    from sglang_omni.utils.device import resolve_concrete_device

    concrete_device = resolve_concrete_device(device, gpu_id)
    dtype = resolve_dtype(dtype)
    runtime = None
    if backend == "sglang":
        from sglang_omni.models.llada2_uni.components.decoder_runtime import (
            initialize_decoder_runtime,
        )

        runtime = initialize_decoder_runtime(
            model_path,
            gpu_id=concrete_device.index if concrete_device.type == "cuda" else None,
            dtype=dtype,
            attention_backend=attention_backend,
        )
    try:
        with runtime.compute_context() if runtime else nullcontext():
            decoder = LLaDA2ImageDecoder(
                model_path=model_path,
                device=str(concrete_device),
                dtype=dtype,
                decode_mode=decode_mode,
                num_steps=num_steps,
                resolution_multiplier=resolution_multiplier,
                backend=backend,
                runtime=runtime,
            )
    except BaseException:
        if runtime:
            runtime.close()
        raise

    def decode_image(payload: StagePayload) -> StagePayload:
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        result = extract_image_vq_tokens(state)
        if result is None:
            if state.task_kind in ("t2i", "edit"):
                raise ValueError(
                    f"{state.task_kind} request did not produce image VQ tokens"
                )
            payload.data = {"events": [], "modality": "image", "skipped": True}
            return payload

        vq_tokens, h, w, params = result
        call_kwargs = {
            target: params[source]
            for source, target in (
                ("decode_mode", "decode_mode"),
                ("decoder_steps", "num_steps"),
                ("seed", "seed"),
            )
            if params.get(source) is not None
        }
        if (
            call_kwargs.get("decode_mode") == "decoder-turbo"
            and "num_steps" not in call_kwargs
        ):
            call_kwargs["num_steps"] = 8
        with runtime.compute_context() if runtime else nullcontext():
            image = decoder.decode(vq_tokens, h, w, **call_kwargs)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        if state.task_kind == "interleaved":
            frame_index = state.stream_state["interleaved"]["frame_index"]
            payload.data = {
                "kind": "interleaved_frame",
                "frame": {
                    "index": frame_index,
                    "image": {
                        "id": f"image-{payload.request_id}-{frame_index - 1}",
                        "data": image_b64,
                        "format": "png",
                        "width": image.width,
                        "height": image.height,
                    },
                },
            }
            return payload
        event = LLaDA2UniEvent(
            type="image_final",
            modality="image",
            payload={"image": image_b64, "format": "png"},
            is_final=True,
        )
        payload.data = {
            "events": [event_to_dict(event)],
            "modality": "image",
            **event.payload,
        }
        return payload

    if runtime is None:
        if interleaved_nonterminal:
            return SimpleScheduler(
                decode_image, allow_multiple_inflight_per_request=True
            )
        return SimpleScheduler(decode_image)

    class ImageDecoderScheduler(SimpleScheduler):
        def start(self) -> None:
            try:
                super().start()
            finally:
                # Shutdown callbacks run before the compute thread has exited.
                runtime.close()

    if interleaved_nonterminal:
        return ImageDecoderScheduler(
            decode_image, allow_multiple_inflight_per_request=True
        )
    return ImageDecoderScheduler(decode_image)
