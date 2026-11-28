# SPDX-License-Identifier: Apache-2.0
"""Stage factories for LLaDA2-Uni pipeline."""

from __future__ import annotations

import io
import logging
from collections.abc import Mapping
from contextlib import nullcontext

import torch
from sglang.srt.arg_groups.model_override_base import resolved_view
from typing_extensions import NotRequired, TypedDict

from sglang_omni.models.llada2_uni.bootstrap import (
    create_dllm_thinker_scheduler,
    register_llada2_uni_cfg,
)
from sglang_omni.models.llada2_uni.cfg_attention_backend import CFG_ATTENTION_BACKEND
from sglang_omni.models.llada2_uni.components.common import load_llada2_tokenizer
from sglang_omni.models.llada2_uni.components.decoder_runtime import (
    initialize_decoder_runtime,
)
from sglang_omni.models.llada2_uni.components.image_decoder import (
    DecoderBackend,
    DecoderMode,
    LLaDA2ImageDecoder,
)
from sglang_omni.models.llada2_uni.components.image_encoder import LLaDA2ImageEncoder
from sglang_omni.models.llada2_uni.components.preprocessor import LLaDA2Preprocessor
from sglang_omni.models.llada2_uni.config import IMAGE_STAGE, THINKER_STAGE
from sglang_omni.models.llada2_uni.merge import (
    LLaDA2UniEventDict,
    build_image_decode_step,
    decode_events,
    event_to_dict,
)
from sglang_omni.models.llada2_uni.payload_types import (
    LLaDA2UniEvent,
    LLaDA2UniPipelineState,
)
from sglang_omni.models.llada2_uni.request_builders import (
    apply_encoder_result,
    build_encoder_request,
    merge_image_tokens_for_thinker,
)
from sglang_omni.models.llada2_uni.serial_offload import get_offload_coordinator
from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
    pin_resolved_device_type,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)


LLaDA2UniDecodeResult = TypedDict(
    "LLaDA2UniDecodeResult",
    {
        "events": list[LLaDA2UniEventDict],
        "text": NotRequired[str | list[str]],
        "modality": NotRequired[str],
        "usage": NotRequired[dict[str, int]],
        "finish_reason": NotRequired[str],
    },
)


def create_preprocessing_executor(
    model_path: str,
    *,
    max_seq_len: int | None = None,
    interleaved: bool = False,
) -> SimpleScheduler[StagePayload, StagePayload]:
    preprocessor = LLaDA2Preprocessor(
        model_path=model_path,
        max_seq_len=max_seq_len,
        interleaved=interleaved,
    )
    return SimpleScheduler(preprocessor)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | torch.dtype | None = None,
    enable_serial_offload: bool = False,
) -> SimpleScheduler[StagePayload, StagePayload]:

    dtype = resolve_dtype(dtype)
    device = str(resolve_concrete_device(device, gpu_id))

    if enable_serial_offload and torch.device(device).type != "cuda":
        raise ValueError("LLaDA2-Uni serial offload requires CUDA")
    else:
        pass
    model = LLaDA2ImageEncoder(
        model_path=model_path,
        device="cpu" if enable_serial_offload else device,
        dtype=dtype,
    )
    model.device = torch.device(device)
    coordinator = (
        get_offload_coordinator(model.device) if enable_serial_offload else None
    )

    def encode_image(payload: StagePayload) -> StagePayload:
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name=IMAGE_STAGE)

        if request.get("_skip"):
            result = request.get("_result", {})
        else:
            with (
                coordinator.use_module(model, "image_encoder")
                if coordinator
                else nullcontext()
            ), torch.no_grad():
                result = model(**request)

        apply_encoder_result(state, stage_name=IMAGE_STAGE, result=result)
        merge_image_tokens_for_thinker(state)
        state.encoder_inputs.clear()
        state.encoder_outs.clear()
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(encode_image)


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
    server_args_overrides: Mapping[str, object] | None = None,
    enable_serial_offload: bool = False,
) -> DllmScheduler:
    """Create an DllmScheduler for the LLaDA2-Uni thinker."""
    concrete_device = resolve_concrete_device(device, gpu_id)
    resolved_gpu_id = concrete_device.index or 0

    overrides: dict[str, object] = {
        "attention_backend": "flashinfer",
        "disable_cuda_graph": True,
        "sampling_backend": "pytorch",
    }
    if dllm_algorithm == "LowConfidenceCFG":
        register_llada2_uni_cfg()
        overrides["attention_backend"] = CFG_ATTENTION_BACKEND
        overrides["dllm_fdfo"] = False
    else:
        pass
    overrides.update(server_args_overrides or {})
    overrides["tp_size"] = tp_size
    if enable_serial_offload:
        if concrete_device.type != "cuda" or int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("LLaDA2-Uni serial offload requires one CUDA GPU and TP=1")
        elif overrides.get("weight_cache_mode", "off") != "off":
            raise ValueError("LLaDA2-Uni serial offload cannot use IPC weight caching")
        elif overrides.get("cpu_offload_gb", 0):
            raise ValueError(
                "LLaDA2-Uni serial offload cannot combine with layer CPU offload"
            )
        else:
            pass
        overrides.update(
            enable_memory_saver=True,
            enable_weights_cpu_backup=True,
            disable_cuda_graph=True,
            enable_torch_compile=False,
            disable_overlap_schedule=True,
        )
    else:
        pass
    pin_resolved_device_type(overrides, concrete_device.type)

    server_args = build_sglang_server_args(
        model_path,
        context_length=max_seq_len,
        dllm_algorithm=dllm_algorithm,
        dllm_algorithm_config=dllm_algorithm_config,
        **overrides,
    )
    if dllm_algorithm == "LowConfidenceCFG":
        # note (Anmuliar): DLLM graph defaults replace custom backends with FlashInfer.
        override_server_args(
            server_args,
            "sglang_omni.llada2_uni.cfg_attention",
            attention_backend=overrides["attention_backend"],
        )
    else:
        pass
    cfg = resolved_view(server_args)
    logger.info(
        f"create_sglang_dllm_thinker_executor_from_config: "
        f"dllm_algorithm={cfg.dllm_algorithm}, "
        f"mem_fraction_static={cfg.mem_fraction_static}"
    )
    return create_dllm_thinker_scheduler(
        server_args,
        resolved_gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        enable_serial_offload=enable_serial_offload,
    )


def create_decode_executor(
    model_path: str,
) -> SimpleScheduler[StagePayload, StagePayload]:
    tokenizer = load_llada2_tokenizer(model_path)

    def decode_text(payload: StagePayload) -> StagePayload:
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            logger.warning(
                f"request {payload.request_id}: thinker produced no output; returning empty text"
            )
            thinker_out = {
                "output_ids": [],
                "is_final": True,
            }
        else:
            pass

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

        result: LLaDA2UniDecodeResult = {"events": event_dicts}
        if events:
            result.update(events[0].payload)
            result.setdefault("modality", events[0].modality)
        else:
            pass

        finish_reason = thinker_out.get("finish_reason")
        if isinstance(finish_reason, str):
            result.setdefault("finish_reason", finish_reason)
        else:
            pass

        input_ids = (
            state.prompt.get("input_ids") if isinstance(state.prompt, dict) else None
        )
        if state.thinking_phase == "image":
            prompt_tokens = state.thinking_prompt_tokens
        elif input_ids is None:
            prompt_tokens = 0
        else:
            prompt_tokens = int(input_ids.numel())

        completion_ids = thinker_out.get("output_ids") or []
        completion_tokens = len(completion_ids) + state.thinking_completion_tokens

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

    return SimpleScheduler(decode_text)


def create_image_decode_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    runtime_gpu_ids: list[int] | None = None,
    dtype: str | torch.dtype | None = None,
    decode_mode: DecoderMode = "normal",
    num_steps: int = 50,
    resolution_multiplier: int = 2,
    backend: DecoderBackend = "diffusers",
    attention_backend: str = "torch_sdpa",
    ring_degree: int = 1,
    enable_serial_offload: bool = False,
) -> SimpleScheduler[StagePayload, StagePayload]:
    if runtime_gpu_ids is not None and len(runtime_gpu_ids) > 1:
        from sglang_omni.models.llada2_uni.native_sp_host import create_native_decoder

        return create_native_decoder(
            model_path,
            device=device,
            gpu_id=gpu_id,
            runtime_gpu_ids=runtime_gpu_ids,
            dtype=dtype,
            backend=backend,
            attention_backend=attention_backend,
            decode_mode=decode_mode,
            num_steps=num_steps,
            resolution_multiplier=resolution_multiplier,
            ring_degree=ring_degree,
            enable_serial_offload=enable_serial_offload,
        )
    elif ring_degree != 1:
        raise ValueError("ring_degree > 1 requires multi-GPU runtime_gpu_ids")
    else:
        pass
    concrete_device = resolve_concrete_device(device, gpu_id)
    dtype = resolve_dtype(dtype)
    if enable_serial_offload and (
        backend != "diffusers" or concrete_device.type != "cuda"
    ):
        raise ValueError(
            "LLaDA2-Uni serial offload requires the CUDA diffusers decoder"
        )
    else:
        pass
    runtime = None
    if backend == "sglang":
        runtime = initialize_decoder_runtime(
            model_path,
            gpu_id=concrete_device.index if concrete_device.type == "cuda" else None,
            dtype=dtype,
            attention_backend=attention_backend,
        )
    else:
        pass
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
                enable_serial_offload=enable_serial_offload,
            )
    except BaseException:
        if runtime:
            runtime.close()
        else:
            pass
        raise

    def decode_png(
        tokens: list[int], height: int, width: int, params: dict[str, int | DecoderMode]
    ) -> bytes:
        with runtime.compute_context() if runtime else nullcontext():
            image = decoder.decode(tokens, height, width, **params)
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        return image_buffer.getvalue()

    decode_step = build_image_decode_step(
        decode_png, enable_serial_offload=enable_serial_offload
    )
    if runtime is None:
        return SimpleScheduler(decode_step)
    else:
        pass

    class ImageDecoderScheduler(SimpleScheduler[StagePayload, StagePayload]):
        def start(self) -> None:
            try:
                super().start()
            finally:
                runtime.close()

    return ImageDecoderScheduler(decode_step)
