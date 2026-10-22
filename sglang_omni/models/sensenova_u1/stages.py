# SPDX-License-Identifier: Apache-2.0
"""Configure and construct SenseNova image-generation schedulers."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer

from sglang_omni.models.sensenova_u1.admission import validate_output_dimensions
from sglang_omni.models.sensenova_u1.cache_dit import SenseNovaCacheDit
from sglang_omni.models.sensenova_u1.cancellation import (
    GenerationScope,
    generation_checkpoint,
)
from sglang_omni.models.sensenova_u1.generation import (
    generate_image,
    generate_images,
    prepare_cache_dit,
    text_to_image_options,
    unmount_cache_dit_after_failure,
)
from sglang_omni.models.sensenova_u1.image_processing import (
    decode_reference_images,
    resolve_reference_output_size,
)
from sglang_omni.models.sensenova_u1.neo_unify import register
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    npu_fia_available,
    npu_swiglu_available,
)
from sglang_omni.models.sensenova_u1.offload import (
    LayerwiseOffloadConfig,
    SenseNovaLayerwiseOffload,
)
from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
    cache_dit_batch_key,
    image_guidance_branch_count,
    resolve_cache_dit_defaults,
)
from sglang_omni.models.sensenova_u1.tensor_parallel import (
    SenseNovaTensorParallelScheduler,
    apply_tensor_parallel,
    initialize_tp,
    validate_tp_device,
    validate_tp_settings,
)
from sglang_omni.models.sensenova_u1.tensor_parallel_scheduler import (
    TensorParallelScheduler,
)
from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device

__all__ = [
    "create_generation_executor",
    "decode_reference_images",
    "generate_image",
    "generate_images",
    "image_generation_batch_key",
    "image_generation_request_cost",
    "prepare_cache_dit",
    "unmount_cache_dit_after_failure",
    "validate_generation_request",
]

logger = logging.getLogger(__name__)


def image_generation_request_cost(payload: StagePayload) -> int:
    """Estimate denoise work as pixel-steps, including the CFG branch."""
    inputs = payload.request.inputs
    if isinstance(inputs, dict) and inputs.get("task") == "image_edit":
        options = SenseNovaU1ImageEditSampling.from_params(payload.request.params)
        branches = image_guidance_branch_count(
            options.guidance_scale, options.img_cfg_scale
        )
    else:
        options = text_to_image_options(payload)
        branches = 1 + int(options.guidance_scale > 1)
    outputs = options.n if isinstance(options, SenseNovaU1Sampling) else 1
    return (
        outputs
        * options.width
        * options.height
        * options.num_inference_steps
        * branches
    )


def image_generation_batch_key(
    payload: StagePayload,
) -> (
    tuple[Literal["image_edit", "text_to_image_multi_output"], str]
    | tuple[Literal["text_to_image"], int, int, int, float, bool | None, str]
):
    """Return a compatibility key; I2I requests always remain single-item."""
    inputs = payload.request.inputs
    if isinstance(inputs, dict) and inputs.get("task") == "image_edit":
        return ("image_edit", payload.request_id)
    else:
        pass
    options = text_to_image_options(payload)
    if options.n > 1:
        return ("text_to_image_multi_output", payload.request_id)
    else:
        pass
    return (
        "text_to_image",
        options.width,
        options.height,
        options.num_inference_steps,
        options.guidance_scale,
        options.enable_cache_dit,
        cache_dit_batch_key(options.cache_dit_params),
    )


def validate_generation_request(payload: StagePayload, model: NEOChatModel) -> None:
    """Reject invalid TP work before the leader dispatches it to other ranks."""
    inputs = payload.request.inputs
    is_edit = isinstance(inputs, dict) and inputs.get("task") == "image_edit"
    if is_edit:
        prompt = inputs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("SenseNova-U1 requires a non-empty image edit prompt")
        else:
            pass
        options = SenseNovaU1ImageEditSampling.from_params(payload.request.params)
        width, height = resolve_reference_output_size(inputs.get("image_b64"), options)
        validate_output_dimensions(payload.request, width, height)
        branch_count = image_guidance_branch_count(
            options.guidance_scale, options.img_cfg_scale
        )
    else:
        options = text_to_image_options(payload)
        branch_count = 2 if options.guidance_scale > 1 else 1
    controller = model.sensenova_cache_dit
    enabled = (
        controller.enabled_by_default
        if options.enable_cache_dit is None
        else options.enable_cache_dit
    )
    if enabled and branch_count <= 2:
        controller.validate_request(
            model.language_model.model,
            params=options.cache_dit_params,
            steps=options.num_inference_steps,
        )
    else:
        pass


def create_generation_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    dtype: str | None,
    cooperative_cancellation: bool,
    max_batch_size: int = 1,
    max_batch_wait_ms: float = 0,
    max_batch_cost: int | None = None,
    enable_cache_dit: bool,
    cache_dit_params: dict[str, int | float] | None = None,
    dit_layerwise_offload: bool,
    dit_offload_prefetch_size: int,
    dit_layerwise_resident_layers: int,
    dit_layerwise_residency_policy: Literal["leading", "strided"],
    pin_cpu_memory: bool,
) -> SimpleScheduler:
    """Load the model once and optionally batch compatible T2I requests."""
    validate_tp_settings(tp_rank, tp_size, max_batch_size)
    resolved_device = resolve_concrete_device(device, gpu_id)
    if tp_size > 1:
        validate_tp_device(resolved_device, nccl_port)
    else:
        pass
    offload_config = None
    if dit_layerwise_offload:
        if resolved_device.type != "cuda":
            raise ValueError("SenseNova layerwise offload currently requires CUDA")
        else:
            pass
        offload_config = LayerwiseOffloadConfig(
            prefetch_size=dit_offload_prefetch_size,
            resident_layers=dit_layerwise_resident_layers,
            residency_policy=dit_layerwise_residency_policy,
            pin_cpu_memory=pin_cpu_memory,
        )
    else:
        pass

    load_started_at_s = time.perf_counter()
    register()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=resolve_dtype(dtype)
    ).eval()
    tp_group = None
    if tp_size > 1:
        tp_group = initialize_tp(resolved_device, tp_rank, tp_size, nccl_port)
        apply_tensor_parallel(model, tp_rank, tp_size, tp_group)
    else:
        pass
    offload = None
    if offload_config is not None:
        offload = SenseNovaLayerwiseOffload(model, resolved_device, offload_config)
    else:
        model = model.to(resolved_device)
    enable_cache_dit, cache_dit_params = resolve_cache_dit_defaults(
        enable_cache_dit, cache_dit_params
    )
    model.sensenova_cache_dit: SenseNovaCacheDit = SenseNovaCacheDit(
        enabled_by_default=enable_cache_dit,
        default_params=cache_dit_params,
    )
    logger.info(
        f"SenseNova-U1 loaded on {resolved_device} with dtype={dtype}, "
        f"tp_rank={tp_rank}/{tp_size} "
        f"in {time.perf_counter() - load_started_at_s:.2f} seconds"
    )
    if resolved_device.type == "npu":
        logger.info(
            f"SenseNova-U1 NPU operators: FIA={npu_fia_available()}, "
            f"SwiGLU={npu_swiglu_available()}"
        )
        npu = getattr(torch, "npu", None)
        if npu is not None and all(
            hasattr(npu, name) for name in ("memory_allocated", "memory_reserved")
        ):
            logger.info(
                "SenseNova-U1 NPU memory after load: "
                f"allocated={npu.memory_allocated(resolved_device)}, "
                f"reserved={npu.memory_reserved(resolved_device)}"
            )
        else:
            pass
    else:
        pass
    batch_enabled = max_batch_size > 1

    def _should_cancel(request_id: str) -> bool:
        stopping = (
            scheduler.stop_requested
            if isinstance(scheduler, TensorParallelScheduler)
            else not scheduler.running
        )
        return stopping or scheduler.is_aborted(request_id)

    def _generate(payload: StagePayload) -> StagePayload:
        cancellation = (
            GenerationScope([payload], _should_cancel, resolved_device, tp_group)
            if cooperative_cancellation
            else nullcontext()
        )
        with cancellation, offload.request() if offload is not None else nullcontext():
            generation_checkpoint()
            result = generate_image(payload, model, tokenizer)
            generation_checkpoint()
            return result

    def _generate_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        cancellation = (
            GenerationScope(payloads, _should_cancel, resolved_device, tp_group)
            if cooperative_cancellation
            else nullcontext()
        )
        with cancellation, offload.request() if offload is not None else nullcontext():
            generation_checkpoint()
            results = generate_images(payloads, model, tokenizer)
            generation_checkpoint()
            return results

    def _shutdown() -> None:
        try:
            try:
                model.sensenova_cache_dit.unmount()
            finally:
                if offload is not None:
                    offload.close()
                else:
                    pass
        finally:
            if tp_group is not None and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            else:
                pass

    if tp_size > 1:
        scheduler_class = (
            SenseNovaTensorParallelScheduler
            if cooperative_cancellation
            else TensorParallelScheduler
        )
        scheduler = scheduler_class(
            _generate,
            validate_request=lambda payload: validate_generation_request(
                payload, model
            ),
            shutdown_callback=_shutdown,
        )
        return scheduler
    else:
        pass

    scheduler = SimpleScheduler(
        _generate,
        batch_compute_fn=_generate_batch if batch_enabled else None,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_key_fn=image_generation_batch_key if batch_enabled else None,
        request_cost_fn=image_generation_request_cost if batch_enabled else None,
        max_batch_cost=max_batch_cost if batch_enabled else None,
        shutdown_callback=_shutdown,
        shutdown_on_compute_thread=True,
    )

    return scheduler
