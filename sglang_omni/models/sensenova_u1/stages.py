# SPDX-License-Identifier: Apache-2.0
"""Configure and construct SenseNova image-generation schedulers."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

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
    reference_image_size,
    resolve_edit_output_size,
)
from sglang_omni.models.sensenova_u1.neo_unify import register
from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    npu_fia_available,
    npu_swiglu_available,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_inputs import (
    build_t2i_queries,
    encode_generation_query,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_conditioned_generation import (
    build_it2i_queries,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.offload import (
    LayerwiseOffloadConfig,
    SenseNovaLayerwiseOffload,
)
from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    image_guidance_branch_count,
    resolve_cache_dit_defaults,
)
from sglang_omni.models.sensenova_u1.tensor_parallel import (
    TP_COLLECTIVE_TIMEOUT,
    SenseNovaTensorParallelScheduler,
    apply_tensor_parallel,
    initialize_tp,
    validate_tp_device,
    validate_tp_settings,
)
from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.pipeline.umm import UMMRequestError
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.tensor_parallel_scheduler import TensorParallelScheduler
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

# A healthy stop ends at the next denoise checkpoint; this caps only a wedged step or TP collective.
GENERATION_SHUTDOWN_TIMEOUT_S = TP_COLLECTIVE_TIMEOUT.total_seconds()


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
    return (
        options.n
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
    return ("text_to_image", *options.batch_key())


def validate_generation_request(
    payload: StagePayload,
    model: NEOChatModel,
    cache_dit: SenseNovaCacheDit,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
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
        references = decode_reference_images(inputs.get("image_b64"))
        try:
            reference_sizes = [
                reference_image_size(
                    image.width, image.height, len(references), options
                )
                for image in references
            ]
        finally:
            for image in references:
                image.close()
        width, height = resolve_edit_output_size(reference_sizes[0], options)
        queries = build_it2i_queries(model, prompt, reference_sizes, options)
        validate_output_dimensions(payload.request, width, height)
        branch_count = image_guidance_branch_count(
            options.guidance_scale, options.img_cfg_scale
        )
    else:
        options = text_to_image_options(payload)
        branch_count = 2 if options.guidance_scale > 1 else 1
        queries = [
            query
            for branch in build_t2i_queries(model, [inputs], options.guidance_scale)
            for query in branch
        ]
    for query in queries:
        encode_generation_query(model, tokenizer, query)
    enabled = (
        cache_dit.enabled_by_default
        if options.enable_cache_dit is None
        else options.enable_cache_dit
    )
    if enabled and branch_count <= 2:
        cache_dit.validate_request(
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
    cache_dit = SenseNovaCacheDit(
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
        logger.info(
            "SenseNova-U1 NPU memory after load: "
            f"allocated={torch.npu.memory_allocated(resolved_device)}, "
            f"reserved={torch.npu.memory_reserved(resolved_device)}"
        )
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
            return generate_image(payload, model, tokenizer, cache_dit)

    def _generate_batch(
        payloads: list[StagePayload],
    ) -> list[StagePayload | UMMRequestError]:
        results: list[StagePayload | UMMRequestError] = []
        valid_payloads: list[StagePayload] = []
        valid_indices: list[int] = []
        for index, payload in enumerate(payloads):
            try:
                validate_generation_request(payload, model, cache_dit, tokenizer)
            except UMMRequestError as error:
                results.append(error)
            else:
                results.append(payload)
                valid_payloads.append(payload)
                valid_indices.append(index)
        if not valid_payloads:
            return results
        else:
            pass
        cancellation = (
            GenerationScope(valid_payloads, _should_cancel, resolved_device, tp_group)
            if cooperative_cancellation
            else nullcontext()
        )
        with cancellation, offload.request() if offload is not None else nullcontext():
            generation_checkpoint()
            generated = generate_images(valid_payloads, model, tokenizer, cache_dit)
        for index, result in zip(valid_indices, generated, strict=True):
            results[index] = result
        return results

    def _shutdown() -> None:
        try:
            try:
                cache_dit.unmount()
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

    def _validate(payload: StagePayload) -> None:
        validate_generation_request(payload, model, cache_dit, tokenizer)

    if tp_size > 1 and cooperative_cancellation:
        scheduler = SenseNovaTensorParallelScheduler(
            _generate,
            report_cancellation=tp_rank == 0,
            shutdown_join_timeout_s=GENERATION_SHUTDOWN_TIMEOUT_S,
            validate_request=_validate,
            shutdown_callback=_shutdown,
        )
        return scheduler
    elif tp_size > 1:
        scheduler = TensorParallelScheduler(
            _generate,
            shutdown_join_timeout_s=GENERATION_SHUTDOWN_TIMEOUT_S,
            validate_request=_validate,
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
        shutdown_join_timeout_s=GENERATION_SHUTDOWN_TIMEOUT_S,
    )

    return scheduler
