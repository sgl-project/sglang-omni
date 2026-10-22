# SPDX-License-Identifier: Apache-2.0
"""Generate SenseNova images individually or in compatible batches."""

from __future__ import annotations

import logging

import torch
from pydantic import JsonValue
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.admission import (
    REFERENCE_MEMORY_ERROR_PREFIX,
    validate_output_dimensions,
)
from sglang_omni.models.sensenova_u1.cache_dit import SenseNovaCacheDit
from sglang_omni.models.sensenova_u1.image_processing import (
    encode_image,
    prepare_reference_images,
    resolve_edit_output_size,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
    cache_dit_batch_key,
    image_guidance_branch_count,
)
from sglang_omni.proto.request import StagePayload

logger = logging.getLogger(__name__)

GENERATION_TIMESTEP_SHIFT = 3.0
GENERATION_TIME_EPSILON = 0.02


def generate_image(
    payload: StagePayload, model: NEOChatModel, tokenizer: PreTrainedTokenizerBase
) -> StagePayload:
    """Run the source model's native T2I or I2I path and return a PNG payload."""
    inputs = payload.request.inputs
    try:
        if isinstance(inputs, dict) and inputs.get("task") == "image_edit":
            return generate_image_edit(payload, model, tokenizer)
        else:
            pass
        return generate_text_to_image(payload, model, tokenizer)
    except Exception:
        unmount_cache_dit_after_failure(model)
        raise


def generate_images(
    payloads: list[StagePayload],
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
) -> list[StagePayload]:
    """Batch compatible T2I requests while preserving request order."""
    if not payloads:
        return []
    else:
        pass

    results: list[StagePayload | None] = [None] * len(payloads)
    groups: dict[
        tuple[int, int, int, float, bool | None, str], list[tuple[int, StagePayload]]
    ] = {}
    for index, payload in enumerate(payloads):
        inputs = payload.request.inputs
        if isinstance(inputs, dict) and inputs.get("task") == "image_edit":
            try:
                results[index] = generate_image_edit(payload, model, tokenizer)
            except Exception:
                unmount_cache_dit_after_failure(model)
                raise
            continue
        else:
            pass
        options = text_to_image_options(payload)
        if options.n > 1:
            results[index] = generate_image(payload, model, tokenizer)
            continue
        else:
            pass
        signature = (
            options.width,
            options.height,
            options.num_inference_steps,
            options.guidance_scale,
            options.enable_cache_dit,
            cache_dit_batch_key(options.cache_dit_params),
        )
        groups.setdefault(signature, []).append((index, payload))

    for indexed_payloads in groups.values():
        indexes, compatible_payloads = zip(*indexed_payloads)
        try:
            if len(compatible_payloads) == 1:
                batch_results = [
                    generate_text_to_image(compatible_payloads[0], model, tokenizer)
                ]
            else:
                batch_results = generate_text_to_image_batch(
                    list(compatible_payloads), model, tokenizer
                )
        except Exception:
            unmount_cache_dit_after_failure(model)
            raise
        for index, result in zip(indexes, batch_results):
            results[index] = result

    completed_results: list[StagePayload] = []
    for result in results:
        if result is None:
            raise RuntimeError("SenseNova-U1 failed to produce every batched result")
        else:
            completed_results.append(result)
    return completed_results


def text_to_image_options(payload: StagePayload) -> SenseNovaU1Sampling:
    prompt = payload.request.inputs
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SenseNova-U1 requires a non-empty text prompt")
    else:
        pass
    return SenseNovaU1Sampling.from_params(payload.request.params)


def generate_text_to_image(
    payload: StagePayload, model: NEOChatModel, tokenizer: PreTrainedTokenizerBase
) -> StagePayload:
    prompt = payload.request.inputs
    options = text_to_image_options(payload)
    prepare_cache_dit(
        model,
        enable_cache_dit=options.enable_cache_dit,
        params=options.cache_dit_params,
        steps=options.num_inference_steps,
        cfg_scale=options.guidance_scale,
    )
    with torch.inference_mode():
        images = model.t2i_generate(
            tokenizer,
            prompt,
            batch_size=options.n,
            thinking_backend=None,
            seed=(
                options.seed
                if options.n == 1
                else [options.seed + index for index in range(options.n)]
            ),
            image_size=(options.width, options.height),
            cfg_scale=options.guidance_scale,
            cfg_norm="none",
            timestep_shift=GENERATION_TIMESTEP_SHIFT,
            enable_timestep_shift=True,
            cfg_interval=(0.0, 1.0),
            num_steps=options.num_inference_steps,
            t_eps=GENERATION_TIME_EPSILON,
            think_mode=False,
        )
    if options.n == 1:
        return encode_image(payload, images, options.width, options.height)
    else:
        pass
    if not isinstance(images, torch.Tensor) or images.shape != (
        options.n,
        3,
        options.height,
        options.width,
    ):
        raise ValueError("SenseNova-U1 returned an invalid multi-output image tensor")
    else:
        pass
    encoded = []
    for index in range(options.n):
        encode_image(payload, images[index : index + 1], options.width, options.height)
        encoded.append(payload.data["image_b64"])
    payload.data = {
        "images_b64": encoded,
        "modality": "image",
        "finish_reason": "stop",
    }
    return payload


def generate_text_to_image_batch(
    payloads: list[StagePayload],
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
) -> list[StagePayload]:
    if not payloads:
        return []
    else:
        pass
    prompts = [payload.request.inputs for payload in payloads]
    options = [text_to_image_options(payload) for payload in payloads]
    first = options[0]
    signature = (
        first.width,
        first.height,
        first.num_inference_steps,
        first.guidance_scale,
        first.enable_cache_dit,
        cache_dit_batch_key(first.cache_dit_params),
    )
    if any(
        (
            sampling.width,
            sampling.height,
            sampling.num_inference_steps,
            sampling.guidance_scale,
            sampling.enable_cache_dit,
            cache_dit_batch_key(sampling.cache_dit_params),
        )
        != signature
        for sampling in options[1:]
    ):
        raise ValueError("SenseNova-U1 can only batch compatible T2I requests")
    else:
        pass

    prepare_cache_dit(
        model,
        enable_cache_dit=first.enable_cache_dit,
        params=first.cache_dit_params,
        steps=first.num_inference_steps,
        cfg_scale=first.guidance_scale,
    )

    logger.info(
        f"SenseNova-U1 T2I batch: size={len(payloads)}, "
        f"image_size={first.width}x{first.height}, "
        f"steps={first.num_inference_steps}, cfg={first.guidance_scale}"
    )
    with torch.inference_mode():
        images = model.t2i_generate(
            tokenizer,
            prompts,
            batch_size=len(payloads),
            thinking_backend=None,
            seed=[sampling.seed for sampling in options],
            image_size=(first.width, first.height),
            cfg_scale=first.guidance_scale,
            cfg_norm="none",
            timestep_shift=GENERATION_TIMESTEP_SHIFT,
            enable_timestep_shift=True,
            cfg_interval=(0.0, 1.0),
            num_steps=first.num_inference_steps,
            t_eps=GENERATION_TIME_EPSILON,
            think_mode=False,
        )
    expected_shape = (len(payloads), 3, first.height, first.width)
    if not isinstance(images, torch.Tensor) or images.shape != expected_shape:
        raise ValueError("SenseNova-U1 returned an invalid batched image tensor")
    else:
        pass
    return [
        encode_image(payload, images[index : index + 1], first.width, first.height)
        for index, payload in enumerate(payloads)
    ]


def generate_image_edit(
    payload: StagePayload, model: NEOChatModel, tokenizer: PreTrainedTokenizerBase
) -> StagePayload:
    inputs = payload.request.inputs
    prompt = inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SenseNova-U1 requires a non-empty image edit prompt")
    else:
        pass
    options = SenseNovaU1ImageEditSampling.from_params(payload.request.params)
    references = prepare_reference_images(inputs.get("image_b64"), options)
    width, height = resolve_edit_output_size(references[0].size, options)
    validate_output_dimensions(payload.request, width, height)
    prepare_cache_dit(
        model,
        enable_cache_dit=options.enable_cache_dit,
        params=options.cache_dit_params,
        steps=options.num_inference_steps,
        cfg_scale=options.guidance_scale,
        img_cfg_scale=options.img_cfg_scale,
        is_edit=True,
    )
    try:
        with torch.inference_mode():
            images = model.it2i_generate(
                tokenizer,
                prompt,
                references,
                batch_size=1,
                seed=options.seed,
                image_size=(width, height),
                cfg_scale=options.guidance_scale,
                img_cfg_scale=options.img_cfg_scale,
                cfg_norm="none",
                timestep_shift=GENERATION_TIMESTEP_SHIFT,
                enable_timestep_shift=True,
                cfg_interval=(0.0, 1.0),
                num_steps=options.num_inference_steps,
                t_eps=GENERATION_TIME_EPSILON,
                think_mode=False,
            )
    except torch.OutOfMemoryError as error:
        device = model.device
        if device.type not in ("cuda", "npu"):
            raise
        else:
            torch.get_device_module(device).empty_cache()
        memory_advice = "reduce references or output size"
        if options.do_resize:
            memory_advice += ", or lower input_max_pixels"
        else:
            pass
        if device.type == "cuda":
            memory_advice += (
                ", or lower generate.factory.dit_layerwise_resident_layers "
                "with layerwise offload enabled"
            )
        else:
            pass
        raise ValueError(
            f"{REFERENCE_MEMORY_ERROR_PREFIX}{len(references)} reference images exceeds "
            f"{device.type.upper()} memory; {memory_advice}"
        ) from error
    else:
        pass
    return encode_image(payload, images, width, height)


def unmount_cache_dit_after_failure(model: NEOChatModel) -> None:
    controller = model.__dict__.get("sensenova_cache_dit")
    if isinstance(controller, SenseNovaCacheDit):
        try:
            controller.unmount()
        except Exception:
            logger.exception(
                "Cache-DiT cleanup failed; the controller retained recovery state"
            )
    else:
        pass


def prepare_cache_dit(
    model: NEOChatModel,
    *,
    enable_cache_dit: bool | None,
    params: JsonValue,
    steps: int,
    cfg_scale: float,
    img_cfg_scale: float = 1.0,
    is_edit: bool = False,
) -> None:
    # note (Wes): Direct generation calls may bypass the factory controller.
    controller = model.__dict__.get("sensenova_cache_dit")
    if not isinstance(controller, SenseNovaCacheDit):
        return
    else:
        pass
    if not is_edit:
        branch_count = 2 if cfg_scale > 1 else 1
    else:
        branch_count = image_guidance_branch_count(cfg_scale, img_cfg_scale)
    controller.prepare(
        model.language_model.model,
        enabled=enable_cache_dit,
        params=params,
        steps=steps,
        branch_count=branch_count,
        cfg_interval=(0.0, 1.0),
    )
