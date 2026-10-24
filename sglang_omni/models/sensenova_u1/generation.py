# SPDX-License-Identifier: Apache-2.0
"""Generate SenseNova images individually or in compatible batches."""

from __future__ import annotations

import logging

import torch
from pydantic import JsonValue
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.admission import validate_output_dimensions
from sglang_omni.models.sensenova_u1.cache_dit import SenseNovaCacheDit
from sglang_omni.models.sensenova_u1.image_processing import (
    encode_image,
    prepare_reference_images,
    resolve_edit_output_size,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_conditioned_generation import (
    it2i_generate,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.neo_unify.text_to_image_generation import (
    t2i_generate,
)
from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
    image_guidance_branch_count,
)
from sglang_omni.pipeline.umm import UMMRequestError
from sglang_omni.proto.request import StagePayload

logger = logging.getLogger(__name__)


def generate_image(
    payload: StagePayload,
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
    cache_dit: SenseNovaCacheDit,
) -> StagePayload:
    """Run the source model's native T2I or I2I path and return a PNG payload."""
    inputs = payload.request.inputs
    try:
        if isinstance(inputs, dict) and inputs.get("task") == "image_edit":
            return generate_image_edit(payload, model, tokenizer, cache_dit)
        else:
            pass
        return generate_text_to_image(payload, model, tokenizer, cache_dit)
    except Exception:
        unmount_cache_dit_after_failure(cache_dit)
        raise


def generate_images(
    payloads: list[StagePayload],
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
    cache_dit: SenseNovaCacheDit,
) -> list[StagePayload]:
    """Generate one scheduler batch, which batch keys limit to matching single-output T2I."""
    try:
        return generate_text_to_image_batch(payloads, model, tokenizer, cache_dit)
    except Exception:
        unmount_cache_dit_after_failure(cache_dit)
        raise


def text_to_image_options(payload: StagePayload) -> SenseNovaU1Sampling:
    prompt = payload.request.inputs
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SenseNova-U1 requires a non-empty text prompt")
    else:
        pass
    return SenseNovaU1Sampling.from_params(payload.request.params)


def generate_text_to_image(
    payload: StagePayload,
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
    cache_dit: SenseNovaCacheDit,
) -> StagePayload:
    prompt = payload.request.inputs
    options = text_to_image_options(payload)
    prepare_cache_dit(
        cache_dit,
        model,
        enable_cache_dit=options.enable_cache_dit,
        params=options.cache_dit_params,
        steps=options.num_inference_steps,
        cfg_scale=options.guidance_scale,
    )
    with torch.inference_mode():
        images = t2i_generate(
            model,
            tokenizer,
            prompt,
            sampling=options,
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
    encoded_images = []
    for index in range(options.n):
        encode_image(payload, images[index : index + 1], options.width, options.height)
        encoded_images.append(payload.data["image_b64"])
    payload.data = {
        "images_b64": encoded_images,
        "modality": "image",
        "finish_reason": "stop",
    }
    return payload


def generate_text_to_image_batch(
    payloads: list[StagePayload],
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
    cache_dit: SenseNovaCacheDit,
) -> list[StagePayload]:
    prompts = [payload.request.inputs for payload in payloads]
    options = [text_to_image_options(payload) for payload in payloads]
    first = options[0]
    assert all(
        sampling.batch_key() == first.batch_key() for sampling in options
    ), "image_generation_batch_key admits only matching T2I requests to a batch"

    prepare_cache_dit(
        cache_dit,
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
        images = t2i_generate(
            model,
            tokenizer,
            prompts,
            sampling=first,
            seeds=[sampling.seed for sampling in options],
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
    payload: StagePayload,
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
    cache_dit: SenseNovaCacheDit,
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
        cache_dit,
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
            images = it2i_generate(
                model,
                tokenizer,
                prompt,
                references,
                sampling=options.model_copy(update={"width": width, "height": height}),
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
        # note (Wes): UMMRequestError's prefix makes every endpoint report this as a client error.
        raise UMMRequestError(
            f"SenseNova-U1 I2I with {len(references)} reference images exceeds "
            f"{device.type.upper()} memory; {memory_advice}"
        ) from error
    else:
        pass
    return encode_image(payload, images, width, height)


def unmount_cache_dit_after_failure(cache_dit: SenseNovaCacheDit) -> None:
    try:
        cache_dit.unmount()
    except Exception:
        logger.exception(
            "Cache-DiT cleanup failed; the controller retained recovery state"
        )


def prepare_cache_dit(
    cache_dit: SenseNovaCacheDit,
    model: NEOChatModel,
    *,
    enable_cache_dit: bool | None,
    params: JsonValue,
    steps: int,
    cfg_scale: float,
    img_cfg_scale: float = 1.0,
    is_edit: bool = False,
) -> None:
    if not is_edit:
        branch_count = 2 if cfg_scale > 1 else 1
    else:
        branch_count = image_guidance_branch_count(cfg_scale, img_cfg_scale)
    cache_dit.prepare(
        model.language_model.model,
        enabled=enable_cache_dit,
        params=params,
        steps=steps,
        branch_count=branch_count,
        cfg_interval=(0.0, 1.0),
    )
