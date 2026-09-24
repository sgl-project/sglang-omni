# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1 native image-generation stage."""

from __future__ import annotations

import base64
import binascii
import io
import logging
import time
from typing import Any

from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
)
from sglang_omni.proto.request import StagePayload

logger = logging.getLogger(__name__)

DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048
MIN_INPUT_MAX_PIXELS = 512 * 512


def generate_image(payload: StagePayload, model: Any, tokenizer: Any) -> StagePayload:
    """Run the source model's native T2I or I2I path and return a PNG payload."""
    inputs = payload.request.inputs
    if isinstance(inputs, dict) and inputs.get("task") == "image_edit":
        return _generate_image_edit(payload, model, tokenizer)
    return _generate_text_to_image(payload, model, tokenizer)


def generate_images(
    payloads: list[StagePayload], model: Any, tokenizer: Any
) -> list[StagePayload]:
    """Batch compatible T2I requests while preserving request order."""
    if not payloads:
        return []

    results: list[StagePayload | None] = [None] * len(payloads)
    groups: dict[tuple[int, int, int, float], list[tuple[int, StagePayload]]] = {}
    for index, payload in enumerate(payloads):
        inputs = payload.request.inputs
        if isinstance(inputs, dict) and inputs.get("task") == "image_edit":
            results[index] = _generate_image_edit(payload, model, tokenizer)
            continue
        options = _text_to_image_options(payload)
        signature = (
            options.width,
            options.height,
            options.num_inference_steps,
            options.guidance_scale,
        )
        groups.setdefault(signature, []).append((index, payload))

    for indexed_payloads in groups.values():
        indexes, compatible_payloads = zip(*indexed_payloads)
        if len(compatible_payloads) == 1:
            batch_results = [
                _generate_text_to_image(compatible_payloads[0], model, tokenizer)
            ]
        else:
            batch_results = _generate_text_to_image_batch(
                list(compatible_payloads), model, tokenizer
            )
        for index, result in zip(indexes, batch_results):
            results[index] = result

    if any(result is None for result in results):
        raise RuntimeError("SenseNova-U1 failed to produce every batched result")
    return [result for result in results if result is not None]


def _text_to_image_options(payload: StagePayload) -> SenseNovaU1Sampling:
    prompt = payload.request.inputs
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SenseNova-U1 requires a non-empty text prompt")
    return SenseNovaU1Sampling.from_params(payload.request.params)


def _generate_text_to_image(
    payload: StagePayload, model: Any, tokenizer: Any
) -> StagePayload:
    import torch

    prompt = payload.request.inputs
    options = _text_to_image_options(payload)
    with torch.inference_mode():
        images = model.t2i_generate(
            tokenizer,
            prompt,
            batch_size=1,
            thinking_backend=None,
            seed=options.seed,
            image_size=(options.width, options.height),
            cfg_scale=options.guidance_scale,
            cfg_norm="none",
            timestep_shift=3.0,
            enable_timestep_shift=True,
            cfg_interval=(0.0, 1.0),
            num_steps=options.num_inference_steps,
            t_eps=0.02,
            think_mode=False,
        )
    return _encode_image(payload, images, options.width, options.height)


def _generate_text_to_image_batch(
    payloads: list[StagePayload], model: Any, tokenizer: Any
) -> list[StagePayload]:
    import torch

    if not payloads:
        return []
    prompts = [payload.request.inputs for payload in payloads]
    options = [_text_to_image_options(payload) for payload in payloads]
    first = options[0]
    signature = (
        first.width,
        first.height,
        first.num_inference_steps,
        first.guidance_scale,
    )
    if any(
        (
            item.width,
            item.height,
            item.num_inference_steps,
            item.guidance_scale,
        )
        != signature
        for item in options[1:]
    ):
        raise ValueError("SenseNova-U1 can only batch compatible T2I requests")

    logger.info(
        "SenseNova-U1 T2I batch: size=%d, image_size=%dx%d, steps=%d, cfg=%s",
        len(payloads),
        first.width,
        first.height,
        first.num_inference_steps,
        first.guidance_scale,
    )
    with torch.inference_mode():
        images = model.t2i_generate(
            tokenizer,
            prompts,
            batch_size=len(payloads),
            thinking_backend=None,
            seed=[item.seed for item in options],
            image_size=(first.width, first.height),
            cfg_scale=first.guidance_scale,
            cfg_norm="none",
            timestep_shift=3.0,
            enable_timestep_shift=True,
            cfg_interval=(0.0, 1.0),
            num_steps=first.num_inference_steps,
            t_eps=0.02,
            think_mode=False,
        )
    return _encode_images(payloads, images, first.width, first.height)


def _generate_image_edit(
    payload: StagePayload, model: Any, tokenizer: Any
) -> StagePayload:
    import torch

    inputs = payload.request.inputs
    prompt = inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SenseNova-U1 requires a non-empty image edit prompt")
    options = SenseNovaU1ImageEditSampling.from_params(payload.request.params)
    references = _prepare_reference_images(inputs.get("image_b64"), options)
    width, height = _resolve_edit_output_size(references, options)
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
            timestep_shift=3.0,
            enable_timestep_shift=True,
            cfg_interval=(0.0, 1.0),
            num_steps=options.num_inference_steps,
            t_eps=0.02,
            think_mode=False,
        )
    return _encode_image(payload, images, width, height)


def _decode_reference_images(value: Any) -> list[Any]:
    from PIL import Image

    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise ValueError("SenseNova-U1 requires at least one reference image")
    images = []
    for item in values:
        try:
            image_bytes = base64.b64decode(item, validate=True)
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except (binascii.Error, OSError, ValueError) as exc:
            raise ValueError(
                "SenseNova-U1 received an invalid reference image"
            ) from exc
        images.append(image)
    return images


def _auto_input_max_pixels(num_images: int) -> int:
    if num_images <= 0:
        raise ValueError("SenseNova-U1 requires at least one reference image")
    if num_images <= 2:
        return DEFAULT_INPUT_MAX_PIXELS
    return max(MIN_INPUT_MAX_PIXELS, 2 * DEFAULT_INPUT_MAX_PIXELS // num_images)


def _prepare_reference_images(
    value: Any, options: SenseNovaU1ImageEditSampling
) -> list[Any]:
    from PIL import Image

    from sglang_omni.models.sensenova_u1.neo_unify.utils import smart_resize

    images = _decode_reference_images(value)
    input_max_pixels = options.input_max_pixels or _auto_input_max_pixels(len(images))
    prepared = []
    for image in images:
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")
        if options.do_resize:
            height, width = smart_resize(
                height=image.height,
                width=image.width,
                factor=32,
                min_pixels=input_max_pixels,
                max_pixels=input_max_pixels,
            )
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
        prepared.append(image)
    return prepared


def _resolve_edit_output_size(
    references: list[Any], options: SenseNovaU1ImageEditSampling
) -> tuple[int, int]:
    from sglang_omni.models.sensenova_u1.neo_unify.utils import smart_resize

    if options.size_explicit:
        return options.width, options.height
    target_pixels = options.width * options.height
    height, width = smart_resize(
        height=references[0].height,
        width=references[0].width,
        factor=32,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return width, height


def _encode_image(
    payload: StagePayload, images: Any, width: int, height: int
) -> StagePayload:
    import torch
    from PIL import Image

    if not isinstance(images, torch.Tensor) or images.shape != (1, 3, height, width):
        raise ValueError("SenseNova-U1 returned an invalid image tensor")
    image = ((images[0].float() + 1.0) * 127.5).clamp(0, 255)
    image = image.permute(1, 2, 0).byte().cpu().numpy()
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    payload.data = {
        "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "modality": "image",
        "finish_reason": "stop",
    }
    return payload


def _encode_images(
    payloads: list[StagePayload], images: Any, width: int, height: int
) -> list[StagePayload]:
    import torch

    expected_shape = (len(payloads), 3, height, width)
    if not isinstance(images, torch.Tensor) or images.shape != expected_shape:
        raise ValueError("SenseNova-U1 returned an invalid batched image tensor")
    return [
        _encode_image(payload, images[index : index + 1], width, height)
        for index, payload in enumerate(payloads)
    ]


def image_generation_request_cost(payload: StagePayload) -> int:
    """Estimate denoise work as pixel-steps, including the CFG branch."""
    inputs = payload.request.inputs
    if isinstance(inputs, dict) and inputs.get("task") == "image_edit":
        options = SenseNovaU1ImageEditSampling.from_params(payload.request.params)
        branches = 1 + int(options.guidance_scale > 1) + int(options.img_cfg_scale > 1)
    else:
        options = _text_to_image_options(payload)
        branches = 1 + int(options.guidance_scale > 1)
    return options.width * options.height * options.num_inference_steps * branches


def image_generation_batch_key(payload: StagePayload) -> tuple[Any, ...]:
    """Return a compatibility key; I2I requests always remain single-item."""
    inputs = payload.request.inputs
    if isinstance(inputs, dict) and inputs.get("task") == "image_edit":
        return ("image_edit", payload.request_id)
    options = _text_to_image_options(payload)
    return (
        "text_to_image",
        options.width,
        options.height,
        options.num_inference_steps,
        options.guidance_scale,
    )


def create_generation_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = "bfloat16",
    max_batch_size: int = 1,
    max_batch_wait_ms: float = 0,
    max_batch_cost: int | None = None,
):
    """Load the model once and optionally batch compatible T2I requests."""
    from transformers import AutoModel, AutoTokenizer

    from sglang_omni.models.sensenova_u1.neo_unify import register
    from sglang_omni.models.weight_loader import resolve_dtype
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    from sglang_omni.utils.device import resolve_concrete_device

    load_started = time.perf_counter()
    register()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=resolve_dtype(dtype)
    ).eval()
    resolved_device = resolve_concrete_device(device, gpu_id)
    model = model.to(resolved_device)
    logger.info(
        "SenseNova-U1 loaded on %s with dtype=%s in %.2f seconds",
        resolved_device,
        dtype,
        time.perf_counter() - load_started,
    )
    if resolved_device.type == "npu":
        import torch

        from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import (
            npu_fia_available,
            npu_swiglu_available,
        )

        logger.info(
            "SenseNova-U1 NPU operators: FIA=%s, SwiGLU=%s",
            npu_fia_available(),
            npu_swiglu_available(),
        )
        npu = getattr(torch, "npu", None)
        if npu is not None and all(
            hasattr(npu, name) for name in ("memory_allocated", "memory_reserved")
        ):
            logger.info(
                "SenseNova-U1 NPU memory after load: allocated=%d, reserved=%d",
                npu.memory_allocated(resolved_device),
                npu.memory_reserved(resolved_device),
            )
    batch_enabled = max_batch_size > 1

    def _generate(payload: StagePayload) -> StagePayload:
        return generate_image(payload, model, tokenizer)

    def _generate_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        return generate_images(payloads, model, tokenizer)

    return SimpleScheduler(
        _generate,
        batch_compute_fn=_generate_batch if batch_enabled else None,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_key_fn=image_generation_batch_key if batch_enabled else None,
        request_cost_fn=image_generation_request_cost if batch_enabled else None,
        max_batch_cost=max_batch_cost if batch_enabled else None,
    )
