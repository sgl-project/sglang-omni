# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1 native image-generation stage."""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any

from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
)
from sglang_omni.proto.request import StagePayload


def generate_image(payload: StagePayload, model: Any, tokenizer: Any) -> StagePayload:
    """Run the source model's native T2I or I2I path and return a PNG payload."""
    inputs = payload.request.inputs
    if isinstance(inputs, dict) and inputs.get("task") == "image_edit":
        return _generate_image_edit(payload, model, tokenizer)
    return _generate_text_to_image(payload, model, tokenizer)


def _generate_text_to_image(
    payload: StagePayload, model: Any, tokenizer: Any
) -> StagePayload:
    import torch

    prompt = payload.request.inputs
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SenseNova-U1 requires a non-empty text prompt")
    options = SenseNovaU1Sampling.from_params(payload.request.params)
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


def _generate_image_edit(
    payload: StagePayload, model: Any, tokenizer: Any
) -> StagePayload:
    import torch

    inputs = payload.request.inputs
    prompt = inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SenseNova-U1 requires a non-empty image edit prompt")
    reference = _decode_reference_image(inputs.get("image_b64"))
    options = SenseNovaU1ImageEditSampling.from_params(payload.request.params)
    with torch.inference_mode():
        images = model.it2i_generate(
            tokenizer,
            prompt,
            [reference],
            batch_size=1,
            seed=options.seed,
            image_size=(options.width, options.height),
            cfg_scale=options.guidance_scale,
            img_cfg_scale=options.img_cfg_scale,
            cfg_norm="none",
            timestep_shift=1.0,
            enable_timestep_shift=True,
            cfg_interval=(0.0, 1.0),
            num_steps=options.num_inference_steps,
            t_eps=0.02,
            think_mode=False,
        )
    return _encode_image(payload, images, options.width, options.height)


def _decode_reference_image(value: Any):
    from PIL import Image

    if not isinstance(value, str) or not value:
        raise ValueError("SenseNova-U1 requires one reference image")
    try:
        image_bytes = base64.b64decode(value, validate=True)
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (binascii.Error, OSError, ValueError) as exc:
        raise ValueError("SenseNova-U1 received an invalid reference image") from exc
    return image


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


def create_generation_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = "bfloat16",
):
    """Load model once per stage process; requests execute serially."""
    from transformers import AutoModel, AutoTokenizer

    from sglang_omni.models.sensenova_u1.neo_unify import register
    from sglang_omni.models.weight_loader import resolve_dtype
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    from sglang_omni.utils.device import resolve_concrete_device

    register()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=resolve_dtype(dtype)
    ).eval()
    model = model.to(resolve_concrete_device(device, gpu_id))
    return SimpleScheduler(lambda payload: generate_image(payload, model, tokenizer))
