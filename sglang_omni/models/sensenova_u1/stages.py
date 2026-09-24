# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1 native image-generation stage."""

from __future__ import annotations

import base64
import io
from typing import Any

from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1Sampling
from sglang_omni.proto.request import StagePayload


def generate_image(payload: StagePayload, model: Any, tokenizer: Any) -> StagePayload:
    """Run the source model's native T2I path and return a PNG payload."""
    import torch
    from PIL import Image

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
    if not isinstance(images, torch.Tensor) or images.shape != (
        1,
        3,
        options.height,
        options.width,
    ):
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
