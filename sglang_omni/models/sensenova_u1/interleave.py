# SPDX-License-Identifier: Apache-2.0
"""SenseNova U1 interleaved text-image generation runner.

This M4 path keeps the official NEOChatModel interleave loop inside an
SGLang-Omni stage. The loop autoregressively emits text, switches into U1
flow-matching when the model emits an image start token, re-encodes the
generated image, and then continues text generation from the updated KV cache.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from sglang_omni.models.sensenova_u1.flow_matching import (
    SenseNovaU1FlowMatchingRunner,
    pil_to_data_url,
    u1_tensor_to_pil,
)
from sglang_omni.models.sensenova_u1.hf_runner import (
    DEFAULT_MODEL_DIR,
    DEFAULT_VENDOR_ROOT,
    _coerce_image,
    _extract_request,
    _extract_text_and_images_from_content,
    _official_block_mask_scope,
)
from sglang_omni.proto import StagePayload


DEFAULT_INTERLEAVE_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with both text and images. "
    "You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a <think></think> block "
    "and place all reasoning inside it. You MUST interleave text with generated images "
    "using tags like <image1>, <image2>. Images can ONLY be generated between <think> and "
    "</think>, and may be referenced in the final answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer without "
    "reasoning. Do not use tags like <image1>, <image2>; present any images naturally "
    "alongside the text.\n\n"
    "After the think block, always provide a concise, user-facing final answer. The "
    "answer may include text, images, or both. Match the user's language in both "
    "reasoning and the final answer."
)


@dataclass(frozen=True, slots=True)
class InterleaveRequestParams:
    prompt: str
    image_size: tuple[int, int] = (256, 256)
    cfg_scale: float = 1.0
    img_cfg_scale: float = 1.0
    cfg_norm: str = "none"
    timestep_shift: float = 1.0
    enable_timestep_shift: bool = True
    cfg_interval: tuple[float, float] = (0.0, 1.0)
    num_steps: int = 2
    max_images: int = 1
    max_new_tokens: int = 512
    t_eps: float = 0.05
    think_mode: bool = True
    seed: int = 20260813
    system_message: str = DEFAULT_INTERLEAVE_SYSTEM_MESSAGE


@dataclass(slots=True)
class InterleaveResult:
    text: str
    token_ids: list[int]
    images: list[Image.Image]
    stats: dict[str, Any]
    elapsed_s: float


def _stage_params(params: dict[str, Any], stage_name: str) -> dict[str, Any]:
    merged = dict(params)
    stage_params = params.get("stage_params")
    if isinstance(stage_params, dict):
        specific = stage_params.get(stage_name)
        if isinstance(specific, dict):
            merged.update(specific)
    stage_sampling = params.get("stage_sampling")
    if isinstance(stage_sampling, dict):
        specific_sampling = stage_sampling.get(stage_name)
        if isinstance(specific_sampling, dict):
            merged.update(specific_sampling)
    return merged


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _cfg_interval(value: Any) -> tuple[float, float]:
    if value is None:
        return (0.0, 1.0)
    interval = tuple(float(x) for x in value)
    if len(interval) != 2:
        raise ValueError("cfg_interval must contain exactly two floats.")
    return (interval[0], interval[1])


def _extract_prompt_images_system(inputs: Any) -> tuple[str, list[Any], str | None]:
    images: list[Any] = []
    system_message: str | None = None

    if isinstance(inputs, str):
        return inputs, images, system_message

    if isinstance(inputs, dict):
        images.extend(_coerce_image(img) for img in inputs.get("images") or [])
        if "image" in inputs:
            images.append(_coerce_image(inputs["image"]))
        if "system_message" in inputs:
            system_message = str(inputs["system_message"])
        if "prompt" in inputs:
            return str(inputs["prompt"]), images, system_message
        if "question" in inputs:
            return str(inputs["question"]), images, system_message
        if "messages" in inputs:
            inputs = inputs["messages"]
        else:
            return str(inputs), images, system_message

    if isinstance(inputs, list):
        prompt_parts: list[str] = []
        for msg in inputs:
            if not isinstance(msg, dict):
                prompt_parts.append(str(msg))
                continue
            role = str(msg.get("role", "user"))
            text, msg_images = _extract_text_and_images_from_content(
                msg.get("content", "")
            )
            images.extend(_coerce_image(img) for img in msg_images)
            if role == "system" and text:
                system_message = text
            elif role == "user" and text:
                prompt_parts.append(text)
        return "\n".join(part for part in prompt_parts if part), images, system_message

    return str(inputs), images, system_message


def _params_from_payload(
    inputs: Any,
    params: dict[str, Any],
) -> tuple[InterleaveRequestParams, list[Any]]:
    prompt, images, system_from_messages = _extract_prompt_images_system(inputs)
    params = _stage_params(params, "u1_interleave")

    data = inputs if isinstance(inputs, dict) else {}
    image_config = params.get("image_config")
    if not isinstance(image_config, dict):
        image_config = data.get("image_config") if isinstance(data, dict) else {}
    if not isinstance(image_config, dict):
        image_config = {}

    width = _positive_int(
        params.get("width", params.get("image_width", image_config.get("width"))),
        256,
    )
    height = _positive_int(
        params.get("height", params.get("image_height", image_config.get("height"))),
        256,
    )

    chat_template_kwargs = params.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    think_mode = bool(params.get("think_mode", True))
    if "enable_thinking" in chat_template_kwargs:
        think_mode = bool(chat_template_kwargs["enable_thinking"])

    seed_value = params.get("seed", image_config.get("seed", 20260813))
    system_message = str(
        params.get(
            "system_message",
            system_from_messages or DEFAULT_INTERLEAVE_SYSTEM_MESSAGE,
        )
    )

    request = InterleaveRequestParams(
        prompt=str(params.get("prompt", prompt)),
        image_size=(width, height),
        cfg_scale=float(params.get("cfg_scale", 1.0)),
        img_cfg_scale=float(params.get("img_cfg_scale", 1.0)),
        cfg_norm=str(params.get("cfg_norm", "none")),
        timestep_shift=float(params.get("timestep_shift", 1.0)),
        enable_timestep_shift=bool(params.get("enable_timestep_shift", True)),
        cfg_interval=_cfg_interval(params.get("cfg_interval", (0.0, 1.0))),
        num_steps=int(params.get("num_steps", 2)),
        max_images=int(params.get("max_images", 1)),
        max_new_tokens=int(params.get("max_new_tokens", params.get("max_tokens", 512))),
        t_eps=float(params.get("t_eps", 0.05)),
        think_mode=think_mode,
        seed=int(seed_value),
        system_message=system_message,
    )
    return request, images


def interleave_tensors_to_pil(image_tensors: list[torch.Tensor]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for tensor in image_tensors:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        images.extend(u1_tensor_to_pil(tensor.detach()))
    return images


def interleave_segments(text: str, image_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts = text.split("<image>")
    segments: list[dict[str, Any]] = []
    for idx, part in enumerate(parts):
        if part:
            segments.append({"type": "text", "text": part})
        if idx < len(parts) - 1:
            if idx < len(image_items):
                segments.append({"type": "image", "image": image_items[idx]})
            else:
                segments.append({"type": "image", "image": None})
    return segments


class SenseNovaU1InterleaveRunner(SenseNovaU1FlowMatchingRunner):
    """End-to-end U1 text-image-text generation runner."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_DIR,
        *,
        vendor_root: str | None = None,
        device: str = "cuda:0",
        dtype: str | torch.dtype = "bfloat16",
        attn_backend: str = "auto",
        load_with_info: bool = False,
    ) -> None:
        super().__init__(
            model_path=model_path,
            vendor_root=vendor_root or DEFAULT_VENDOR_ROOT,
            device=device,
            dtype=dtype,
            attn_backend=attn_backend,
            load_with_info=load_with_info,
        )

    @torch.inference_mode()
    def generate_interleave(
        self,
        request: InterleaveRequestParams,
        images: list[Any] | None = None,
        *,
        use_official_hybrid_mask: bool = False,
    ) -> InterleaveResult:
        assert self.model is not None and self.tokenizer is not None
        ctx = _official_block_mask_scope() if use_official_hybrid_mask else nullcontext()
        generation_config = SimpleNamespace(max_new_tokens=request.max_new_tokens)
        start = time.perf_counter()
        with ctx:
            text, image_tensors = self.model.interleave_gen(
                self.tokenizer,
                request.prompt,
                images=[_coerce_image(image) for image in (images or [])],
                generation_config=generation_config,
                cfg_scale=request.cfg_scale,
                img_cfg_scale=request.img_cfg_scale,
                cfg_norm=request.cfg_norm,
                max_images=request.max_images,
                enable_timestep_shift=request.enable_timestep_shift,
                timestep_shift=request.timestep_shift,
                image_size=request.image_size,
                num_steps=request.num_steps,
                cfg_interval=request.cfg_interval,
                t_eps=request.t_eps,
                verbose=False,
                system_message=request.system_message,
                think_mode=request.think_mode,
                seed=request.seed,
            )
        elapsed = time.perf_counter() - start
        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
        stats = dict(getattr(self.model, "last_interleave_generation_stats", {}) or {})
        return InterleaveResult(
            text=str(text),
            token_ids=[int(x) for x in token_ids],
            images=interleave_tensors_to_pil(list(image_tensors)),
            stats=stats,
            elapsed_s=elapsed,
        )

    def complete_payload(self, payload: StagePayload) -> dict[str, Any]:
        inputs, params, request_id = _extract_request(payload)
        request, images = _params_from_payload(inputs, params)
        result = self.generate_interleave(request, images)
        image_items = [
            {
                "type": "image",
                "format": "png",
                "data": pil_to_data_url(image),
                "width": image.width,
                "height": image.height,
                "index": idx,
            }
            for idx, image in enumerate(result.images)
        ]
        prompt_token_ids = self.tokenizer(  # type: ignore[operator]
            request.prompt,
            add_special_tokens=False,
        )["input_ids"]
        rollout = {
            "type": "sensenova_u1_interleave",
            "segments": interleave_segments(result.text, image_items),
            "images": image_items,
            "text": result.text,
            "token_ids": result.token_ids,
            "stats": result.stats,
            "request": asdict(request),
            "backend": "hf_compatible_interleave_fallback",
        }
        return {
            "request_id": request_id,
            "text": result.text,
            "token_ids": result.token_ids,
            "images": image_items,
            "omni_rollout": rollout,
            "finish_reason": result.stats.get("stop_reason", "stop"),
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(result.token_ids),
                "total_tokens": len(prompt_token_ids) + len(result.token_ids),
                "engine_time_s": result.elapsed_s,
                "generated_images": len(result.images),
            },
            "stage_name": "u1_interleave",
            "backend": "hf_compatible_interleave_fallback",
        }


__all__ = [
    "DEFAULT_INTERLEAVE_SYSTEM_MESSAGE",
    "InterleaveRequestParams",
    "InterleaveResult",
    "SenseNovaU1InterleaveRunner",
    "interleave_segments",
    "interleave_tensors_to_pil",
]
