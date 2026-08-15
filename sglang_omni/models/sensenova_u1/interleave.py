# SPDX-License-Identifier: Apache-2.0
"""SenseNova U1 interleaved text-image generation runner."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from PIL import Image

from sglang_omni.models.sensenova_u1.flow_matching import (
    FlowRequestParams,
    NativeFlowPrefix,
    SenseNovaU1FlowMatchingRunner,
    _build_neo_prompt,
    pil_to_data_url,
    u1_tensor_to_pil,
)
from sglang_omni.models.sensenova_u1.hf_runner import (
    DEFAULT_MODEL_DIR,
    DEFAULT_VENDOR_ROOT,
    _coerce_image,
    _extract_request,
    _extract_text_and_images_from_content,
)
from sglang_omni.models.sensenova_u1.limits import (
    U1_MAX_TOTAL_TOKENS,
    generated_image_span_token_count,
    parse_int_param,
    validate_image_count,
    validate_image_size,
    validate_input_image_count,
    validate_max_new_tokens,
    validate_num_steps,
    validate_token_budget_components,
    validate_total_token_budget,
)
from sglang_omni.models.sensenova_u1.sglang_model import (
    _blocked_hf_modeling_modules,
    assert_no_hf_modeling_imported,
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


def _cfg_interval(value: Any) -> tuple[float, float]:
    if value is None:
        return (0.0, 1.0)
    try:
        interval = tuple(float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cfg_interval must contain exactly two floats.") from exc
    if len(interval) != 2:
        raise ValueError("cfg_interval must contain exactly two floats.")
    return (interval[0], interval[1])


def _extract_prompt_images_system(inputs: Any) -> tuple[str, list[Any], str | None]:
    images: list[Any] = []
    system_message: str | None = None

    if isinstance(inputs, str):
        return inputs, images, system_message

    if isinstance(inputs, dict):
        raw_images = inputs.get("images")
        if raw_images is None:
            raw_images = []
        if not isinstance(raw_images, (list, tuple)):
            raise ValueError("images must be a list or tuple.")
        images.extend(_coerce_image(img) for img in raw_images)
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

    width_value = params.get(
        "width",
        params.get("image_width", image_config.get("width", 256)),
    )
    height_value = params.get(
        "height",
        params.get("image_height", image_config.get("height", 256)),
    )
    width, height = validate_image_size(
        256 if width_value is None else width_value,
        256 if height_value is None else height_value,
    )
    validate_input_image_count(images)

    chat_template_kwargs = params.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    think_mode = bool(params.get("think_mode", True))
    if "enable_thinking" in chat_template_kwargs:
        think_mode = bool(chat_template_kwargs["enable_thinking"])

    seed_value = params.get("seed", image_config.get("seed", 20260813))
    num_steps = validate_num_steps(params.get("num_steps", 2))
    max_images = validate_image_count(
        params.get("max_images", 1),
        name="max_images",
    )
    max_new_tokens = validate_max_new_tokens(
        params.get("max_new_tokens", params.get("max_tokens", 512))
    )
    validate_total_token_budget(
        image_size=(width, height),
        image_count=max_images,
        max_new_tokens=max_new_tokens,
    )
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
        num_steps=num_steps,
        max_images=max_images,
        max_new_tokens=max_new_tokens,
        t_eps=float(params.get("t_eps", 0.05)),
        think_mode=think_mode,
        seed=parse_int_param(seed_value, name="seed"),
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


def _apply_prompt_image_prefix(prompt: str, image_count: int) -> str:
    prompt_image_count = prompt.count("<image>")
    if image_count > prompt_image_count:
        prompt = "<image>\n" * (image_count - prompt_image_count) + prompt
    return prompt


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
        max_total_tokens: int = U1_MAX_TOTAL_TOKENS,
        eager_prefix_cache_max_entries: int = 4,
        eager_decode_graph_cache_max_entries: int = 2,
        eager_decode_graph_max_captures: int = 4,
        eager_prefix_cache_max_tokens: int = 2048,
        eager_decode_graph_max_total_tokens: int = 1024,
    ) -> None:
        super().__init__(
            model_path=model_path,
            vendor_root=vendor_root or DEFAULT_VENDOR_ROOT,
            device=device,
            dtype=dtype,
            attn_backend=attn_backend,
            load_with_info=load_with_info,
            max_total_tokens=max_total_tokens,
            eager_prefix_cache_max_entries=eager_prefix_cache_max_entries,
            eager_decode_graph_cache_max_entries=(
                eager_decode_graph_cache_max_entries
            ),
            eager_decode_graph_max_captures=(
                eager_decode_graph_max_captures
            ),
            eager_prefix_cache_max_tokens=eager_prefix_cache_max_tokens,
            eager_decode_graph_max_total_tokens=(
                eager_decode_graph_max_total_tokens
            ),
        )

    def _build_condition_prefix(
        self,
        *,
        prompt: str,
        images: list[Any],
        generated_image_tensors: list[torch.Tensor] | None = None,
        system_message: str,
        think_mode: bool,
        assistant_append: str | None = None,
        prompt_image_count: int | None = None,
        reserved_text_tokens: int = 0,
        reserved_image_tokens: int = 0,
    ) -> NativeFlowPrefix:
        prompt_image_count = len(images) if prompt_image_count is None else int(prompt_image_count)
        prompt = _apply_prompt_image_prefix(prompt, prompt_image_count)
        pixel_values, grid_hw = self._load_interleave_prefix_images(
            images,
            generated_image_tensors or [],
        )
        if assistant_append is None:
            assistant_append = None if think_mode else "<think>\n\n</think>\n\n"
        query = _build_neo_prompt(
            user_text=prompt,
            system_message=system_message,
            assistant_append=assistant_append,
        )
        query = self._replace_image_tokens(query, grid_hw)
        input_ids = self.tokenizer(query, return_tensors="pt")["input_ids"][0]
        validate_token_budget_components(
            prefix_tokens=int(input_ids.numel()),
            text_tokens=int(reserved_text_tokens),
            image_tokens=int(reserved_image_tokens),
            max_total_tokens=self.max_total_tokens,
        )
        indexes = self._get_thw_indexes(input_ids, grid_hw)
        image_token_tag = input_ids == self.img_context_token_id
        if pixel_values is not None and grid_hw is not None:
            input_embeds = self.executor.compose_input_embeds(
                input_ids=input_ids,
                image_token_tag=image_token_tag,
                pixel_values=pixel_values,
                grid_hw=grid_hw,
            )
            if input_embeds is None:
                raise RuntimeError("native interleave image compose returned no embeds")
        else:
            input_embeds = self._token_embeds(input_ids)
        return NativeFlowPrefix(
            input_ids=input_ids.to(dtype=torch.long),
            indexes=indexes.to(dtype=torch.long),
            image_token_tag=image_token_tag.to(dtype=torch.bool),
            input_embeds=input_embeds.detach(),
            cache_extra_key=None,
            cache_insert_log={
                "skipped": True,
                "reason": "native_interleave_text_state_machine_no_prefix_insert",
                "prefix_tokens": int(input_ids.numel()),
                "image_token_count": int(image_token_tag.sum().item()),
            },
        )

    def _generated_tensor_to_native_pixels(
        self,
        image_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = image_tensor.detach()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(
                "generated interleave image tensor must have shape [3,H,W] or [1,3,H,W]"
            )
        pred_img = tensor.unsqueeze(0).to(
            device=self.torch_device,
            dtype=torch.bfloat16,
        )
        raw_img = pred_img * 0.5 + 0.5
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=raw_img.dtype,
            device=raw_img.device,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=raw_img.dtype,
            device=raw_img.device,
        ).view(1, 3, 1, 1)
        pixel_values = (raw_img - mean) / std
        c, h, w = pixel_values[0].shape
        ps = self.patch_size
        if h % ps != 0 or w % ps != 0:
            raise ValueError(
                f"generated image shape {(h, w)} must be divisible by patch_size={ps}"
            )
        grid_h = h // ps
        grid_w = w // ps
        flat = (
            pixel_values[0]
            .view(c, grid_h, ps, grid_w, ps)
            .permute(1, 3, 0, 2, 4)
            .reshape(grid_h * grid_w, c * ps**2)
        )
        return (
            flat.cpu(),
            torch.tensor([[grid_h, grid_w]], dtype=torch.long),
        )

    def _load_interleave_prefix_images(
        self,
        input_images: list[Any],
        generated_image_tensors: list[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pixel_parts: list[torch.Tensor] = []
        grid_parts: list[torch.Tensor] = []
        if input_images:
            pixel_values, grid_hw = self._load_input_images(input_images)
            if pixel_values is not None and grid_hw is not None:
                pixel_parts.append(pixel_values.cpu())
                grid_parts.append(grid_hw.cpu())
        for image_tensor in generated_image_tensors:
            pixel_values, grid_hw = self._generated_tensor_to_native_pixels(image_tensor)
            pixel_parts.append(pixel_values)
            grid_parts.append(grid_hw)
        if not pixel_parts:
            return None, None
        return torch.cat(pixel_parts, dim=0), torch.cat(grid_parts, dim=0)

    def _generate_text_until_boundary(
        self,
        *,
        prefix: NativeFlowPrefix,
        max_new_tokens: int,
        suppress_token_ids: list[int] | None = None,
    ) -> tuple[str, list[int], dict[str, Any]]:
        eos_token_id = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        max_new_tokens = max(int(max_new_tokens), 0)
        if max_new_tokens == 0:
            return "", [], {
                "stop_reason": "max_new_tokens",
                "generated_tokens": 0,
                "raw_generated_token_ids": [],
                "terminal_token_id": None,
                "eos_token_id": eos_token_id,
                "img_start_token_id": self.img_start_token_id,
                "native_decode_stats": None,
                "native_decode_mode": self._native_interleave_text_decode_mode(),
            }

        decode_mode = self._native_interleave_text_decode_mode()
        if decode_mode == "eager_text_decode":
            decode_result = self.executor.run_eager_text_decode(
                input_ids=prefix.input_ids,
                indexes=prefix.indexes,
                image_token_tag=prefix.image_token_tag,
                input_embeds=prefix.input_embeds,
                decode_steps=max_new_tokens,
                suppress_token_ids=suppress_token_ids,
            )
            raw_ids = [int(x) for x in decode_result.generated_token_ids]
        else:
            if suppress_token_ids:
                raise NotImplementedError(
                    "native interleave image-cap suppression currently requires eager text decode"
                )
            decode_result = self.executor.run_greedy_decode_batch(
                [
                    {
                        "request_id": f"u1-native-interleave-text-{time.time_ns()}",
                        "input_ids": prefix.input_ids,
                        "indexes": prefix.indexes,
                        "image_token_tag": prefix.image_token_tag,
                        "input_embeds": prefix.input_embeds,
                        "cache_extra_key": prefix.cache_extra_key,
                    }
                ],
                decode_steps=max_new_tokens,
                suppress_token_ids=suppress_token_ids,
            )
            raw_ids = [int(x) for x in decode_result.generated_token_ids[0]]
        text_token_ids: list[int] = []
        terminal_token_id: int | None = None
        stop_reason = "max_new_tokens"
        for token_id in raw_ids:
            if token_id == eos_token_id:
                terminal_token_id = token_id
                stop_reason = "eos"
                break
            if token_id == self.img_start_token_id:
                terminal_token_id = token_id
                stop_reason = "image_start_pending_native_flow"
                break
            text_token_ids.append(token_id)
            if len(text_token_ids) >= max_new_tokens:
                stop_reason = "max_new_tokens"
                break
        text = self.tokenizer.decode(text_token_ids, skip_special_tokens=True)
        return text, text_token_ids, {
            "stop_reason": stop_reason,
            "generated_tokens": len(text_token_ids),
            "raw_generated_token_ids": raw_ids,
            "terminal_token_id": terminal_token_id,
            "eos_token_id": eos_token_id,
            "img_start_token_id": self.img_start_token_id,
            "native_decode_mode": decode_mode,
            "native_decode_stats": decode_result.to_dict(),
        }

    def _base_assistant_append(self, think_mode: bool) -> str:
        return "" if think_mode else "<think>\n\n</think>\n\n"

    def _flow_request_from_interleave(
        self,
        request: InterleaveRequestParams,
        *,
        seed: int,
    ) -> FlowRequestParams:
        return FlowRequestParams(
            mode="t2i",
            prompt=request.prompt,
            image_size=request.image_size,
            cfg_scale=request.cfg_scale,
            img_cfg_scale=request.img_cfg_scale,
            cfg_norm=request.cfg_norm,
            timestep_shift=request.timestep_shift,
            enable_timestep_shift=request.enable_timestep_shift,
            cfg_interval=request.cfg_interval,
            num_steps=request.num_steps,
            batch_size=1,
            t_eps=request.t_eps,
            think_mode=False,
            seed=seed,
        )

    @staticmethod
    def _native_interleave_text_decode_mode() -> str:
        value = os.environ.get(
            "SENSENOVA_U1_NATIVE_INTERLEAVE_EAGER_TEXT_DECODE",
            "",
        ).lower()
        if value in {"0", "false", "no", "off"}:
            return "sglang_cached_decode"
        return "eager_text_decode"

    @torch.inference_mode()
    def generate_interleave(
        self,
        request: InterleaveRequestParams,
        images: list[Any] | None = None,
        *,
        use_official_hybrid_mask: bool = False,
    ) -> InterleaveResult:
        if use_official_hybrid_mask:
            raise RuntimeError("native interleave runner cannot use official HF mask")
        assert_no_hf_modeling_imported(context="before native interleave generation")
        image_size = validate_image_size(*request.image_size)
        num_steps = validate_num_steps(request.num_steps)
        max_images = validate_image_count(
            request.max_images,
            name="max_images",
        )
        max_new_tokens = validate_max_new_tokens(request.max_new_tokens)
        validate_total_token_budget(
            image_size=image_size,
            image_count=max_images,
            max_new_tokens=max_new_tokens,
            max_total_tokens=self.max_total_tokens,
        )
        start = time.perf_counter()
        input_images = [_coerce_image(image) for image in (images or [])]
        validate_input_image_count(input_images)
        base_assistant = self._base_assistant_append(request.think_mode)
        generated_text = ""
        generated_image_tensors: list[torch.Tensor] = []
        generated_images: list[Image.Image] = []
        decode_stats_by_segment: list[dict[str, Any]] = []
        flow_stats_by_image: list[dict[str, Any]] = []
        image_elapsed_s: list[float] = []
        current_generated_tokens = 0
        forced_image_cap = False
        forced_image_requests = 0
        terminal_reason = "max_new_tokens"
        prefix: NativeFlowPrefix | None = None
        generated_image_span_tokens = generated_image_span_token_count(
            image_size
        )

        while current_generated_tokens < max_new_tokens:
            remaining_tokens = max_new_tokens - current_generated_tokens
            remaining_image_slots = max_images - len(generated_images)
            suppress_image = remaining_image_slots <= 0
            prefix = self._build_condition_prefix(
                prompt=request.prompt,
                system_message=request.system_message,
                images=input_images,
                generated_image_tensors=generated_image_tensors,
                think_mode=request.think_mode,
                assistant_append=base_assistant + generated_text,
                prompt_image_count=len(input_images),
                reserved_text_tokens=remaining_tokens,
                reserved_image_tokens=(
                    max(remaining_image_slots, 0) * generated_image_span_tokens
                ),
            )
            text, segment_token_ids, decode_stats = self._generate_text_until_boundary(
                prefix=prefix,
                max_new_tokens=remaining_tokens,
                suppress_token_ids=[self.img_start_token_id] if suppress_image else None,
            )
            native_decode_stats = decode_stats.get("native_decode_stats") or {}
            if suppress_image:
                suppressed_hits = int(native_decode_stats.get("suppressed_token_hits", 0))
                if suppressed_hits > 0:
                    forced_image_cap = True
                    forced_image_requests += suppressed_hits
            generated_text += text
            current_generated_tokens += len(segment_token_ids)
            decode_stats_by_segment.append(decode_stats)

            stop_reason = str(decode_stats["stop_reason"])
            if stop_reason == "image_start_pending_native_flow" and not suppress_image:
                image_started_at = time.perf_counter()
                flow_assistant_prefix = base_assistant + generated_text + "<img>"
                flow_request = self._flow_request_from_interleave(
                    request,
                    seed=int(request.seed),
                )
                if flow_request.num_steps != num_steps:
                    raise RuntimeError("interleave flow num_steps changed unexpectedly")
                image_tensor, flow_stats = self.generate_interleave_image_tensor(
                    flow_request,
                    input_images,
                    system_message=request.system_message,
                    assistant_prefix=flow_assistant_prefix,
                )
                generated_image_tensors.append(image_tensor.detach())
                generated_images.extend(u1_tensor_to_pil(image_tensor.detach()))
                generated_text += "<image>"
                image_elapsed_s.append(time.perf_counter() - image_started_at)
                flow_stats_by_image.append(flow_stats.to_dict())
                continue

            terminal_reason = "eos" if stop_reason == "eos" else "max_new_tokens"
            break

        elapsed = time.perf_counter() - start
        guard_after = _blocked_hf_modeling_modules()
        assert_no_hf_modeling_imported(context="after native interleave generation")
        stop_reason = (
            f"forced_cap_then_{terminal_reason}"
            if forced_image_cap
            else terminal_reason
        )
        token_ids = [
            int(x)
            for x in self.tokenizer(
                generated_text,
                add_special_tokens=False,
            )["input_ids"]
        ]
        stats = {
            "forced_native_single_image": False,
            "text_only_state_machine": len(generated_images) == 0,
            "image_state_machine_implemented": True,
            "native_full_prefill_interleave_state_machine": True,
            "forced_image_cap": forced_image_cap,
            "forced_image_requests": forced_image_requests,
            "natural_eos": terminal_reason == "eos" and not forced_image_cap,
            "eos_reached": terminal_reason == "eos",
            "stop_reason": stop_reason,
            "generated_tokens": current_generated_tokens,
            "generated_images": len(generated_images),
            "image_elapsed_s": image_elapsed_s,
            "condition_prefix_tokens": int(prefix.input_ids.numel()) if prefix else 0,
            "condition_prefix_image_tokens": (
                int(prefix.image_token_tag.sum().item()) if prefix else 0
            ),
            "native_decode_mode": (
                decode_stats_by_segment[-1]["native_decode_mode"]
                if decode_stats_by_segment
                else self._native_interleave_text_decode_mode()
            ),
            "native_decode_segments": decode_stats_by_segment,
            "native_flow_stats": {
                "runs": flow_stats_by_image,
                "hf_modeling_imported_after": guard_after,
            },
            "raw_generated_token_ids": [
                int(token_id)
                for segment in decode_stats_by_segment
                for token_id in segment["raw_generated_token_ids"]
            ],
            "terminal_token_id": (
                decode_stats_by_segment[-1]["terminal_token_id"]
                if decode_stats_by_segment
                else None
            ),
            "eos_token_id": (
                decode_stats_by_segment[-1]["eos_token_id"]
                if decode_stats_by_segment
                else int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
            ),
            "img_start_token_id": self.img_start_token_id,
            "hf_modeling_imported_after": guard_after,
        }
        return InterleaveResult(
            text=str(generated_text),
            token_ids=[int(x) for x in token_ids],
            images=generated_images,
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
            "backend": "native_sglang_interleave_text_state_machine",
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
            "backend": "native_sglang_interleave_text_state_machine",
        }


__all__ = [
    "DEFAULT_INTERLEAVE_SYSTEM_MESSAGE",
    "InterleaveRequestParams",
    "InterleaveResult",
    "SenseNovaU1InterleaveRunner",
    "interleave_segments",
    "interleave_tensors_to_pil",
]
