# SPDX-License-Identifier: Apache-2.0
"""HF-compatible SenseNova U1 understanding runner.

This is the M1 fallback path. It lives inside SGLang-Omni's pipeline registry
and request lifecycle, but delegates U1 math to the official NEOChatModel
implementation. The native SGLang attention path still needs the M2 hybrid
image-span mask before it can claim parity.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from sglang_omni.proto import StagePayload

SENSENOVA_U1_VENDOR_ROOT_ENV = "SENSENOVA_U1_VENDOR_ROOT"
SENSENOVA_U1_MODEL_PATH_ENV = "SENSENOVA_U1_MODEL_PATH"
DEFAULT_VENDOR_ROOT: str | None = None
DEFAULT_MODEL_DIR = os.environ.get(
    SENSENOVA_U1_MODEL_PATH_ENV,
    "sensenova/SenseNova-U1-8B-MoT-Interleaved",
)
DEFAULT_VQA_MIN_PIXELS = 65536
DEFAULT_VQA_MAX_PIXELS = 4194304


@dataclass(slots=True)
class PreparedVQA:
    question: str
    query: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    indexes: torch.Tensor
    image_token_tag: torch.Tensor
    pixel_values: torch.Tensor | None
    grid_hw: torch.Tensor | None


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized == "auto":
        return torch.bfloat16
    raise ValueError(f"Unsupported SenseNova U1 dtype: {name!r}")


def _ensure_vendor_import(vendor_root: str | None) -> Path:
    root_value = vendor_root or os.environ.get(SENSENOVA_U1_VENDOR_ROOT_ENV)
    if not root_value:
        spec = importlib.util.find_spec("sensenova_u1")
        if spec is not None and spec.origin:
            return Path(spec.origin).resolve().parent
        raise FileNotFoundError(
            "SenseNova-U1 Python package is not importable. Install it or set "
            f"{SENSENOVA_U1_VENDOR_ROOT_ENV} to a checkout containing src/sensenova_u1."
        )

    root = Path(root_value)
    src = root / "src"
    if not src.exists():
        raise FileNotFoundError(
            f"SenseNova-U1 source not found at {src}. Set {SENSENOVA_U1_VENDOR_ROOT_ENV}."
        )
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return root


def _load_raw_config(model_path: str) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def _missing_attr(obj: Any, name: str) -> bool:
    try:
        getattr(obj, name)
    except AttributeError:
        return True
    return False


def _patch_llm_config_compat(config: Any, model_path: str) -> None:
    """Restore raw U1 config fields hidden by newer Transformers Qwen3Config.

    The public U1 source expects legacy attributes such as ``rope_theta``.
    Transformers 4.57 can normalize Qwen3 rotary settings into newer internal
    fields, which breaks the unmodified U1 implementation at construction time.
    """

    llm_config = getattr(config, "llm_config", None)
    if llm_config is None:
        return
    raw_llm = (_load_raw_config(model_path).get("llm_config") or {})
    if not isinstance(raw_llm, dict):
        raw_llm = {}

    for key, value in raw_llm.items():
        if _missing_attr(llm_config, key):
            setattr(llm_config, key, value)

    rope_parameters = getattr(llm_config, "rope_parameters", None)
    if _missing_attr(llm_config, "rope_theta"):
        if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
            setattr(llm_config, "rope_theta", rope_parameters["rope_theta"])
        elif "rope_theta" in raw_llm:
            setattr(llm_config, "rope_theta", raw_llm["rope_theta"])

    layer_types = getattr(llm_config, "layer_types", None)
    num_layers = int(getattr(llm_config, "num_hidden_layers"))
    if not layer_types or len(layer_types) != num_layers:
        use_swa = bool(getattr(llm_config, "use_sliding_window", False)) and (
            getattr(llm_config, "sliding_window", None) is not None
        )
        max_window_layers = int(getattr(llm_config, "max_window_layers", 0) or 0)
        setattr(
            llm_config,
            "layer_types",
            [
                "sliding_attention" if (use_swa and i >= max_window_layers) else "full_attention"
                for i in range(num_layers)
            ],
        )


def _patch_model_class_compat() -> None:
    import sensenova_u1.models.neo_unify.modeling_neo_chat as modeling_neo_chat
    import sensenova_u1.models.neo_unify.modeling_qwen3 as modeling_qwen3
    import sensenova_u1.models.neo_unify.modeling_qwen3_moe as modeling_qwen3_moe

    from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel
    from sensenova_u1.models.neo_unify.modeling_qwen3 import Qwen3RotaryEmbedding
    from sglang_omni.models.sensenova_u1.hybrid_attention import create_u1_hybrid_mask
    from transformers.masking_utils import create_causal_mask as hf_create_causal_mask

    if not hasattr(NEOChatModel, "all_tied_weights_keys"):
        # U1 checkpoints set tie_word_embeddings=false. Transformers 4.57's
        # loader finalizer still expects the new instance attribute even when
        # the model class does not call post_init().
        NEOChatModel.all_tied_weights_keys = {}
    if not hasattr(Qwen3RotaryEmbedding, "compute_default_rope_parameters"):
        # Transformers 4.57's generic initializer recognizes any class named
        # *RotaryEmbedding and calls this newer helper for default RoPE. U1's
        # implementation already stores the correct initializer as rope_init_fn.
        def _compute_default_rope_parameters(self, config=None, device=None, **_kwargs):
            return self.rope_init_fn(config or self.config, device)

        Qwen3RotaryEmbedding.compute_default_rope_parameters = (
            _compute_default_rope_parameters
        )

    if not getattr(modeling_qwen3, "_sglang_omni_mask_compat", False):
        def _create_causal_mask_compat(*args, **kwargs):
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            kwargs.pop("cache_position", None)
            return hf_create_causal_mask(*args, **kwargs)

        modeling_qwen3.create_causal_mask = _create_causal_mask_compat
        if hasattr(modeling_qwen3_moe, "create_causal_mask"):
            modeling_qwen3_moe.create_causal_mask = _create_causal_mask_compat
        modeling_qwen3._sglang_omni_mask_compat = True

    for module in (modeling_qwen3, modeling_qwen3_moe, modeling_neo_chat):
        if hasattr(module, "create_block_causal_mask"):
            if not hasattr(module, "_sglang_omni_original_create_block_causal_mask"):
                module._sglang_omni_original_create_block_causal_mask = (
                    module.create_block_causal_mask
                )
            module.create_block_causal_mask = create_u1_hybrid_mask


def _refresh_official_vision_rope_cache(vision_model: Any) -> None:
    """Rebuild non-persistent NEOVision RoPE buffers after HF loading.

    Transformers 4.57 can leave the public U1 vision tower's non-persistent
    cos/sin buffers with invalid values after ``AutoModel.from_pretrained``.
    The official source computes those buffers deterministically in
    ``NEOVisionEmbeddings.__init__``; refresh them here so HF oracle runs use
    the intended vision path rather than corrupted compatibility-load state.
    """

    embeddings = getattr(vision_model, "embeddings", None)
    config = getattr(vision_model, "config", None)
    if embeddings is None or config is None:
        return

    device = next(vision_model.parameters()).device
    embed_dim = int(getattr(config, "hidden_size"))
    rope_dim_part = embed_dim // 2
    max_position = int(getattr(config, "max_position_embeddings_vision"))
    base = float(getattr(config, "rope_theta_vision"))
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rope_dim_part, 2, device=device, dtype=torch.float32)
            / rope_dim_part
        )
    )
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    embeddings.cos_cached_x = cos
    embeddings.sin_cached_x = sin
    embeddings.cos_cached_y = cos.clone()
    embeddings.sin_cached_y = sin.clone()


def _refresh_official_u1_rope_caches(model: Any) -> None:
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        _refresh_official_vision_rope_cache(vision_model)
    fm_modules = getattr(model, "fm_modules", None)
    if fm_modules is not None and "vision_model_mot_gen" in fm_modules:
        _refresh_official_vision_rope_cache(fm_modules["vision_model_mot_gen"])


def _force_official_llm_attn_implementation(model: Any, backend: str) -> None:
    """Apply an explicit Transformers backend after U1 model construction.

    Public U1 resets ``llm_config._attn_implementation`` to ``"eager"``
    inside ``NEOChatModel.__init__``. The package-level backend flag therefore
    does not switch the understanding tower by itself.
    """

    if backend not in {"eager", "sdpa"}:
        return
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return
    configs: list[Any] = [
        getattr(getattr(model, "config", None), "llm_config", None),
        getattr(language_model, "config", None),
        getattr(getattr(language_model, "model", None), "config", None),
    ]
    configs.extend(
        getattr(module, "config", None) for module in language_model.modules()
    )
    seen: set[int] = set()
    for config in configs:
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        config._attn_implementation = backend


@contextmanager
def _official_block_mask_scope():
    """Temporarily restore official U1 block mask functions for references."""

    import sensenova_u1.models.neo_unify.modeling_neo_chat as modeling_neo_chat
    import sensenova_u1.models.neo_unify.modeling_qwen3 as modeling_qwen3
    import sensenova_u1.models.neo_unify.modeling_qwen3_moe as modeling_qwen3_moe

    modules = [modeling_qwen3, modeling_qwen3_moe, modeling_neo_chat]
    saved: list[tuple[Any, Any]] = []
    for module in modules:
        if not hasattr(module, "create_block_causal_mask"):
            continue
        saved.append((module, module.create_block_causal_mask))
        original = getattr(module, "_sglang_omni_original_create_block_causal_mask", None)
        if original is not None:
            module.create_block_causal_mask = original
    try:
        yield
    finally:
        for module, function in saved:
            module.create_block_causal_mask = function


def _image_from_data_url(value: str) -> Image.Image:
    _, _, encoded = value.partition(",")
    if not encoded:
        raise ValueError("image data URL is missing base64 payload")
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def _coerce_image(value: Any) -> Any:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if "path" in value:
            return value["path"]
        if "image" in value:
            return _coerce_image(value["image"])
        if "image_url" in value:
            image_url = value["image_url"]
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            return _coerce_image(image_url)
        if "url" in value:
            return _coerce_image(value["url"])
    if isinstance(value, str) and value.startswith("data:image/"):
        return _image_from_data_url(value)
    return value


def _extract_text_and_images_from_content(content: Any) -> tuple[str, list[Any]]:
    images: list[Any] = []
    if isinstance(content, str):
        return content, images
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                typ = item.get("type")
                if typ == "text" or "text" in item:
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif typ in {"image_url", "image"} or "image_url" in item or "image" in item:
                    images.append(_coerce_image(item))
        return "\n".join(part for part in parts if part), images
    return str(content), images


def _extract_request(payload_or_inputs: StagePayload | Any) -> tuple[Any, dict[str, Any], str]:
    if isinstance(payload_or_inputs, StagePayload):
        return (
            payload_or_inputs.request.inputs,
            payload_or_inputs.request.params,
            payload_or_inputs.request_id,
        )
    return payload_or_inputs, {}, "local"


def _extract_question_images_history(inputs: Any) -> tuple[str, list[Any], list[tuple[str, str]]]:
    images: list[Any] = []
    history: list[tuple[str, str]] = []

    if isinstance(inputs, str):
        return inputs, images, history

    if isinstance(inputs, dict):
        images.extend(_coerce_image(img) for img in inputs.get("images") or [])
        if "image" in inputs:
            images.append(_coerce_image(inputs["image"]))
        if "question" in inputs:
            return str(inputs["question"]), images, history
        if "prompt" in inputs:
            return str(inputs["prompt"]), images, history
        if "messages" in inputs:
            inputs = inputs["messages"]
        else:
            return str(inputs), images, history

    if isinstance(inputs, list):
        pending_user: str | None = None
        for msg in inputs:
            if not isinstance(msg, dict):
                pending_user = str(msg)
                continue
            role = str(msg.get("role", "user"))
            text, msg_images = _extract_text_and_images_from_content(msg.get("content", ""))
            images.extend(_coerce_image(img) for img in msg_images)
            if role == "user":
                pending_user = text
            elif role == "assistant" and pending_user is not None:
                history.append((pending_user, text))
                pending_user = None
        return pending_user or "", images, history

    return str(inputs), images, history


class SenseNovaU1UnderstandingRunner:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_DIR,
        *,
        vendor_root: str | None = None,
        device: str = "cuda:0",
        dtype: str | torch.dtype = "bfloat16",
        attn_backend: str = "auto",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        load_with_info: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.vendor_root = _ensure_vendor_import(vendor_root)
        self.device = str(device)
        self.dtype = dtype if isinstance(dtype, torch.dtype) else _dtype_from_name(dtype)
        self.attn_backend = attn_backend
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model = None
        self.tokenizer = None
        self.loading_info: dict[str, Any] = {}
        self.load(load_with_info=load_with_info)

    def load(self, *, load_with_info: bool = False) -> None:
        import sensenova_u1
        from sensenova_u1 import check_checkpoint_compatibility
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        _patch_model_class_compat()
        sensenova_u1.set_attn_backend(self.attn_backend)
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        _patch_llm_config_compat(config, self.model_path)
        check_checkpoint_compatibility(config)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        kwargs = {"config": config, "torch_dtype": self.dtype}
        if load_with_info:
            model, info = AutoModel.from_pretrained(
                self.model_path,
                output_loading_info=True,
                trust_remote_code=True,
                **kwargs,
            )
            self.loading_info = dict(info)
        else:
            model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                **kwargs,
            )
            self.loading_info = {}
        self.model = model.eval().to(self.device)
        _force_official_llm_attn_implementation(self.model, self.attn_backend)
        _refresh_official_u1_rope_caches(self.model)
        self.tokenizer = tokenizer

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def prepare(
        self,
        question: str,
        images: list[Any] | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> PreparedVQA:
        from sensenova_u1.models.neo_unify.conversation import get_conv_template
        from sensenova_u1.models.neo_unify.utils import load_image_native

        assert self.model is not None and self.tokenizer is not None
        images = list(images or [])
        if history is None:
            history = []
        if images and "<image>" not in question:
            question = "<image>\n" + question

        pixel_values_list: list[torch.Tensor] = []
        grid_hw_list: list[torch.Tensor] = []
        for image in images:
            kwargs: dict[str, Any] = {
                "patch_size": self.model.patch_size,
                "downsample_ratio": self.model.downsample_ratio,
            }
            if self.min_pixels is not None:
                kwargs["min_pixels"] = self.min_pixels
            if self.max_pixels is not None:
                kwargs["max_pixels"] = self.max_pixels
            cur_pixel_values, cur_grid_hw = load_image_native(_coerce_image(image), **kwargs)
            pixel_values_list.append(cur_pixel_values)
            grid_hw_list.append(cur_grid_hw)

        pixel_values = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
        grid_hw = torch.cat(grid_hw_list, dim=0) if grid_hw_list else None
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.torch_device, dtype=self.model.dtype)
        if grid_hw is not None:
            grid_hw = grid_hw.to(self.torch_device)

        img_context_token = "<IMG_CONTEXT>"
        img_start_token = "<img>"
        img_end_token = "</img>"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(img_context_token)
        self.model.img_start_token_id = self.tokenizer.convert_tokens_to_ids(img_start_token)

        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if grid_hw is not None:
            for i in range(grid_hw.shape[0]):
                num_patch_token = int(
                    grid_hw[i, 0] * grid_hw[i, 1] * self.model.downsample_ratio**2
                )
                image_tokens = img_start_token + img_context_token * num_patch_token + img_end_token
                query = query.replace("<image>", image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.torch_device)
        indexes = self.model.get_thw_indexes(input_ids[0], grid_hw)
        image_token_tag = input_ids[0] == self.model.img_context_token_id
        return PreparedVQA(
            question=question,
            query=query,
            input_ids=input_ids,
            attention_mask=model_inputs["attention_mask"].to(self.torch_device),
            indexes=indexes,
            image_token_tag=image_token_tag,
            pixel_values=pixel_values,
            grid_hw=grid_hw,
        )

    def prefill_logits(self, prepared: PreparedVQA) -> torch.Tensor:
        assert self.model is not None
        input_ids = prepared.input_ids
        if prepared.pixel_values is not None:
            vit_embeds = self.model.extract_feature(
                prepared.pixel_values,
                grid_hw=prepared.grid_hw,
            )
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            bsz, seqlen, channels = input_embeds.shape
            flat_embeds = input_embeds.reshape(bsz * seqlen, channels)
            flat_ids = input_ids.reshape(bsz * seqlen)
            selected = flat_ids == self.model.img_context_token_id
            flat_embeds[selected] = vit_embeds.reshape(-1, channels).to(flat_embeds.device)
            input_embeds = flat_embeds.reshape(bsz, seqlen, channels)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            indexes=prepared.indexes,
            attention_mask=prepared.attention_mask,
            use_cache=True,
        )
        return outputs.logits[:, -1, :].detach().float()

    def generate_ids(
        self,
        prepared: PreparedVQA,
        *,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> torch.Tensor:
        assert self.model is not None and self.tokenizer is not None
        from sensenova_u1.models.neo_unify.conversation import get_conv_template

        template = get_conv_template(self.model.template)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": eos_token_id,
        }
        if do_sample:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p
            if top_k is not None:
                generation_config["top_k"] = top_k
        if repetition_penalty is not None:
            generation_config["repetition_penalty"] = repetition_penalty

        return self.model.generate(
            pixel_values=prepared.pixel_values,
            input_ids=prepared.input_ids,
            grid_hw=prepared.grid_hw,
            attention_mask=prepared.attention_mask,
            **generation_config,
        )

    def decode_generated(self, output_ids: torch.Tensor) -> str:
        assert self.model is not None and self.tokenizer is not None
        from sensenova_u1.models.neo_unify.conversation import get_conv_template

        template = get_conv_template(self.model.template)
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response.split(template.sep.strip())[0].strip()

    def answer(self, question: str, images: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        prepared = self.prepare(question, images)
        start = time.perf_counter()
        output_ids = self.generate_ids(prepared, **kwargs)
        elapsed = time.perf_counter() - start
        text = self.decode_generated(output_ids)
        token_ids = output_ids[0].detach().cpu().tolist()
        return {
            "text": text,
            "token_ids": token_ids,
            "prompt_tokens": int(prepared.input_ids.numel()),
            "completion_tokens": len(token_ids),
            "engine_time_s": elapsed,
            "finish_reason": "stop",
        }

    def reference_chat_capture(
        self,
        question: str,
        images: list[Any] | None = None,
        *,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        use_official_hybrid_mask: bool = False,
    ) -> dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None
        from sensenova_u1.models.neo_unify.utils import load_image_native

        images = list(images or [])
        pixel_values = None
        grid_hw = None
        if images:
            pixel_values_list = []
            grid_hw_list = []
            for image in images:
                cur_pixel_values, cur_grid_hw = load_image_native(
                    _coerce_image(image),
                    patch_size=self.model.patch_size,
                    downsample_ratio=self.model.downsample_ratio,
                    min_pixels=(
                        self.min_pixels
                        if self.min_pixels is not None
                        else DEFAULT_VQA_MIN_PIXELS
                    ),
                    max_pixels=(
                        self.max_pixels
                        if self.max_pixels is not None
                        else DEFAULT_VQA_MAX_PIXELS
                    ),
                )
                pixel_values_list.append(cur_pixel_values)
                grid_hw_list.append(cur_grid_hw)
            pixel_values = torch.cat(pixel_values_list, dim=0).to(
                self.torch_device, dtype=self.model.dtype
            )
            grid_hw = torch.cat(grid_hw_list, dim=0).to(self.torch_device)

        captured: dict[str, Any] = {}
        original_generate = self.model.generate

        def _capture_generate(*args: Any, **kwargs: Any):
            output = original_generate(*args, **kwargs)
            captured["output_ids"] = output.detach().cpu()
            return output

        self.model.generate = _capture_generate
        try:
            start = time.perf_counter()
            if use_official_hybrid_mask:
                with _official_block_mask_scope():
                    text = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        question,
                        {"max_new_tokens": max_new_tokens, "do_sample": do_sample},
                        grid_hw=grid_hw,
                    )
            else:
                text = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    {"max_new_tokens": max_new_tokens, "do_sample": do_sample},
                    grid_hw=grid_hw,
                )
            elapsed = time.perf_counter() - start
        finally:
            self.model.generate = original_generate

        output_ids = captured["output_ids"]
        return {
            "text": text,
            "token_ids": output_ids[0].tolist(),
            "engine_time_s": elapsed,
        }

    def complete_payload(
        self,
        payload: StagePayload,
        *,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> dict[str, Any]:
        inputs, params, request_id = _extract_request(payload)
        question, images, history = _extract_question_images_history(inputs)
        prepared = self.prepare(question, images, history)
        start = time.perf_counter()
        output_ids = self.generate_ids(
            prepared,
            max_new_tokens=int(params.get("max_new_tokens", max_new_tokens)),
            do_sample=bool(params.get("do_sample", do_sample)),
            temperature=float(params.get("temperature", temperature)),
            top_p=float(params.get("top_p", top_p)),
            top_k=params.get("top_k", top_k),
            repetition_penalty=params.get("repetition_penalty", repetition_penalty),
        )
        elapsed = time.perf_counter() - start
        token_ids = output_ids[0].detach().cpu().tolist()
        return {
            "request_id": request_id,
            "text": self.decode_generated(output_ids),
            "token_ids": token_ids,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": int(prepared.input_ids.numel()),
                "completion_tokens": len(token_ids),
                "total_tokens": int(prepared.input_ids.numel()) + len(token_ids),
                "engine_time_s": elapsed,
            },
            "stage_name": "u1_vqa",
            "backend": "hf_compatible_fallback",
        }


def logits_similarity(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().float().reshape(1, -1).cpu()
    b = b.detach().float().reshape(1, -1).cpu()
    return {
        "cosine": float(F.cosine_similarity(a, b, dim=-1).item()),
        "max_abs_diff": float((a - b).abs().max().item()),
    }
