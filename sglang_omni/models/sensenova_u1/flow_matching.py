# SPDX-License-Identifier: Apache-2.0
"""SenseNova U1 flow-matching image generation runner.

This is the M3 HF-compatible in-pipeline path. It keeps U1 generation inside a
SGLang-Omni stage lifecycle, while delegating the exact numerical kernels to
the official NEOChatModel implementation already used by the M1/M2 fallback.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer

from sglang_omni.models.sensenova_u1.native_serving import (
    SenseNovaU1NativeServingExecutor,
)
from sglang_omni.models.sensenova_u1.native_vision import (
    SenseNovaU1NativeVisionModel,
)
from sglang_omni.models.sensenova_u1.sglang_model import (
    _blocked_hf_modeling_modules,
    assert_no_hf_modeling_imported,
    block_hf_modeling_imports,
)
from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.proto import StagePayload
from sglang_omni.models.sensenova_u1.hf_runner import (
    DEFAULT_MODEL_DIR,
    DEFAULT_VENDOR_ROOT,
    SenseNovaU1UnderstandingRunner,
    _coerce_image,
    _extract_request,
    _official_block_mask_scope,
)

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SYSTEM_MESSAGE_FOR_GEN = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language."
)


@dataclass(frozen=True, slots=True)
class FlowRequestParams:
    mode: str
    prompt: str
    image_size: tuple[int, int] = (256, 256)
    cfg_scale: float = 1.0
    img_cfg_scale: float = 1.0
    cfg_norm: str = "none"
    timestep_shift: float = 1.0
    enable_timestep_shift: bool = True
    cfg_interval: tuple[float, float] = (0.0, 1.0)
    num_steps: int = 2
    batch_size: int = 1
    t_eps: float = 0.05
    think_mode: bool = False
    seed: int = 20260813


@dataclass(slots=True)
class NativeFlowPrefix:
    input_ids: torch.Tensor
    indexes: torch.Tensor
    image_token_tag: torch.Tensor
    input_embeds: torch.Tensor
    cache_extra_key: str | None
    cache_insert_log: dict[str, Any]
    cache_reuse_enabled: bool = False


@dataclass(slots=True)
class NativeFlowRunStats:
    prefix_tokens: int
    image_tokens: int
    forward_elapsed_s: list[float]
    backend_name: str | None = None
    hf_modeling_imported_after: list[str] | None = None
    hidden_prefill_logs: list[dict[str, Any]] | None = None
    prefix_cache_enabled: bool = False
    prefix_cache_log: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_tokens": int(self.prefix_tokens),
            "image_tokens": int(self.image_tokens),
            "forward_elapsed_s": [float(x) for x in self.forward_elapsed_s],
            "forward_elapsed_sum_s": float(sum(self.forward_elapsed_s)),
            "backend_name": self.backend_name,
            "hf_modeling_imported_after": self.hf_modeling_imported_after or [],
            "hidden_prefill_logs": self.hidden_prefill_logs or [],
            "prefix_cache_enabled": bool(self.prefix_cache_enabled),
            "prefix_cache_log": self.prefix_cache_log or {},
        }


class NativeTimestepEmbedder(nn.Module):
    """Native copy of U1's small timestep MLP; avoids importing HF modeling code."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = int(frequency_embedding_size)

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: float = 10000.0,
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class SenseNovaU1NativeFlowModules(nn.Module):
    """Native U1 flow modules for the public Interleaved checkpoint."""

    def __init__(self, model_path: str | Path, *, params_dtype: torch.dtype) -> None:
        super().__init__()
        self.raw_config = _load_raw_config(model_path)
        llm_config = self.raw_config["llm_config"]
        hidden_size = int(llm_config["hidden_size"])
        patch_size = int(self.raw_config.get("patch_size", 16))
        merge_size = int(1 / float(self.raw_config.get("downsample_ratio", 0.5)))
        output_dim = 3 * (patch_size * merge_size) ** 2
        if bool(self.raw_config.get("use_pixel_head", False)):
            raise NotImplementedError("native U1 flow path does not support use_pixel_head")
        if int(self.raw_config.get("fm_head_layers", 2)) > 2:
            raise NotImplementedError("native U1 flow path does not support deep fm_head")

        self.vision_model_mot_gen = SenseNovaU1NativeVisionModel.from_model_path(
            model_path,
            params_dtype=params_dtype,
        )
        self.timestep_embedder = NativeTimestepEmbedder(hidden_size)
        self.fm_head = nn.Sequential(
            nn.Linear(hidden_size, 4096, bias=True),
            nn.GELU(),
            nn.Linear(4096, output_dim, bias=True),
        )
        self.add_noise_scale_embedding = bool(
            self.raw_config.get("add_noise_scale_embedding", False)
        )
        if self.add_noise_scale_embedding:
            self.noise_scale_embedder = NativeTimestepEmbedder(hidden_size)
        self.to(dtype=params_dtype)

    def load_weights(self, model_path: str | Path) -> dict[str, Any]:
        expected = set(dict(self.named_parameters()))
        loaded: set[str] = set()
        unexpected: list[str] = []
        errors: list[str] = []
        model_path = Path(model_path)
        index_file = model_path / "model.safetensors.index.json"
        weight_map = json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
        shards: dict[str, list[str]] = {}
        for key, shard_name in weight_map.items():
            if key.startswith("fm_modules."):
                shards.setdefault(shard_name, []).append(key)

        params = dict(self.named_parameters())
        for shard_name in sorted(shards):
            with safe_open(str(model_path / shard_name), framework="pt", device="cpu") as f:
                for full_key in sorted(shards[shard_name]):
                    name = full_key[len("fm_modules.") :]
                    if name.startswith("vision_model_mot_gen.embeddings."):
                        name = (
                            "vision_model_mot_gen."
                            + name[len("vision_model_mot_gen.embeddings.") :]
                        )
                    tensor = f.get_tensor(full_key)
                    param = params.get(name)
                    if param is None:
                        unexpected.append(name)
                        continue
                    if tuple(param.shape) != tuple(tensor.shape):
                        errors.append(
                            f"{name}: param={tuple(param.shape)} checkpoint={tuple(tensor.shape)}"
                        )
                        continue
                    with torch.no_grad():
                        param.copy_(tensor.to(device=param.device, dtype=param.dtype))
                    loaded.add(name)

        missing = sorted(expected - loaded)
        return {
            "loaded_keys": sorted(loaded),
            "loaded_count": len(loaded),
            "missing_keys": missing,
            "missing_count": len(missing),
            "unexpected_keys": sorted(unexpected),
            "unexpected_count": len(unexpected),
            "errors": errors,
            "error_count": len(errors),
            "ok": not missing and not unexpected and not errors,
        }


def _load_raw_config(model_path: str | Path) -> dict[str, Any]:
    return json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute image aspect ratio must be smaller than 200")
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def load_image_native_tensor(
    image: Any,
    *,
    patch_size: int,
    downsample_ratio: float,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
    upscale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = _coerce_image(image)
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background.convert("RGB")
    else:
        image = image.convert("RGB")
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    factor = int(patch_size // downsample_ratio)
    resized_h, resized_w = _smart_resize(
        image.height,
        image.width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_w, resized_h))
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    pixel_values = transform(image).to(torch.float32)
    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size
    flat = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size**2)
    )
    return flat, torch.tensor([[grid_h, grid_w]], dtype=torch.long)


def _build_neo_prompt(
    *,
    user_text: str,
    system_message: str,
    assistant_append: str | None = None,
) -> str:
    prompt = ""
    if system_message:
        prompt += f"<|im_start|>system\n{system_message}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_text}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    if assistant_append is not None:
        prompt += assistant_append
    return prompt


def build_abs_positions_from_grid_hw(
    grid_hw: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_hw = grid_hw.reshape(-1, 2).to(dtype=torch.long)
    hs: list[torch.Tensor] = []
    ws: list[torch.Tensor] = []
    for h, w in grid_hw.detach().cpu().tolist():
        idx = torch.arange(int(h) * int(w), device=device, dtype=torch.long)
        hs.append(idx // int(w))
        ws.append(idx % int(w))
    return torch.cat(ws), torch.cat(hs)


def denormalize_u1_image_tensor(batch: torch.Tensor) -> torch.Tensor:
    """Convert U1 normalized image tensors from [-1, 1] into [0, 1]."""

    mean = torch.tensor(NORM_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch * std + mean).clamp(0, 1)


def u1_tensor_to_pil(batch: torch.Tensor) -> list[Image.Image]:
    arr = denormalize_u1_image_tensor(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def pil_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def compare_pil_images(a: Image.Image, b: Image.Image) -> dict[str, Any]:
    if a.size != b.size:
        raise ValueError(f"Cannot compare images with different sizes: {a.size} vs {b.size}")
    arr_a_u8 = np.asarray(a.convert("RGB"), dtype=np.uint8)
    arr_b_u8 = np.asarray(b.convert("RGB"), dtype=np.uint8)
    diff = arr_a_u8.astype(np.int16) - arr_b_u8.astype(np.int16)
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = math.inf if mse == 0.0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    a_float = arr_a_u8.astype(np.float64) / 255.0
    b_float = arr_b_u8.astype(np.float64) / 255.0
    mu_a = float(a_float.mean())
    mu_b = float(b_float.mean())
    var_a = float(((a_float - mu_a) ** 2).mean())
    var_b = float(((b_float - mu_b) ** 2).mean())
    cov = float(((a_float - mu_a) * (b_float - mu_b)).mean())
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return {
        "width": a.size[0],
        "height": a.size[1],
        "mse_uint8": mse,
        "psnr_db": psnr,
        "ssim_global_rgb": float(ssim),
        "pixel_max_abs_diff_uint8": int(np.abs(diff).max()),
        "pixel_mean_abs_diff_uint8": float(np.abs(diff).mean()),
        "exact_png_pixels": bool(np.array_equal(arr_a_u8, arr_b_u8)),
        "ssim_implementation": "global_rgb_formula",
    }


class SenseNovaU1FlowMatchingFallbackRunner(SenseNovaU1UnderstandingRunner):
    """T2I/IT2I runner for U1 flow matching inside SGLang-Omni."""

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

    def _flow_params_from_request(self, inputs: Any, params: dict[str, Any]) -> tuple[FlowRequestParams, list[Any]]:
        data = inputs if isinstance(inputs, dict) else {"prompt": str(inputs)}
        merged = {**data, **params}
        mode = str(merged.get("mode") or merged.get("task") or "t2i").lower()
        if mode in {"text_to_image", "txt2img"}:
            mode = "t2i"
        if mode in {"image_to_image", "edit", "editing"}:
            mode = "it2i"
        if mode not in {"t2i", "it2i"}:
            raise ValueError(f"Unsupported U1 flow mode: {mode!r}")
        width = int(merged.get("width", merged.get("image_width", 256)))
        height = int(merged.get("height", merged.get("image_height", 256)))
        cfg_interval_value = merged.get("cfg_interval", (0.0, 1.0))
        cfg_interval = tuple(float(x) for x in cfg_interval_value)
        if len(cfg_interval) != 2:
            raise ValueError("cfg_interval must contain exactly two floats.")
        images = [_coerce_image(img) for img in (merged.get("images") or [])]
        if "image" in merged:
            images.append(_coerce_image(merged["image"]))
        request = FlowRequestParams(
            mode=mode,
            prompt=str(merged.get("prompt", "")),
            image_size=(width, height),
            cfg_scale=float(merged.get("cfg_scale", 1.0)),
            img_cfg_scale=float(merged.get("img_cfg_scale", 1.0)),
            cfg_norm=str(merged.get("cfg_norm", "none")),
            timestep_shift=float(merged.get("timestep_shift", 1.0)),
            enable_timestep_shift=bool(merged.get("enable_timestep_shift", True)),
            cfg_interval=(float(cfg_interval[0]), float(cfg_interval[1])),
            num_steps=int(merged.get("num_steps", 2)),
            batch_size=int(merged.get("batch_size", 1)),
            t_eps=float(merged.get("t_eps", 0.05)),
            think_mode=bool(merged.get("think_mode", False)),
            seed=int(merged.get("seed", 20260813)),
        )
        return request, images

    def scheduler_params(self, request: FlowRequestParams) -> dict[str, Any]:
        assert self.model is not None
        model = self.model
        width, height = request.image_size
        merge_size = int(1 / model.downsample_ratio)
        token_h = height // (model.patch_size * merge_size)
        token_w = width // (model.patch_size * merge_size)
        grid_h = height // model.patch_size
        grid_w = width // model.patch_size
        image_seq_len = token_h * token_w

        noise_scale = float(model.noise_scale)
        if model.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(model.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base)
            noise_scale = scale * float(model.noise_scale)
            if model.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, float(model.noise_scale_max_value))

        timesteps = torch.linspace(0.0, 1.0, request.num_steps + 1, device=self.torch_device)
        if request.enable_timestep_shift:
            timesteps = model._apply_time_schedule(
                timesteps,
                image_seq_len,
                request.timestep_shift,
            )
        t_values = [float(x) for x in timesteps.detach().cpu().tolist()]
        step_table = [
            {
                "step": i,
                "t": t_values[i],
                "t_next": t_values[i + 1],
                "dt": t_values[i + 1] - t_values[i],
                "use_cfg": bool(
                    (
                        t_values[i] > request.cfg_interval[0]
                        and t_values[i] < request.cfg_interval[1]
                    )
                    or request.cfg_interval[0] == 0
                ),
            }
            for i in range(request.num_steps)
        ]
        config_keys = [
            "fm_head_dim",
            "fm_head_layers",
            "fm_head_mlp_ratio",
            "timestep_shift",
            "time_schedule",
            "time_shift_type",
            "base_shift",
            "max_shift",
            "base_image_seq_len",
            "max_image_seq_len",
            "noise_scale",
            "noise_scale_mode",
            "noise_scale_base_image_seq_len",
            "noise_scale_max_value",
            "add_noise_scale_embedding",
            "P_mean",
            "P_std",
            "t_eps",
            "use_pixel_head",
            "use_adaLN",
        ]
        raw_config = {
            key: getattr(model.config, key, getattr(model, key, None))
            for key in config_keys
        }
        return {
            "backend": "sglang_omni_sensenova_u1_flow_matching_hf_compatible_official_mask",
            "request": asdict(request),
            "patch_size": int(model.patch_size),
            "downsample_ratio": float(model.downsample_ratio),
            "merge_size": merge_size,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "token_h": token_h,
            "token_w": token_w,
            "image_seq_len": image_seq_len,
            "computed_noise_scale": noise_scale,
            "raw_config": raw_config,
            "timesteps": t_values,
            "step_table": step_table,
        }

    @torch.inference_mode()
    def generate_t2i_tensor(
        self,
        request: FlowRequestParams,
        *,
        use_official_hybrid_mask: bool = True,
    ) -> tuple[torch.Tensor, str]:
        assert self.model is not None and self.tokenizer is not None
        ctx = _official_block_mask_scope() if use_official_hybrid_mask else nullcontext()
        with ctx:
            output = self.model.t2i_generate(
                self.tokenizer,
                request.prompt,
                image_size=request.image_size,
                cfg_scale=request.cfg_scale,
                cfg_norm=request.cfg_norm,
                timestep_shift=request.timestep_shift,
                enable_timestep_shift=request.enable_timestep_shift,
                cfg_interval=request.cfg_interval,
                num_steps=request.num_steps,
                batch_size=request.batch_size,
                t_eps=request.t_eps,
                think_mode=request.think_mode,
                seed=request.seed,
            )
        if request.think_mode:
            tensor, think_text = output
            return tensor.detach(), str(think_text)
        return output.detach(), ""

    @torch.inference_mode()
    def generate_it2i_tensor(
        self,
        request: FlowRequestParams,
        images: list[Any],
        *,
        use_official_hybrid_mask: bool = True,
    ) -> tuple[torch.Tensor, str]:
        assert self.model is not None and self.tokenizer is not None
        if not images:
            raise ValueError("IT2I generation requires at least one input image.")
        ctx = _official_block_mask_scope() if use_official_hybrid_mask else nullcontext()
        with ctx:
            output = self.model.it2i_generate(
                self.tokenizer,
                request.prompt,
                [_coerce_image(image) for image in images],
                image_size=request.image_size,
                cfg_scale=request.cfg_scale,
                img_cfg_scale=request.img_cfg_scale,
                cfg_norm=request.cfg_norm,
                timestep_shift=request.timestep_shift,
                enable_timestep_shift=request.enable_timestep_shift,
                cfg_interval=request.cfg_interval,
                num_steps=request.num_steps,
                batch_size=request.batch_size,
                t_eps=request.t_eps,
                think_mode=request.think_mode,
                seed=request.seed,
            )
        if request.think_mode:
            tensor, think_text = output
            return tensor.detach(), str(think_text)
        return output.detach(), ""

    def complete_payload(self, payload: StagePayload) -> dict[str, Any]:
        inputs, params, request_id = _extract_request(payload)
        request, images = self._flow_params_from_request(inputs, params)
        start = time.perf_counter()
        if request.mode == "t2i":
            tensor, think_text = self.generate_t2i_tensor(request)
        else:
            tensor, think_text = self.generate_it2i_tensor(request, images)
        elapsed = time.perf_counter() - start
        pil_images = u1_tensor_to_pil(tensor)
        return {
            "request_id": request_id,
            "mode": request.mode,
            "images": [
                {
                    "type": "image",
                    "format": "png",
                    "data": pil_to_data_url(img),
                    "width": img.width,
                    "height": img.height,
                }
                for img in pil_images
            ],
            "think_text": think_text,
            "scheduler_params": self.scheduler_params(request),
            "usage": {
                "image_count": len(pil_images),
                "num_steps": request.num_steps,
                "engine_time_s": elapsed,
            },
            "stage_name": "u1_flow",
            "backend": "hf_compatible_flow_matching_fallback_official_mask",
        }


class SenseNovaU1FlowMatchingRunner:
    """Native SGLang U1 T2I/IT2I flow-matching runner.

    The runner avoids official U1 HF modeling imports. Prefixes are inserted
    through the existing SGLang/RadixAttention serving executor; each denoise
    step extends that prefix with generated image tokens, captures the native
    MoT hidden states, and applies native-loaded flow modules.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_DIR,
        *,
        vendor_root: str | None = None,
        device: str = "cuda:0",
        dtype: str | torch.dtype = "bfloat16",
        attn_backend: str = "triton",
        attention_backend: str | None = None,
        load_with_info: bool = False,
        mem_fraction_static: float = 0.65,
        max_total_tokens: int = 8192,
        max_running_requests: int = 2,
    ) -> None:
        del vendor_root, load_with_info
        assert_no_hf_modeling_imported(context="before native flow runner init")
        self.model_path = str(model_path)
        self.device = str(device)
        self.torch_device = torch.device(device)
        self.dtype = dtype if isinstance(dtype, torch.dtype) else resolve_dtype(dtype)
        self.raw_config = _load_raw_config(model_path)
        self.config = _to_namespace(self.raw_config)
        self.llm_config = _to_namespace(self.raw_config["llm_config"])
        self.patch_size = int(self.raw_config.get("patch_size", 16))
        self.downsample_ratio = float(self.raw_config.get("downsample_ratio", 0.5))
        self.noise_scale = float(self.raw_config.get("noise_scale", 1.0))
        self.noise_scale_mode = str(self.raw_config.get("noise_scale_mode", "constant"))
        self.noise_scale_base_image_seq_len = int(
            self.raw_config.get("noise_scale_base_image_seq_len", 64)
        )
        self.add_noise_scale_embedding = bool(
            self.raw_config.get("add_noise_scale_embedding", False)
        )
        self.noise_scale_max_value = float(
            self.raw_config.get("noise_scale_max_value", 1.0)
        )
        self.time_schedule = str(self.raw_config.get("time_schedule", "standard"))
        self.time_shift_type = str(self.raw_config.get("time_shift_type", "exponential"))
        self.base_shift = float(self.raw_config.get("base_shift", 0.5))
        self.max_shift = float(self.raw_config.get("max_shift", 1.15))
        self.base_image_seq_len = int(self.raw_config.get("base_image_seq_len", 64))
        self.max_image_seq_len = int(self.raw_config.get("max_image_seq_len", 4096))
        self.last_native_flow_stats: dict[str, Any] = {}
        self.native_flow_prefill_cuda_graph_enabled = (
            self._native_flow_prefill_cuda_graph_enabled()
        )
        if isinstance(dtype, str):
            dtype_arg = dtype
        elif dtype is torch.bfloat16:
            dtype_arg = "bfloat16"
        elif dtype is torch.float16:
            dtype_arg = "float16"
        elif dtype is torch.float32:
            dtype_arg = "float32"
        else:
            dtype_arg = "bfloat16"

        backend_arg = attention_backend or attn_backend
        if backend_arg == "auto":
            backend_arg = "triton"

        with block_hf_modeling_imports():
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            self.executor = SenseNovaU1NativeServingExecutor(
                model_path,
                device=device,
                dtype=dtype_arg,
                attention_backend=backend_arg,
                mem_fraction_static=mem_fraction_static,
                max_total_tokens=max_total_tokens,
                max_running_requests=max_running_requests,
                enable_radix_cache=True,
                prefill_cuda_graph_backend=(
                    "breakable"
                    if self.native_flow_prefill_cuda_graph_enabled
                    else "disabled"
                ),
                prefill_cuda_graph_bs=[64],
            )
            self.fm_modules = SenseNovaU1NativeFlowModules(
                model_path,
                params_dtype=next(
                    self.executor.model_worker.model_runner.model.parameters()
                ).dtype,
            ).to(device=self.torch_device)
            self.fm_modules.eval()
            self.flow_load_report = self.fm_modules.load_weights(model_path)
            if not self.flow_load_report["ok"]:
                raise RuntimeError(
                    "native U1 flow module load failed: "
                    f"{self.flow_load_report}"
                )
        self.img_context_token_id = int(
            self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        )
        self.img_start_token_id = int(self.tokenizer.convert_tokens_to_ids("<img>"))
        self.img_end_token_id = int(self.tokenizer.convert_tokens_to_ids("</img>"))
        assert_no_hf_modeling_imported(context="after native flow runner init")

    def _calculate_dynamic_mu(self, image_seq_len: int) -> float:
        denom = self.max_image_seq_len - self.base_image_seq_len
        if denom == 0:
            return float(self.base_shift)
        m = (self.max_shift - self.base_shift) / denom
        b = self.base_shift - m * self.base_image_seq_len
        return float(image_seq_len) * m + b

    def _apply_time_schedule(
        self,
        t: torch.Tensor,
        image_seq_len: int,
        timestep_shift: float,
    ) -> torch.Tensor:
        schedule = self.time_schedule
        if timestep_shift != 1:
            schedule = "standard"
        sigma = 1 - t
        if schedule == "standard":
            shift = timestep_shift
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
        elif schedule == "dynamic":
            mu_t = t.new_tensor(self._calculate_dynamic_mu(image_seq_len))
            if self.time_shift_type == "exponential":
                shift = torch.exp(mu_t)
                sigma = shift * sigma / (1 + (shift - 1) * sigma)
            elif self.time_shift_type == "linear":
                sigma = mu_t / (mu_t + (1 / sigma - 1))
            else:
                raise ValueError(f"Unsupported time_shift_type: {self.time_shift_type}")
        else:
            raise ValueError(f"Unsupported time_schedule: {schedule}")
        return 1 - sigma

    def scheduler_params(self, request: FlowRequestParams) -> dict[str, Any]:
        width, height = request.image_size
        merge_size = int(1 / self.downsample_ratio)
        token_h = height // (self.patch_size * merge_size)
        token_w = width // (self.patch_size * merge_size)
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        image_seq_len = token_h * token_w
        noise_scale = self._computed_noise_scale(grid_h, grid_w, merge_size)
        timesteps = torch.linspace(0.0, 1.0, request.num_steps + 1, device=self.torch_device)
        if request.enable_timestep_shift:
            timesteps = self._apply_time_schedule(
                timesteps,
                image_seq_len,
                request.timestep_shift,
            )
        t_values = [float(x) for x in timesteps.detach().cpu().tolist()]
        raw_keys = [
            "fm_head_dim",
            "fm_head_layers",
            "fm_head_mlp_ratio",
            "timestep_shift",
            "time_schedule",
            "time_shift_type",
            "base_shift",
            "max_shift",
            "base_image_seq_len",
            "max_image_seq_len",
            "noise_scale",
            "noise_scale_mode",
            "noise_scale_base_image_seq_len",
            "noise_scale_max_value",
            "add_noise_scale_embedding",
            "P_mean",
            "P_std",
            "t_eps",
            "use_pixel_head",
            "use_adaLN",
        ]
        return {
            "backend": "sglang_native_sensenova_u1_full_sequence_flow",
            "request": asdict(request),
            "patch_size": self.patch_size,
            "downsample_ratio": self.downsample_ratio,
            "merge_size": merge_size,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "token_h": token_h,
            "token_w": token_w,
            "image_seq_len": image_seq_len,
            "computed_noise_scale": noise_scale,
            "raw_config": {key: self.raw_config.get(key) for key in raw_keys},
            "timesteps": t_values,
            "step_table": [
                {
                    "step": i,
                    "t": t_values[i],
                    "t_next": t_values[i + 1],
                    "dt": t_values[i + 1] - t_values[i],
                    "use_cfg": bool(
                        (
                            t_values[i] > request.cfg_interval[0]
                            and t_values[i] < request.cfg_interval[1]
                        )
                        or request.cfg_interval[0] == 0
                    ),
                }
                for i in range(request.num_steps)
            ],
            "flow_load_report": self.flow_load_report,
        }

    @staticmethod
    def patchify(
        images: torch.Tensor,
        patch_size: int,
        *,
        channel_first: bool = False,
    ) -> torch.Tensor:
        h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
        x = images.reshape(images.shape[0], 3, h, patch_size, w, patch_size)
        if channel_first:
            x = torch.einsum("nchpwq->nhwcpq", x)
        else:
            x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(images.shape[0], h * w, patch_size**2 * 3)

    @staticmethod
    def unpatchify(
        x: torch.Tensor,
        patch_size: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        h_tokens = h // patch_size
        w_tokens = w // patch_size
        x = x.reshape(x.shape[0], h_tokens, w_tokens, patch_size, patch_size, 3)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], 3, h, w)

    def _computed_noise_scale(self, grid_h: int, grid_w: int, merge_size: int) -> float:
        noise_scale = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base)
            noise_scale = scale * float(self.noise_scale)
            if self.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
        return min(noise_scale, self.noise_scale_max_value)

    def _build_t2i_image_indexes(
        self,
        token_h: int,
        token_w: int,
        text_len: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        t_image = torch.full(
            (token_h * token_w,),
            int(text_len),
            dtype=torch.long,
            device=device,
        )
        idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
        h_image = idx // token_w
        w_image = idx % token_w
        return torch.stack([t_image, h_image, w_image], dim=0)

    def _get_thw_indexes(
        self,
        input_ids: torch.Tensor,
        grid_hw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        img_start_shift = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=input_ids.device),
                (input_ids == self.img_start_token_id).long(),
            ],
            dim=0,
        )[:-1]
        not_img_token = (input_ids != self.img_context_token_id).long()
        t_indexes = (img_start_shift + not_img_token).cumsum(0) - 1
        h_indexes = torch.zeros_like(t_indexes)
        w_indexes = torch.zeros_like(t_indexes)
        if grid_hw is not None:
            selected = input_ids == self.img_context_token_id
            if bool(selected.any().item()):
                merge_size = int(1 / self.downsample_ratio)
                abs_w, abs_h = build_abs_positions_from_grid_hw(
                    grid_hw // merge_size,
                    device=input_ids.device,
                )
                h_indexes[selected] = abs_h.to(t_indexes.dtype)
                w_indexes[selected] = abs_w.to(t_indexes.dtype)
        return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

    def _token_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        model = self.executor.model_worker.model_runner.model
        device = torch.device(self.executor.model_worker.device)
        dtype = next(model.parameters()).dtype
        flat_ids = input_ids.reshape(-1).to(device=device, dtype=torch.long)
        with torch.inference_mode(), block_hf_modeling_imports():
            return model.get_input_embeddings()(flat_ids).to(dtype=dtype)

    @staticmethod
    def _native_flow_prefix_cache_enabled() -> bool:
        value = os.environ.get(
            "SENSENOVA_U1_NATIVE_FLOW_PREFIX_CACHE",
            "",
        ).lower()
        return value not in {"0", "false", "no", "off"}

    @staticmethod
    def _native_flow_prefill_cuda_graph_enabled() -> bool:
        value = os.environ.get(
            "SENSENOVA_U1_NATIVE_FLOW_PREFILL_CUDA_GRAPH",
            "",
        ).lower()
        return value not in {"0", "false", "no", "off"}

    def _load_input_images(
        self,
        images: list[Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not images:
            return None, None
        pixel_values = []
        grid_hw = []
        max_pixels = min(2048 * 2048, (4096 * 4096) // max(1, len(images)))
        for image in images:
            cur_pixels, cur_grid = load_image_native_tensor(
                image,
                patch_size=self.patch_size,
                downsample_ratio=self.downsample_ratio,
                min_pixels=512 * 512,
                max_pixels=max_pixels,
                upscale=False,
            )
            pixel_values.append(cur_pixels)
            grid_hw.append(cur_grid)
        return torch.cat(pixel_values, dim=0), torch.cat(grid_hw, dim=0)

    def _replace_image_tokens(self, query: str, grid_hw: torch.Tensor | None) -> str:
        if grid_hw is None:
            return query
        for i in range(int(grid_hw.shape[0])):
            num_patch_token = int(
                grid_hw[i, 0] * grid_hw[i, 1] * self.downsample_ratio**2
            )
            image_tokens = (
                "<img>"
                + "<IMG_CONTEXT>" * num_patch_token
                + "</img>"
            )
            query = query.replace("<image>", image_tokens, 1)
        return query

    def _build_prefix(
        self,
        *,
        prompt: str,
        images: list[Any],
        system_message: str,
        assistant_append: str,
    ) -> NativeFlowPrefix:
        image_count = prompt.count("<image>")
        if len(images) > image_count:
            if image_count == 0 and len(images) > 1:
                prompt = "".join(
                    f"Image-{idx + 1}:<image>\n" for idx in range(len(images))
                ) + prompt
            else:
                prompt = "<image>\n" * (len(images) - image_count) + prompt
        pixel_values, grid_hw = self._load_input_images(images)
        query = _build_neo_prompt(
            user_text=prompt,
            system_message=system_message,
            assistant_append=assistant_append,
        )
        query = self._replace_image_tokens(query, grid_hw)
        input_ids = self.tokenizer(query, return_tensors="pt")["input_ids"][0]
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
                raise RuntimeError("native prefix image compose returned no embeds")
        else:
            input_embeds = self._token_embeds(input_ids)

        cache_reuse_enabled = self._native_flow_prefix_cache_enabled()
        cache_extra_key = (
            self.executor._cache_extra_key(
                image_token_tag=image_token_tag,
                pixel_values=pixel_values,
                grid_hw=grid_hw,
                input_embeds=input_embeds,
            )
            if cache_reuse_enabled
            else None
        )
        return NativeFlowPrefix(
            input_ids=input_ids.to(dtype=torch.long),
            indexes=indexes.to(dtype=torch.long),
            image_token_tag=image_token_tag.to(dtype=torch.bool),
            input_embeds=input_embeds.detach(),
            cache_extra_key=cache_extra_key,
            cache_insert_log={
                "skipped": not cache_reuse_enabled,
                "reason": (
                    "pending_static_prefix_insert"
                    if cache_reuse_enabled
                    else "native_flow_prefix_cache_disabled"
                ),
                "prefix_tokens": int(input_ids.numel()),
                "image_token_count": int(image_token_tag.sum().item()),
            },
            cache_reuse_enabled=cache_reuse_enabled,
        )

    def _prime_prefix_cache(self, prefix: NativeFlowPrefix) -> None:
        if not prefix.cache_reuse_enabled:
            return
        cached_prefix_len = self.executor.cached_prefix_length(
            input_ids=prefix.input_ids,
            cache_extra_key=prefix.cache_extra_key,
        )
        if cached_prefix_len == int(prefix.input_ids.numel()):
            prefix.cache_insert_log = {
                "skipped": True,
                "reason": "static_prefix_already_cached",
                "prefix_tokens": int(prefix.input_ids.numel()),
                "image_token_count": int(prefix.image_token_tag.sum().item()),
                "cache_extra_key": prefix.cache_extra_key,
                "cache_hit_tokens": cached_prefix_len,
            }
            return
        result = self.executor.run_prefill(
            request_id=f"u1-native-flow-prefix-{time.time_ns()}",
            input_ids=prefix.input_ids,
            indexes=prefix.indexes,
            image_token_tag=prefix.image_token_tag,
            image_gen_indicators=torch.zeros_like(
                prefix.image_token_tag,
                dtype=torch.bool,
            ),
            input_embeds=prefix.input_embeds,
            cache_extra_key=prefix.cache_extra_key,
            cache_insert=True,
        )
        prefix.cache_extra_key = result.cache_extra_key
        prefix.cache_insert_log = {
            "skipped": False,
            "reason": "static_prefix_inserted",
            "prefix_tokens": int(prefix.input_ids.numel()),
            "image_token_count": int(prefix.image_token_tag.sum().item()),
            "cache_extra_key": result.cache_extra_key,
            "cache_inserted": bool(result.cache_inserted),
            "cache_hit_tokens_before_insert": cached_prefix_len,
            "forward_elapsed_s": float(result.forward_elapsed_s),
            "forward_batch_log": result.forward_batch_log,
        }
        if not result.cache_inserted:
            raise RuntimeError("native flow static prefix was not inserted into Radix cache")

    def _predict_v(
        self,
        *,
        prefix: NativeFlowPrefix,
        image_embeds: torch.Tensor,
        indexes_image: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        request_id: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        bsz, image_token_num, _ = image_embeds.shape
        if bsz != 1:
            raise NotImplementedError("native full-sequence flow currently supports batch_size=1")
        device = self.torch_device
        img_ids = torch.full(
            (image_token_num,),
            self.img_context_token_id,
            dtype=torch.long,
        )
        full_input_ids = torch.cat([prefix.input_ids.cpu(), img_ids.cpu()], dim=0)
        full_indexes = torch.cat(
            [prefix.indexes.cpu(), indexes_image.detach().cpu()],
            dim=1,
        )
        full_image_tag = torch.cat(
            [
                prefix.image_token_tag.cpu(),
                torch.ones(image_token_num, dtype=torch.bool),
            ],
            dim=0,
        )
        full_gen = torch.cat(
            [
                torch.zeros(prefix.input_ids.numel(), dtype=torch.bool),
                torch.ones(image_token_num, dtype=torch.bool),
            ],
            dim=0,
        )
        full_embeds = torch.cat(
            [
                prefix.input_embeds.to(device=device),
                image_embeds.reshape(image_token_num, -1).to(device=device),
            ],
            dim=0,
        )
        hidden_result = self.executor.run_hidden_prefill(
            {
                "request_id": request_id,
                "input_ids": full_input_ids,
                "indexes": full_indexes,
                "image_token_tag": full_image_tag,
                "image_gen_indicators": full_gen,
                "input_embeds": full_embeds,
                "cache_extra_key": prefix.cache_extra_key,
            },
            cache_insert=False,
        )
        if prefix.cache_reuse_enabled:
            prefix_lens = list(
                hidden_result.forward_batch_log.get(
                    "extend_prefix_lens_cpu",
                    [],
                )
            )
            extend_lens = list(
                hidden_result.forward_batch_log.get(
                    "extend_seq_lens_cpu",
                    [],
                )
            )
            expected_prefix_len = int(prefix.input_ids.numel())
            if prefix_lens != [expected_prefix_len]:
                raise RuntimeError(
                    "native flow prefix cache miss: "
                    f"expected_prefix_len={expected_prefix_len} actual={prefix_lens}"
                )
            if extend_lens != [image_token_num]:
                raise RuntimeError(
                    "native flow cached extend length mismatch: "
                    f"expected_image_tokens={image_token_num} actual={extend_lens}"
                )
        image_hidden = hidden_result.hidden_states[-image_token_num:].view(
            bsz,
            image_token_num,
            -1,
        )
        x_pred = self.fm_modules.fm_head(image_hidden).view_as(z).to(z.device)
        v_pred = (x_pred - z) / (1 - t).clamp_min(float(self.raw_config.get("t_eps", 0.05)))
        return v_pred, hidden_result.to_dict()

    def _generate_conditioned_tensor(
        self,
        request: FlowRequestParams,
        *,
        images: list[Any],
        system_message: str,
        assistant_append: str,
    ) -> tuple[torch.Tensor, NativeFlowRunStats]:
        if request.batch_size != 1:
            raise NotImplementedError("native U1 flow supports batch_size=1 in this path")
        if request.cfg_scale != 1.0 or request.img_cfg_scale != 1.0:
            raise NotImplementedError("native U1 flow path currently supports cfg scales of 1.0")
        assert_no_hf_modeling_imported(context="before native flow generation")
        width, height = request.image_size
        merge_size = int(1 / self.downsample_ratio)
        token_h = height // (self.patch_size * merge_size)
        token_w = width // (self.patch_size * merge_size)
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        image_token_num = token_h * token_w
        gen_grid_hw = torch.tensor(
            [[grid_h, grid_w]],
            dtype=torch.long,
            device=self.torch_device,
        )
        prefix = self._build_prefix(
            prompt=request.prompt,
            images=images,
            system_message=system_message,
            assistant_append=assistant_append,
        )
        self._prime_prefix_cache(prefix)
        image_t_index = int(prefix.indexes[0].max().item()) + 1
        indexes_image = self._build_t2i_image_indexes(
            token_h,
            token_w,
            image_t_index,
            device=self.torch_device,
        )
        noise_scale = self._computed_noise_scale(grid_h, grid_w, merge_size)
        generator = torch.Generator(self.torch_device).manual_seed(request.seed)
        image_prediction = noise_scale * torch.randn(
            (1, 3, height, width),
            device=self.torch_device,
            dtype=next(self.fm_modules.parameters()).dtype,
            generator=generator,
        )
        timesteps = torch.linspace(
            0.0,
            1.0,
            request.num_steps + 1,
            device=self.torch_device,
        )
        if request.enable_timestep_shift:
            timesteps = self._apply_time_schedule(
                timesteps,
                image_token_num,
                request.timestep_shift,
            )

        forward_elapsed: list[float] = []
        hidden_logs: list[dict[str, Any]] = []
        backend_name = None
        for step_i in range(request.num_steps):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]
            z = self.patchify(image_prediction, self.patch_size * merge_size)
            image_input = self.patchify(
                image_prediction,
                self.patch_size,
                channel_first=True,
            )
            image_embeds = self.fm_modules.vision_model_mot_gen(
                image_input.view(grid_h * grid_w, -1),
                gen_grid_hw,
            ).view(1, image_token_num, -1)
            t_expanded = t.expand(image_token_num)
            timestep_embeddings = self.fm_modules.timestep_embedder(
                t_expanded,
            ).view(1, image_token_num, -1)
            if self.add_noise_scale_embedding:
                noise_scale_tensor = torch.full_like(
                    t_expanded,
                    noise_scale / self.noise_scale_max_value,
                )
                timestep_embeddings = timestep_embeddings + self.fm_modules.noise_scale_embedder(
                    noise_scale_tensor,
                ).view(1, image_token_num, -1)
            image_embeds = image_embeds + timestep_embeddings.to(image_embeds.dtype)
            v_pred, hidden_log = self._predict_v(
                prefix=prefix,
                image_embeds=image_embeds,
                indexes_image=indexes_image,
                t=t,
                z=z,
                request_id=f"u1-native-flow-step-{time.time_ns()}",
            )
            forward_elapsed.append(float(hidden_log["forward_elapsed_s"]))
            backend_name = str(hidden_log["backend_name"])
            hidden_logs.append(hidden_log)
            z = z + (t_next - t) * v_pred
            image_prediction = self.unpatchify(
                z,
                self.patch_size * merge_size,
                height,
                width,
            )

        stats = NativeFlowRunStats(
            prefix_tokens=int(prefix.input_ids.numel()),
            image_tokens=image_token_num,
            forward_elapsed_s=forward_elapsed,
            backend_name=backend_name,
            hf_modeling_imported_after=_blocked_hf_modeling_modules(),
            hidden_prefill_logs=hidden_logs,
            prefix_cache_enabled=prefix.cache_reuse_enabled,
            prefix_cache_log=prefix.cache_insert_log,
        )
        self.last_native_flow_stats = stats.to_dict()
        assert_no_hf_modeling_imported(context="after native flow generation")
        return image_prediction.detach(), stats

    def _flow_params_from_request(self, inputs: Any, params: dict[str, Any]) -> tuple[FlowRequestParams, list[Any]]:
        data = inputs if isinstance(inputs, dict) else {"prompt": str(inputs)}
        merged = {**data, **params}
        mode = str(merged.get("mode") or merged.get("task") or "t2i").lower()
        if mode in {"text_to_image", "txt2img"}:
            mode = "t2i"
        if mode in {"image_to_image", "edit", "editing"}:
            mode = "it2i"
        if mode not in {"t2i", "it2i"}:
            raise ValueError(f"Unsupported U1 flow mode: {mode!r}")
        width = int(merged.get("width", merged.get("image_width", 256)))
        height = int(merged.get("height", merged.get("image_height", 256)))
        cfg_interval_value = merged.get("cfg_interval", (0.0, 1.0))
        cfg_interval = tuple(float(x) for x in cfg_interval_value)
        if len(cfg_interval) != 2:
            raise ValueError("cfg_interval must contain exactly two floats.")
        images = [_coerce_image(img) for img in (merged.get("images") or [])]
        if "image" in merged:
            images.append(_coerce_image(merged["image"]))
        request = FlowRequestParams(
            mode=mode,
            prompt=str(merged.get("prompt", "")),
            image_size=(width, height),
            cfg_scale=float(merged.get("cfg_scale", 1.0)),
            img_cfg_scale=float(merged.get("img_cfg_scale", 1.0)),
            cfg_norm=str(merged.get("cfg_norm", "none")),
            timestep_shift=float(merged.get("timestep_shift", 1.0)),
            enable_timestep_shift=bool(merged.get("enable_timestep_shift", True)),
            cfg_interval=(float(cfg_interval[0]), float(cfg_interval[1])),
            num_steps=int(merged.get("num_steps", 2)),
            batch_size=int(merged.get("batch_size", 1)),
            t_eps=float(merged.get("t_eps", 0.05)),
            think_mode=bool(merged.get("think_mode", False)),
            seed=int(merged.get("seed", 20260813)),
        )
        return request, images

    @torch.inference_mode()
    def generate_t2i_tensor(
        self,
        request: FlowRequestParams,
        *,
        use_official_hybrid_mask: bool = False,
    ) -> tuple[torch.Tensor, str]:
        if use_official_hybrid_mask:
            raise RuntimeError("native flow runner cannot use official HF hybrid mask")
        if request.think_mode:
            raise NotImplementedError("native flow path currently supports think_mode=False")
        return self._generate_conditioned_tensor(
            request,
            images=[],
            system_message=SYSTEM_MESSAGE_FOR_GEN,
            assistant_append="<think>\n\n</think>\n\n<img>",
        )[0], ""

    @torch.inference_mode()
    def generate_it2i_tensor(
        self,
        request: FlowRequestParams,
        images: list[Any],
        *,
        use_official_hybrid_mask: bool = False,
    ) -> tuple[torch.Tensor, str]:
        if use_official_hybrid_mask:
            raise RuntimeError("native flow runner cannot use official HF hybrid mask")
        if not images:
            raise ValueError("IT2I generation requires at least one input image.")
        if request.think_mode:
            raise NotImplementedError("native flow path currently supports think_mode=False")
        return self._generate_conditioned_tensor(
            request,
            images=[_coerce_image(image) for image in images],
            system_message=SYSTEM_MESSAGE_FOR_GEN,
            assistant_append="<think>\n\n</think>\n\n<img>",
        )[0], ""

    @torch.inference_mode()
    def generate_interleave_image_tensor(
        self,
        request: FlowRequestParams,
        images: list[Any],
        *,
        system_message: str,
        assistant_prefix: str = "<think>\n\n</think>\n\n<img>",
    ) -> tuple[torch.Tensor, NativeFlowRunStats]:
        return self._generate_conditioned_tensor(
            request,
            images=[_coerce_image(image) for image in images],
            system_message=system_message,
            assistant_append=assistant_prefix,
        )

    def complete_payload(self, payload: StagePayload) -> dict[str, Any]:
        inputs, params, request_id = _extract_request(payload)
        request, images = self._flow_params_from_request(inputs, params)
        start = time.perf_counter()
        if request.mode == "t2i":
            tensor, think_text = self.generate_t2i_tensor(request)
        else:
            tensor, think_text = self.generate_it2i_tensor(request, images)
        elapsed = time.perf_counter() - start
        pil_images = u1_tensor_to_pil(tensor)
        return {
            "request_id": request_id,
            "mode": request.mode,
            "images": [
                {
                    "type": "image",
                    "format": "png",
                    "data": pil_to_data_url(img),
                    "width": img.width,
                    "height": img.height,
                }
                for img in pil_images
            ],
            "think_text": think_text,
            "scheduler_params": self.scheduler_params(request),
            "native_flow_stats": self.last_native_flow_stats,
            "usage": {
                "image_count": len(pil_images),
                "num_steps": request.num_steps,
                "engine_time_s": elapsed,
            },
            "stage_name": "u1_flow",
            "backend": "native_sglang_flow_matching",
        }


__all__ = [
    "FlowRequestParams",
    "SenseNovaU1FlowMatchingFallbackRunner",
    "SenseNovaU1FlowMatchingRunner",
    "SenseNovaU1NativeFlowModules",
    "compare_pil_images",
    "denormalize_u1_image_tensor",
    "load_image_native_tensor",
    "pil_to_data_url",
    "u1_tensor_to_pil",
]
