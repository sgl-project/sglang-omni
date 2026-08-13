# SPDX-License-Identifier: Apache-2.0
"""SenseNova U1 flow-matching image generation runner.

This is the M3 HF-compatible in-pipeline path. It keeps U1 generation inside a
SGLang-Omni stage lifecycle, while delegating the exact numerical kernels to
the official NEOChatModel implementation already used by the M1/M2 fallback.
"""

from __future__ import annotations

import base64
import io
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

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


class SenseNovaU1FlowMatchingRunner(SenseNovaU1UnderstandingRunner):
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
            "backend": "sglang_omni_sensenova_u1_flow_matching_hf_compatible",
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
        use_official_hybrid_mask: bool = False,
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
        use_official_hybrid_mask: bool = False,
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
            "backend": "hf_compatible_flow_matching_fallback",
        }


__all__ = [
    "FlowRequestParams",
    "SenseNovaU1FlowMatchingRunner",
    "compare_pil_images",
    "denormalize_u1_image_tensor",
    "pil_to_data_url",
    "u1_tensor_to_pil",
]
