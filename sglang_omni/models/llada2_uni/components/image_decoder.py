# SPDX-License-Identifier: Apache-2.0
# Adapted and modified from inclusionAI/LLaDA2.0-Uni (Apache-2.0):
# decoder/decode.py and decoder/transport/ at commit
# 3457030a9c737f77f38ad5ff657e7659243d3444.
"""Native LLaDA2-Uni VQ decoder using Diffusers ZImage and a VAE."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from PIL import Image
from torch import nn

from sglang_omni.models.llada2_uni.components.sigvq import SigVQ
from sglang_omni.models.weight_loader import resolve_model_path

logger = logging.getLogger(__name__)


def _remap_zimage_checkpoint_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map the reference decoder's semantic embedder to Diffusers ZImage."""
    return {
        key.replace("semantic_embedder.", "cap_embedder.", 1): value
        for key, value in state_dict.items()
    }


def euler_sample(
    initial: torch.Tensor,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    num_steps: int,
    time_shifting_factor: float = 6.0,
) -> torch.Tensor:
    """Integrate the reference linear velocity path with fixed Euler steps."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    times = torch.linspace(
        0.0,
        1.0,
        num_steps,
        dtype=torch.float32,
        device=initial.device,
    )
    if time_shifting_factor:
        times = times / (times + time_shifting_factor - time_shifting_factor * times)
    value = initial.float()
    for current, following in zip(times[:-1], times[1:]):
        time_batch = current.expand(value.shape[0])
        value = value + (following - current) * velocity_fn(value, time_batch).float()
    return value


def _decoder_velocity_fn(
    model: nn.Module,
    cap_pos: list[torch.Tensor],
    cap_neg: list[torch.Tensor],
    *,
    cfg_scale: float,
    patch_size: int,
    frame_patch_size: int,
    dtype: torch.dtype,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def _forward(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if cfg_scale > 0.0:
            output = model(
                x=list(value.to(dtype).repeat(2, 1, 1, 1, 1).unbind(0)),
                t=time.repeat(2),
                cap_feats=cap_pos + cap_neg,
                patch_size=patch_size,
                f_patch_size=frame_patch_size,
                return_dict=False,
            )[0]
            positive = torch.stack([item.float() for item in output[: len(cap_pos)]])
            negative = torch.stack([item.float() for item in output[len(cap_pos) :]])
            guided = positive + cfg_scale * (positive - negative)
            positive_norm = torch.linalg.vector_norm(
                positive.flatten(1), dim=1, keepdim=True
            )
            guided_norm = torch.linalg.vector_norm(
                guided.flatten(1), dim=1, keepdim=True
            ).clamp_min(1e-6)
            scale = torch.minimum(
                torch.ones_like(guided_norm), positive_norm / guided_norm
            )
            return guided * scale.view(-1, 1, 1, 1, 1)

        output = model(
            x=list(value.to(dtype).unbind(0)),
            t=time,
            cap_feats=cap_pos,
            patch_size=patch_size,
            f_patch_size=frame_patch_size,
            return_dict=False,
        )[0]
        return torch.stack([item.float() for item in output])

    return _forward


class LLaDA2ImageDecoder:
    """Lazy single-device SigVQ -> ZImage -> VAE native decoder."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        decode_mode: str = "normal",
        num_steps: int = 50,
        resolution_multiplier: int = 2,
    ):
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        self.model_path = Path(
            resolve_model_path(
                model_path,
                local_files_only=Path(model_path).exists(),
            )
        )
        self.device = resolved_device
        self.dtype = dtype
        self.decode_mode = decode_mode
        self.num_steps = num_steps
        self.resolution_multiplier = resolution_multiplier

        self._sigvq: SigVQ | None = None
        self._diffusion_model: nn.Module | None = None
        self._diffusion_mode: str | None = None
        self._diffusion_config: dict[str, Any] | None = None
        self._vae: nn.Module | None = None

    @staticmethod
    def _validate_decode_inputs(
        *,
        token_ids: list[int],
        height: int,
        width: int,
        num_steps: int,
    ) -> None:
        if height < 1 or width < 1:
            raise ValueError("decoder grid dimensions must be positive")
        if len(token_ids) != height * width:
            raise ValueError(
                "image decoder requires exactly h * w VQ tokens: "
                f"got {len(token_ids)} for {height}x{width}"
            )
        if any(token_id < 0 or token_id >= 16384 for token_id in token_ids):
            raise ValueError("image decoder codebook ids must be in [0, 16384)")
        if num_steps < 1:
            raise ValueError("image decoder num_steps must be positive")

    def _ensure_sigvq(self) -> SigVQ:
        if self._sigvq is None:
            model = SigVQ().to(self.device, dtype=self.dtype)
            model.load_state_dict(
                torch.load(
                    self.model_path / "image_tokenizer" / "sigvq_embedding.pt",
                    map_location=self.device,
                    weights_only=True,
                )
            )
            self._sigvq = model.eval()
        return self._sigvq

    def _ensure_diffusion_model(self, mode: str) -> tuple[nn.Module, dict[str, Any]]:
        if self._diffusion_model is not None and self._diffusion_mode == mode:
            assert self._diffusion_config is not None
            return self._diffusion_model, self._diffusion_config

        from diffusers.models.transformers.transformer_z_image import (
            ZImageTransformer2DModel,
        )
        from safetensors.torch import load_file

        decoder_dir = self.model_path / (
            "decoder-turbo" if mode == "decoder-turbo" else "decoder"
        )
        with (decoder_dir / "config.json").open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        config["axes_lens"] = [32768, 1024, 1024]
        config["cap_feat_dim"] = 4096
        kwargs = {
            "all_patch_size": tuple(config["all_patch_size"]),
            "all_f_patch_size": tuple(config["all_f_patch_size"]),
            "in_channels": config["in_channels"],
            "dim": config["dim"],
            "n_layers": config["n_layers"],
            "n_refiner_layers": config["n_refiner_layers"],
            "n_heads": config["n_heads"],
            "n_kv_heads": config["n_kv_heads"],
            "norm_eps": config["norm_eps"],
            "qk_norm": config["qk_norm"],
            "cap_feat_dim": config["cap_feat_dim"],
            "rope_theta": config["rope_theta"],
            "t_scale": config["t_scale"],
            "axes_dims": list(config["axes_dims"]),
            "axes_lens": list(config["axes_lens"]),
        }
        with torch.device("meta"):
            model = ZImageTransformer2DModel(**kwargs)
        state_dict = load_file(
            str(decoder_dir / "model.safetensors"), device=str(self.device)
        )
        state_dict = _remap_zimage_checkpoint_keys(state_dict)
        missing, unexpected = model.load_state_dict(
            state_dict, strict=False, assign=True
        )
        if missing or unexpected:
            raise RuntimeError(
                "invalid ZImage decoder checkpoint: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        self._diffusion_model = model.to(device=self.device, dtype=self.dtype).eval()
        self._diffusion_mode = mode
        self._diffusion_config = config
        return self._diffusion_model, config

    def _ensure_vae(self) -> nn.Module:
        if self._vae is None:
            from diffusers import AutoencoderKL

            self._vae = (
                AutoencoderKL.from_pretrained(
                    self.model_path / "vae", torch_dtype=self.dtype
                )
                .to(self.device)
                .eval()
            )
        return self._vae

    @torch.inference_mode()
    def decode(
        self,
        token_ids: list[int],
        height: int,
        width: int,
        *,
        decode_mode: str | None = None,
        num_steps: int | None = None,
        resolution_multiplier: int | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        mode = decode_mode or self.decode_mode
        if mode not in {"normal", "decoder-turbo"}:
            raise ValueError(f"unsupported image decoder mode: {mode}")
        steps = num_steps if num_steps is not None else self.num_steps
        multiplier = (
            resolution_multiplier
            if resolution_multiplier is not None
            else self.resolution_multiplier
        )
        self._validate_decode_inputs(
            token_ids=token_ids,
            height=height,
            width=width,
            num_steps=steps,
        )
        if mode == "decoder-turbo":
            raise NotImplementedError("decoder-turbo is not supported")
        if multiplier < 1:
            raise ValueError("resolution_multiplier must be positive")

        tokens = torch.tensor(token_ids, device=self.device).view(1, 1, height, width)
        upsampled = (
            functional.interpolate(tokens.float(), scale_factor=2, mode="nearest")
            .long()
            .flatten(1)
        )
        cap_pos = [self._ensure_sigvq()(upsampled).squeeze(0)]
        cap_neg = [torch.zeros_like(cap_pos[0])]

        model, config = self._ensure_diffusion_model(mode)
        pixel_height = height * 16 * multiplier
        pixel_width = width * 16 * multiplier
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        latent = torch.randn(
            [
                1,
                16,
                1,
                2 * (pixel_height // 16),
                2 * (pixel_width // 16),
            ],
            device=self.device,
            generator=generator,
        )
        velocity_fn = _decoder_velocity_fn(
            model,
            cap_pos,
            cap_neg,
            cfg_scale=0.0 if mode == "decoder-turbo" else 1.0,
            patch_size=config.get("all_patch_size", (2,))[0],
            frame_patch_size=config.get("all_f_patch_size", (1,))[0],
            dtype=self.dtype,
        )
        sample = euler_sample(latent, velocity_fn, num_steps=steps).squeeze(2)

        vae = self._ensure_vae()
        latent = (sample.to(self.dtype) / vae.config.scaling_factor) + (
            vae.config.shift_factor
        )
        pixels = ((vae.decode(latent, return_dict=False)[0] + 1) / 2).clamp_(0, 1)
        array = (
            pixels[0]
            .float()
            .mul(255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return Image.fromarray(array)

    def decode_to_bytes(
        self,
        token_ids: list[int],
        height: int,
        width: int,
        **decode_kwargs: Any,
    ) -> tuple[bytes, int, int]:
        image = self.decode(token_ids, height, width, **decode_kwargs)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue(), image.width, image.height
