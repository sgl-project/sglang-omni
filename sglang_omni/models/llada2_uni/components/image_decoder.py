"""Decode LLaDA2 semantic tokens through SigVQ, diffusion, and a VAE."""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL.Image import Image
from torchvision.transforms.functional import to_pil_image

from sglang_omni.models.llada2_uni.components.decoder_model import (
    DecoderConfig,
    ZImageTransformer2DModelWrapper,
)
from sglang_omni.models.llada2_uni.components.decoder_runtime import (
    DecoderRuntimeHandle,
)
from sglang_omni.models.llada2_uni.components.sigvq import SigVQ
from sglang_omni.models.llada2_uni.components.transport import (
    VelocityModel,
    sample_velocity,
)
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)

DecoderBackend = Literal["diffusers", "sglang"]
DecoderMode = Literal["normal", "decoder-turbo"]


def create_decoder_model_fn(
    model: ZImageTransformer2DModelWrapper,
    positive_caption_features: list[torch.Tensor],
    negative_caption_features: list[torch.Tensor],
    cfg_scale: float,
    patch_size: int,
    f_patch_size: int,
    dtype: torch.dtype,
) -> VelocityModel:
    batch_size = len(positive_caption_features)
    combined_caption_features = positive_caption_features + negative_caption_features

    def predict_velocity(
        latents: torch.Tensor, timestep: torch.Tensor | float
    ) -> torch.Tensor:
        batch_timesteps = (
            torch.tensor([timestep], device=latents.device, dtype=torch.float32)
            if not isinstance(timestep, torch.Tensor)
            else timestep.float()
        )
        if batch_timesteps.dim() == 0:
            batch_timesteps = batch_timesteps.unsqueeze(0)
        else:
            pass
        if batch_timesteps.shape[0] == 1 and latents.shape[0] > 1:
            batch_timesteps = batch_timesteps.expand(latents.shape[0])
        else:
            pass
        if cfg_scale > 0:
            model_output = model(
                x=list(latents.to(dtype).repeat(2, 1, 1, 1, 1).unbind(0)),
                t=batch_timesteps.repeat(2),
                cap_feats=combined_caption_features,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                return_dict=False,
            )
            positive_predictions = model_output[0][:batch_size]
            negative_predictions = model_output[0][batch_size:]
            guided_predictions = []
            for positive, negative in zip(positive_predictions, negative_predictions):
                positive, negative = positive.float(), negative.float()
                guided = positive + cfg_scale * (positive - negative)
                positive_norm = torch.linalg.vector_norm(positive)
                guided_norm = torch.linalg.vector_norm(guided)
                safe_norm = torch.where(
                    guided_norm == 0, torch.ones_like(guided_norm), guided_norm
                )
                guided *= torch.where(
                    guided_norm > positive_norm,
                    positive_norm / safe_norm,
                    torch.ones_like(guided_norm),
                )
                guided_predictions.append(guided)
            return torch.stack(guided_predictions)
        else:
            pass
        model_output = model(
            x=list(latents.to(dtype).unbind(0)),
            t=batch_timesteps,
            cap_feats=positive_caption_features,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            return_dict=False,
        )
        return torch.stack([prediction.float() for prediction in model_output[0]])

    return predict_velocity


class LLaDA2ImageDecoder:
    """Decode semantic tokens using Diffusers or a caller-owned SGLang runtime.

    The caller must enter runtime.compute_context() for SGLang operations.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        decode_mode: DecoderMode = "normal",
        num_steps: int = 50,
        resolution_multiplier: int = 2,
        *,
        backend: DecoderBackend = "diffusers",
        runtime: DecoderRuntimeHandle | None = None,
    ) -> None:
        if backend not in {"diffusers", "sglang"}:
            raise ValueError(f"Unsupported image decoder backend: {backend!r}")
        else:
            pass
        if backend == "sglang" and runtime is None:
            raise ValueError("sglang image decoder requires an initialized runtime")
        else:
            pass
        if backend == "diffusers" and runtime is not None:
            raise ValueError("diffusers image decoder cannot use an SGLang runtime")
        else:
            pass
        self.validate_settings(decode_mode, num_steps, resolution_multiplier)
        self.backend = backend
        requested_device = torch.device(device)
        self.device = resolve_concrete_device(
            requested_device.type, requested_device.index
        )
        self.dtype = dtype
        self.runtime = runtime
        if runtime is not None:
            if runtime.device != self.device or runtime.dtype != self.dtype:
                raise ValueError("Decoder model and runtime device/dtype must match")
            else:
                pass
        else:
            pass
        self.model_path = str(resolve_model_path(model_path))
        self.decode_mode: DecoderMode = decode_mode
        self.num_steps = num_steps
        self.resolution_multiplier = resolution_multiplier

        self.sigvq: SigVQ | None = None
        self.diff_model: ZImageTransformer2DModelWrapper | None = None
        self.diff_model_mode: DecoderMode | None = None
        self.vae: AutoencoderKL | None = None
        self.diff_config: DecoderConfig | None = None

    @staticmethod
    def validate_settings(
        mode: DecoderMode, steps: int, resolution_multiplier: int
    ) -> None:
        if mode not in {"normal", "decoder-turbo"}:
            raise ValueError(f"Unsupported image decoder mode: {mode!r}")
        else:
            pass
        minimum_steps = 1 if mode == "decoder-turbo" else 2
        if not isinstance(steps, int) or steps < minimum_steps:
            raise ValueError(
                f"Image decoder {mode} num_steps must be at least {minimum_steps}"
            )
        else:
            pass
        if not isinstance(resolution_multiplier, int) or resolution_multiplier < 1:
            raise ValueError("Image decoder resolution_multiplier must be positive")
        else:
            pass

    def ensure_sigvq(self) -> None:
        if self.sigvq is not None:
            return
        else:
            pass
        sigvq_path = os.path.join(
            self.model_path, "image_tokenizer", "sigvq_embedding.pt"
        )
        sigvq = SigVQ(vocab_size=16384, inner_dim=4096).to(
            self.device, dtype=self.dtype
        )
        sigvq.load_state_dict(
            torch.load(sigvq_path, map_location=self.device, weights_only=True)
        )
        self.sigvq = sigvq.eval()
        logger.info(f"SigVQ loaded from {sigvq_path}")

    def ensure_diff_model(self, decode_mode: DecoderMode) -> None:
        if self.diff_model is not None and self.diff_model_mode == decode_mode:
            return
        else:
            pass
        if self.diff_model is not None:
            logger.info(
                f"Switching diffusion model: {self.diff_model_mode} -> "
                f"{decode_mode} (releasing GPU memory)"
            )
            del self.diff_model
            self.diff_model = None
            self.diff_config = None
            self.diff_model_mode = None
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            else:
                pass
        else:
            pass

        if decode_mode == "decoder-turbo":
            decoder_dir = os.path.join(self.model_path, "decoder-turbo")
        else:
            decoder_dir = os.path.join(self.model_path, "decoder")

        config_path = os.path.join(decoder_dir, "config.json")
        with open(config_path) as config_file:
            config: DecoderConfig = json.load(config_file)
        # Override axes_lens and cap_feat_dim to match SigVQ output
        config["axes_lens"] = [32768, 1024, 1024]
        config["cap_feat_dim"] = 4096
        model = ZImageTransformer2DModelWrapper(
            decoder_dir=decoder_dir,
            cfg=config,
            device=self.device,
            dtype=self.dtype,
            backend=self.backend,
            runtime=self.runtime,
        )
        self.diff_model = model
        self.diff_config = config
        self.diff_model_mode = decode_mode
        logger.info(f"Diffusion model loaded from {decoder_dir} ({decode_mode} mode)")

    def ensure_vae(self) -> None:
        if self.vae is not None:
            return
        else:
            pass
        vae_dir = os.path.join(self.model_path, "vae")
        self.vae = (
            AutoencoderKL.from_pretrained(vae_dir, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        logger.info(f"VAE loaded from {vae_dir}")

    @torch.inference_mode()
    def decode(
        self,
        token_ids: list[int],
        h: int,
        w: int,
        *,
        decode_mode: DecoderMode | None = None,
        num_steps: int | None = None,
        resolution_multiplier: int | None = None,
        seed: int | None = None,
    ) -> Image:
        """Decode unoffset VQ IDs on an h-by-w semantic grid into a PIL image.

        Switching decode_mode reloads weights; seed=None uses the global RNG.
        """
        mode: DecoderMode = decode_mode if decode_mode is not None else self.decode_mode
        steps = num_steps if num_steps is not None else self.num_steps
        output_resolution_multiplier = (
            resolution_multiplier
            if resolution_multiplier is not None
            else self.resolution_multiplier
        )
        self.validate_settings(mode, steps, output_resolution_multiplier)
        if not isinstance(h, int) or not isinstance(w, int) or h < 1 or w < 1:
            raise ValueError("Image decoder grid dimensions must be positive integers")
        else:
            pass
        if len(token_ids) != h * w:
            raise ValueError("Image decoder requires exactly h * w VQ tokens")
        else:
            pass
        if any(not isinstance(i, int) or not 0 <= i < 16384 for i in token_ids):
            raise ValueError(
                "Image decoder VQ token IDs must be integers in [0, 16383]"
            )
        else:
            pass

        image_height = h * 16 * output_resolution_multiplier
        image_width = w * 16 * output_resolution_multiplier
        self.ensure_sigvq()
        token_grid = torch.tensor(token_ids).view(1, 1, h, w).float().to(self.device)
        upsampled_tokens = (
            F.interpolate(token_grid, scale_factor=2, mode="nearest").long().view(1, -1)
        )
        positive_caption_features = [
            self.sigvq(upsampled_tokens).squeeze(0).contiguous()
        ]
        negative_caption_features = [torch.zeros_like(positive_caption_features[0])]

        self.ensure_diff_model(mode)
        decoder_config = self.diff_config
        noise_shape = [1, 16, 1, 2 * (image_height // 16), 2 * (image_width // 16)]
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            initial_latents = torch.randn(
                noise_shape, device=self.device, generator=generator
            )
        else:
            generator = None
            initial_latents = torch.randn(noise_shape, device=self.device)
        velocity_model = create_decoder_model_fn(
            self.diff_model,
            positive_caption_features,
            negative_caption_features,
            cfg_scale=0.0 if mode == "decoder-turbo" else 1.0,
            patch_size=decoder_config.get("all_patch_size", (2,))[0],
            f_patch_size=decoder_config.get("all_f_patch_size", (1,))[0],
            dtype=self.dtype,
        )

        samples = sample_velocity(
            initial_latents,
            velocity_model,
            num_steps=steps,
            turbo=mode == "decoder-turbo",
            generator=generator,
        ).squeeze(2)

        self.ensure_vae()
        vae_latents = samples.to(self.dtype)
        vae_latents = (
            vae_latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        pixels = ((self.vae.decode(vae_latents, return_dict=False)[0] + 1) / 2).clamp_(
            0, 1
        )
        return to_pil_image(pixels[0].float())
