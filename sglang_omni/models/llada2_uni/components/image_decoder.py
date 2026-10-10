"""LLaDA2 image decoder: SigVQ + ZImage Diffusion + VAE.

Converts discrete VQ token IDs produced by the thinker into a PIL image.
"""

from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torchvision.transforms.functional import to_pil_image

from sglang_omni.models.llada2_uni.components.decoder_model import (
    ZImageTransformer2DModelWrapper,
)
from sglang_omni.models.llada2_uni.components.decoder_runtime import (
    DecoderRuntimeHandle,
)
from sglang_omni.models.llada2_uni.components.sigvq import SigVQ
from sglang_omni.models.llada2_uni.components.transport import sample_velocity
from sglang_omni.models.weight_loader import resolve_model_path

logger = logging.getLogger(__name__)


def create_decoder_model_fn(
    model,
    positive_caption_features,
    negative_caption_features,
    cfg_scale,
    patch_size,
    f_patch_size,
    dtype,
):
    batch_size = len(positive_caption_features)
    combined_caption_features = positive_caption_features + negative_caption_features

    def predict_velocity(latents, timestep, **kwargs):
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
    """3-stage image decoder: SigVQ -> Diffusion ODE -> VAE.

    Args:
        model_path: Hugging Face model ID or root model directory (parent of
            decoder/, vae/, image_tokenizer/).
        device: Torch device string.
        dtype: Model dtype (default: bfloat16).
        decode_mode: "normal" for standard 50-step decoder,
            "decoder-turbo" for distilled 8-step decoder. Used as the default
            when decode is called without an explicit decode_mode.
        num_steps: Default number of ODE sampling steps.
        resolution_multiplier: Default upscale factor (2 = 1024px from 512px tokens).
        backend: diffusers (default) or explicitly sglang.
        runtime: Caller-owned SGLang diffusion runtime. Required by the
            sglang backend and rejected by the diffusers backend.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        decode_mode: str = "normal",
        num_steps: int = 50,
        resolution_multiplier: int = 2,
        *,
        backend: str = "diffusers",
        runtime: DecoderRuntimeHandle | None = None,
    ):
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
        self.device = torch.device(device)
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
        self.decode_mode = decode_mode
        self.num_steps = num_steps
        self.resolution_multiplier = resolution_multiplier

        self.sigvq: SigVQ | None = None
        self.diff_model: ZImageTransformer2DModelWrapper | None = None
        self.diff_model_mode: str | None = None
        self.vae: AutoencoderKL | None = None
        self.diff_config: dict | None = None

    @staticmethod
    def validate_settings(mode, steps, resolution_multiplier):
        if mode not in {"normal", "decoder-turbo"}:
            raise ValueError(f"Unsupported image decoder mode: {mode!r}")
        else:
            pass
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("Image decoder num_steps must be positive")
        else:
            pass
        if not isinstance(resolution_multiplier, int) or resolution_multiplier < 1:
            raise ValueError("Image decoder resolution_multiplier must be positive")
        else:
            pass

    def ensure_sigvq(self):
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

    def ensure_diff_model(self, decode_mode: str):
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
        with open(config_path) as f:
            cfg = json.load(f)
        # Override axes_lens and cap_feat_dim to match SigVQ output
        cfg["axes_lens"] = [32768, 1024, 1024]
        cfg["cap_feat_dim"] = 4096
        model = ZImageTransformer2DModelWrapper(
            decoder_dir=decoder_dir,
            cfg=cfg,
            device=self.device,
            dtype=self.dtype,
            backend=self.backend,
            runtime=self.runtime,
        )
        self.diff_model = model
        self.diff_config = cfg
        self.diff_model_mode = decode_mode
        logger.info(f"Diffusion model loaded from {decoder_dir} ({decode_mode} mode)")

    def ensure_vae(self):
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
        decode_mode: str | None = None,
        num_steps: int | None = None,
        resolution_multiplier: int | None = None,
        seed: int | None = None,
    ):
        """Decode VQ token IDs into a PIL Image.

        Args:
            token_ids: List of VQ token IDs (without the +157184 offset).
            h: Semantic grid height (image_pixels // 16).
            w: Semantic grid width (image_pixels // 16).
            decode_mode: Override instance default; switches diffusion weights
                between decoder/ and decoder-turbo/ (single-slot reload).
            num_steps: Override default ODE step count.
            resolution_multiplier: Override default upscale factor.
            seed: If set, draws initial noise with a deterministic generator.
                If None, each call draws from the global RNG.

        Returns:
            PIL.Image.Image
        """
        mode = decode_mode if decode_mode is not None else self.decode_mode
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
