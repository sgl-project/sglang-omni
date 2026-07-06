# SPDX-License-Identifier: Apache-2.0
"""LLaDA2-Uni image decoder: VQ tokens -> image bytes/base64."""

from __future__ import annotations

import base64
import io
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from sglang_omni.models.llada2_uni.components.common import resolve_local_model_dir
from sglang_omni.models.weight_loader import resolve_dtype, resolve_model_path

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_FORMATS = {
    "png": ("PNG", "image/png"),
    "jpg": ("JPEG", "image/jpeg"),
    "jpeg": ("JPEG", "image/jpeg"),
}


class _LinearWrapper(nn.Module):
    """Wrap ``nn.Linear`` inside ``.proj`` to match decoder checkpoints."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _FeedForward(nn.Module):
    """SiLU feed-forward matching the SigVQ checkpoint key layout."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _LinearWrapper(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SigVQ(nn.Module):
    """Semantic token embedding extractor used by the LLaDA2 decoder."""

    def __init__(self, vocab_size: int = 16384, inner_dim: int = 4096) -> None:
        super().__init__()
        self.prior_token_embedding = nn.Embedding(vocab_size, inner_dim)
        self.prior_projector = _FeedForward(dim=inner_dim, hidden_dim=inner_dim)
        self.requires_grad_(False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.prior_projector(self.prior_token_embedding(token_ids))


@dataclass(frozen=True)
class DecodedImage:
    """Decoded image payload with both raw bytes and JSON-safe base64."""

    image_bytes: bytes
    data: str
    format: str
    mime_type: str
    width: int
    height: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "format": self.format,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
        }


def normalize_image_format(image_format: str | None) -> tuple[str, str, str]:
    key = (image_format or "png").lower()
    if key not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            "LLaDA2-Uni image decoder format must be one of: png, jpeg, jpg"
        )
    pil_format, mime_type = SUPPORTED_IMAGE_FORMATS[key]
    return ("jpeg" if key == "jpg" else key), pil_format, mime_type


def pil_image_to_bytes(
    image: Image.Image,
    *,
    image_format: str = "png",
) -> tuple[bytes, str, str]:
    """Encode a PIL image as PNG/JPEG bytes."""
    normalized_format, pil_format, mime_type = normalize_image_format(image_format)
    if pil_format == "JPEG" and image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")

    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {}
    if pil_format == "JPEG":
        save_kwargs.update({"quality": 95, "subsampling": 0})
    image.save(buffer, format=pil_format, **save_kwargs)
    return buffer.getvalue(), normalized_format, mime_type


def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def _create_decoder_model_fn(
    model,
    cap_pos,
    cap_neg,
    cfg_scale: float,
    patch_size: int,
    f_patch_size: int,
    dtype: torch.dtype,
):
    n = len(cap_pos)
    doubled = cap_pos + cap_neg

    def fn(x, t, **kw):
        if isinstance(t, torch.Tensor):
            t_t = t.float()
        else:
            t_t = torch.tensor([t], device=x.device, dtype=torch.float32)
        if t_t.dim() == 0:
            t_t = t_t.unsqueeze(0)
        if t_t.shape[0] == 1 and x.shape[0] > 1:
            t_t = t_t.expand(x.shape[0])
        if cfg_scale > 0:
            out = model(
                x=list(x.to(dtype).repeat(2, 1, 1, 1, 1).unbind(0)),
                t=t_t.repeat(2),
                cap_feats=doubled,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                return_dict=False,
            )
            pos, neg = out[0][:n], out[0][n:]
            res = []
            for p, ng in zip(pos, neg):
                p, ng = p.float(), ng.float()
                pred = p + cfg_scale * (p - ng)
                original_norm = torch.linalg.vector_norm(p)
                new_norm = torch.linalg.vector_norm(pred)
                if new_norm > original_norm:
                    pred *= original_norm / new_norm
                res.append(pred)
            return torch.stack(res)
        out = model(
            x=list(x.to(dtype).unbind(0)),
            t=t_t,
            cap_feats=cap_pos,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            return_dict=False,
        )
        return torch.stack([o.float() for o in out[0]])

    return fn


class LLaDA2ImageDecoder:
    """Decode LLaDA2 VQ token grids into image bytes/base64.

    The decoder loads three model-side components:
    ``image_tokenizer/sigvq_embedding.pt`` for semantic token features,
    either ``decoder`` or ``decoder-turbo`` for diffusion sampling, and
    ``vae`` for pixel reconstruction.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = resolve_dtype(dtype) or torch.bfloat16
        self.model_dir = Path(resolve_local_model_dir(model_path))
        self._original_model_path = model_path

        self._sigvq: SigVQ | None = None
        self._diff_model = None
        self._diff_config: dict[str, Any] | None = None
        self._diff_mode: str | None = None
        self._vae = None

        self._ensure_model_dir()

    def _ensure_model_dir(self) -> None:
        expected = self.model_dir / "image_tokenizer" / "sigvq_embedding.pt"
        if expected.exists():
            return
        if Path(self._original_model_path).exists():
            raise FileNotFoundError(expected)
        self.model_dir = Path(
            resolve_model_path(self._original_model_path, local_files_only=False)
        )

    def _empty_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _seed_decode(self, seed: int | None) -> None:
        if seed is None:
            return
        seed = int(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_sigvq(self) -> SigVQ:
        if self._sigvq is not None:
            return self._sigvq

        sigvq_path = self.model_dir / "image_tokenizer" / "sigvq_embedding.pt"
        extractor = SigVQ(vocab_size=16384, inner_dim=4096).to(
            self.device, dtype=self.dtype
        )
        extractor.load_state_dict(
            torch.load(
                sigvq_path,
                map_location=self.device,
                weights_only=True,
            )
        )
        extractor.eval()
        self._sigvq = extractor
        logger.info("Loaded LLaDA2-Uni SigVQ image tokenizer from %s", sigvq_path)
        return extractor

    def _decoder_dir(self, decode_mode: str) -> tuple[str, Path]:
        mode = "decoder-turbo" if decode_mode == "decoder-turbo" else "normal"
        dirname = "decoder-turbo" if mode == "decoder-turbo" else "decoder"
        return mode, self.model_dir / dirname

    def _load_diffusion_model(self, decode_mode: str):
        mode, decoder_dir = self._decoder_dir(decode_mode)
        if self._diff_model is not None and self._diff_mode == mode:
            return self._diff_model, self._diff_config or {}

        if self._diff_model is not None:
            del self._diff_model
            self._diff_model = None
            self._empty_cache()

        from safetensors.torch import load_file

        from sglang_omni.models.llada2_uni.components.decoder_model import (
            ZImageTransformer2DModel,
        )

        config_path = decoder_dir / "config.json"
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["axes_lens"] = [32768, 1024, 1024]
        cfg["cap_feat_dim"] = 4096

        with torch.device("meta"):
            diff_model = ZImageTransformer2DModel(**cfg)
        ckpt = decoder_dir / "model.safetensors"
        diff_model.load_state_dict(load_file(ckpt, device=str(self.device)), assign=True)
        diff_model = diff_model.to(dtype=self.dtype).eval()

        self._diff_model = diff_model
        self._diff_config = cfg
        self._diff_mode = mode
        logger.info("Loaded LLaDA2-Uni diffusion decoder from %s", decoder_dir)
        return diff_model, cfg

    def _load_vae(self):
        if self._vae is not None:
            return self._vae

        from diffusers import AutoencoderKL

        vae_dir = self.model_dir / "vae"
        self._vae = AutoencoderKL.from_pretrained(
            vae_dir,
            torch_dtype=self.dtype,
        ).to(self.device)
        self._vae.eval()
        logger.info("Loaded LLaDA2-Uni VAE from %s", vae_dir)
        return self._vae

    def _semantic_features(
        self,
        token_ids: list[int],
        *,
        token_grid_h: int,
        token_grid_w: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        extractor = self._load_sigvq()
        tok = torch.tensor(token_ids).view(1, 1, token_grid_h, token_grid_w)
        tok = tok.float().to(self.device)
        up = F.interpolate(tok, scale_factor=2, mode="nearest").long().view(1, -1)
        cap_pos = [extractor(up).squeeze(0)]
        cap_neg = [torch.zeros_like(cap_pos[0])]
        return cap_pos, cap_neg

    @torch.inference_mode()
    def decode_to_pil(
        self,
        token_ids: list[int],
        *,
        token_grid_h: int,
        token_grid_w: int,
        resolution_multiplier: int = 2,
        num_steps: int = 50,
        decode_mode: str = "normal",
        seed: int | None = None,
    ) -> Image.Image:
        expected_tokens = token_grid_h * token_grid_w
        if len(token_ids) != expected_tokens:
            raise ValueError(
                "LLaDA2-Uni image decoder expected "
                f"{expected_tokens} VQ tokens for grid "
                f"{token_grid_h}x{token_grid_w}, got {len(token_ids)}"
            )
        if any(token_id < 0 for token_id in token_ids):
            raise ValueError("LLaDA2-Uni image token ids must be non-negative")
        if any(token_id >= 16384 for token_id in token_ids):
            raise ValueError("LLaDA2-Uni image token ids must be < 16384")

        from torchvision.transforms.functional import to_pil_image

        from sglang_omni.models.llada2_uni.components.decoder_transport import (
            Sampler,
            create_transport,
        )

        self._seed_decode(seed)
        cap_pos, cap_neg = self._semantic_features(
            token_ids,
            token_grid_h=token_grid_h,
            token_grid_w=token_grid_w,
        )
        diff_model, cfg = self._load_diffusion_model(decode_mode)

        th = token_grid_h * 16 * resolution_multiplier
        tw = token_grid_w * 16 * resolution_multiplier
        z = torch.randn(
            [1, 16, 1, 2 * (th // 16), 2 * (tw // 16)],
            device=self.device,
        )

        model_fn = _create_decoder_model_fn(
            diff_model,
            cap_pos,
            cap_neg,
            cfg_scale=0.0 if decode_mode == "decoder-turbo" else 1.0,
            patch_size=cfg.get("all_patch_size", (2,))[0],
            f_patch_size=cfg.get("all_f_patch_size", (1,))[0],
            dtype=self.dtype,
        )
        sampler = Sampler(create_transport("Linear", "velocity", None))
        sample_fn = sampler.sample_ode(
            sampling_method="euler",
            num_steps=num_steps,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            time_shifting_factor=6,
            stochast_ratio=1.0 if decode_mode == "decoder-turbo" else 0.0,
        )
        samples = sample_fn(z, model_fn)[-1].squeeze(2)

        vae = self._load_vae()
        s = samples.to(self.dtype)
        s = (s / vae.config.scaling_factor) + vae.config.shift_factor
        px = ((vae.decode(s, return_dict=False)[0] + 1) / 2).clamp_(0, 1)
        return to_pil_image(px[0].float())

    def decode(
        self,
        token_ids: list[int],
        *,
        token_grid_h: int,
        token_grid_w: int,
        resolution_multiplier: int = 2,
        num_steps: int = 50,
        decode_mode: str = "normal",
        image_format: str = "png",
        seed: int | None = None,
    ) -> DecodedImage:
        image = self.decode_to_pil(
            token_ids,
            token_grid_h=token_grid_h,
            token_grid_w=token_grid_w,
            resolution_multiplier=resolution_multiplier,
            num_steps=num_steps,
            decode_mode=decode_mode,
            seed=seed,
        )
        image_bytes, normalized_format, mime_type = pil_image_to_bytes(
            image,
            image_format=image_format,
        )
        return DecodedImage(
            image_bytes=image_bytes,
            data=image_bytes_to_base64(image_bytes),
            format=normalized_format,
            mime_type=mime_type,
            width=image.width,
            height=image.height,
        )
