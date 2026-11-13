# SPDX-License-Identifier: Apache-2.0
"""YuE2 weight loading and acoustic rendering from generated semantic tokens."""

import json
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from sglang_omni.models.yue2.acoustic import synthesize_latents
from sglang_omni.models.yue2.layers import Yue2ModelConfig
from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.models.yue2.request import (
    CODEC_OFFSET,
    CancellationCheck,
    check_cancelled,
)
from sglang_omni.models.yue2.shared_weights import (
    attach_ar_weights,
    drop_ar_weights,
    is_acoustic_only,
)
from sglang_omni.models.yue2.vae import VaeConfig, Yue2Vae
from sglang_omni.utils.checkpoint import resolve_checkpoint


def load_weights(
    module: nn.Module, directory: Path, *, skipped: Callable[[str], bool]
) -> None:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        filenames = sorted(set(mapping.values()))
    else:
        filenames = ["model.safetensors"]
    weights: dict[str, torch.Tensor] = {}
    for filename in filenames:
        with safe_open(directory / filename, framework="pt", device="cpu") as archive:
            for name in archive.keys():
                if skipped(name):
                    continue
                elif name in weights:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                else:
                    weights[name] = archive.get_tensor(name)
    module.load_state_dict(weights, strict=True, assign=True)


class Yue2Runtime:
    def __init__(
        self,
        model: Yue2Model,
        vae: Yue2Vae,
        *,
        vae_core_frames: int,
        vae_halo_frames: int,
        fuse_projections: bool = False,
        fused_kernels: bool = False,
        ar_weights: tuple[str, str] | None = None,
    ) -> None:
        self.model = model
        self.fuse_projections = fuse_projections
        self.fused_kernels = fused_kernels
        # note (ruiling): the AR stage may still be loading, so its weights are attached on the first song.
        self.ar_weights = ar_weights
        if ar_weights is None:
            self.fuse()
        else:
            pass
        self.vae = vae
        self.vae_core_frames = vae_core_frames
        self.vae_halo_frames = vae_halo_frames
        if vae_core_frames < 1 or vae_halo_frames < vae.required_halo(vae_core_frames):
            raise ValueError(
                "YuE2 VAE tile configuration does not cover the decoder receptive field"
            )
        else:
            pass

    def fuse(self) -> None:
        if self.fused_kernels:
            self.model.enable_fused_kernels()
        elif self.fuse_projections:
            for layer in self.model.model.layers:
                layer.self_attn.fuse_projections()
                layer.nar_self_attn.fuse_projections()
                layer.mlp.fuse_projections()
                layer.nar_mlp.fuse_projections()
        else:
            pass

    def borrow_ar_weights(self) -> None:
        assert self.ar_weights is not None
        directory, checkpoint = self.ar_weights
        self.ar_stage_monitor = attach_ar_weights(
            self.model,
            directory,
            checkpoint,
            fused=self.fused_kernels or self.fuse_projections,
        )
        self.ar_weights = None
        self.fuse()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        vae_path: str,
        device: torch.device,
        *,
        vae_core_frames: int,
        vae_halo_frames: int,
        fuse_projections: bool = False,
        fused_kernels: bool = False,
        shared_weights_dir: str | None = None,
    ) -> "Yue2Runtime":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("highest")
        model_directory = Path(resolve_checkpoint(model_path))
        vae_directory = Path(resolve_checkpoint(vae_path))
        model_config = Yue2ModelConfig.model_validate_json(
            (model_directory / "config.json").read_text()
        )
        vae_config = VaeConfig.model_validate_json(
            (vae_directory / "config.json").read_text()
        )
        with torch.device("meta"):
            model = Yue2Model(model_config)
            vae = Yue2Vae(vae_config)
        if shared_weights_dir is None:
            # note (ruiling): the token head is only needed by the AR stage on the SGLang engine.
            load_weights(
                model, model_directory, skipped=lambda name: name.startswith("lm_head.")
            )
        else:
            drop_ar_weights(model)
            load_weights(
                model, model_directory, skipped=lambda name: not is_acoustic_only(name)
            )
        load_weights(
            vae, vae_directory, skipped=lambda name: name.startswith("encoder.")
        )
        model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        torch.cuda.empty_cache()
        return cls(
            model,
            vae,
            vae_core_frames=vae_core_frames,
            vae_halo_frames=vae_halo_frames,
            fuse_projections=fuse_projections,
            fused_kernels=fused_kernels,
            ar_weights=(
                None
                if shared_weights_dir is None
                else (shared_weights_dir, str(model_directory))
            ),
        )

    @torch.inference_mode()
    def render(
        self,
        prefix: list[int],
        semantic_tokens: list[int],
        seed: int,
        ode_steps: int,
        cancelled: CancellationCheck,
    ) -> torch.Tensor:
        check_cancelled(cancelled)
        if self.ar_weights is None:
            pass
        else:
            self.borrow_ar_weights()
        latents = synthesize_latents(
            self.model,
            prefix,
            [token - CODEC_OFFSET for token in semantic_tokens],
            seed,
            ode_steps,
            cancelled,
        )
        # note (ruiling): flow matching and decoding cache blocks the other cannot reuse.
        torch.cuda.empty_cache()
        audio = self.vae.decode_tiled(
            latents.T[None],
            self.vae_core_frames,
            self.vae_halo_frames,
            cancelled,
        )[0]
        torch.cuda.empty_cache()
        check_cancelled(cancelled)
        return audio
