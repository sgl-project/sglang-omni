# SPDX-License-Identifier: Apache-2.0
"""Load official AuK checkpoints into native MLX components."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

import mlx.core as mx
from safetensors import safe_open

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.hf_config import (
    AuKDitConfig,
    AuKVAEConfig,
    make_runtime_config,
)
from sglang_omni.models.auk.mlx.dit import AuKDit
from sglang_omni.models.auk.mlx.flow_matching import AuKFlowMatching
from sglang_omni.models.auk.mlx.vae import BigVGANFlowVAE
from sglang_omni.models.auk.weight_loader import (
    normalize_state_dict,
    resolve_vae_file,
    resolve_weight_file,
    strip_prefix,
)
from sglang_omni.platforms import current_platform
from sglang_omni.utils.device import resolve_concrete_device

MLX_DTYPES = {
    "float32": mx.float32,
    "bfloat16": mx.bfloat16,
}


def validate_device(device: str | None, gpu_id: int | None) -> None:
    resolved = resolve_concrete_device(device, gpu_id)
    if not current_platform.is_mps() or resolved.type != "mps":
        raise ValueError("AuK native MLX requires an Apple Metal device")
    elif resolved.index not in (None, 0) or not mx.metal.is_available():
        raise ValueError("AuK native MLX requires the available Apple Metal device 0")
    else:
        pass


def resolve_dtype(name: str) -> mx.Dtype:
    try:
        return MLX_DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported AuK MLX dtype: {name!r}") from exc


def load_fusion(checkpoint: str) -> tuple[mx.array, mx.array]:
    with safe_open(str(resolve_weight_file(checkpoint)), framework="pt") as weights:
        names = {strip_prefix(key): key for key in weights.keys()}
        values = tuple(
            mx.array(weights.get_tensor(names[name]).float().numpy())
            for name in C.FUSION_PARAMETERS
        )
    mx.eval(values)
    return values


@lru_cache(maxsize=1)
def load_vae(checkpoint: str) -> BigVGANFlowVAE:
    config = make_runtime_config(checkpoint)
    path = resolve_vae_file(checkpoint)
    if path is None:
        raise FileNotFoundError(f"No AuK VAE weights found under {checkpoint}")
    else:
        model = BigVGANFlowVAE(AuKVAEConfig.from_dict(config.vae_init_kwargs))
        weights = normalize_state_dict(mx.load(str(path)))
        model.load_weights(list(model.sanitize(weights).items()), strict=True)
        del weights
        model.eval()
        mx.eval(model.parameters())
        return model


def load_flow(checkpoint: str, dtype: mx.Dtype) -> AuKFlowMatching:
    config = make_runtime_config(checkpoint)
    arch = asdict(AuKDitConfig.from_dict(config.arch))
    # note (Codex): The native backbone excludes training-only hyperparameters.
    arch.pop("depth")
    arch.pop("dropout")
    arch["latent_dim"] = config.latent_dim
    weights = normalize_state_dict(mx.load(str(resolve_weight_file(checkpoint))))
    flow = AuKFlowMatching(AuKDit(**arch), num_llm_layers=weights["layer_weights"].size)
    flow.load_weights(list(flow.sanitize(weights).items()), strict=True)
    del weights
    rotary_frequency = flow.transformer.rotary_embed.inv_freq.astype(mx.float32)
    flow.transformer.set_dtype(dtype)
    flow.transformer.rotary_embed.inv_freq = rotary_frequency
    flow.eval()
    mx.eval(flow.parameters())
    return flow
