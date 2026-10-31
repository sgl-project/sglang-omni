# SPDX-License-Identifier: Apache-2.0 AND MIT
# Inference recipe adapted from Tencent-Hunyuan/AuK, Copyright (C) 2026 Tencent.
# See LICENSE for the upstream MIT permission notice.
"""AuK component loading and sampling warmup."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from functools import lru_cache

import torch
from safetensors import safe_open
from torch.nn.utils import remove_weight_norm

from sglang_omni.models.auk.dit import AuKDit, AuKDitConfig
from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem
from sglang_omni.models.auk.hf_config import make_runtime_config
from sglang_omni.models.auk.step_cuda_graph import AuKStepCudaGraphRunner
from sglang_omni.models.auk.vae import AuKVAEConfig, BigVGANFlowVAE
from sglang_omni.models.auk.weight_loader import (
    load_dit_weights,
    load_vae_weights,
    resolve_weight_file,
)

logger = logging.getLogger(__name__)


TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def pin_attention_backend(device: torch.device) -> None:
    """Use one SDPA kernel for resident and offloaded AuK.

    Restoring paused weight memory can make the dispatcher select another kernel for the same shape. Flash attention also rejects the additive key bias.
    """
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    else:
        pass


def resolve_dtype(*, field: str, name: str) -> torch.dtype:
    if name not in TORCH_DTYPES:
        raise ValueError(
            f"AuK {field} must be one of {', '.join(TORCH_DTYPES)}, got {name!r}"
        )
    else:
        pass
    return TORCH_DTYPES[name]


def autocast(device: torch.device, dtype: torch.dtype) -> torch.autocast:
    # fp32 means "no autocast": the weights already carry the compute dtype.
    return torch.autocast(
        device_type=device.type, dtype=dtype, enabled=dtype != torch.float32
    )


@lru_cache(maxsize=None)
def load_vae(
    checkpoint: str, device: str, fold_weight_norm: bool = False
) -> BigVGANFlowVAE:
    config = make_runtime_config(checkpoint)
    vae = BigVGANFlowVAE(AuKVAEConfig.from_dict(config.vae_init_kwargs))
    load_vae_weights(vae, checkpoint)
    vae = vae.to(device=device).eval().requires_grad_(False)
    if fold_weight_norm:
        for module in vae.modules():
            parameters = dict(module.named_parameters(recurse=False))
            if "weight_g" in parameters and "weight_v" in parameters:
                remove_weight_norm(module)
            else:
                pass
    else:
        pass
    return vae


# Order matches the fuse_hidden_states signature.
FUSION_PARAMETERS = ("layer_weights", "layer_scale")


@lru_cache(maxsize=None)
def load_fusion(checkpoint: str, device: str) -> tuple[torch.Tensor, ...]:
    # Read straight from the file so a conditioning-only process does not have to
    # hold the 1.5B DiT these two tensors are stored next to.
    with safe_open(str(resolve_weight_file(checkpoint)), framework="pt") as weights:
        keys = {key.rsplit(".", 1)[-1]: key for key in weights.keys()}
        return tuple(
            weights.get_tensor(keys[name]).to(device=device, dtype=torch.float32)
            for name in FUSION_PARAMETERS
        )


@lru_cache(maxsize=None)
def load_flow(
    checkpoint: str,
    device: str,
    backbone_dtype: torch.dtype,
    compile_blocks: bool,
    finalize_on_cpu: bool = False,
) -> AuKFlowMatching:
    config = make_runtime_config(checkpoint)
    layer_weights, _ = load_fusion(checkpoint, device)
    dit_config = AuKDitConfig.from_dict(config.arch)
    dit = AuKDit(**{**dit_config.__dict__, "latent_dim": config.latent_dim})
    flow = AuKFlowMatching(dit, num_llm_layers=layer_weights.numel())
    load_dit_weights(flow, checkpoint)
    if finalize_on_cpu:
        flow.float()
        flow.transformer.to(dtype=backbone_dtype)
        flow = flow.to(device=device).eval().requires_grad_(False)
    else:
        flow = flow.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    # note(Dayuxiaoshui): keyed by dtype and by the compile flag because both
    # mutate the backbone, and a cached one would retroactively change every
    # executor already built on it.
    flow.transformer.to(dtype=backbone_dtype)
    if compile_blocks:
        flow.transformer.enable_compiled_blocks()
    else:
        pass
    return flow


def warmup_items(
    flow: AuKFlowMatching,
    device: torch.device,
    *,
    batch: int,
    frames: int,
    ref: int,
    text: int,
) -> list[AuKSampleItem]:
    """A synthetic sampling batch shaped like one the server will be given."""
    return [
        AuKSampleItem(
            torch.zeros(text, flow.transformer.txt_proj.in_features, device=device),
            torch.ones(text, dtype=torch.bool, device=device),
            frames,
            (
                torch.zeros(ref, flow.transformer.latent_dim, device=device)
                if ref
                else None
            ),
            seed=0,
            ref_length=ref,
        )
        for _ in range(batch)
    ]


def warmup_flow(
    flow: AuKFlowMatching,
    device: torch.device,
    dtype: torch.dtype,
    sampling: Mapping[str, int | float | tuple[float, ...] | None],
    step_graph: AuKStepCudaGraphRunner | None = None,
    *,
    serial: bool = False,
) -> None:
    """Pay the block compile, and every declared graph capture, at startup.

    The warmup enters at sample_batch, where a request does, so it compiles the
    shapes a request runs rather than shapes that merely resemble them.
    """
    started = time.perf_counter()
    # note(Dayuxiaoshui): the released time grid overrides steps where a
    # checkpoint declares one, and only the shapes matter here, so it goes.
    one_step = {**sampling, "steps": 1, "t_grid": None}
    # note(Dayuxiaoshui): under inference_mode like the request path, so
    # dynamo compiles once.
    with torch.inference_mode(), autocast(device, dtype):
        # note(Dayuxiaoshui): the eager path a request takes when no declared
        # graph covers it, over the three combinations dynamo guards on: a lone
        # item carries no rope positions, and without a reference it carries no
        # attention bias either. The frame count shares no model axis, so
        # dynamo cannot duck-size it to one and guard the audio length on it.
        shapes = ((1, 0), (1, 32)) if serial else ((1, 0), (1, 32), (2, 32))
        for batch, ref in shapes:
            items = warmup_items(flow, device, batch=batch, frames=72, ref=ref, text=16)
            flow.sample_batch(items, **one_step)
        if step_graph is not None:
            step_graph.capture_declared(
                lambda shape: flow.sample_batch(
                    warmup_items(
                        flow,
                        device,
                        batch=shape.batch,
                        frames=shape.frames,
                        ref=shape.ref,
                        text=shape.text,
                    ),
                    **one_step,
                    step_graph=step_graph,
                )
            )
        else:
            pass
    logger.info(f"AuK DiT: warmed the sampler in {time.perf_counter() - started:.1f}s")
