# SPDX-License-Identifier: Apache-2.0
"""Tests for the srt-based MiniCPM-o image encoder.

The golden-parity test compares the srt-module encoder against the
checkpoint's remote-code path (modeling_navit_siglip.SiglipVisionTransformer
+ modeling_minicpmo.Resampler) on the real checkpoint weights, so it needs
a full checkpoint (weights included) and a CUDA device for the srt vision
attention. Set MINICPMO_CHECKPOINT or place MiniCPM-o-4_6 / MiniCPM-o-4_5
in the repo root; the test skips otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder

REPO_ROOT = Path(__file__).resolve().parents[3]


def _checkpoint_dir() -> Path | None:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    candidates = [Path(env)] if env else []
    candidates += [REPO_ROOT / "MiniCPM-o-4_6", REPO_ROOT / "MiniCPM-o-4_5"]
    for path in candidates:
        if (path / "model.safetensors.index.json").exists() and (
            path / "modeling_navit_siglip.py"
        ).exists():
            return path
    return None


def test_padding_does_not_change_image_embeddings() -> None:
    batch_size = 2
    encoder = object.__new__(MiniCPMOImageEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.vision_batch_size = batch_size

    def run_vpm(pixel_values, patch_attn_mask, tgt_sizes, patch_counts_cpu):
        features = pixel_values.mean(dim=1)
        pooled = (features * patch_attn_mask).sum(dim=-1) / patch_attn_mask.sum(dim=-1)
        return pooled.unsqueeze(-1)

    encoder.run_vpm = run_vpm
    encoder.resampler = lambda features, tgt_sizes: features
    tgt_sizes = torch.tensor([[1, 6], [1, 1], [1, 4]], dtype=torch.int32)
    pixel_values = [
        torch.full((3, 1, count), float(i + 1)) for i, count in enumerate([6, 1, 4])
    ]

    batched = encoder(pixel_values=pixel_values, tgt_sizes=tgt_sizes)["image_embeds"]
    individual = torch.cat(
        [
            encoder(pixel_values=[pixels], tgt_sizes=tgt_sizes[i : i + 1])[
                "image_embeds"
            ]
            for i, pixels in enumerate(pixel_values)
        ]
    )

    torch.testing.assert_close(batched, individual)
    torch.testing.assert_close(batched[:, 0], torch.tensor([1.0, 2.0, 3.0]))


def _build_remote_encoder(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    """The pre-srt remote-code path this component replaced, as golden."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from sglang_omni.models.weight_loader import load_module

    model_dir = str(checkpoint)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    siglip_cls = get_class_from_dynamic_module(
        "modeling_navit_siglip.SiglipVisionTransformer", model_dir
    )
    resampler_cls = get_class_from_dynamic_module(
        "modeling_minicpmo.Resampler", model_dir
    )
    vision_config = config.vision_config
    vision_config._attn_implementation = "eager"
    vpm = siglip_cls(vision_config)
    if getattr(config, "drop_vision_last_layer", False):
        vpm.encoder.layers = vpm.encoder.layers[:-1]
    vpm = load_module(vpm, model_dir, prefix=("vpm.",), dtype=dtype, device=str(device))

    embed_dim = config.hidden_size
    resampler = resampler_cls(
        num_queries=config.query_num,
        embed_dim=embed_dim,
        num_heads=embed_dim // 128,
        kv_dim=vision_config.hidden_size,
        adaptive=True,
    )
    resampler = load_module(
        resampler, model_dir, prefix=("resampler.",), dtype=dtype, device=str(device)
    )
    resampler._set_2d_pos_cache(resampler.max_size, device=str(device))
    return config, vpm.eval(), resampler.eval()


def _remote_forward(vpm, resampler, pixel_values, tgt_sizes, device, dtype):
    from torch.nn.utils.rnn import pad_sequence

    tgt_sizes = tgt_sizes.to(device, dtype=torch.int32)
    all_pixel_values = [
        v.to(device, dtype=dtype).flatten(end_dim=1).permute(1, 0) for v in pixel_values
    ]
    all_pixel_values = pad_sequence(
        all_pixel_values, batch_first=True, padding_value=0.0
    )
    B, L, _ = all_pixel_values.shape
    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
    patch_counts = tgt_sizes[:, 0] * tgt_sizes[:, 1]
    max_patches = int(patch_counts.max().item())
    patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
    for i in range(B):
        patch_attn_mask[i, 0, : patch_counts[i]] = True
    vision_embedding = vpm(
        all_pixel_values,
        patch_attention_mask=patch_attn_mask,
        tgt_sizes=tgt_sizes,
    ).last_hidden_state
    return resampler(vision_embedding, tgt_sizes)


def test_golden_parity_vs_remote_code() -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None:
        pytest.skip("no MiniCPM-o checkpoint with weights")
    if not torch.cuda.is_available():
        pytest.skip("srt vision attention requires CUDA")

    from sglang_omni.models.minicpm_o.components.image_encoder import (
        MiniCPMOImageEncoder,
    )

    # srt VisionAttention's flash-attn backend only supports fp16/bf16, so the
    # srt encoder cannot run an fp32 bitwise-parity pass. Instead, both the
    # srt path and the remote-code path run in bf16 against an fp32
    # remote-code golden, and the srt path's error must stay within the
    # remote path's own bf16 rounding error (plus slack) — i.e. the module
    # swap adds no error beyond dtype noise.
    device = torch.device("cuda")

    torch.manual_seed(0)
    config, vpm32, resampler32 = _build_remote_encoder(
        checkpoint, device, torch.float32
    )
    patch = config.vision_config.patch_size
    # Variable-resolution slices (h, w) in patch units, incl. a 1-patch-high one.
    tgt_sizes = torch.tensor([[8, 12], [3, 5], [1, 9]], dtype=torch.int32)
    pixel_values = [
        torch.randn(3, patch, int(h * w) * patch) for h, w in tgt_sizes.tolist()
    ]
    with torch.no_grad():
        golden = _remote_forward(
            vpm32, resampler32, pixel_values, tgt_sizes, device, torch.float32
        ).float()
    del vpm32, resampler32
    torch.cuda.empty_cache()

    config, vpm16, resampler16 = _build_remote_encoder(
        checkpoint, device, torch.bfloat16
    )
    with torch.no_grad():
        remote_bf16 = _remote_forward(
            vpm16, resampler16, pixel_values, tgt_sizes, device, torch.bfloat16
        ).float()
    del vpm16, resampler16
    torch.cuda.empty_cache()

    native = MiniCPMOImageEncoder(str(checkpoint), device="cuda", dtype="bfloat16")
    with torch.no_grad():
        got = (
            native(pixel_values=pixel_values, tgt_sizes=tgt_sizes)["image_embeds"]
            .float()
            .view(golden.shape)
        )

    remote_err = (remote_bf16 - golden).abs()
    native_err = (got - golden).abs()
    assert native_err.mean() <= remote_err.mean() * 1.5, (
        f"srt path error {native_err.mean():.6f} exceeds remote bf16 "
        f"rounding error {remote_err.mean():.6f}"
    )
    cos = torch.nn.functional.cosine_similarity(got, golden, dim=-1)
    remote_cos = torch.nn.functional.cosine_similarity(remote_bf16, golden, dim=-1)
    assert cos.min() >= remote_cos.min() - 0.01, (
        f"srt path cos_min {cos.min():.6f} below remote bf16 "
        f"cos_min {remote_cos.min():.6f}"
    )
