# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
"""LLaDA2 image encoder: ViT + VQ-VAE for converting images to discrete VQ token IDs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.models.llada2_uni.components.common import resolve_local_model_dir
from sglang_omni.models.weight_loader import (
    load_module,
    resolve_dtype,
    resolve_model_path,
)


def _load_image_tokenizer_config(model_dir: str | Path) -> dict:
    with open(Path(model_dir) / "image_tokenizer" / "config.json", "r") as f:
        return json.load(f)


def _make_vision_config(raw: dict) -> SimpleNamespace:
    vc = raw.get("vision_config", raw)
    return SimpleNamespace(
        hidden_size=vc["hidden_size"],
        intermediate_size=vc["intermediate_size"],
        num_heads=vc["num_heads"],
        depth=vc["depth"],
        patch_size=vc["patch_size"],
        image_size=vc["image_size"],
        in_channels=vc.get("in_channels", 3),
        attention_bias=vc.get("attention_bias", True),
        layer_norm_eps=vc.get("layer_norm_eps", 1e-6),
        spatial_merge_size=vc.get("spatial_merge_size", 1),
    )


def _make_vq_config(raw: dict) -> SimpleNamespace:
    vq = raw.get("vq_config", raw)
    return SimpleNamespace(
        num_embeddings=vq["num_embeddings"],
        embed_dim=vq["embed_dim"],
        latent_channels=vq["latent_channels"],
    )


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.fc2(self.activation_fn(self.fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(
            config.hidden_size, config.hidden_size * 3, bias=config.attention_bias
        )
        self.proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.scaling = self.head_dim**-0.5

    def forward(self, hidden_states, cu_seqlens, **kwargs):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q = query_states.transpose(0, 1).unsqueeze(0)
        k = key_states.transpose(0, 1).unsqueeze(0)
        v = value_states.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = []
        for qc, kc, vc in zip(
            torch.split(q, lengths, dim=2),
            torch.split(k, lengths, dim=2),
            torch.split(v, lengths, dim=2),
        ):
            outputs.append(
                F.scaled_dot_product_attention(qc, kc, vc, scale=self.scaling)
            )

        attn_output = (
            torch.cat(outputs, dim=2)
            .transpose(1, 2)
            .reshape(seq_length, -1)
            .contiguous()
        )
        return self.proj(attn_output)


class VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        target_dtype = self.proj.weight.dtype
        x = x.view(-1, self.in_channels, self.patch_size, self.patch_size)
        return self.proj(x.to(dtype=target_dtype)).view(-1, self.embed_dim)


class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, self.embed_dim)

    def forward(self, embeddings, lengths, image_shapes, h_coords, w_coords):
        pos_w = self.position_embedding.weight
        hidden_size = pos_w.shape[1]
        device = pos_w.device

        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=device, dtype=torch.long)

        orig_size = int(pos_w.shape[0] ** 0.5)
        pos_2d = (
            pos_w.view(orig_size, orig_size, hidden_size)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )

        target_h = torch.cat(
            [image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]
        ).to(device=device, dtype=torch.float32)
        target_w = torch.cat(
            [image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]
        ).to(device=device, dtype=torch.float32)

        norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
        norm_h = ((h_coords + 0.5) / target_h) * 2 - 1
        grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

        adapted = F.grid_sample(
            pos_2d, grid, mode="bilinear", align_corners=False, padding_mode="border"
        )
        adapted = (
            adapted.squeeze(0)
            .squeeze(-1)
            .permute(1, 0)
            .to(pos_w.dtype)
            .to(embeddings.device)
        )
        return embeddings + adapted


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, hidden_states, cu_seqlens, **kwargs):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionEncoder(nn.Module):
    """Vision transformer encoder that produces per-patch features."""

    def __init__(self, config):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.embeddings = VisionEmbeddings(config)
        self.patch_embed = VisionPatchEmbed(config)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos = hpos.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos = hpos.permute(0, 2, 1, 3).flatten()

            wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos = wpos.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos = wpos.permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
        return torch.cat(pos_ids, dim=0)

    def forward(self, pixel_values, grid_thw):
        hidden_states = self.patch_embed(pixel_values)
        image_type_ids = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = F.pad(cu_seqlens.cumsum(0, dtype=torch.int32), (1, 0), value=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        hidden_states = self.embeddings(
            hidden_states,
            seqlens,
            grid_thw,
            image_type_ids[:, 0].to(hidden_states.device),
            image_type_ids[:, 1].to(hidden_states.device),
        )
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens)
        return hidden_states


class VQVAEVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embed_dim
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, hidden_state):
        hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
        flat = hidden_state.view(-1, self.embedding_dim)
        flat = F.normalize(flat, p=2, dim=-1)
        emb = F.normalize(self.embedding.weight, p=2, dim=-1)
        distances = (
            torch.sum(flat**2, dim=1, keepdim=True)
            + torch.sum(emb**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", flat, emb.t())
        )
        return torch.argmin(distances, dim=1)


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quantize = VQVAEVectorQuantizer(config)
        self.quant_conv = nn.Conv2d(config.latent_channels, config.embed_dim, 1)

    def encode(self, hidden_states):
        return self.quantize(self.quant_conv(hidden_states))


class LLaDA2ImageEncoder(nn.Module):
    """Image encoder that converts pixel values to discrete VQ token IDs.

    This is the encoder stage model used by the sglang-omni pipeline.
    It takes preprocessed pixel_values and image_grid_thw from the
    preprocessor, runs ViT + VQ-VAE, and returns VQ token IDs.

    Args:
        model_path: Root model directory (parent of image_tokenizer/).
        device: Torch device.
        dtype: Model dtype (default: bfloat16).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model_dir = resolve_local_model_dir(model_path)
        torch_dtype = resolve_dtype(dtype)
        self.dtype = torch_dtype

        try:
            raw_config = _load_image_tokenizer_config(self.model_dir)
        except (FileNotFoundError, OSError):
            if Path(model_path).exists():
                raise
            self.model_dir = str(resolve_model_path(model_path, local_files_only=False))
            raw_config = _load_image_tokenizer_config(self.model_dir)
        vision_cfg = _make_vision_config(raw_config)
        vq_cfg = _make_vq_config(raw_config)

        tokenizer_path = str(Path(self.model_dir) / "image_tokenizer")
        self.visual = load_module(
            VisionEncoder(vision_cfg),
            tokenizer_path,
            prefix="model.visual.",
            dtype=torch_dtype,
            device=device,
            strict=False,
        )
        self.vqmodel = load_module(
            VQVAE(vq_cfg),
            tokenizer_path,
            prefix="model.vqmodel.",
            dtype=torch_dtype,
            device=device,
            strict=False,
        )

    @torch.no_grad()
    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, **kwargs
    ) -> dict[str, Any]:
        """Run ViT encoder + VQ-VAE quantization.

        Args:
            pixel_values: [total_patches, patch_dim] preprocessed image patches.
            image_grid_thw: [num_images, 3] grid dimensions (t, h, w) per image.

        Returns:
            dict with:
                image_token_ids: list[list[int]] — VQ token IDs per image (without offset)
        """
        hidden_states = self.visual(
            pixel_values.to(self.device, self.dtype),
            grid_thw=image_grid_thw.to(self.device),
        )

        hidden_size = hidden_states.shape[-1]
        split_sizes = image_grid_thw.prod(dim=-1).tolist()
        all_token_ids = []
        for i, hs in enumerate(torch.split(hidden_states, split_sizes)):
            gt, gh, gw = image_grid_thw[i].tolist()
            hs = hs.view(gt, gh, gw, hidden_size).permute(0, 3, 1, 2).contiguous()
            tokens = self.vqmodel.encode(hs)
            all_token_ids.append(tokens.flatten().tolist())

        return {"image_token_ids": all_token_ids}
