# SPDX-License-Identifier: Apache-2.0
"""Image encoder component for MiniCPM-V 2.6.

MiniCPM-V uses SigLIP ViT as the vision backbone and a Perceiver Resampler
for token compression. This is different from Qwen3-Omni which uses a
custom ViT with DeepStack multi-scale features.

Key components:
- vpm (Vision Perception Module): SigLIP ViT-400M backbone
- resampler: Perceiver Resampler that compresses patch tokens to fixed queries

The forward pass:
1. SigLIP encodes each image slice into patch embeddings
2. Resampler compresses patch embeddings using cross-attention with learnable queries
3. Output shape: [total_queries, hidden_dim] where total_queries = num_slices * query_num
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from sglang_omni.models.minicpm_v.components.common import (
    RESAMPLER_PREFIX,
    VISUAL_PREFIX,
    load_minicpm_config,
)
from sglang_omni.models.weight_loader import load_weights_by_prefix, resolve_dtype

logger = logging.getLogger(__name__)


def _build_vpm(
    model_path: str,
    *,
    config: Any,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    """Build and load the SigLIP vision encoder (vpm).

    MiniCPM-V uses SigLIP-400M as the vision backbone. The module is loaded
    with trust_remote_code=True to handle the custom architecture.
    """
    from transformers import AutoModel

    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        raise ValueError("MiniCPM-V config missing vision_config")

    # Build the vision model from config
    # MiniCPM-V's vpm is typically a SigLIP model
    vpm = AutoModel.from_config(vision_config, trust_remote_code=True)

    # Load weights with prefix filtering
    state_dict = load_weights_by_prefix(model_path, prefix=VISUAL_PREFIX)
    if state_dict:
        try:
            vpm.load_state_dict(state_dict, strict=True, assign=True)
        except (TypeError, RuntimeError):
            vpm.load_state_dict(state_dict, strict=True)

    vpm.eval()
    if torch_dtype is not None:
        vpm = vpm.to(dtype=torch_dtype)
    if device:
        vpm = vpm.to(device=device)

    return vpm


def _build_resampler(
    model_path: str,
    *,
    config: Any,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    """Build and load the Perceiver Resampler.

    The Resampler compresses variable-length patch embeddings from SigLIP
    into a fixed number of query tokens per image slice. This uses
    cross-attention with learnable query embeddings.
    """
    # Get resampler config - MiniCPM-V stores this in the main config
    # The resampler architecture varies by version
    hidden_size = getattr(config, "hidden_size", 2304)
    query_num = getattr(config, "query_num", 64)
    vision_config = getattr(config, "vision_config", None)

    if vision_config is not None:
        vision_hidden_size = getattr(vision_config, "hidden_size", 1152)
    else:
        vision_hidden_size = hidden_size

    # Try to instantiate the resampler from the HF model
    # MiniCPM-V defines a custom Resampler class
    try:
        from transformers import AutoConfig

        full_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(full_config, "resampler"):
            resampler_cls = type(full_config.resampler)
            resampler = resampler_cls(full_config)
        else:
            # Fallback: build a simple Perceiver Resampler
            resampler = _build_simple_resampler(
                query_num=query_num,
                vision_hidden_size=vision_hidden_size,
                hidden_size=hidden_size,
            )
    except Exception as e:
        logger.warning(f"Failed to load resampler from config: {e}, using simple resampler")
        resampler = _build_simple_resampler(
            query_num=query_num,
            vision_hidden_size=vision_hidden_size,
            hidden_size=hidden_size,
        )

    # Load weights with prefix filtering
    state_dict = load_weights_by_prefix(model_path, prefix=RESAMPLER_PREFIX)
    if state_dict:
        try:
            resampler.load_state_dict(state_dict, strict=False, assign=True)
        except (TypeError, RuntimeError):
            resampler.load_state_dict(state_dict, strict=False)

    resampler.eval()
    if torch_dtype is not None:
        resampler = resampler.to(dtype=torch_dtype)
    if device:
        resampler = resampler.to(device=device)

    return resampler


def _build_simple_resampler(
    query_num: int,
    vision_hidden_size: int,
    hidden_size: int,
) -> nn.Module:
    """Build a simple Perceiver Resampler as fallback.

    This is a minimal implementation for cases where the full HF model
    architecture cannot be loaded.
    """

    class SimpleResampler(nn.Module):
        def __init__(self, query_num: int, vision_dim: int, output_dim: int):
            super().__init__()
            self.query_num = query_num
            self.queries = nn.Parameter(torch.randn(query_num, output_dim))
            self.proj = nn.Linear(vision_dim, output_dim)
            self.attn = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(output_dim)

        def forward(
            self,
            x: torch.Tensor,
            tgt_sizes: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            # x: [batch, seq_len, vision_dim] or [seq_len, vision_dim]
            if x.dim() == 2:
                x = x.unsqueeze(0)
            batch_size = x.shape[0]
            x = self.proj(x)
            queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
            out, _ = self.attn(queries, x, x)
            return self.norm(out)

    return SimpleResampler(query_num, vision_hidden_size, hidden_size)


class MiniCPMVImageEncoder(nn.Module):
    """Vision encoder extracted from MiniCPM-V 2.6.

    This module combines:
    - vpm: SigLIP ViT for patch embedding extraction
    - resampler: Perceiver Resampler for token compression

    Input:
        pixel_values: [total_slices, C, H, W] - all image slices concatenated
        tgt_sizes: [total_slices, 2] - (height, width) patch grid for each slice

    Output:
        image_embeds: [total_tokens, hidden_dim] where total_tokens = total_slices * query_num
        tgt_sizes: passed through for position encoding in LLM
        slice_lengths: number of slices per image (for reconstruction)
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        config = load_minicpm_config(model_path)
        self._device = torch.device(device)

        # Build vision backbone (SigLIP)
        self.vpm = _build_vpm(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Build Perceiver Resampler
        self.resampler = _build_resampler(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Store config for reference
        self.query_num = getattr(config, "query_num", 64)

    @property
    def dtype(self) -> torch.dtype:
        """Get the model dtype from vpm."""
        for param in self.vpm.parameters():
            return param.dtype
        return torch.float32

    def forward(
        self,
        *,
        pixel_values: torch.Tensor | None = None,
        tgt_sizes: torch.Tensor | None = None,
        slice_lengths: list[int] | None = None,
        **_: object,
    ) -> dict[str, Any]:
        """Forward pass through SigLIP + Resampler.

        Args:
            pixel_values: [total_slices, C, H, W] all image slices
            tgt_sizes: [total_slices, 2] patch grid sizes
            slice_lengths: number of slices per original image

        Returns:
            dict with image_embeds, tgt_sizes, slice_lengths
        """
        outputs: dict[str, Any] = {}

        if pixel_values is None:
            return outputs

        pixel_values = pixel_values.to(device=self._device, dtype=self.dtype)
        if tgt_sizes is not None:
            tgt_sizes = tgt_sizes.to(self._device, dtype=torch.long)

        # Pass through SigLIP to get patch embeddings
        # SigLIP output: [batch, num_patches, hidden_dim]
        with torch.no_grad():
            vpm_output = self.vpm(pixel_values)
            # Handle different output formats
            if hasattr(vpm_output, "last_hidden_state"):
                patch_embeds = vpm_output.last_hidden_state
            elif isinstance(vpm_output, tuple):
                patch_embeds = vpm_output[0]
            else:
                patch_embeds = vpm_output

        # Pass through Resampler to compress to fixed query tokens
        # Resampler expects patch embeddings and produces [batch, query_num, hidden_dim]
        with torch.no_grad():
            if tgt_sizes is not None:
                image_embeds = self.resampler(patch_embeds, tgt_sizes=tgt_sizes)
            else:
                image_embeds = self.resampler(patch_embeds)

        # Flatten to [total_tokens, hidden_dim] for embedding injection
        if image_embeds.dim() == 3:
            # [batch, query_num, hidden_dim] -> [total_tokens, hidden_dim]
            image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])

        outputs["image_embeds"] = image_embeds
        if tgt_sizes is not None:
            outputs["tgt_sizes"] = tgt_sizes
        if slice_lengths is not None:
            outputs["slice_lengths"] = slice_lengths

        # Calculate token counts per image for embedding scatter
        if slice_lengths:
            token_counts = [length * self.query_num for length in slice_lengths]
            outputs["image_token_counts"] = torch.tensor(
                token_counts, device=self._device, dtype=torch.long
            )

        return outputs
