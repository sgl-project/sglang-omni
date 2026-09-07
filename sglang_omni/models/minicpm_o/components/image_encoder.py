# SPDX-License-Identifier: Apache-2.0
"""Image encoder component for MiniCPM-o.

Wraps the checkpoint's remote-code SigLIP NaViT vision tower (``vpm.``) and
perceiver Resampler (``resampler.``). The forward pass mirrors the remote
code's ``get_vision_embedding``: variable-resolution slices are padded into a
patch batch, run through the vision tower with a patch attention mask, then
compressed to ``query_num`` tokens per slice by the resampler.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang_omni.models.weight_loader import (
    load_module,
    resolve_dtype,
    resolve_model_path,
)

logger = logging.getLogger(__name__)


class MiniCPMOImageEncoder(nn.Module):
    """SigLIP NaViT vision tower + Resampler extracted from the remote code."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        model_dir = str(resolve_model_path(model_path))
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self._device = torch.device(device)
        self._dtype = torch_dtype

        # Mirrors the remote code's init_vision_module / init_resampler.
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
        self.vpm = load_module(
            vpm, model_dir, prefix=("vpm.",), dtype=torch_dtype, device=device
        )

        embed_dim = config.hidden_size
        resampler = resampler_cls(
            num_queries=config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_config.hidden_size,
            adaptive=True,
        )
        self.resampler = load_module(
            resampler,
            model_dir,
            prefix=("resampler.",),
            dtype=torch_dtype,
            device=device,
        )
        # pos_embed is a non-persistent buffer absent from the checkpoint, so
        # load_module leaves it unmaterialized. Recompute the fp32 sincos
        # cache on device (the forward casts it per slice).
        self.resampler._set_2d_pos_cache(self.resampler.max_size, device=device)
        self.vision_batch_size = int(getattr(config, "vision_batch_size", 16))

    @torch.no_grad()
    def forward(
        self,
        *,
        pixel_values: list[torch.Tensor] | None = None,
        tgt_sizes: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Encode a flat list of image slices.

        Args:
            pixel_values: one ``(3, patch_size, num_patches * patch_size)``
                tensor per slice, as produced by the checkpoint processor.
            tgt_sizes: ``(num_slices, 2)`` patch grid ``(h, w)`` per slice.

        Returns:
            ``image_embeds``: flat ``(num_slices * query_num, hidden)`` rows in
            slice order, matching the placeholder token layout.
        """
        if not pixel_values:
            return {}
        tgt_sizes = tgt_sizes.to(self._device, dtype=torch.int32)

        # get_vision_embedding: flatten each slice to (num_patches, 3*p*p)
        # rows, pad across slices, then restore (B, 3, p, max_patches*p).
        all_pixel_values = [
            v.to(self._device, dtype=self._dtype).flatten(end_dim=1).permute(1, 0)
            for v in pixel_values
        ]
        all_pixel_values = pad_sequence(
            all_pixel_values, batch_first=True, padding_value=0.0
        )
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

        patch_counts = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        max_patches = int(patch_counts.max().item())
        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=self._device
        )
        for i in range(B):
            patch_attn_mask[i, 0, : patch_counts[i]] = True

        chunk = self.vision_batch_size
        if B > chunk:
            hs = []
            for start in range(0, B, chunk):
                end = start + chunk
                hs.append(
                    self.vpm(
                        all_pixel_values[start:end],
                        patch_attention_mask=patch_attn_mask[start:end],
                        tgt_sizes=tgt_sizes[start:end],
                    ).last_hidden_state
                )
            vision_embedding = torch.vstack(hs)
        else:
            vision_embedding = self.vpm(
                all_pixel_values,
                patch_attention_mask=patch_attn_mask,
                tgt_sizes=tgt_sizes,
            ).last_hidden_state

        # (B, query_num, hidden) → flat placeholder rows in slice order.
        vision_embedding = self.resampler(vision_embedding, tgt_sizes)
        return {"image_embeds": vision_embedding.flatten(0, 1)}
