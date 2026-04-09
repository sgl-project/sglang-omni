# SPDX-License-Identifier: Apache-2.0
"""Standalone image encoder for Ming-Omni pipeline.

Loads the vision encoder + projector from checkpoint and runs:
  pixel_values → MingOmniVisionEncoder → VisionProjector → L2 normalize
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.models.ming_omni.components.common import load_ming_config
from sglang_omni.models.ming_omni.components.projectors import VisionProjector
from sglang_omni.models.ming_omni.components.vision_encoder import MingOmniVisionEncoder
from sglang_omni.models.weight_loader import resolve_model_path

logger = logging.getLogger(__name__)


def _iter_weights_by_prefix(model_dir: Path, prefix: str):
    """Iterate checkpoint weights with given prefix, stripping it."""
    from safetensors import safe_open

    index_file = model_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]

    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shards.setdefault(shard, []).append(key)

    for shard, keys in sorted(shards.items()):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for key in keys:
                yield key[len(prefix) :], f.get_tensor(key)


class MingImageEncoder(nn.Module):
    """Image encoder for Ming-Omni pipeline.

    Loads vision encoder + projector from checkpoint and produces
    L2-normalized image embeddings ready for injection into the thinker.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | None = None,
    ) -> None:
        super().__init__()

        resolved_path = resolve_model_path(model_path)
        model_dir = Path(resolved_path)
        config = load_ming_config(model_path)

        vision_cfg = config.vision_config
        mlp_depth = config.mlp_depth

        # Need sglang TP context for VisionAttention
        self._init_sglang_tp()

        # Build vision encoder
        from transformers import PretrainedConfig

        vision_config_obj = PretrainedConfig(**self._vision_dict(vision_cfg))
        self.visual = MingOmniVisionEncoder(
            vision_config_obj, quant_config=None, prefix="visual"
        )

        # Build projector
        vision_dim = vision_cfg.out_hidden_size
        llm_dim = config.llm_config.hidden_size
        self.linear_proj = VisionProjector(
            vision_dim=vision_dim, llm_dim=llm_dim, mlp_depth=mlp_depth
        )

        # Load weights
        loaded_vis = self.visual.load_weights(
            _iter_weights_by_prefix(model_dir, "vision.")
        )
        loaded_proj = self.linear_proj.load_weights(
            _iter_weights_by_prefix(model_dir, "linear_proj.")
        )
        logger.info(
            "MingImageEncoder loaded: %d vision + %d projector weights",
            len(loaded_vis),
            len(loaded_proj),
        )

        # Store spatial merge size for token count computation
        self._spatial_merge_size = vision_cfg.spatial_merge_size

        # Move to device
        torch_dtype = _resolve_dtype(dtype)
        self.to(device=device, dtype=torch_dtype)
        self.eval()

        # Release model parallel state so the thinker (loaded later) can
        # reinitialize it.  At runtime, the thinker's TP context will be
        # active and this encoder can use it transparently (TP=1).
        self._cleanup_sglang_tp()

    @staticmethod
    def _vision_dict(vision_cfg: Any) -> dict:
        """Convert VisionConfig dataclass to plain dict for PretrainedConfig."""
        if hasattr(vision_cfg, "__dataclass_fields__"):
            from dataclasses import asdict

            return asdict(vision_cfg)
        return {k: v for k, v in vars(vision_cfg).items() if not k.startswith("_")}

    _did_init_tp = False  # Track whether we initialized TP ourselves

    @classmethod
    def _init_sglang_tp(cls):
        """Initialize minimal sglang TP=1 context if not already done."""
        import os

        import sglang.srt.layers.dp_attention as dp

        if getattr(dp, "_ATTN_TP_SIZE", None) is not None and dp._ATTN_TP_SIZE > 0:
            return  # Already initialized

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])

        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        try:
            set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        except Exception:
            pass  # Already set

        from sglang.srt.distributed import parallel_state

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.init_distributed_environment(
                backend="nccl", world_size=1, rank=0, local_rank=0
            )
            parallel_state.initialize_model_parallel()
            cls._did_init_tp = True

        dp._ATTN_TP_SIZE = 1
        dp._ATTN_TP_RANK = 0

    @classmethod
    def _cleanup_sglang_tp(cls):
        """Destroy model parallel state so a later component (thinker) can reinit.

        Only cleans up if we were the ones who initialized it.
        torch.distributed stays alive — only the TP/PP groups are removed.
        """
        if not cls._did_init_tp:
            return
        cls._did_init_tp = False

        from sglang.srt.distributed import parallel_state

        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
            logger.info("Cleaned up model parallel state for thinker reuse")

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Encode images and return embeddings.

        Args:
            pixel_values: Flattened pixel values [total_patches, patch_dim].
            image_grid_thw: [num_images, 3] tensor of (t, h, w).

        Returns:
            Dict with:
            - ``image_embeds`` [seq_len, llm_hidden_size] (L2-normalized)
            - ``image_grid_thw`` [num_images, 3]
            - ``image_token_counts`` [num_images] per-image token counts
        """
        pixel_values = pixel_values.to(
            device=self.visual.device, dtype=self.visual.dtype
        )
        image_grid_thw = image_grid_thw.to(device=self.visual.device)

        with torch.no_grad():
            image_embeds = self.visual(pixel_values, image_grid_thw)

            # Deepstack: use only base merger output for projection
            if self.visual.use_deepstack:
                image_embeds = image_embeds[:, : self.visual.image_emb_dim]

            image_embeds = self.linear_proj(image_embeds)
            image_embeds = F.normalize(image_embeds, dim=-1)

        # Per-image token counts after spatial merge: t * h * w / merge^2
        merge_sq = self._spatial_merge_size**2
        image_token_counts = (
            image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        ) // merge_sq

        return {
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
            "image_token_counts": image_token_counts,
        }


def _resolve_dtype(dtype: str | None) -> torch.dtype:
    if dtype is None or dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    return torch.bfloat16
