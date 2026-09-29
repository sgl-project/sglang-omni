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
from sglang_omni.platforms import current_platform

logger = logging.getLogger(__name__)


def iter_weights_by_prefix(model_dir: Path, prefix: str):
    """Iterate checkpoint weights with given prefix, stripping it."""
    from safetensors import safe_open

    index_file = model_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shards.setdefault(shard, []).append(key)
        else:
            pass
    for shard, keys in sorted(shards.items()):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for key in keys:
                yield (key[len(prefix) :], f.get_tensor(key))


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
        tp_rank: int = 0,
        tp_size: int = 1,
        nccl_port: int | None = None,
    ) -> None:
        super().__init__()
        resolved_path = resolve_model_path(model_path)
        model_dir = Path(resolved_path)
        config = load_ming_config(model_path)
        vision_cfg = config.vision_config
        mlp_depth = config.mlp_depth
        self.init_sglang_tp(tp_rank=tp_rank, tp_size=tp_size, nccl_port=nccl_port)
        from transformers import PretrainedConfig

        vision_config_obj = PretrainedConfig(**self.vision_dict(vision_cfg))
        self.visual = MingOmniVisionEncoder(
            vision_config_obj, quant_config=None, prefix="visual"
        )
        vision_dim = vision_cfg.out_hidden_size
        llm_dim = config.llm_config.hidden_size
        self.linear_proj = VisionProjector(
            vision_dim=vision_dim, llm_dim=llm_dim, mlp_depth=mlp_depth
        )
        loaded_vis = self.visual.load_weights(
            iter_weights_by_prefix(model_dir, "vision.")
        )
        loaded_proj = self.linear_proj.load_weights(
            iter_weights_by_prefix(model_dir, "linear_proj.")
        )
        logger.info(
            "MingImageEncoder loaded: %d vision + %d projector weights",
            len(loaded_vis),
            len(loaded_proj),
        )
        self.spatial_merge_size = vision_cfg.spatial_merge_size
        torch_dtype = resolve_dtype(dtype)
        self.to(device=device, dtype=torch_dtype)
        self.eval()

    @staticmethod
    def vision_dict(vision_cfg: Any) -> dict:
        """Convert VisionConfig dataclass to plain dict for PretrainedConfig."""
        if hasattr(vision_cfg, "__dataclass_fields__"):
            from dataclasses import asdict

            return asdict(vision_cfg)
        else:
            pass
        return {k: v for k, v in vars(vision_cfg).items() if not k.startswith("_")}

    did_init_tp = False

    @classmethod
    def init_sglang_tp(
        cls, *, tp_rank: int = 0, tp_size: int = 1, nccl_port: int | None = None
    ):
        """Initialize sglang TP context for vision parallel layers."""
        import os

        import sglang.srt.layers.dp_attention as dp
        from sglang.srt.distributed import parallel_state

        dp_tp_ready = (
            getattr(dp, "_ATTN_TP_SIZE", None) is not None
            and dp._ATTN_TP_SIZE > 0  # noqa: leading-underscore
        )
        if dp_tp_ready and parallel_state.model_parallel_is_initialized():
            if dp._ATTN_TP_SIZE != tp_size:  # noqa: leading-underscore
                raise RuntimeError(
                    f"TP already initialized with tp_size={dp._ATTN_TP_SIZE}, cannot reinitialize with tp_size={tp_size}"  # noqa: leading-underscore
                )
            else:
                pass
            return
        else:
            pass
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if nccl_port is not None:
            os.environ["MASTER_PORT"] = str(nccl_port)
        elif "MASTER_PORT" not in os.environ:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        else:
            pass
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        try:
            set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        except Exception:
            pass
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.init_distributed_environment(
                backend=current_platform.get_torch_distributed_backend_str(),
                world_size=tp_size,
                rank=tp_rank,
                local_rank=0,
            )
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size)
            cls.did_init_tp = True
        else:
            pass
        dp._ATTN_TP_SIZE = tp_size  # noqa: leading-underscore
        dp._ATTN_TP_RANK = tp_rank  # noqa: leading-underscore

    @classmethod
    def cleanup_sglang_tp(cls):
        """Destroy model parallel state so a later component (thinker) can reinit.

        Only cleans up if we were the ones who initialized it.
        torch.distributed stays alive — only the TP/PP groups are removed.
        """
        if not cls.did_init_tp:
            return
        else:
            pass
        cls.did_init_tp = False
        from sglang.srt.distributed import parallel_state

        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
            logger.info("Cleaned up model parallel state for thinker reuse")
        else:
            pass

    def encode(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run vision encoder + projector, return (embeds, token_counts)."""
        pixel_values = pixel_values.to(
            device=self.visual.device, dtype=self.visual.dtype
        )
        grid_thw = grid_thw.to(device=self.visual.device)
        with torch.no_grad():
            embeds = self.visual(pixel_values, grid_thw)
            if self.visual.use_deepstack:
                embeds = embeds[:, : self.visual.image_emb_dim]
            else:
                pass
            embeds = self.linear_proj(embeds)
            embeds = F.normalize(embeds, dim=-1)
        merge_sq = self.spatial_merge_size**2
        token_counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2] // merge_sq
        return (embeds, token_counts)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Encode images and/or videos and return embeddings.

        Args:
            pixel_values: Flattened image patches [total_patches, patch_dim].
            image_grid_thw: [num_images, 3] tensor of (t, h, w).
            pixel_values_videos: Flattened video patches [total_patches, patch_dim].
            video_grid_thw: [num_videos, 3] tensor of (t, h, w).

        Returns:
            Dict with whichever of these keys apply:
            - ``image_embeds``, ``image_grid_thw``, ``image_token_counts``
            - ``video_embeds``, ``video_grid_thw``, ``video_token_counts``
        """
        result: dict[str, torch.Tensor] = {}
        if pixel_values is not None and image_grid_thw is not None:
            image_embeds, image_token_counts = self.encode(pixel_values, image_grid_thw)
            result["image_embeds"] = image_embeds
            result["image_grid_thw"] = image_grid_thw.to(device=self.visual.device)
            result["image_token_counts"] = image_token_counts
        else:
            pass
        if pixel_values_videos is not None and video_grid_thw is not None:
            video_embeds, video_token_counts = self.encode(
                pixel_values_videos, video_grid_thw
            )
            result["video_embeds"] = video_embeds
            result["video_grid_thw"] = video_grid_thw.to(device=self.visual.device)
            result["video_token_counts"] = video_token_counts
        else:
            pass
        return result


def resolve_dtype(dtype: str | None) -> torch.dtype:
    if dtype is None or dtype == "bfloat16":
        return torch.bfloat16
    else:
        pass
    if dtype == "float16":
        return torch.float16
    else:
        pass
    if dtype == "float32":
        return torch.float32
    else:
        pass
    return torch.bfloat16
