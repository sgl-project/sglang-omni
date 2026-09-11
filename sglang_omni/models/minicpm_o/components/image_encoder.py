# SPDX-License-Identifier: Apache-2.0
"""Image encoder component for MiniCPM-o.

SigLIP NaViT vision tower (``vpm.``) + perceiver Resampler (``resampler.``),
assembled from the sglang srt modules the in-engine MiniCPM-V models already
use (``Idefics2VisionTransformer`` + ``Resampler2_5``, see srt
``minicpmv.init_vision_module``/``init_resampler``). The srt vision tower is
semantically identical to the checkpoint's remote-code
``SiglipVisionTransformer`` but runs srt ``VisionAttention`` (backend
selectable) instead of eager attention. The forward pass mirrors the remote
code's ``get_vision_embedding``: variable-resolution slices are padded into a
patch batch, run through the vision tower, then compressed to ``query_num``
tokens per slice by the resampler.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, PretrainedConfig

from sglang_omni.models.weight_loader import (
    load_weights_by_prefix,
    resolve_dtype,
    resolve_model_path,
)

logger = logging.getLogger(__name__)

# srt VisionAttention stacks q/k/v into one qkv_proj and renames out_proj.
_STACKED_QKV = [
    ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
    ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
    ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
]


def _vision_config_object(config: PretrainedConfig) -> PretrainedConfig:
    vision_config = config.vision_config
    if isinstance(vision_config, dict):
        return PretrainedConfig.from_dict(vision_config)
    return vision_config


def _init_sglang_tp() -> None:
    """Initialize the TP=1 sglang context the srt vision modules require.

    The srt parallel layers (``VisionAttention``, ``ReplicatedLinear``) need
    an initialized model-parallel group and global server args. When the
    stage shares a process with an sglang engine that already initialized
    them, reuse that state; otherwise bring up a single-rank group.
    """
    import os

    import sglang.srt.layers.dp_attention as dp
    from sglang.srt.distributed import parallel_state

    if parallel_state.model_parallel_is_initialized():
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size != 1:
            raise RuntimeError(
                "MiniCPM-o image encoder requires tp_size=1 but the process "
                f"already initialized tp_size={tp_size}"
            )
        return

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

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=0,
        )
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

    dp._ATTN_TP_SIZE = 1
    dp._ATTN_TP_RANK = 0


def _load_srt_weights(module: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Load checkpoint weights into an srt module (srt minicpmv convention).

    Remaps ``self_attn.out_proj`` → ``self_attn.proj`` and routes the q/k/v
    projections into the stacked ``qkv_proj`` via per-shard weight loaders.
    """
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    params_dict = dict(module.named_parameters())
    loaded = set()
    for name, tensor in weights.items():
        name = name.replace("self_attn.out_proj", "self_attn.proj")
        for param_name, weight_name, shard_id in _STACKED_QKV:
            if weight_name not in name:
                continue
            target = name.replace(weight_name, param_name)
            if target not in params_dict:
                continue
            param = params_dict[target]
            param.weight_loader(param, tensor, shard_id)
            loaded.add(target)
            break
        else:
            if name not in params_dict:
                raise KeyError(f"unexpected checkpoint weight: {name}")
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
            loaded.add(name)
    missing = set(params_dict) - loaded
    if missing:
        raise KeyError(f"checkpoint missing weights for: {sorted(missing)[:8]}")


class MiniCPMOImageEncoder(nn.Module):
    """srt ``Idefics2VisionTransformer`` (``vpm.``) + ``Resampler2_5``
    (``resampler.``)."""

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

        _init_sglang_tp()
        from sglang.srt.models.idefics2 import Idefics2VisionTransformer
        from sglang.srt.models.minicpmv import Resampler2_5

        vision_config = _vision_config_object(config)
        vpm = Idefics2VisionTransformer(vision_config)
        if getattr(config, "drop_vision_last_layer", False):
            vpm.encoder.layers = vpm.encoder.layers[:-1]
        _load_srt_weights(vpm, load_weights_by_prefix(model_dir, prefix=("vpm.",)))
        self.vpm = vpm

        embed_dim = config.hidden_size
        resampler = Resampler2_5(
            num_queries=config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_config.hidden_size,
        )
        _load_srt_weights(
            resampler, load_weights_by_prefix(model_dir, prefix=("resampler.",))
        )
        self.resampler = resampler

        self.eval()
        self.to(device=self._device, dtype=torch_dtype)
        # pos_embed is an fp32 non-persistent buffer; keep it out of the
        # bf16 cast (the resampler forward casts per slice).
        self.resampler._set_2d_pos_cache(self.resampler.max_size, device=device)

        self.vision_batch_size = int(getattr(config, "vision_batch_size", 16))

    def _run_vpm(
        self,
        pixel_values: torch.Tensor,
        patch_attn_mask: torch.Tensor,
        tgt_sizes: torch.Tensor,
        patch_counts_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Run the srt vision tower on a padded slice batch.

        srt ``VisionAttention`` interprets ``cu_seqlens`` over a *packed*
        layout (valid patches back to back), while the conv patch embedding
        needs the padded ``(B, 3, p, max_patches*p)`` batch. Bridge the two:
        embed padded, gather the valid patches into one ``(1, total, D)``
        packed sequence for the encoder, then scatter back to the padded
        ``(B, max_patches, D)`` shape the resampler consumes.
        """
        from sglang.srt.layers.attention.vision import (
            prepare_vision_attention_metadata,
        )

        embeds = self.vpm.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        valid = patch_attn_mask[:, 0, :]  # (B, max_patches)
        packed = embeds[valid].unsqueeze(0)  # (1, total, D)

        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(patch_counts_cpu.to(torch.int32), dim=0),
            ]
        ).to(embeds.device)
        packed = self.vpm.encoder(
            packed,
            cu_seqlens=cu_seqlens,
            forward_metadata=prepare_vision_attention_metadata(
                cu_seqlens, device=embeds.device
            ),
        )
        packed = self.vpm.post_layernorm(packed)

        out = torch.zeros_like(embeds)
        out[valid] = packed.squeeze(0)
        return out

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
        if not pixel_values or tgt_sizes is None:
            return {}
        tgt_sizes_cpu = tgt_sizes.to("cpu", dtype=torch.int32)
        tgt_sizes = tgt_sizes_cpu.to(self._device)

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

        # Patch counts stay host-side: max_patches and the validity mask come
        # from CPU tgt_sizes, so no GPU→CPU sync on the hot path.
        patch_counts_cpu = tgt_sizes_cpu[:, 0] * tgt_sizes_cpu[:, 1]
        max_patches = int(patch_counts_cpu.max())
        patch_range = torch.arange(max_patches, device=self._device)
        patch_attn_mask = (
            patch_range[None, :] < patch_counts_cpu.to(self._device)[:, None]
        ).unsqueeze(1)

        chunk = self.vision_batch_size
        if B > chunk:
            hs = []
            for start in range(0, B, chunk):
                end = start + chunk
                hs.append(
                    self._run_vpm(
                        all_pixel_values[start:end],
                        patch_attn_mask[start:end],
                        tgt_sizes[start:end],
                        patch_counts_cpu[start:end],
                    )
                )
            vision_embedding = torch.vstack(hs)
        else:
            vision_embedding = self._run_vpm(
                all_pixel_values, patch_attn_mask, tgt_sizes, patch_counts_cpu
            )

        # (B, query_num, hidden) → flat placeholder rows in slice order.
        vision_embedding = self.resampler(vision_embedding, tgt_sizes)
        return {"image_embeds": vision_embedding.flatten(0, 1)}
