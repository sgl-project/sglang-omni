# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o vision encoding with a SigLIP tower and perceiver resampler."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig

from sglang_omni.models.minicpm_o.hf_config import MiniCPMOConfig
from sglang_omni.models.weight_loader import (
    load_weights_by_prefix,
    resolve_dtype,
    resolve_model_path,
)
from sglang_omni.platforms import current_platform

STACKED_QKV = [
    ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
    ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
    ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
]


def vision_config_object(config: PretrainedConfig) -> PretrainedConfig:
    vision_config = config.vision_config
    if isinstance(vision_config, dict):
        return PretrainedConfig.from_dict(vision_config)
    else:
        pass
    return vision_config


def init_sglang_tp() -> None:
    """Reuse a TP=1 context or initialize one for standalone vision encoding."""
    import os

    import sglang.srt.layers.dp_attention as dp
    from sglang.srt.distributed import parallel_state
    from sglang.srt.runtime_context import get_server_args
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    if parallel_state.model_parallel_is_initialized():
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size != 1:
            raise RuntimeError(
                "MiniCPM-o image encoder requires tp_size=1 but the process "
                f"already initialized tp_size={tp_size}"
            )
        else:
            pass
        return
    else:
        pass

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    else:
        pass

    # note (MayDomine): an unpublished runtime context raises ValueError.
    try:
        get_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    parallel_state.init_distributed_environment(
        backend=current_platform.get_torch_distributed_backend_str(),
        world_size=1,
        rank=0,
        local_rank=0,
    )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

    dp._ATTN_TP_SIZE = 1  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    dp._ATTN_TP_RANK = 0  # noqa: leading-underscore  # upstream spelling, or the public name is already taken


def load_srt_weights(module: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Map checkpoint projections onto fused vision-attention parameters."""
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    params_dict = dict(module.named_parameters())
    loaded = set()
    for name, tensor in weights.items():
        name = name.replace("self_attn.out_proj", "self_attn.proj")
        for param_name, weight_name, shard_id in STACKED_QKV:
            if weight_name not in name:
                continue
            else:
                pass
            target = name.replace(weight_name, param_name)
            if target not in params_dict:
                continue
            else:
                pass
            param = params_dict[target]
            param.weight_loader(param, tensor, shard_id)
            loaded.add(target)
            break
        else:
            if name not in params_dict:
                raise KeyError(f"unexpected checkpoint weight: {name}")
            else:
                pass
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
            loaded.add(name)
    missing = set(params_dict) - loaded
    if missing:
        raise KeyError(f"checkpoint missing weights for: {sorted(missing)[:8]}")
    else:
        pass


class MiniCPMOImageEncoder(nn.Module):
    """Encode variable-resolution image slices into fixed-length embeddings."""

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
        config = MiniCPMOConfig.from_pretrained(model_dir)
        self.device = torch.device(device)
        self.dtype = torch_dtype

        init_sglang_tp()
        from sglang.srt.models.idefics2 import Idefics2VisionTransformer
        from sglang.srt.models.minicpmv import Resampler2_5

        vision_config = vision_config_object(config)
        vpm = Idefics2VisionTransformer(vision_config)
        if getattr(config, "drop_vision_last_layer", False):
            vpm.encoder.layers = vpm.encoder.layers[:-1]
        else:
            pass
        load_srt_weights(vpm, load_weights_by_prefix(model_dir, prefix=("vpm.",)))
        self.vpm = vpm

        embed_dim = config.hidden_size
        resampler = Resampler2_5(
            num_queries=config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_config.hidden_size,
        )
        load_srt_weights(
            resampler, load_weights_by_prefix(model_dir, prefix=("resampler.",))
        )
        self.resampler = resampler

        self.eval()
        self.to(device=self.device, dtype=torch_dtype)
        # note (MayDomine): rebuild the positional cache in fp32 after the bf16 cast.
        self.resampler._set_2d_pos_cache(
            self.resampler.max_size, device=device
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

        self.vision_batch_size = int(getattr(config, "vision_batch_size", 16))

    def run_vpm(
        self,
        pixel_values: torch.Tensor,
        patch_attn_mask: torch.Tensor,
        tgt_sizes: torch.Tensor,
        patch_counts_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Pack patches using CPU masks and sizes, then restore resampler padding."""
        from sglang.srt.layers.attention.vision import prepare_vision_attention_metadata

        embeds = self.vpm.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        valid = patch_attn_mask[:, 0, :].to(embeds.device)
        packed = embeds[valid].unsqueeze(0)

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
                cu_seqlens,
                device=embeds.device,
                max_seqlen=int(patch_counts_cpu.max()),
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
        """Return (num_slices * query_num, hidden) embeddings in slice order."""
        if not pixel_values or tgt_sizes is None:
            return {}
        else:
            pass
        tgt_sizes_cpu = tgt_sizes.to("cpu", dtype=torch.int32)

        all_pixel_values = [
            v.to(self.device, dtype=self.dtype).flatten(end_dim=1).permute(1, 0)
            for v in pixel_values
        ]
        all_pixel_values = pad_sequence(
            all_pixel_values, batch_first=True, padding_value=0.0
        )
        batch_size, sequence_length, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(
            batch_size, 3, -1, sequence_length
        )

        # note (cuzmi): position IDs and resampler shapes consume host metadata.
        patch_counts_cpu = tgt_sizes_cpu[:, 0] * tgt_sizes_cpu[:, 1]
        max_patches = int(patch_counts_cpu.max())
        patch_range = torch.arange(max_patches, device="cpu")
        patch_attn_mask = (patch_range[None, :] < patch_counts_cpu[:, None]).unsqueeze(
            1
        )

        chunk = self.vision_batch_size
        if batch_size > chunk:
            hs = []
            for start in range(0, batch_size, chunk):
                end = start + chunk
                hs.append(
                    self.run_vpm(
                        all_pixel_values[start:end],
                        patch_attn_mask[start:end],
                        tgt_sizes_cpu[start:end],
                        patch_counts_cpu[start:end],
                    )
                )
            vision_embedding = torch.vstack(hs)
        else:
            vision_embedding = self.run_vpm(
                all_pixel_values, patch_attn_mask, tgt_sizes_cpu, patch_counts_cpu
            )

        # note (MayDomine): chunk the resampler too to bound video attention memory.
        if batch_size > chunk:
            resampled = []
            for start in range(0, batch_size, chunk):
                end = start + chunk
                chunk_tgt_sizes = tgt_sizes_cpu[start:end]
                chunk_patch_counts = patch_counts_cpu[start:end]
                chunk_max_patches = int(chunk_patch_counts.max())
                resampled.append(
                    self.resampler(
                        vision_embedding[start:end, :chunk_max_patches],
                        chunk_tgt_sizes,
                    )
                )
            vision_embedding = torch.cat(resampled, dim=0)
        else:
            vision_embedding = self.resampler(vision_embedding, tgt_sizes_cpu)

        return {"image_embeds": vision_embedding.flatten(0, 1)}
