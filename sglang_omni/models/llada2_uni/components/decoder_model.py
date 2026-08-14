# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Z-Image adapter for LLaDA2-Uni decoder SP."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ZImageParallelConfig:
    sp_rank: int = 0
    sp_size: int = 1
    ulysses_degree: int | None = None
    ring_degree: int = 1
    attention_backend: str = "fa"

    def __post_init__(self) -> None:
        if self.sp_size < 1:
            raise ValueError("Z-Image SP size must be positive")
        if not 0 <= self.sp_rank < self.sp_size:
            raise ValueError(f"invalid Z-Image SP rank {self.sp_rank}/{self.sp_size}")
        if self.sp_size & (self.sp_size - 1):
            raise ValueError("Z-Image SP size must be a power of two")
        if self.ring_degree < 1:
            raise ValueError("Z-Image ring degree must be positive")
        if self.sp_size != self.resolved_ulysses_degree * self.ring_degree:
            raise ValueError(
                "Z-Image sp_size must equal ulysses_degree * ring_degree: "
                f"{self.sp_size} != {self.resolved_ulysses_degree} * "
                f"{self.ring_degree}"
            )

    @property
    def resolved_ulysses_degree(self) -> int:
        return self.sp_size if self.ulysses_degree is None else self.ulysses_degree


@dataclass(frozen=True)
class SGLangZImageRuntime:
    dit_config: Any
    pipeline_config: Any


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _torch_dtype_to_sglang_precision(dtype: torch.dtype) -> str:
    precision_by_dtype = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    try:
        return precision_by_dtype[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported SGLang Z-Image dtype: {dtype}") from exc


def _get_sglang_runtime_symbols() -> SimpleNamespace:
    from sglang.multimodal_gen.configs.models.dits.zimage import (
        ZImageArchConfig,
        ZImageDitConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
        ZImagePipelineConfig,
    )
    from sglang.multimodal_gen.runtime.distributed import (
        get_sp_parallel_rank,
        get_sp_world_size,
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args,
    )
    from sglang.multimodal_gen.utils import set_mixed_precision_policy

    return SimpleNamespace(
        ZImageArchConfig=ZImageArchConfig,
        ZImageDitConfig=ZImageDitConfig,
        ZImagePipelineConfig=ZImagePipelineConfig,
        ServerArgs=ServerArgs,
        get_global_server_args=get_global_server_args,
        set_global_server_args=set_global_server_args,
        set_mixed_precision_policy=set_mixed_precision_policy,
        model_parallel_is_initialized=model_parallel_is_initialized,
        maybe_init_distributed_environment_and_model_parallel=(
            maybe_init_distributed_environment_and_model_parallel
        ),
        get_sp_world_size=get_sp_world_size,
        get_sp_parallel_rank=get_sp_parallel_rank,
    )


def _build_sglang_zimage_configs(
    config: dict[str, Any], symbols: SimpleNamespace
) -> tuple[Any, Any]:
    arch_config = symbols.ZImageArchConfig(
        all_patch_size=tuple(config["all_patch_size"]),
        all_f_patch_size=tuple(config["all_f_patch_size"]),
        in_channels=int(config["in_channels"]),
        dim=int(config["dim"]),
        num_layers=int(config["n_layers"]),
        n_refiner_layers=int(config["n_refiner_layers"]),
        num_attention_heads=int(config["n_heads"]),
        n_kv_heads=int(config["n_kv_heads"]),
        norm_eps=float(config["norm_eps"]),
        qk_norm=bool(config["qk_norm"]),
        cap_feat_dim=int(config["cap_feat_dim"]),
        rope_theta=float(config["rope_theta"]),
        t_scale=float(config["t_scale"]),
        axes_dims=tuple(config["axes_dims"]),
        axes_lens=tuple(config["axes_lens"]),
    )
    dit_config = symbols.ZImageDitConfig(arch_config=arch_config)
    return dit_config, symbols.ZImagePipelineConfig(dit_config=dit_config)


def ensure_sglang_zimage_runtime(
    *,
    model_path: str,
    config: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
    parallel_config: ZImageParallelConfig,
) -> SGLangZImageRuntime:
    """Initialize SGLang's model-parallel state without taking cleanup ownership."""
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("SGLang Z-Image SP requires an explicitly indexed CUDA device")
    os.environ["LOCAL_RANK"] = str(device.index)

    symbols = _get_sglang_runtime_symbols()
    dit_config, pipeline_config = _build_sglang_zimage_configs(config, symbols)
    pipeline_config.dit_precision = _torch_dtype_to_sglang_precision(dtype)
    symbols.set_mixed_precision_policy(
        param_dtype=dtype,
        reduce_dtype=dtype,
        output_dtype=dtype,
    )

    if not symbols.model_parallel_is_initialized():
        try:
            symbols.get_global_server_args()
        except ValueError:
            symbols.set_global_server_args(
                symbols.ServerArgs(
                    model_path=model_path,
                    trust_remote_code=True,
                    backend="sglang",
                    attention_backend=parallel_config.attention_backend,
                    num_gpus=parallel_config.sp_size,
                    tp_size=1,
                    sp_degree=parallel_config.sp_size,
                    ulysses_degree=parallel_config.resolved_ulysses_degree,
                    ring_degree=parallel_config.ring_degree,
                    enable_cfg_parallel=False,
                    pipeline_config=pipeline_config,
                    dit_cpu_offload=False,
                    dit_layerwise_offload=False,
                    text_encoder_cpu_offload=False,
                    image_encoder_cpu_offload=False,
                    use_fsdp_inference=False,
                    base_gpu_id=device.index,
                )
            )
        distributed_init_method = (
            "env://"
            if parallel_config.sp_size > 1
            else f"tcp://127.0.0.1:{_free_tcp_port()}"
        )
        symbols.maybe_init_distributed_environment_and_model_parallel(
            tp_size=1,
            sp_size=parallel_config.sp_size,
            cfg_degree=1,
            ulysses_degree=parallel_config.resolved_ulysses_degree,
            ring_degree=parallel_config.ring_degree,
            distributed_init_method=distributed_init_method,
        )

    actual_size = int(symbols.get_sp_world_size())
    actual_rank = int(symbols.get_sp_parallel_rank())
    if actual_size != parallel_config.sp_size:
        raise RuntimeError(
            "SGLang Z-Image SP world size does not match stage metadata: "
            f"{actual_size} != {parallel_config.sp_size}"
        )
    if actual_rank != parallel_config.sp_rank:
        raise RuntimeError(
            "SGLang Z-Image SP rank does not match stage metadata: "
            f"{actual_rank} != {parallel_config.sp_rank}"
        )
    return SGLangZImageRuntime(
        dit_config=dit_config,
        pipeline_config=pipeline_config,
    )


def _get_sglang_loader_symbols() -> SimpleNamespace:
    from sglang.multimodal_gen.runtime.loader.fsdp_load import (
        load_model_from_full_model_state_dict,
    )
    from sglang.multimodal_gen.runtime.loader.utils import (
        get_param_names_mapping,
        set_default_torch_dtype,
    )
    from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
    from sglang.multimodal_gen.runtime.loader.weight_utils import (
        safetensors_weights_iterator,
    )
    from sglang.multimodal_gen.runtime.models.dits.zimage import (
        ZImageTransformer2DModel,
    )

    return SimpleNamespace(
        ZImageTransformer2DModel=ZImageTransformer2DModel,
        WeightLoadPlan=WeightLoadPlan,
        get_param_names_mapping=get_param_names_mapping,
        load_model_from_full_model_state_dict=load_model_from_full_model_state_dict,
        safetensors_weights_iterator=safetensors_weights_iterator,
        set_default_torch_dtype=set_default_torch_dtype,
    )


def _get_sglang_forward_context():
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    return set_forward_context


def load_sglang_zimage_model(
    *,
    decoder_dir: str | Path,
    config: dict[str, Any],
    dit_config: Any,
    device: torch.device,
    dtype: torch.dtype,
    checkpoint_load_device: torch.device,
) -> nn.Module:
    """Load the native Z-Image DiT with SGLang 0.5.16's loader contract."""
    symbols = _get_sglang_loader_symbols()
    device = torch.device(device)
    checkpoint_load_device = torch.device(checkpoint_load_device)
    with symbols.set_default_torch_dtype(dtype), torch.device("meta"):
        model = symbols.ZImageTransformer2DModel(
            config=dit_config,
            hf_config=dict(config),
        )

    mapping = dict(model.param_names_mapping)
    mapping[r"semantic_embedder\.(.*)$"] = r"cap_embedder.\1"
    mapping_fn = symbols.get_param_names_mapping(mapping)
    weight_load_plan = symbols.WeightLoadPlan(
        checkpoint_load_device=checkpoint_load_device
    )
    weights = symbols.safetensors_weights_iterator(
        [str(Path(decoder_dir) / "model.safetensors")],
        weight_load_plan=weight_load_plan,
    )
    symbols.load_model_from_full_model_state_dict(
        model=model,
        full_sd_iterator=weights,
        checkpoint_load_device=checkpoint_load_device,
        param_dtype=dtype,
        strict=True,
        cpu_offload=False,
        param_names_mapping=mapping_fn,
    )
    model = model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


class SGLangZImageModelAdapter(nn.Module):
    """Expose SGLang's native Z-Image model through the decoder's model API."""

    def __init__(self, model: nn.Module, pipeline_config: Any):
        super().__init__()
        self.model = model
        self.pipeline_config = pipeline_config
        self._batch: SimpleNamespace | None = None

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError(
                "Z-Image latents must have shape [batch, channels, frames, height, width]"
            )
        self._batch = SimpleNamespace(
            height=int(latents.shape[-2]) * 8,
            width=int(latents.shape[-1]) * 8,
            raw_latent_shape=tuple(latents.shape),
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_seq_lens=None,
            negative_prompt_seq_lens=None,
        )
        local_latents, _ = self.pipeline_config.shard_latents_for_sp(
            self._batch, latents
        )
        return local_latents

    def gather_latents(
        self, latents: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor:
        if self._batch is None:
            raise RuntimeError("prepare_latents must run before gather_latents")
        if isinstance(latents, list):
            latents = torch.stack(latents)
        return self.pipeline_config.gather_latents_for_sp(latents, self._batch)

    def forward(
        self,
        *,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
        return_dict: bool = True,
        **_: Any,
    ):
        if self._batch is None:
            raise RuntimeError("prepare_latents must run before Z-Image forward")
        if len(x) != len(cap_feats):
            raise ValueError("Z-Image requires one caption feature tensor per latent")
        try:
            prompt_embeds = torch.stack(cap_feats)
        except RuntimeError as exc:
            raise ValueError("Z-Image caption feature lengths must match") from exc
        self._batch.prompt_embeds = [prompt_embeds]
        self._batch.prompt_seq_lens = [[int(item.shape[0]) for item in cap_feats]]

        time = t.float()
        if time.ndim == 0:
            time = time.reshape(1)
        if time.shape[0] == 1 and len(x) > 1:
            time = time.expand(len(x))
        if time.shape[0] != len(x):
            raise ValueError("Z-Image timestep batch does not match latent batch")
        timestep = (1.0 - time) * 1000.0
        cond_kwargs = self.pipeline_config.prepare_pos_cond_kwargs(
            self._batch,
            x[0].device,
            self.model.rotary_emb,
            x[0].dtype,
        )
        with _get_sglang_forward_context()(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=None,
        ):
            prediction = self.model(
                hidden_states=x,
                encoder_hidden_states=cap_feats,
                timestep=timestep,
                guidance=0,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                **cond_kwargs,
            )
        outputs = list((-prediction).unbind(dim=0))
        if not return_dict:
            return (outputs,)

        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=outputs)


class ZImageTransformer2DModelWrapper(nn.Module):
    """Load and run the SGLang 0.5.16 native Z-Image SP implementation."""

    def __init__(
        self,
        *,
        model_path: str,
        decoder_dir: str | Path,
        config: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        parallel_config: ZImageParallelConfig,
        checkpoint_load_device: torch.device,
    ):
        super().__init__()
        runtime = ensure_sglang_zimage_runtime(
            model_path=model_path,
            config=config,
            dtype=dtype,
            device=device,
            parallel_config=parallel_config,
        )
        model = load_sglang_zimage_model(
            decoder_dir=decoder_dir,
            config=config,
            dit_config=runtime.dit_config,
            device=device,
            dtype=dtype,
            checkpoint_load_device=checkpoint_load_device,
        )
        self.adapter = SGLangZImageModelAdapter(model, runtime.pipeline_config)

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.adapter.prepare_latents(latents)

    def gather_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.adapter.gather_latents(latents)

    def forward(self, **kwargs):
        return self.adapter(**kwargs)
