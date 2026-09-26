# SPDX-License-Identifier: Apache-2.0
"""Adapt Diffusers and SGLang ZImage backbones for semantic-only LLaDA2-Uni."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal, TypedDict

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from safetensors.torch import load_file
from torch import nn

from sglang_omni.models.llada2_uni.components.decoder_runtime import (
    DecoderRuntimeHandle,
)


class DecoderConfig(TypedDict, total=False):
    """Checkpoint architecture overrides; omitted fields use decoder defaults."""

    all_patch_size: list[int] | tuple[int, ...]
    all_f_patch_size: list[int] | tuple[int, ...]
    in_channels: int
    dim: int
    n_layers: int
    n_refiner_layers: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    qk_norm: bool
    cap_feat_dim: int
    rope_theta: float
    t_scale: float
    axes_dims: list[int] | tuple[int, ...]
    axes_lens: list[int] | tuple[int, ...]


def decoder_config(cfg: DecoderConfig) -> DecoderConfig:
    defaults: DecoderConfig = {
        "all_patch_size": (2,),
        "all_f_patch_size": (1,),
        "in_channels": 16,
        "dim": 3840,
        "n_layers": 30,
        "n_refiner_layers": 2,
        "n_heads": 30,
        "n_kv_heads": 30,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "cap_feat_dim": 4096,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "axes_dims": (32, 48, 48),
        "axes_lens": (32768, 1024, 1024),
    }
    # HF metadata and the original unused siglip_feat_dim are not architecture.
    unknown = set(cfg) - defaults.keys() - {"siglip_feat_dim"}
    unknown = {key for key in unknown if not key.startswith("_")}
    if unknown:
        raise ValueError(f"Unsupported decoder config keys: {sorted(unknown)}")
    else:
        pass
    defaults.update({key: cfg[key] for key in defaults.keys() & cfg.keys()})
    defaults["all_patch_size"] = tuple(defaults["all_patch_size"])
    defaults["all_f_patch_size"] = tuple(defaults["all_f_patch_size"])
    defaults["axes_dims"] = tuple(defaults["axes_dims"])
    defaults["axes_lens"] = tuple(defaults["axes_lens"])
    return defaults


def semantic_checkpoint(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    seen: set[str] = set()
    for name, value in weights:
        if name.startswith("semantic_embedder."):
            name = "cap_embedder." + name.removeprefix("semantic_embedder.")
        else:
            pass
        if name in seen:
            raise ValueError(f"Duplicate decoder checkpoint parameter: {name}")
        else:
            pass
        seen.add(name)
        yield name, value


class SequenceParallelBlocks(nn.Module):
    """Shard a contiguous global sequence across native transformer blocks."""

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        adaln_input: torch.Tensor | None = None,
        rope_cos_sin_cache: torch.Tensor | None = None,
        rope_positions: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        attn_mask_meta: dict[str, torch.Tensor | int] | None = None,
        **kwargs: int,
    ) -> torch.Tensor:
        from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
            build_shard_plan,
            gather_seq,
            shard_like,
        )

        shard = build_shard_plan(x.shape[1])
        if shard.num_pad:
            raise ValueError("SP degree must divide the padded decoder sequence length")
        else:
            pass
        assert attn_mask is None and attn_mask_meta is None
        freqs_cis = tuple(shard_like(freq, shard, dim=-2) for freq in freqs_cis)
        if rope_positions is not None:
            rope_positions = shard_like(
                rope_positions.view(x.shape[0], x.shape[1]), shard
            ).reshape(-1)
        elif rope_cos_sin_cache is not None:
            rope_cos_sin_cache = shard_like(rope_cos_sin_cache, shard, dim=0)
        else:
            pass
        x = shard_like(x, shard).contiguous()
        # A singleton batch can retain its pre-shard stride despite contiguous().
        # Canonical strides keep the native fused normalization path enabled.
        x = x.view(-1, x.shape[-1]).view(x.shape)
        # The global image/caption sequence is already sharded together.
        kwargs["num_replicated_suffix"] = 0
        for layer in self.layers:
            x = layer(
                x,
                freqs_cis,
                adaln_input,
                rope_cos_sin_cache=rope_cos_sin_cache,
                rope_positions=rope_positions,
                **kwargs,
            )
        return gather_seq(x, shard.orig_len)


class ZImageTransformer2DModelWrapper(nn.Module):
    """Load semantic checkpoints and return positive velocity for t in [0, 1].

    Inputs are lists of [C, F, H, W] latents and [L, D] semantic features.
    """

    def __init__(
        self,
        decoder_dir: str | Path,
        cfg: DecoderConfig,
        device: torch.device | str,
        dtype: torch.dtype,
        *,
        backend: Literal["diffusers", "sglang"] = "diffusers",
        runtime: DecoderRuntimeHandle | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.cfg = decoder_config(cfg)
        self.native_cache: (
            tuple[
                tuple[
                    tuple[torch.Size, ...],
                    tuple[torch.Size, ...],
                    torch.device,
                    torch.dtype,
                    int,
                    int,
                ],
                tuple[torch.Tensor, torch.Tensor],
            ]
            | None
        ) = None
        if backend == "sglang":
            self.runtime = runtime
            self.model = self.load_sglang_model(decoder_dir, self.runtime.device, dtype)
            return
        else:
            pass
        from diffusers.models.transformers.transformer_z_image import (
            ZImageTransformer2DModel,
        )

        with torch.device("meta"):
            model = ZImageTransformer2DModel(**self.cfg)
        checkpoint = str(Path(decoder_dir) / "model.safetensors")
        state = dict(semantic_checkpoint(load_file(checkpoint, device="cpu").items()))
        model.load_state_dict(state, strict=True, assign=True)
        self.model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    def load_sglang_model(
        self, decoder_dir: str | Path, device: torch.device, dtype: torch.dtype
    ) -> nn.Module:
        from sglang.multimodal_gen.configs.models.dits.zimage import (
            ZImageArchConfig,
            ZImageDitConfig,
        )
        from sglang.multimodal_gen.runtime.layers.attention.selector import (
            component_attn_backend_context_manager,
        )
        from sglang.multimodal_gen.runtime.loader.fsdp_load import (
            load_model_from_full_model_state_dict,
        )
        from sglang.multimodal_gen.runtime.loader.utils import (
            get_param_names_mapping,
            set_default_torch_dtype,
        )
        from sglang.multimodal_gen.runtime.loader.weight_utils import (
            safetensors_weights_iterator,
        )
        from sglang.multimodal_gen.runtime.models.dits.zimage import (
            ZImageTransformer2DModel,
        )
        from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

        if (
            self.cfg["n_kv_heads"] != self.cfg["n_heads"]
            or self.cfg["n_heads"] % self.runtime.ulysses_degree
        ):
            raise ValueError(
                "Native decoder requires equal Q/KV heads divisible by Ulysses degree"
            )
        else:
            pass
        if self.cfg["all_patch_size"] != (2,) or self.cfg["all_f_patch_size"] != (1,):
            raise ValueError(
                "Native spatial decoder supports only patch_size=2, f_patch_size=1"
            )
        else:
            pass
        try:
            attention = AttentionBackendEnum[self.runtime.attention_backend.upper()]
        except KeyError as exc:
            raise ValueError(
                f"Unknown decoder attention backend: {self.runtime.attention_backend}"
            ) from exc
        arch = dict(self.cfg)
        arch["num_layers"] = arch.pop("n_layers")
        arch["num_attention_heads"] = arch.pop("n_heads")
        config = ZImageDitConfig(arch_config=ZImageArchConfig(**arch))
        with (
            component_attn_backend_context_manager(
                attention,
                component_name="llada2_uni_decoder",
                require_backend_selection=True,
            ),
            set_default_torch_dtype(dtype),
            torch.device("meta"),
        ):
            model = ZImageTransformer2DModel(config=config, hf_config=self.cfg)
        load_model_from_full_model_state_dict(
            model=model,
            full_sd_iterator=semantic_checkpoint(
                safetensors_weights_iterator(
                    [str(Path(decoder_dir) / "model.safetensors")]
                )
            ),
            checkpoint_load_device=device,
            param_dtype=dtype,
            strict=True,
            cpu_offload=False,
            param_names_mapping=get_param_names_mapping(model.param_names_mapping),
        )
        if self.runtime.sp_size > 1:
            model.noise_refiner = nn.ModuleList(
                [SequenceParallelBlocks(model.noise_refiner)]
            )
            model.layers = nn.ModuleList([SequenceParallelBlocks(model.layers)])
        else:
            pass
        return model.eval().requires_grad_(False)

    def native_forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ) -> list[torch.Tensor]:
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        key = (
            tuple(image.shape for image in x),
            tuple(cap.shape for cap in cap_feats),
            x[0].device,
            x[0].dtype,
            patch_size,
            f_patch_size,
        )
        if self.native_cache is None or self.native_cache[0] != key:
            with self.runtime.preparation("native input preparation"):
                self.runtime.validate()
                if any(image.shape != x[0].shape for image in x) or any(
                    cap.shape != cap_feats[0].shape for cap in cap_feats
                ):
                    raise ValueError(
                        "Native decoder requires a uniform latent/semantic batch"
                    )
                else:
                    pass
                if x[0].shape[1] != 1:
                    raise ValueError("Native decoder supports one image frame")
                else:
                    pass
                freqs_cis = self.model._build_single_sample_freqs_cis(  # noqa: leading-underscore  # SGLang API
                    x[0], cap_feats[0], patch_size, f_patch_size
                )
            self.native_cache = (key, freqs_cis)
        else:
            pass
        _, freqs_cis = self.native_cache
        full = torch.stack(x)
        with set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=None
        ):
            prediction = self.model(
                hidden_states=full,
                encoder_hidden_states=cap_feats,
                timestep=1000.0 - t * self.cfg["t_scale"],
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                freqs_cis=freqs_cis,
            )
        if not isinstance(prediction, torch.Tensor) or prediction.shape != full.shape:
            raise RuntimeError(
                "Native decoder must return the full [B, C, F, H, W] shape"
            )
        else:
            pass
        # Native forward returns negative velocity.
        return list((-prediction).unbind(0))

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor | float,
        cap_feats: list[torch.Tensor],
        return_dict: bool = True,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> tuple[list[torch.Tensor]] | Transformer2DModelOutput:
        t = torch.as_tensor(t, dtype=torch.float32, device=x[0].device)
        t = t.reshape(-1).expand(len(x))
        if self.backend == "sglang":
            outputs = self.native_forward(x, t, cap_feats, patch_size, f_patch_size)
            if not return_dict:
                return (outputs,)
            else:
                pass
            return Transformer2DModelOutput(sample=outputs)
        else:
            pass
        return self.model(
            x=x,
            t=t,
            cap_feats=cap_feats,
            return_dict=return_dict,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )
