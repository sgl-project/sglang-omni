# SPDX-License-Identifier: Apache-2.0
"""MOSS-Audio-Tokenizer runtime."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.moss_tts.attention import (
    AUTO_ATTENTION_BACKEND,
    PACKED_FLASH_ATTENTION_BACKEND,
    SDPA_ATTENTION_BACKEND,
    AttentionBackendResolution,
    MossAudioTokenizerAttention,
    MossPackedRopeCache,
    PositionIdsCache,
    merge_attention_backend_resolutions,
    pack_padded_sequence,
    pack_padded_sequence_from_host_lengths,
    pack_unpadded_sequence,
    unpack_packed_sequence,
    unpack_packed_sequence_from_indices,
    unpack_unpadded_sequence,
    validate_attention_backend,
)
from sglang_omni.models.weight_loader import (
    load_module,
    load_weights_by_prefix,
    resolve_model_path,
)

logger = logging.getLogger(__name__)

DEFAULT_MOSS_TTS_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
_LOUDNESS_TARGET_DBFS = -20.0
_LOUDNESS_GAIN_MIN_DB = -3.0
_LOUDNESS_GAIN_MAX_DB = 3.0
_HF_FLASH_ATTENTION_IMPLEMENTATION = "flash_attention_2"
_SUPPORTED_ATTENTION_IMPLEMENTATIONS = (
    None,
    _HF_FLASH_ATTENTION_IMPLEMENTATION,
    SDPA_ATTENTION_BACKEND,
)


def resolve_moss_audio_attention_backend(
    attention_backend: str,
    attention_implementation: str | None,
) -> str:
    validate_attention_backend(attention_backend)
    if attention_implementation not in _SUPPORTED_ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            "attention_implementation must be None, 'flash_attention_2', "
            f"or 'sdpa'; got {attention_implementation!r}"
        )
    if attention_backend != AUTO_ATTENTION_BACKEND:
        return attention_backend
    if attention_implementation == SDPA_ATTENTION_BACKEND:
        return SDPA_ATTENTION_BACKEND
    return AUTO_ATTENTION_BACKEND


# Note (Zhang Yiyang): Prefer the runtime model value, then the canonical config
# field, and finally the legacy config alias for checkpoint compatibility.
def resolve_moss_audio_sample_rate(model: Any, config: Any) -> int:
    for value in (
        getattr(model, "sampling_rate", None),
        getattr(config, "sampling_rate", None),
        getattr(config, "sample_rate", None),
    ):
        if value is not None:
            return int(value)
    raise ValueError(
        "MOSS-Audio-Tokenizer model/config lacks sampling_rate or sample_rate"
    )


def _attention_backend_label(resolution: AttentionBackendResolution) -> str:
    if resolution.fallback_reason is None:
        return resolution.backend
    return f"{resolution.backend} (fallback: {resolution.fallback_reason})"


class _MossAudioTokenizerV1FeedForward(nn.Module):
    """Expose MOSS-Audio-Tokenizer v1 Linear-GELU-Linear weights."""

    def __init__(
        self,
        linear1: nn.Module,
        linear2: nn.Module,
        activation: Any,
    ) -> None:
        super().__init__()
        self.linear1 = linear1
        self.linear2 = linear2
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


def _feed_forward(module: nn.Module) -> nn.Module:
    ffn = getattr(module, "ffn", None)
    if ffn is not None:
        return ffn
    if (
        getattr(module, "linear1", None) is None
        or getattr(module, "linear2", None) is None
    ):
        raise ValueError("MOSS-Audio-Tokenizer transformer layer has no supported FFN")
    return _MossAudioTokenizerV1FeedForward(
        module.linear1,
        module.linear2,
        module.activation,
    )


class MossAudioTokenizerTransformerLayer(nn.Module):
    """One shared MOSS-Audio-Tokenizer transformer layer."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        context: int | None = None,
        rope: nn.Module | None = None,
        moss_audio_tokenizer_v1_weights: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
        packed_rope_cache: MossPackedRopeCache | None = None,
        source_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if source_module is not None:
            if config is not None:
                raise ValueError(
                    "MOSS-Audio-Tokenizer transformer layer accepts either "
                    "config or source_module, not both"
                )
            norm1 = source_module.norm1
            selected_attention_backend = resolve_moss_audio_attention_backend(
                attention_backend,
                getattr(source_module.self_attn, "attention_implementation", None),
            )
            self_attn = MossAudioTokenizerAttention.from_module(
                source_module.self_attn,
                attention_backend=selected_attention_backend,
                packed_rope_cache=packed_rope_cache,
            )
            layer_scale_1 = source_module.layer_scale_1
            norm2 = source_module.norm2
            ffn = _feed_forward(source_module)
            layer_scale_2 = source_module.layer_scale_2
        elif config is not None:
            d_model = int(config["d_model"])
            num_heads = int(config["num_heads"])
            dim_feedforward = int(config.get("dim_feedforward", 2048))
            causal = bool(config.get("causal", False))
            norm = str(config.get("norm", "layer_norm"))
            layer_scale = config.get("layer_scale")
            gating = str(config.get("gating", "none"))
            if gating != "none":
                raise ValueError(
                    "repository-local MOSS encoder currently supports gating='none'; "
                    f"got {gating!r}"
                )
            if moss_audio_tokenizer_v1_weights:
                ffn = _MossAudioTokenizerV1FeedForward(
                    nn.Linear(
                        d_model,
                        dim_feedforward,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    ),
                    nn.Linear(
                        dim_feedforward,
                        d_model,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    ),
                    nn.GELU(),
                )
            else:
                ffn = nn.Sequential(
                    nn.Linear(
                        d_model,
                        dim_feedforward,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        dim_feedforward,
                        d_model,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    ),
                )
            if layer_scale is None:
                layer_scale_1 = nn.Identity()
                layer_scale_2 = nn.Identity()
            else:
                layer_scale_1 = _LayerScale(
                    d_model,
                    init=float(layer_scale),
                    device=device,
                    dtype=dtype,
                )
                layer_scale_2 = _LayerScale(
                    d_model,
                    init=float(layer_scale),
                    device=device,
                    dtype=dtype,
                )
            selected_attention_backend = resolve_moss_audio_attention_backend(
                attention_backend,
                (
                    None
                    if moss_audio_tokenizer_v1_weights
                    else config.get(
                        "attention_implementation",
                        _HF_FLASH_ATTENTION_IMPLEMENTATION,
                    )
                ),
            )
            self_attn = MossAudioTokenizerAttention(
                in_proj=nn.Linear(
                    d_model,
                    3 * d_model,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                out_proj=nn.Linear(
                    d_model,
                    d_model,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                embed_dim=d_model,
                num_heads=num_heads,
                causal=causal,
                context=context,
                rope=rope,
                attention_backend=selected_attention_backend,
                packed_rope_cache=packed_rope_cache,
            )
            norm1 = _create_norm(norm, d_model, device=device, dtype=dtype)
            norm2 = _create_norm(norm, d_model, device=device, dtype=dtype)
        else:
            raise ValueError(
                "MOSS-Audio-Tokenizer transformer layer requires config or "
                "source_module"
            )
        self.norm1 = norm1
        self.self_attn = self_attn
        self.layer_scale_1 = layer_scale_1
        self.norm2 = norm2
        self.ffn = ffn
        self.layer_scale_2 = layer_scale_2
        assert callable(self.ffn), "MOSS-Audio-Tokenizer layer requires an FFN"

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
        packed_rope_cache: MossPackedRopeCache | None = None,
    ) -> MossAudioTokenizerTransformerLayer:
        return cls(
            source_module=module,
            attention_backend=attention_backend,
            packed_rope_cache=packed_rope_cache,
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = residual.to(x) + self.layer_scale_1(self.self_attn(x, **kwargs))
        residual = x
        x = self.norm2(x)
        x = residual.to(x) + self.layer_scale_2(self.ffn(x))
        return x


class MossAudioTokenizerTransformer(nn.Module):
    """Shared MOSS-Audio-Tokenizer transformer body."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        context: int | None = None,
        moss_audio_tokenizer_v1_weights: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
        source_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if source_module is not None:
            if config is not None:
                raise ValueError(
                    "MOSS-Audio-Tokenizer transformer accepts either config or "
                    "source_module, not both"
                )
            max_period = float(source_module.max_period)
            packed_rope_cache = MossPackedRopeCache(max_period=max_period)
            layers = [
                MossAudioTokenizerTransformerLayer.from_module(
                    layer,
                    attention_backend=attention_backend,
                    packed_rope_cache=packed_rope_cache,
                )
                for layer in source_module.layers
            ]
            positional_embedding = source_module.positional_embedding
            positional_scale = float(source_module.positional_scale)
        elif config is not None:
            positional_embedding = str(config.get("positional_embedding", "sin"))
            max_period = float(config.get("max_period", 10_000))
            positional_scale = float(config.get("positional_scale", 1.0))
            rope = (
                _RotaryEmbedding(max_period)
                if positional_embedding in {"rope", "sin_rope"}
                else None
            )
            packed_rope_cache = MossPackedRopeCache(max_period=max_period)
            layers = [
                MossAudioTokenizerTransformerLayer(
                    config,
                    context=context,
                    rope=rope,
                    moss_audio_tokenizer_v1_weights=moss_audio_tokenizer_v1_weights,
                    device=device,
                    dtype=dtype,
                    attention_backend=attention_backend,
                    packed_rope_cache=packed_rope_cache,
                )
                for _ in range(int(config["num_layers"]))
            ]
        else:
            raise ValueError(
                "MOSS-Audio-Tokenizer transformer requires config or source_module"
            )
        self.layers = nn.ModuleList(layers)
        self.positional_embedding = positional_embedding
        self.positional_scale = float(positional_scale)
        self.max_period = float(max_period)

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> MossAudioTokenizerTransformer:
        return cls(
            source_module=module,
            attention_backend=attention_backend,
        )

    def resolve_attention_backend(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> AttentionBackendResolution:
        return merge_attention_backend_resolutions(
            [
                layer.self_attn.resolve_attention_backend(device, dtype)
                for layer in self.layers
            ]
        )

    def supports_packed_attention(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        return bool(self.layers) and all(
            layer.self_attn.supports_packed_attention(device, dtype)
            for layer in self.layers
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.positional_embedding in {"sin", "sin_rope"}:
            if x.dim() == 3:
                positions = torch.arange(x.shape[1], device=x.device).view(1, -1)
            else:
                positions = kwargs.get("position_ids")
                if positions is None:
                    raise ValueError(
                        "packed transformer inputs require position_ids for "
                        "sinusoidal embeddings"
                    )
            pos_emb = create_sin_embedding(
                positions,
                x.shape[-1],
                max_period=self.max_period,
                dtype=x.dtype,
            )
            x = x + self.positional_scale * pos_emb
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


class MossAudioTokenizerProjectedTransformer(nn.Module):
    """Shared projected Transformer stage with the MOSS-Audio-Tokenizer tensor layout."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        context: int | None = None,
        moss_audio_tokenizer_v1_weights: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
        source_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if source_module is not None:
            if config is not None:
                raise ValueError(
                    "MOSS-Audio-Tokenizer projected Transformer accepts either "
                    "config or source_module, not both"
                )
            input_proj = source_module.input_proj
            transformer = MossAudioTokenizerTransformer.from_module(
                source_module.transformer,
                attention_backend=attention_backend,
            )
            output_proj = source_module.output_proj
        elif config is not None:
            input_dimension = int(config["input_dimension"])
            output_dimension = int(config["output_dimension"])
            d_model = int(config["d_model"])
            input_proj = (
                nn.Linear(
                    input_dimension,
                    d_model,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                if not moss_audio_tokenizer_v1_weights or input_dimension != d_model
                else nn.Identity()
            )
            output_proj = (
                nn.Linear(
                    d_model,
                    output_dimension,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                if not moss_audio_tokenizer_v1_weights or d_model != output_dimension
                else nn.Identity()
            )
            transformer = MossAudioTokenizerTransformer(
                config,
                context=context,
                moss_audio_tokenizer_v1_weights=moss_audio_tokenizer_v1_weights,
                device=device,
                dtype=dtype,
                attention_backend=attention_backend,
            )
        else:
            raise ValueError(
                "MOSS-Audio-Tokenizer projected Transformer requires config or "
                "source_module"
            )
        self.module_type = "Transformer"
        self.downsample_ratio = 1
        self.input_proj = input_proj
        self.transformer = transformer
        self.output_proj = output_proj
        self._position_ids_cache = PositionIdsCache()

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> MossAudioTokenizerProjectedTransformer:
        return cls(
            source_module=module,
            attention_backend=attention_backend,
        )

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        input_lengths_cpu: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x.transpose(1, 2))
        backend = self.transformer.resolve_attention_backend(
            x.device,
            MossAudioTokenizerAttention.get_backend_dtype(x),
        ).backend
        if backend == PACKED_FLASH_ATTENTION_BACKEND:
            batch_size, max_seqlen, _ = x.shape
            if input_lengths_cpu is not None:
                if len(input_lengths_cpu) != batch_size:
                    raise ValueError(
                        "input_lengths_cpu must match the decoder batch size"
                    )
                max_valid_seqlen = max(map(int, input_lengths_cpu), default=0)
            else:
                max_valid_seqlen = int(input_lengths.max().item()) if max_seqlen else 0
            if max_valid_seqlen == 0:
                x = x.new_zeros(x.shape)
            else:
                is_unpadded_single = batch_size == 1 and max_valid_seqlen == max_seqlen
                if is_unpadded_single:
                    packed_x, cu_seqlens, position_ids = pack_unpadded_sequence(
                        x,
                        self._position_ids_cache,
                    )
                    valid_mask = None
                    flat_indices = None
                elif input_lengths_cpu is not None:
                    packed_x, flat_indices, cu_seqlens, position_ids = (
                        pack_padded_sequence_from_host_lengths(
                            x,
                            input_lengths,
                            input_lengths_cpu,
                        )
                    )
                    valid_mask = None
                else:
                    packed_x, valid_mask, cu_seqlens, position_ids = (
                        pack_padded_sequence(x, input_lengths)
                    )
                    flat_indices = None
                first_attention = self.transformer.layers[0].self_attn
                local_flash_plan = first_attention.build_local_causal_flash_plan(
                    cu_seqlens,
                    max_seqlen=max_valid_seqlen,
                    sequence_lengths=input_lengths_cpu,
                )
                packed_x = self.transformer(
                    packed_x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_valid_seqlen,
                    position_ids=position_ids,
                    input_lengths=input_lengths,
                    local_flash_plan=local_flash_plan,
                    **kwargs,
                )
                if is_unpadded_single:
                    x = unpack_unpadded_sequence(packed_x)
                elif flat_indices is not None:
                    x = unpack_packed_sequence_from_indices(
                        packed_x,
                        flat_indices,
                        batch_size,
                        max_seqlen,
                    )
                else:
                    assert valid_mask is not None
                    x = unpack_packed_sequence(
                        packed_x,
                        valid_mask,
                        batch_size,
                        max_seqlen,
                    )
        else:
            x = self.transformer(x, input_lengths=input_lengths, **kwargs)
        return self.output_proj(x).transpose(1, 2), input_lengths

    def supports_packed_attention(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        return self.transformer.supports_packed_attention(device, dtype)

    def resolve_attention_backend(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None,
    ) -> AttentionBackendResolution:
        return self.transformer.resolve_attention_backend(torch.device(device), dtype)


def _update_decoder_cpu_lengths(
    stage: nn.Module,
    input_lengths: Sequence[int],
) -> list[int]:
    if isinstance(stage, MossAudioTokenizerProjectedTransformer):
        return list(map(int, input_lengths))
    patch_size = int(getattr(stage, "patch_size", 0))
    if patch_size <= 0:
        raise ValueError(
            "MOSS-Audio-Tokenizer patched pretransform requires patch_size > 0"
        )
    if bool(getattr(stage, "is_downsample", False)):
        return [int(length) // patch_size for length in input_lengths]
    return [int(length) * patch_size for length in input_lengths]


# note (Zhang Yiyang): Keep one non-streaming decoder stage container for both
# repository-owned and source decoder modules.


class MossAudioTokenizerVocoderDecoder(nn.ModuleList):
    """MOSS-Audio-Tokenizer vocoder decoder stage container."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        moss_audio_tokenizer_v1_weights: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
        source_decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if source_decoder is not None:
            if config is not None:
                raise ValueError(
                    "MOSS-Audio-Tokenizer vocoder decoder accepts either config "
                    "or source_decoder, not both"
                )
            stages = []
            for stage in source_decoder:
                module_type = getattr(stage, "module_type", None)
                if module_type == "Transformer":
                    stage = MossAudioTokenizerProjectedTransformer.from_module(
                        stage,
                        attention_backend=attention_backend,
                    )
                elif module_type != "PatchedPretransform":
                    raise ValueError(
                        f"unsupported MOSS-Audio-Tokenizer vocoder decoder stage "
                        f"{stage.__class__.__name__} with module_type={module_type!r}"
                    )
                stages.append(stage)
        elif config is not None:
            sampling_rate = config.get("sampling_rate") or config.get("sample_rate")
            if sampling_rate is None:
                raise ValueError("MOSS audio-tokenizer config lacks sampling_rate")
            downsample_rate = int(config["downsample_rate"])
            default_context_duration = float(
                config.get("causal_transformer_context_duration", 10.0)
            )
            frame_rate = float(sampling_rate) / downsample_rate
            stages = []
            for stage_config_raw in config["decoder_kwargs"]:
                stage_config = dict(stage_config_raw)
                module_type = stage_config["module_type"]
                if module_type == "PatchedPretransform":
                    stage = _PatchedPretransform(
                        int(stage_config["patch_size"]),
                        is_downsample=False,
                    )
                elif module_type == "Transformer":
                    stage_config.setdefault(
                        "attention_implementation",
                        config.get(
                            "attention_implementation",
                            _HF_FLASH_ATTENTION_IMPLEMENTATION,
                        ),
                    )
                    context_duration = float(
                        stage_config.pop("context_duration", default_context_duration)
                    )
                    stage = MossAudioTokenizerProjectedTransformer(
                        stage_config,
                        context=int(round(frame_rate * context_duration)),
                        moss_audio_tokenizer_v1_weights=(
                            moss_audio_tokenizer_v1_weights
                        ),
                        device=device,
                        dtype=dtype,
                        attention_backend=attention_backend,
                    )
                else:
                    raise ValueError(
                        "unsupported MOSS-Audio-Tokenizer decoder stage: "
                        f"{module_type!r}"
                    )
                stages.append(stage)
                if isinstance(stage, _PatchedPretransform):
                    frame_rate *= stage.patch_size
        else:
            raise ValueError(
                "MOSS-Audio-Tokenizer vocoder decoder requires config or "
                "source_decoder"
            )
        stages = list(stages)
        if not stages:
            raise ValueError(
                "MOSS-Audio-Tokenizer vocoder decoder must be a non-empty stage list"
            )
        self.attention_backend = validate_attention_backend(attention_backend)
        self.extend(stages)

    @classmethod
    def from_module(
        cls,
        decoder: nn.Module,
        *,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> MossAudioTokenizerVocoderDecoder:
        if isinstance(decoder, cls):
            return decoder
        return cls(
            source_decoder=decoder,
            attention_backend=attention_backend,
        )

    def supports_packed_attention(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        device = torch.device(device)
        transformer_stages = tuple(
            stage
            for stage in self
            if isinstance(stage, MossAudioTokenizerProjectedTransformer)
        )
        return bool(transformer_stages) and all(
            stage.supports_packed_attention(device, dtype)
            for stage in transformer_stages
        )

    def resolve_attention_backend(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None,
    ) -> AttentionBackendResolution:
        device = torch.device(device)
        return merge_attention_backend_resolutions(
            [
                stage.resolve_attention_backend(device, dtype)
                for stage in self
                if isinstance(stage, MossAudioTokenizerProjectedTransformer)
            ]
        )

    def output_lengths(self, input_lengths: Sequence[int]) -> list[int]:
        output_lengths = list(map(int, input_lengths))
        for stage in self:
            output_lengths = _update_decoder_cpu_lengths(stage, output_lengths)
        return output_lengths

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        input_lengths_cpu: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cpu_lengths = (
            None if input_lengths_cpu is None else list(map(int, input_lengths_cpu))
        )
        for stage in self:
            if isinstance(stage, MossAudioTokenizerProjectedTransformer):
                x, input_lengths = stage(
                    x,
                    input_lengths,
                    input_lengths_cpu=cpu_lengths,
                )
            else:
                x, input_lengths = stage(x, input_lengths)
            if cpu_lengths is not None:
                cpu_lengths = _update_decoder_cpu_lengths(stage, cpu_lengths)
        return x, input_lengths


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    *,
    max_period: float = 10_000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create the sinusoidal embedding expected by the shared stage wrapper."""

    if dim % 2:
        raise ValueError(f"sinusoidal embedding requires an even dim, got {dim}")
    half_dim = dim // 2
    if half_dim <= 1:
        raise ValueError(f"sinusoidal embedding requires dim >= 4, got {dim}")
    positions = positions.to(dtype).unsqueeze(-1)
    dimensions = torch.arange(half_dim, device=positions.device, dtype=dtype)
    period = torch.full(
        (),
        float(max_period),
        device=positions.device,
        dtype=dtype,
    )
    phase = positions / (period ** (dimensions / (half_dim - 1)))
    return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


_AUDIO_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def resolve_moss_audio_dtype(
    dtype: str | torch.dtype | None,
    *,
    name: str,
    allow_none: bool,
) -> torch.dtype | None:
    if dtype is None:
        if allow_none:
            return None
    elif isinstance(dtype, str):
        resolved = _AUDIO_DTYPES.get(dtype.lower())
        if resolved is not None:
            return resolved
    elif isinstance(dtype, torch.dtype) and dtype in _AUDIO_DTYPES.values():
        return dtype
    allowed = "float32, bfloat16"
    if allow_none:
        allowed += ", or null"
    raise ValueError(f"{name} must be {allowed}; got {dtype!r}")


def _validate_audio_dtypes(
    *,
    component_dtype: torch.dtype,
    component_name: str,
    compute_dtype: torch.dtype | None,
) -> None:
    if component_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"{component_name} must be torch.float32 or torch.bfloat16; "
            f"got {component_dtype!r}"
        )
    if compute_dtype not in (None, torch.float32, torch.bfloat16):
        raise ValueError(
            "compute_dtype must be torch.float32, torch.bfloat16, or None; "
            f"got {compute_dtype!r}"
        )


@dataclass
class MossAudioTokenizerEncoderOutput:
    """Output contract shared with the upstream audio-tokenizer encoder."""

    audio_codes: torch.Tensor
    audio_codes_lengths: torch.Tensor
    encoder_hidden_states: torch.Tensor


class _LayerScale(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        init: float,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((channels,), init, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class _RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.compute_dtype = compute_dtype
        self.alpha = nn.Parameter(torch.ones((1, 1, dim), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)
        variance = self.eps + torch.mean(x**2, dim=-1, keepdim=True)
        alpha = self.alpha.to(variance)
        if x.dim() == 2:
            alpha = alpha.view(1, -1)
        return (x * (alpha * torch.rsqrt(variance))).to(output_dtype)


def _create_norm(
    norm: str,
    dim: int,
    *,
    device: str | torch.device | None,
    dtype: torch.dtype | None,
) -> nn.Module:
    if norm == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
    if norm == "rms_norm":
        return _RMSNorm(dim, eps=1e-5, device=device, dtype=dtype)
    if norm == "rms_norm_f32":
        return _RMSNorm(
            dim,
            eps=1e-8,
            device=device,
            # note (Zhang Yiyang): This norm explicitly computes in FP32. Keep
            # its scale in FP32 as well so the forward path does not recast the
            # parameter every call.
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )
    raise ValueError(f"unsupported MOSS audio-tokenizer norm: {norm!r}")


def _restore_fp32_compute_parameters(module: nn.Module) -> None:
    """Keep parameters of explicitly FP32 compute modules in FP32."""
    for submodule in module.modules():
        if not isinstance(submodule, _RMSNorm):
            continue
        if submodule.compute_dtype is not torch.float32:
            continue
        with torch.no_grad():
            submodule.alpha.data = submodule.alpha.data.to(dtype=torch.float32)


class _RotaryEmbedding(nn.Module):
    def __init__(self, max_period: float) -> None:
        super().__init__()
        self.max_period = float(max_period)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, sequence_length, head_dim = q.shape
        frequencies = torch.exp(
            torch.arange(
                head_dim // 2,
                device=q.device,
                dtype=torch.float32,
            )
            * (-math.log(self.max_period) * 2 / head_dim)
        )
        positions = offset.float().view(batch_size, 1, 1, 1) + torch.arange(
            sequence_length,
            device=q.device,
            dtype=torch.float32,
        ).view(1, 1, sequence_length, 1)
        phase = positions * frequencies.view(1, 1, 1, -1)
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            pairs = x.float().view(*shape[:-1], head_dim // 2, 2)
            real, imag = pairs[..., 0], pairs[..., 1]
            return (
                torch.stack(
                    (real * cos - imag * sin, real * sin + imag * cos),
                    dim=-1,
                )
                .to(x.dtype)
                .view(shape)
            )

        return rotate(q), rotate(k)


class _PatchedPretransform(nn.Module):
    def __init__(self, patch_size: int, *, is_downsample: bool) -> None:
        super().__init__()
        self.module_type = "PatchedPretransform"
        self.patch_size = int(patch_size)
        self.downsample_ratio = self.patch_size
        self.is_downsample = bool(is_downsample)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, length = x.shape
        if self.is_downsample:
            x = (
                x.reshape(batch_size, channels, -1, self.patch_size)
                .permute(0, 1, 3, 2)
                .reshape(batch_size, channels * self.patch_size, -1)
            )
            return x, input_lengths // self.patch_size
        if channels % self.patch_size:
            raise ValueError(
                "MOSS vocoder patch stage requires channels divisible by "
                f"patch_size, got channels={channels}, patch_size={self.patch_size}"
            )
        output_channels = channels // self.patch_size
        x = (
            x.reshape(batch_size, output_channels, self.patch_size, length)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, output_channels, length * self.patch_size)
        )
        return x, input_lengths * self.patch_size


def _weight_normalized_conv1d(*args: Any, **kwargs: Any) -> nn.Module:
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


class _LFQ(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        device: str | torch.device | None,
    ) -> None:
        super().__init__()
        self.in_proj = (
            _weight_normalized_conv1d(
                input_dim,
                codebook_dim,
                kernel_size=1,
                device=device,
                dtype=torch.float32,
            )
            if input_dim != codebook_dim
            else nn.Identity()
        )
        self.out_proj = (
            _weight_normalized_conv1d(
                codebook_dim,
                input_dim,
                kernel_size=1,
                device=device,
                dtype=torch.float32,
            )
            if input_dim != codebook_dim
            else nn.Identity()
        )
        self.codebook = nn.Embedding(
            codebook_size,
            codebook_dim,
            device=device,
            dtype=torch.float32,
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.in_proj(z.float()).float()
        flat = F.normalize(encoded.transpose(1, 2).reshape(-1, encoded.shape[1]))
        codebook = F.normalize(self.codebook.weight.float())
        distance = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-distance).max(1)[1].reshape(z.shape[0], -1)
        quantized = F.embedding(indices, self.codebook.weight).transpose(1, 2)
        quantized = encoded + (quantized - encoded).detach()
        return self.out_proj(quantized.float()).float(), indices

    def decode_code(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = F.embedding(indices, self.codebook.weight).transpose(1, 2)
        return self.out_proj(quantized.float()).float()


class _ResidualLFQ(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        device: str | torch.device | None,
    ) -> None:
        super().__init__()
        input_dim = int(config.get("input_dim", 1024))
        rvq_dim = int(config.get("rvq_dim") or input_dim)
        output_dim = int(config.get("output_dim") or input_dim)
        self.rvq_dim = rvq_dim
        self.num_quantizers = int(config.get("num_quantizers", 32))
        codebook_size = int(config.get("codebook_size", 1024))
        codebook_dim = int(config.get("codebook_dim", 8))
        self.input_proj = (
            _weight_normalized_conv1d(
                input_dim,
                rvq_dim,
                kernel_size=1,
                device=device,
                dtype=torch.float32,
            )
            if input_dim != rvq_dim
            else nn.Identity()
        )
        self.output_proj = (
            _weight_normalized_conv1d(
                rvq_dim,
                output_dim,
                kernel_size=1,
                device=device,
                dtype=torch.float32,
            )
            if rvq_dim != output_dim
            else nn.Identity()
        )
        self.quantizers = nn.ModuleList(
            [
                _LFQ(
                    input_dim=rvq_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    device=device,
                )
                for _ in range(self.num_quantizers)
            ]
        )

    @torch.no_grad()
    def forward(
        self,
        z: torch.Tensor,
        input_lengths: torch.Tensor,
        num_quantizers: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda", enabled=False):
            z = self.input_proj(z.float()).float()
            batch_size, _, max_time = z.shape
            mask = torch.arange(max_time, device=z.device).expand(
                batch_size, max_time
            ) < input_lengths.unsqueeze(1)
            residual = z.clone()
            quantized = torch.zeros_like(z)
            indices = []
            count = (
                self.num_quantizers if num_quantizers is None else int(num_quantizers)
            )
            if not 0 < count <= self.num_quantizers:
                raise ValueError(
                    f"num_quantizers must be in [1, {self.num_quantizers}], got {count}"
                )
            update_mask = mask.unsqueeze(1)
            for quantizer in self.quantizers[:count]:
                current, current_indices = quantizer(residual * update_mask)
                quantized += current * update_mask
                residual -= current * update_mask
                indices.append(current_indices)
            return (
                self.output_proj(quantized.float()).float(),
                torch.stack(indices),
                input_lengths,
            )

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=codes.device.type, enabled=False):
            if codes.ndim != 3:
                raise ValueError(
                    "MOSS quantizer codes must be [N, B, T], got "
                    f"{tuple(codes.shape)}"
                )
            count, batch_size, frames = map(int, codes.shape)
            if not 0 < count <= self.num_quantizers:
                raise ValueError(
                    "MOSS quantizer codebook count must be within "
                    f"[1, {self.num_quantizers}], got {count}"
                )
            decoded = torch.zeros(
                batch_size,
                self.rvq_dim,
                frames,
                device=codes.device,
                dtype=torch.float32,
            )
            for index, quantizer in enumerate(self.quantizers[:count]):
                decoded += quantizer.decode_code(codes[index]).float()
            return self.output_proj(decoded.float()).float()


class MossAudioTokenizerEncoder(nn.Module):
    """Inference-only MOSS codec encoder with repository-owned execution code."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        parameter_device: str | torch.device | None = None,
        compute_dtype: torch.dtype | None = None,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(**config)
        sampling_rate = config.get("sampling_rate") or config.get("sample_rate")
        if sampling_rate is None:
            raise ValueError("MOSS audio-tokenizer config lacks sampling_rate")
        self.sampling_rate = int(sampling_rate)
        self.downsample_rate = int(config["downsample_rate"])
        self.number_channels = int(config.get("number_channels", 1))
        self.enable_channel_interleave = bool(
            config.get("enable_channel_interleave", self.number_channels > 1)
        )
        configured_compute_dtype = str(config.get("compute_dtype") or "bfloat16")
        resolved_compute_dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": None,
            "float32": None,
        }.get(configured_compute_dtype)
        if configured_compute_dtype not in {
            "bf16",
            "bfloat16",
            "fp32",
            "float32",
        }:
            raise ValueError(
                f"unsupported codec compute_dtype: {configured_compute_dtype!r}"
            )
        requested_compute_dtype = (
            resolved_compute_dtype if compute_dtype is None else compute_dtype
        )
        if requested_compute_dtype not in (
            None,
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError(
                "compute_dtype must be torch.float32, torch.bfloat16, or None; "
                f"got {requested_compute_dtype!r}"
            )
        effective_encoder_dtype = (
            torch.float32
            if requested_compute_dtype is None
            or requested_compute_dtype is torch.float32
            else requested_compute_dtype
        )
        self.compute_dtype = (
            None
            if requested_compute_dtype is torch.float32
            else requested_compute_dtype
        )
        # note (Zhang Yiyang): The compute policy is also the materialized dtype
        # for the encoder weights. This avoids relying on autocast to recast
        # FP32 parameters on every request. The quantizer is intentionally kept
        # FP32 below.
        self.encoder_dtype = effective_encoder_dtype
        self.attention_backend = validate_attention_backend(attention_backend)
        self._uses_moss_audio_tokenizer_v1_weights = "number_channels" not in config

        default_context_duration = float(
            config.get("causal_transformer_context_duration", 10.0)
        )
        channel_factor = (
            self.number_channels
            if self.enable_channel_interleave and self.number_channels > 1
            else 1
        )
        frame_rate = float(self.sampling_rate * channel_factor)
        stages: list[nn.Module] = []
        for stage_config_raw in config["encoder_kwargs"]:
            stage_config = dict(stage_config_raw)
            module_type = stage_config["module_type"]
            if module_type == "PatchedPretransform":
                stage = _PatchedPretransform(
                    int(stage_config["patch_size"]),
                    is_downsample=True,
                )
            elif module_type == "Transformer":
                stage_config.setdefault(
                    "attention_implementation",
                    config.get(
                        "attention_implementation",
                        _HF_FLASH_ATTENTION_IMPLEMENTATION,
                    ),
                )
                context_duration = float(
                    stage_config.pop("context_duration", default_context_duration)
                )
                stage = MossAudioTokenizerProjectedTransformer(
                    stage_config,
                    context=int(round(frame_rate * context_duration)),
                    moss_audio_tokenizer_v1_weights=(
                        self._uses_moss_audio_tokenizer_v1_weights
                    ),
                    device=parameter_device,
                    dtype=self.encoder_dtype,
                    attention_backend=self.attention_backend,
                )
            else:
                raise ValueError(f"unsupported MOSS encoder stage: {module_type!r}")
            stages.append(stage)
            frame_rate /= int(getattr(stage, "downsample_ratio", 1))
        self.encoder = nn.ModuleList(stages)

        quantizer_config = dict(config["quantizer_kwargs"])
        quantizer_type = quantizer_config.get(
            "quantizer_type", config.get("quantizer_type", "rlfq")
        )
        if quantizer_type not in {"rlfq", "random_prefix_rlfq"}:
            raise ValueError(
                "repository-local MOSS encoder supports residual LFQ checkpoints; "
                f"got quantizer_type={quantizer_type!r}"
            )
        self.quantizer = _ResidualLFQ(
            quantizer_config,
            device=parameter_device,
        )

    def supports_packed_attention(self) -> bool:
        device = next(self.parameters()).device
        return all(
            not isinstance(stage, MossAudioTokenizerProjectedTransformer)
            or stage.supports_packed_attention(device, self.encoder_dtype)
            for stage in self.encoder
        )

    def resolve_attention_backend(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> AttentionBackendResolution:
        device = torch.device(device)
        return merge_attention_backend_resolutions(
            [
                stage.resolve_attention_backend(
                    device,
                    self.encoder_dtype if dtype is None else dtype,
                )
                for stage in self.encoder
                if isinstance(stage, MossAudioTokenizerProjectedTransformer)
            ]
        )

    def _prepare_waveform_batch(
        self,
        waveforms: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if not waveforms:
            raise ValueError("waveforms must contain at least one item")
        device = waveforms[0].device
        dtype = waveforms[0].dtype
        normalized = []
        lengths_cpu = []
        for index, waveform in enumerate(waveforms):
            if self.number_channels == 1 and waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.dim() != 2 or waveform.shape[0] != self.number_channels:
                raise ValueError(
                    f"waveforms[{index}] must have shape "
                    f"({self.number_channels}, T), got {tuple(waveform.shape)}"
                )
            normalized.append(waveform)
            lengths_cpu.append(int(waveform.shape[-1]))
        max_length = max(lengths_cpu)
        batch = torch.zeros(
            len(normalized),
            self.number_channels,
            max_length,
            device=device,
            dtype=dtype,
        )
        for index, waveform in enumerate(normalized):
            batch[index, :, : waveform.shape[-1]] = waveform
        lengths = torch.tensor(lengths_cpu, device=device, dtype=torch.long)
        return batch, lengths, lengths_cpu

    def _flatten_channels(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor,
        input_lengths_cpu: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        remainder = input_values.shape[-1] % self.downsample_rate
        if remainder:
            input_values = F.pad(
                input_values,
                (0, self.downsample_rate - remainder),
            )
        if self.number_channels > 1 and self.enable_channel_interleave:
            input_values = (
                input_values.transpose(1, 2)
                .contiguous()
                .view(input_values.shape[0], 1, -1)
            )
            input_lengths = input_lengths * self.number_channels
            input_lengths_cpu = [
                length * self.number_channels for length in input_lengths_cpu
            ]
        return input_values, input_lengths, input_lengths_cpu

    @torch.no_grad()
    def batch_encode(
        self,
        waveforms: list[torch.Tensor],
        num_quantizers: int | None = None,
        chunk_duration: float | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        if chunk_duration is not None:
            raise ValueError(
                "repository-local MOSS encoder only supports full non-streaming "
                "batch_encode (chunk_duration=None)"
            )
        hidden, lengths, lengths_cpu = self._prepare_waveform_batch(waveforms)
        hidden, lengths, lengths_cpu = self._flatten_channels(
            hidden,
            lengths,
            lengths_cpu,
        )
        hidden = hidden.to(dtype=self.encoder_dtype)
        for stage in self.encoder:
            if isinstance(stage, MossAudioTokenizerProjectedTransformer):
                hidden, lengths = stage(
                    hidden,
                    lengths,
                    input_lengths_cpu=lengths_cpu,
                )
            else:
                hidden, lengths = stage(hidden, lengths)
                lengths_cpu = [
                    length // int(stage.downsample_ratio) for length in lengths_cpu
                ]
        _, codes, code_lengths = self.quantizer(
            hidden.float(),
            lengths,
            num_quantizers,
        )
        max_valid_length = max(lengths_cpu, default=0)
        return MossAudioTokenizerEncoderOutput(
            audio_codes=codes[:, :, :max_valid_length],
            audio_codes_lengths=code_lengths,
            encoder_hidden_states=hidden[:, :, :max_valid_length].float(),
        )


class MossAudioEncoder:
    """Prepare reference audio and encode it with a shared MOSS encoder."""

    def __init__(self, model: MossAudioTokenizerEncoder, *, device: str) -> None:
        self.model = model
        self.device = str(device)
        config = model.config
        self.sample_rate = resolve_moss_audio_sample_rate(model, config)
        self.number_channels = int(
            getattr(model, "number_channels", getattr(config, "number_channels", 1))
        )

    def encode_paths(
        self,
        paths: list[str | PathLike[str]],
        *,
        num_quantizers: int,
    ) -> list[torch.Tensor]:
        if not paths:
            raise ValueError("paths must contain at least one audio path")
        return self.encode_waveforms(
            self.load_paths(paths),
            num_quantizers=num_quantizers,
        )

    def load_paths(
        self,
        paths: list[str | PathLike[str]],
    ) -> list[tuple[torch.Tensor, int]]:
        import torchaudio

        waveforms = []
        for path in paths:
            waveform, sample_rate = torchaudio.load(path)
            if int(sample_rate) != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform=waveform,
                    orig_freq=int(sample_rate),
                    new_freq=self.sample_rate,
                )
            waveforms.append((waveform, self.sample_rate))
        return waveforms

    def encode_wavs(
        self,
        waveforms: list[torch.Tensor],
        sample_rate: int,
        *,
        num_quantizers: int,
    ) -> list[torch.Tensor]:
        return self.encode_waveforms(
            [(waveform, int(sample_rate)) for waveform in waveforms],
            num_quantizers=num_quantizers,
        )

    def encode_waveforms(
        self,
        waveforms: list[tuple[torch.Tensor, int]],
        *,
        num_quantizers: int,
    ) -> list[torch.Tensor]:
        if not waveforms:
            raise ValueError("waveforms must contain at least one waveform")
        prepared = [
            self._prepare_waveform(waveform, sample_rate)
            for waveform, sample_rate in waveforms
        ]
        with torch.inference_mode():
            encoded = self.model.batch_encode(
                prepared,
                num_quantizers=int(num_quantizers),
            )
        codes = encoded.audio_codes
        lengths = encoded.audio_codes_lengths
        if codes is None or lengths is None:
            raise RuntimeError(
                "MOSS audio encoder returned empty audio_codes/audio_codes_lengths"
            )
        codes = codes.detach().to(device="cpu", dtype=torch.long)
        lengths = lengths.detach().to("cpu")
        return [
            codes[:, index, : int(lengths[index])].transpose(0, 1).contiguous()
            for index in range(int(codes.shape[1]))
        ]

    def _prepare_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(
                "expected waveform with shape [channels, samples], got "
                f"{tuple(waveform.shape)}"
            )
        if self.number_channels == 1:
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(self.number_channels, 1)
            elif waveform.shape[0] > self.number_channels:
                waveform = waveform[: self.number_channels]
            if waveform.shape[0] != self.number_channels:
                raise ValueError(
                    f"expected {self.number_channels} audio channels, "
                    f"got {waveform.shape[0]}"
                )
        if int(sample_rate) != self.sample_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(
                waveform=waveform,
                orig_freq=int(sample_rate),
                new_freq=self.sample_rate,
            )
        waveform = self._loudness_normalize(waveform)
        if self.number_channels == 1:
            waveform = waveform.squeeze(0)
        return waveform.to(device=self.device, dtype=torch.float32)

    @staticmethod
    def _loudness_normalize(waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(torch.float32)
        if waveform.numel() == 0:
            return waveform
        current_dbfs = 10.0 * torch.log10(torch.mean(waveform**2) + 1e-9)
        gain = float(_LOUDNESS_TARGET_DBFS - current_dbfs)
        gain = max(_LOUDNESS_GAIN_MIN_DB, min(gain, _LOUDNESS_GAIN_MAX_DB))
        return waveform * (10.0 ** (gain / 20.0))


def _normalize_moss_audio_tokenizer_v1_transformer_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map MOSS v1 weight names onto the shared Transformer modules."""

    replacements = (
        (".self_attn.in_projs.0.", ".self_attn.in_proj."),
        (".self_attn.out_projs.0.", ".self_attn.out_proj."),
        (".linear1.", ".ffn.linear1."),
        (".linear2.", ".ffn.linear2."),
    )
    normalized = {}
    for name, tensor in state_dict.items():
        normalized_name = name
        for old, new in replacements:
            normalized_name = normalized_name.replace(old, new)
        normalized[normalized_name] = tensor
    return normalized


def _load_moss_audio_config(model_path: str) -> tuple[Path, dict[str, Any]]:
    resolved_path = resolve_model_path(str(model_path))
    config_path = resolved_path / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("model_type") != "moss-audio-tokenizer":
        raise ValueError(
            f"expected model_type='moss-audio-tokenizer' in {config_path}, "
            f"got {config.get('model_type')!r}"
        )
    return resolved_path, config


def _load_moss_audio_component(
    module: nn.Module,
    model_path: Path,
    *,
    prefix: str,
    dtype: torch.dtype,
    device: str | torch.device,
    v1_weights: bool,
) -> nn.Module:
    if v1_weights:
        state_dict = load_weights_by_prefix(str(model_path), prefix=prefix)
        state_dict = _normalize_moss_audio_tokenizer_v1_transformer_state_dict(
            state_dict
        )
        try:
            module.load_state_dict(state_dict, strict=True, assign=True)
        except TypeError:
            module.load_state_dict(state_dict, strict=True)
        module = module.to(device=device, dtype=dtype)
        module.eval()
    else:
        module = load_module(
            module,
            str(model_path),
            prefix=prefix,
            dtype=dtype,
            device=device,
            strict=True,
        )
    _restore_fp32_compute_parameters(module)
    return module


def _load_moss_audio_quantizer(
    module: nn.Module,
    model_path: Path,
    *,
    device: str | torch.device,
) -> nn.Module:
    return load_module(
        module,
        str(model_path),
        prefix="quantizer.",
        dtype=torch.float32,
        device=device,
        strict=True,
    )


def load_moss_audio_encoder(
    model_path: str,
    *,
    device: str | torch.device,
    compute_dtype: torch.dtype | None = None,
    attention_backend: str = AUTO_ATTENTION_BACKEND,
) -> MossAudioEncoder:
    """Load only encoder/quantizer weights without executing checkpoint code."""

    resolved_path, config = _load_moss_audio_config(model_path)

    model = MossAudioTokenizerEncoder(
        config,
        parameter_device="meta",
        compute_dtype=compute_dtype,
        attention_backend=attention_backend,
    )
    target_device = torch.device(device)
    backend_resolution = model.resolve_attention_backend(target_device)
    backend_label = _attention_backend_label(backend_resolution)
    _load_moss_audio_component(
        model.encoder,
        resolved_path,
        prefix="encoder.",
        dtype=model.encoder_dtype,
        device=device,
        v1_weights=model._uses_moss_audio_tokenizer_v1_weights,
    )
    _load_moss_audio_quantizer(
        model.quantizer,
        resolved_path,
        device=device,
    )
    model.eval()
    logger.info(
        "Loaded repository-local MOSS-Audio-Tokenizer encoder from %s on %s "
        "(channels=%d, attention_backend=%s, encoder_dtype=%s, "
        "compute_dtype=%s)",
        resolved_path,
        device,
        model.number_channels,
        backend_label,
        model.encoder_dtype,
        model.compute_dtype,
    )
    return MossAudioEncoder(model, device=str(device))


class MossAudioTokenizerVocoder(nn.Module):
    """Inference-only MOSS quantizer and waveform decoder."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        parameter_device: str | torch.device | None = None,
        decoder_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype | None = None,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> None:
        super().__init__()
        _validate_audio_dtypes(
            component_dtype=decoder_dtype,
            component_name="decoder_dtype",
            compute_dtype=compute_dtype,
        )
        self.config = SimpleNamespace(**config)
        sampling_rate = config.get("sampling_rate") or config.get("sample_rate")
        if sampling_rate is None:
            raise ValueError("MOSS audio-tokenizer config lacks sampling_rate")
        self.sampling_rate = int(sampling_rate)
        self.downsample_rate = int(config["downsample_rate"])
        self.decoder_dtype = decoder_dtype if compute_dtype is None else compute_dtype
        self.compute_dtype = None if compute_dtype is torch.float32 else compute_dtype
        self.attention_backend = validate_attention_backend(attention_backend)
        self._uses_moss_audio_tokenizer_v1_weights = "number_channels" not in config

        quantizer_config = dict(config["quantizer_kwargs"])
        quantizer_type = quantizer_config.get(
            "quantizer_type", config.get("quantizer_type", "rlfq")
        )
        if quantizer_type not in {"rlfq", "random_prefix_rlfq"}:
            raise ValueError(
                "repository-local MOSS vocoder supports residual LFQ checkpoints; "
                f"got quantizer_type={quantizer_type!r}"
            )
        self.quantizer = _ResidualLFQ(
            quantizer_config,
            device=parameter_device,
        )

        self.decoder = MossAudioTokenizerVocoderDecoder(
            config,
            moss_audio_tokenizer_v1_weights=(
                self._uses_moss_audio_tokenizer_v1_weights
            ),
            device=parameter_device,
            dtype=self.decoder_dtype,
            attention_backend=self.attention_backend,
        )

    def resolve_attention_backend(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> AttentionBackendResolution:
        decoder_dtype = self.decoder_dtype if dtype is None else dtype
        return self.decoder.resolve_attention_backend(device, decoder_dtype)


class MossAudioVocoder:
    """Narrow wrapper used by the MOSS-TTS Delay vocoder stage."""

    def __init__(self, model: MossAudioTokenizerVocoder, *, device: str) -> None:
        self.model = model
        self.device = str(device)
        self.sample_rate = int(model.sampling_rate)

    @torch.no_grad()
    def decode_codes(
        self,
        codes: torch.Tensor | list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if isinstance(codes, torch.Tensor):
            codes = [codes]
        if not codes:
            return []
        codes_nq_t = [
            item.transpose(0, 1).contiguous().to(device=self.device, dtype=torch.long)
            for item in codes
        ]
        count = int(codes_nq_t[0].shape[0])
        if any(int(item.shape[0]) != count for item in codes_nq_t):
            raise ValueError("all audio-code rows must use the same quantizer count")
        lengths_cpu = [int(item.shape[1]) for item in codes_nq_t]
        max_length = max(lengths_cpu)
        audio_codes = torch.zeros(
            count,
            len(codes_nq_t),
            max_length,
            device=self.device,
            dtype=torch.long,
        )
        for index, item in enumerate(codes_nq_t):
            audio_codes[:, index, : item.shape[1]] = item
        lengths = torch.tensor(lengths_cpu, device=self.device, dtype=torch.int32)
        hidden = self.model.quantizer.decode_codes(audio_codes)
        hidden = hidden.to(dtype=self.model.decoder_dtype)
        audio, _ = self.model.decoder(
            hidden,
            lengths,
            input_lengths_cpu=lengths_cpu,
        )
        output_lengths = self.model.decoder.output_lengths(lengths_cpu)
        return [
            audio[index, 0, :length].detach().to(device="cpu", dtype=torch.float32)
            for index, length in enumerate(output_lengths)
        ]


def load_moss_audio_vocoder(
    model_path: str,
    *,
    device: str | torch.device,
    decoder_dtype: torch.dtype = torch.bfloat16,
    compute_dtype: torch.dtype | None = None,
    attention_backend: str = AUTO_ATTENTION_BACKEND,
) -> MossAudioVocoder:
    """Load only quantizer/decoder weights without executing checkpoint code."""

    resolved_path, config = _load_moss_audio_config(model_path)

    model = MossAudioTokenizerVocoder(
        config,
        parameter_device="meta",
        decoder_dtype=decoder_dtype,
        compute_dtype=compute_dtype,
        attention_backend=attention_backend,
    )
    target_device = torch.device(device)
    backend_resolution = model.resolve_attention_backend(target_device)
    backend_label = _attention_backend_label(backend_resolution)
    _load_moss_audio_quantizer(
        model.quantizer,
        resolved_path,
        device=device,
    )
    _load_moss_audio_component(
        model.decoder,
        resolved_path,
        prefix="decoder.",
        dtype=model.decoder_dtype,
        device=device,
        v1_weights=model._uses_moss_audio_tokenizer_v1_weights,
    )
    model.eval()
    logger.info(
        "Loaded repository-local MOSS-Audio-Tokenizer vocoder from %s on %s "
        "(attention_backend=%s, decoder_dtype=%s, compute_dtype=%s)",
        resolved_path,
        device,
        backend_label,
        model.decoder_dtype,
        model.compute_dtype,
    )
    return MossAudioVocoder(model, device=str(device))
