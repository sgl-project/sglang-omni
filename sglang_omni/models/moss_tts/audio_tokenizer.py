# SPDX-License-Identifier: Apache-2.0
"""MOSS-Audio-Tokenizer runtime."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from typing import Any

import torch
import torchaudio
from torch import nn
from transformers import AutoModel

from sglang_omni.models.moss_tts.attention import (
    AUTO_ATTENTION_BACKEND,
    PACKED_FLASH_ATTENTION_BACKEND,
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
from sglang_omni.models.moss_tts.hf_loading import moss_transformers_processor_compat

logger = logging.getLogger(__name__)

DEFAULT_MOSS_TTS_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
_LOUDNESS_TARGET_DBFS = -20.0
_LOUDNESS_GAIN_MIN_DB = -3.0
_LOUDNESS_GAIN_MAX_DB = 3.0


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
        *,
        norm1: nn.Module,
        self_attn: MossAudioTokenizerAttention,
        layer_scale_1: nn.Module,
        norm2: nn.Module,
        ffn: nn.Module,
        layer_scale_2: nn.Module,
    ) -> None:
        super().__init__()
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
            norm1=module.norm1,
            self_attn=MossAudioTokenizerAttention.from_module(
                module.self_attn,
                attention_backend=attention_backend,
                packed_rope_cache=packed_rope_cache,
            ),
            layer_scale_1=module.layer_scale_1,
            norm2=module.norm2,
            ffn=_feed_forward(module),
            layer_scale_2=module.layer_scale_2,
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
        *,
        layers: Sequence[MossAudioTokenizerTransformerLayer],
        positional_embedding: str,
        positional_scale: float,
        max_period: float,
    ) -> None:
        super().__init__()
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
        max_period = float(module.max_period)
        packed_rope_cache = MossPackedRopeCache(max_period=max_period)
        return cls(
            layers=[
                MossAudioTokenizerTransformerLayer.from_module(
                    layer,
                    attention_backend=attention_backend,
                    packed_rope_cache=packed_rope_cache,
                )
                for layer in module.layers
            ],
            positional_embedding=module.positional_embedding,
            positional_scale=float(module.positional_scale),
            max_period=max_period,
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
        *,
        input_proj: nn.Module,
        transformer: MossAudioTokenizerTransformer,
        output_proj: nn.Module,
    ) -> None:
        super().__init__()
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
            input_proj=module.input_proj,
            transformer=MossAudioTokenizerTransformer.from_module(
                module.transformer,
                attention_backend=attention_backend,
            ),
            output_proj=module.output_proj,
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


# note (Zhang Yiyang): Non-streaming decoder wrapper.


class MossAudioTokenizerVocoderDecoder(nn.Module):
    """Iterable MOSS-Audio-Tokenizer vocoder decoder with patched projected transformers."""

    def __init__(
        self,
        decoder: nn.Module,
        *,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> None:
        super().__init__()
        stages = list(decoder)
        assert (
            stages
        ), "MOSS-Audio-Tokenizer vocoder decoder must be a non-empty stage list"
        self.attention_backend = validate_attention_backend(attention_backend)
        self.stages = nn.ModuleList([self._wrap_stage(stage) for stage in stages])

    def _wrap_stage(self, stage: nn.Module) -> nn.Module:
        module_type = stage.module_type
        if module_type == "Transformer":
            return MossAudioTokenizerProjectedTransformer.from_module(
                stage,
                attention_backend=self.attention_backend,
            )
        if module_type == "PatchedPretransform":
            return stage
        raise ValueError(
            f"unsupported MOSS-Audio-Tokenizer vocoder decoder stage {stage.__class__.__name__} "
            f"with module_type={module_type!r}"
        )

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, index: int) -> nn.Module:
        return self.stages[index]

    def supports_packed_attention(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        device = torch.device(device)
        transformer_stages = [
            stage
            for stage in self.stages
            if isinstance(stage, MossAudioTokenizerProjectedTransformer)
        ]
        return bool(transformer_stages) and all(
            stage.supports_packed_attention(device, dtype)
            for stage in transformer_stages
        )

    @staticmethod
    def _update_cpu_lengths(
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

    def output_lengths(self, input_lengths: Sequence[int]) -> list[int]:
        output_lengths = list(map(int, input_lengths))
        for stage in self.stages:
            output_lengths = self._update_cpu_lengths(stage, output_lengths)
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
        for stage in self.stages:
            if isinstance(stage, MossAudioTokenizerProjectedTransformer):
                x, input_lengths = stage(
                    x,
                    input_lengths,
                    input_lengths_cpu=cpu_lengths,
                )
            else:
                x, input_lengths = stage(x, input_lengths)
            if cpu_lengths is not None:
                cpu_lengths = self._update_cpu_lengths(stage, cpu_lengths)
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


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def _model_floating_dtype(model: Any) -> torch.dtype:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return torch.float32
    return next(
        (
            parameter.dtype
            for parameter in parameters()
            if parameter.is_floating_point()
        ),
        torch.float32,
    )


class MossTTSAudioTokenizer:
    """Processor-compatible wrapper around a separately loaded codec model."""

    def __init__(self, model: Any, *, device: str) -> None:
        self.model = model
        self.device = str(device)
        self.dtype = _model_floating_dtype(model)
        self.sample_rate = int(model.config.sampling_rate)

    def _autocast(self) -> Any:
        device_type = torch.device(self.device).type
        if device_type == "cuda" and self.dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type=device_type, dtype=self.dtype)
        return nullcontext()

    def encode_waveforms(
        self,
        waveforms: list[tuple[torch.Tensor, int]],
        *,
        num_quantizers: int | None = None,
    ) -> list[torch.Tensor]:
        if not waveforms:
            raise ValueError("waveforms must contain at least one waveform")
        prepared = [
            self._prepare_waveform(wav, sample_rate) for wav, sample_rate in waveforms
        ]

        with torch.inference_mode(), self._autocast():
            if hasattr(self.model, "batch_encode"):
                encoded = self.model.batch_encode(
                    prepared,
                    num_quantizers=num_quantizers,
                )
            else:
                max_length = max(int(wav.shape[-1]) for wav in prepared)
                input_values = torch.zeros(
                    len(prepared),
                    1,
                    max_length,
                    device=self.device,
                    dtype=torch.float32,
                )
                padding_mask = torch.zeros(
                    len(prepared),
                    max_length,
                    device=self.device,
                    dtype=torch.bool,
                )
                for index, wav in enumerate(prepared):
                    length = int(wav.shape[-1])
                    input_values[index, 0, :length] = wav
                    padding_mask[index, :length] = True
                encoded = self.model.encode(
                    input_values,
                    padding_mask=padding_mask,
                    num_quantizers=num_quantizers,
                    return_dict=True,
                )

        audio_codes = encoded.audio_codes
        audio_codes_lengths = encoded.audio_codes_lengths
        if audio_codes is None or audio_codes_lengths is None:
            raise RuntimeError(
                "MOSS-Audio-Tokenizer encode returned empty "
                "audio_codes/audio_codes_lengths"
            )
        codes_cpu = audio_codes.detach().to(device="cpu", dtype=torch.long)
        lengths_cpu = audio_codes_lengths.detach().to("cpu")
        return [
            codes_cpu[:, index, : int(lengths_cpu[index])].transpose(0, 1).contiguous()
            for index in range(int(codes_cpu.shape[1]))
        ]

    def _prepare_waveform(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim != 2:
            raise ValueError(
                f"expected waveform with shape [channels, samples], got {tuple(wav.shape)}"
            )
        if int(wav.shape[0]) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if int(sample_rate) != self.sample_rate:
            wav = torchaudio.functional.resample(
                waveform=wav,
                orig_freq=int(sample_rate),
                new_freq=self.sample_rate,
            )
        wav = self._loudness_normalize(wav.squeeze(0))
        return wav.to(device=self.device, dtype=torch.float32)

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
        num_quantizers = int(codes_nq_t[0].shape[0])
        if any(int(item.shape[0]) != num_quantizers for item in codes_nq_t):
            raise ValueError("all audio-code rows must use the same quantizer count")
        max_length = max(int(item.shape[1]) for item in codes_nq_t)
        audio_codes = torch.zeros(
            num_quantizers,
            len(codes_nq_t),
            max_length,
            device=self.device,
            dtype=torch.long,
        )
        padding_mask = torch.zeros(
            len(codes_nq_t),
            max_length,
            device=self.device,
            dtype=torch.bool,
        )
        for index, item in enumerate(codes_nq_t):
            length = int(item.shape[1])
            audio_codes[:, index, :length] = item
            padding_mask[index, :length] = True

        with torch.inference_mode(), self._autocast():
            decoded = self.model.decode(
                audio_codes,
                padding_mask=padding_mask,
                return_dict=True,
                chunk_duration=8,
            )
        audio = decoded.audio
        audio_lengths = decoded.audio_lengths
        if audio is None or audio_lengths is None:
            raise RuntimeError(
                "MOSS-Audio-Tokenizer decode returned empty audio/audio_lengths"
            )
        audio_cpu = audio.detach().to(device="cpu", dtype=torch.float32)
        lengths_cpu = audio_lengths.detach().to("cpu")
        return [
            audio_cpu[index, 0, : int(lengths_cpu[index])].contiguous()
            for index in range(int(audio_cpu.shape[0]))
        ]

    @staticmethod
    def _loudness_normalize(wav: torch.Tensor) -> torch.Tensor:
        wav = wav.to(torch.float32)
        if wav.numel() == 0:
            return wav
        current_dbfs = 10.0 * torch.log10(torch.mean(wav**2) + 1e-9)
        gain = float(_LOUDNESS_TARGET_DBFS - current_dbfs)
        gain = max(_LOUDNESS_GAIN_MIN_DB, min(gain, _LOUDNESS_GAIN_MAX_DB))
        return wav * (10.0 ** (gain / 20.0))


def load_moss_tts_audio_tokenizer(
    model_path: str = DEFAULT_MOSS_TTS_AUDIO_TOKENIZER,
    *,
    device: str = "cpu",
    dtype: str | torch.dtype = "float32",
) -> MossTTSAudioTokenizer:
    logger.info(f"Loading MOSS-Audio-Tokenizer from {model_path} on {device}")
    try:
        with moss_transformers_processor_compat():
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
    except Exception as exc:
        raise RuntimeError(
            "MOSS-TTS support requires OpenMOSS-Team/MOSS-Audio-Tokenizer"
        ) from exc
    model.eval()
    move_kwargs: dict[str, Any] = {"device": device}
    if device != "cpu":
        move_kwargs["dtype"] = _torch_dtype(dtype)
    model.to(**move_kwargs)
    return MossTTSAudioTokenizer(model, device=device)
