# SPDX-License-Identifier: Apache-2.0
"""MOSS-Audio-Tokenizer-v2 codec loader for MOSS-TTS Local."""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
from torch import nn

from sglang_omni.models.moss_tts.audio_tokenizer import (
    AUTO_ATTENTION_BACKEND,
    PACKED_FLASH_ATTENTION_BACKEND,
    SDPA_ATTENTION_BACKEND,
    MossAudioEncoder,
    load_moss_audio_encoder,
    resolve_moss_audio_attention_backend,
    resolve_moss_audio_dtype,
    resolve_moss_audio_sample_rate,
    validate_attention_backend,
)
from sglang_omni.models.moss_tts.hf_loading import moss_transformers_processor_compat
from sglang_omni.models.weight_loader import load_module, resolve_model_path

logger = logging.getLogger(__name__)

DEFAULT_MOSS_TTS_LOCAL_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
_REFERENCE_CHANNELS = 2
_LOUDNESS_TARGET_DBFS = -20.0
_LOUDNESS_GAIN_MIN_DB = -3.0
_LOUDNESS_GAIN_MAX_DB = 3.0


class MossTTSLocalAudioTokenizer:
    """Encode wrapper around a separately loaded MOSS-Audio-Tokenizer-v2 model."""

    def __init__(
        self,
        model: Any,
        *,
        device: str,
        encoder: MossAudioEncoder | None = None,
    ) -> None:
        self.model = model
        self._encoder = encoder
        self.device = str(device)
        config = getattr(model, "config", None)
        if config is None and encoder is not None:
            config = encoder.model.config
        if config is None:
            raise ValueError("MOSS-TTS Local audio tokenizer model lacks config")
        self.sample_rate = resolve_moss_audio_sample_rate(model, config)

    def encode_paths(
        self,
        paths: list[str],
        *,
        num_quantizers: int,
    ) -> list[torch.Tensor]:
        if not paths:
            raise ValueError("paths must contain at least one audio path")
        return self.encode_waveforms(
            self.load_paths(paths),
            num_quantizers=num_quantizers,
        )

    def load_paths(self, paths: list[str]) -> list[tuple[torch.Tensor, int]]:
        import torchaudio

        waveforms = []
        for path in paths:
            wav, sample_rate = torchaudio.load(path)
            if int(sample_rate) != self.sample_rate:
                wav = torchaudio.functional.resample(
                    waveform=wav,
                    orig_freq=int(sample_rate),
                    new_freq=self.sample_rate,
                )
            waveforms.append((wav, self.sample_rate))
        return waveforms

    def encode_wavs(
        self,
        wavs: list[torch.Tensor],
        sample_rate: int,
        *,
        num_quantizers: int,
    ) -> list[torch.Tensor]:
        return self.encode_waveforms(
            [(wav, int(sample_rate)) for wav in wavs],
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
            self._prepare_waveform(wav, sample_rate) for wav, sample_rate in waveforms
        ]

        with torch.inference_mode():
            encoder_model = (
                self._encoder.model if self._encoder is not None else self.model
            )
            encoded = encoder_model.batch_encode(
                prepared,
                num_quantizers=int(num_quantizers),
            )
        audio_codes = encoded.audio_codes
        audio_codes_lengths = encoded.audio_codes_lengths
        if audio_codes is None or audio_codes_lengths is None:
            raise RuntimeError(
                "MOSS-TTS Local audio tokenizer encode returned empty "
                "audio_codes/audio_codes_lengths"
            )
        codes_cpu = audio_codes.detach().to("cpu", torch.long)
        lengths_cpu = audio_codes_lengths.detach().to("cpu")
        return [
            codes_cpu[:, index, : int(lengths_cpu[index])].transpose(0, 1).contiguous()
            for index in range(int(codes_cpu.shape[1]))
        ]

    def _prepare_waveform(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] == 1:
            wav = wav.repeat(_REFERENCE_CHANNELS, 1)
        elif wav.shape[0] > _REFERENCE_CHANNELS:
            wav = wav[:_REFERENCE_CHANNELS]
        if wav.shape[0] != _REFERENCE_CHANNELS:
            raise ValueError(
                f"expected {_REFERENCE_CHANNELS} audio channels, got {wav.shape[0]}"
            )
        if int(sample_rate) != self.sample_rate:
            import torchaudio

            wav = torchaudio.functional.resample(
                waveform=wav,
                orig_freq=int(sample_rate),
                new_freq=self.sample_rate,
            )
        wav = self._loudness_normalize(wav)
        return wav.to(device=self.device, dtype=torch.float32)

    @staticmethod
    def _loudness_normalize(wav: torch.Tensor) -> torch.Tensor:
        wav = wav.to(torch.float32)
        if wav.numel() == 0:
            return wav
        current_dbfs = 10.0 * torch.log10(torch.mean(wav**2) + 1e-9)
        gain = float(_LOUDNESS_TARGET_DBFS - current_dbfs)
        gain = max(_LOUDNESS_GAIN_MIN_DB, min(gain, _LOUDNESS_GAIN_MAX_DB))
        return wav * (10.0 ** (gain / 20.0))


def load_moss_tts_local_audio_tokenizer(
    model_path: str = DEFAULT_MOSS_TTS_LOCAL_AUDIO_TOKENIZER,
    *,
    device: str = "cuda:0",
    compute_dtype: torch.dtype | None = None,
    attention_backend: str = AUTO_ATTENTION_BACKEND,
) -> MossTTSLocalAudioTokenizer:
    encoder = load_moss_audio_encoder(
        model_path,
        device=device,
        compute_dtype=compute_dtype,
        attention_backend=attention_backend,
    )
    logger.info(
        "Loaded repository-local MOSS-Audio-Tokenizer encoder for MOSS-TTS Local "
        "from %s on %s (encoder_dtype=%s, compute_dtype=%s)",
        model_path,
        device,
        encoder.model.encoder_dtype,
        encoder.model.compute_dtype,
    )
    return MossTTSLocalAudioTokenizer(
        encoder.model,
        device=device,
        encoder=encoder,
    )


class MossTTSLocalAudioVocoder:
    """V2 streaming codec shell with only quantizer and decoder weights."""

    def __init__(self, model: Any, *, device: str) -> None:
        self.model = model
        self.device = str(device)
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("MOSS-TTS Local vocoder model lacks config")
        self.sample_rate = resolve_moss_audio_sample_rate(model, config)


def _resolve_local_codec_dtype(
    value: str | torch.dtype | None,
    *,
    name: str,
    allow_none: bool,
) -> torch.dtype | None:
    if isinstance(value, str):
        value = {
            "bf16": "bfloat16",
            "fp32": "float32",
        }.get(value.lower(), value)
    return resolve_moss_audio_dtype(value, name=name, allow_none=allow_none)


def _load_local_streaming_codec_config(model_path: str) -> tuple[Any, Any]:
    resolved_path = resolve_model_path(str(model_path))
    config_path = resolved_path / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config_dict = json.load(config_file)
    if config_dict.get("model_type") != "moss-audio-tokenizer":
        raise ValueError(
            f"expected model_type='moss-audio-tokenizer' in {config_path}, "
            f"got {config_dict.get('model_type')!r}"
        )
    from transformers import AutoConfig

    with moss_transformers_processor_compat():
        config = AutoConfig.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
    return resolved_path, config


def load_moss_tts_local_audio_vocoder(
    model_path: str = DEFAULT_MOSS_TTS_LOCAL_AUDIO_TOKENIZER,
    *,
    device: str = "cuda:0",
    decoder_dtype: torch.dtype = torch.bfloat16,
    compute_dtype: torch.dtype | None = None,
    attention_backend: str = AUTO_ATTENTION_BACKEND,
) -> MossTTSLocalAudioVocoder:
    """Load the Local streaming shell with quantizer/decoder prefixes only."""
    validate_attention_backend(attention_backend)
    resolved_path, config = _load_local_streaming_codec_config(model_path)
    configured_compute_dtype = _resolve_local_codec_dtype(
        getattr(config, "compute_dtype", "bfloat16"),
        name="compute_dtype",
        allow_none=True,
    )
    effective_compute_dtype = (
        configured_compute_dtype if compute_dtype is None else compute_dtype
    )
    if isinstance(effective_compute_dtype, str):
        effective_compute_dtype = _resolve_local_codec_dtype(
            effective_compute_dtype,
            name="compute_dtype",
            allow_none=True,
        )
    if effective_compute_dtype is None:
        effective_decoder_dtype = decoder_dtype
    else:
        effective_decoder_dtype = effective_compute_dtype

    config_attention_implementation = getattr(
        config,
        "attention_implementation",
        "flash_attention_2",
    )
    selected_backend = resolve_moss_audio_attention_backend(
        attention_backend,
        config_attention_implementation,
    )
    if selected_backend == SDPA_ATTENTION_BACKEND:
        config.attention_implementation = SDPA_ATTENTION_BACKEND
    elif selected_backend == PACKED_FLASH_ATTENTION_BACKEND:
        config.attention_implementation = "flash_attention_2"

    from transformers import AutoModel

    with moss_transformers_processor_compat(), torch.device("meta"):
        model = AutoModel.from_config(config, trust_remote_code=True)
    if not hasattr(model, "decoder") or not hasattr(model, "quantizer"):
        raise ValueError("MOSS-Audio-Tokenizer-v2 model lacks decoder/quantizer")
    model.encoder = nn.ModuleList()
    model.quantizer = load_module(
        model.quantizer,
        str(resolved_path),
        prefix="quantizer.",
        dtype=torch.float32,
        device=device,
        strict=True,
    )
    model.decoder = load_module(
        model.decoder,
        str(resolved_path),
        prefix="decoder.",
        dtype=effective_decoder_dtype,
        device=device,
        strict=True,
    )
    model.compute_dtype = (
        None
        if effective_compute_dtype is None or effective_compute_dtype is torch.float32
        else effective_compute_dtype
    )
    model.compute_dtype_name = "fp32" if model.compute_dtype is None else "bf16"
    model.config.compute_dtype = model.compute_dtype_name
    model.eval()
    logger.info(
        "Loaded MOSS-TTS Local streaming codec from %s on %s "
        "(encoder_weights=0, decoder_dtype=%s, compute_dtype=%s, backend=%s)",
        resolved_path,
        device,
        effective_decoder_dtype,
        model.compute_dtype,
        selected_backend,
    )
    return MossTTSLocalAudioVocoder(model, device=device)
