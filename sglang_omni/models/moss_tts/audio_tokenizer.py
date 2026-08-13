# SPDX-License-Identifier: Apache-2.0
"""Standalone MOSS-Audio-Tokenizer loader for MOSS-TTS Delay."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import torch
import torchaudio
from transformers import AutoModel

from sglang_omni.models.moss_tts.hf_loading import moss_transformers_processor_compat

logger = logging.getLogger(__name__)

DEFAULT_MOSS_TTS_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
_LOUDNESS_TARGET_DBFS = -20.0
_LOUDNESS_GAIN_MIN_DB = -3.0
_LOUDNESS_GAIN_MAX_DB = 3.0


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
                "MOSS-TTS audio tokenizer encode returned empty "
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
                "MOSS-TTS audio tokenizer decode returned empty audio/audio_lengths"
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
    logger.info(f"Loading MOSS-TTS audio tokenizer from {model_path} on {device}")
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
