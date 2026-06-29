# SPDX-License-Identifier: Apache-2.0
"""MOSS-Audio-Tokenizer-v2 codec loader for MOSS-TTS Local."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.models.moss_tts.hf_loading import (
    moss_transformers_processor_compat,
    resolve_moss_checkpoint,
)

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
    ) -> None:
        self.model = model
        self.device = str(device)
        config = model.config
        self.sample_rate = int(config.sampling_rate)

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
            encoded = self.model.batch_encode(
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
) -> MossTTSLocalAudioTokenizer:
    checkpoint_dir = resolve_moss_checkpoint(model_path)
    logger.info(
        f"Loading MOSS-TTS Local audio tokenizer from {checkpoint_dir} on {device}"
    )
    try:
        from transformers import AutoModel

        with moss_transformers_processor_compat():
            model = AutoModel.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
                codec_weight_dtype="bf16",
            )
    except Exception as exc:
        raise RuntimeError(
            "MOSS-TTS Local support requires OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
        ) from exc
    model.eval()
    model.to(device)
    return MossTTSLocalAudioTokenizer(
        model,
        device=device,
    )
