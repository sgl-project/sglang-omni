# SPDX-License-Identifier: Apache-2.0
"""MOSS-Audio-Tokenizer-Nano reference encoder wrapper."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.models.moss_tts.audio_tokenizer import (
    AUTO_ATTENTION_BACKEND,
    MossAudioEncoder,
    load_moss_audio_encoder,
)
from sglang_omni.models.moss_tts_local.audio_tokenizer import MossTTSLocalAudioTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MOSS_TTS_NANO_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"


class MossTTSNanoAudioTokenizer(MossTTSLocalAudioTokenizer):
    """Encode references without the Local-v1.5 loudness adjustment."""

    def __init__(
        self,
        model: Any,
        *,
        device: str,
        encoder: MossAudioEncoder | None = None,
    ) -> None:
        super().__init__(model, device=device, encoder=encoder)
        config = getattr(model, "config", None)
        self.number_channels = int(getattr(config, "number_channels", 2))

    def _prepare_waveform(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if int(sample_rate) != self.sample_rate:
            import torchaudio

            wav = torchaudio.functional.resample(
                waveform=wav,
                orig_freq=int(sample_rate),
                new_freq=self.sample_rate,
            )
        if int(wav.shape[0]) == self.number_channels:
            pass
        elif int(wav.shape[0]) == 1 and self.number_channels > 1:
            wav = wav.repeat(self.number_channels, 1)
        elif int(wav.shape[0]) > 1 and self.number_channels == 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif int(wav.shape[0]) > self.number_channels:
            wav = wav[: self.number_channels]
        else:
            raise ValueError(
                "unsupported MOSS-TTS-Nano reference channel conversion: "
                f"{int(wav.shape[0])} -> {self.number_channels}"
            )
        return wav.to(device=self.device, dtype=torch.float32)


def load_moss_tts_nano_audio_tokenizer(
    model_path: str = DEFAULT_MOSS_TTS_NANO_AUDIO_TOKENIZER,
    *,
    device: str = "cuda:0",
    compute_dtype: torch.dtype | None = None,
    attention_backend: str = AUTO_ATTENTION_BACKEND,
) -> MossTTSNanoAudioTokenizer:
    encoder = load_moss_audio_encoder(
        model_path,
        device=device,
        compute_dtype=compute_dtype,
        attention_backend=attention_backend,
    )
    logger.info(
        "Loaded MOSS-Audio-Tokenizer-Nano encoder from %s on %s "
        "(encoder_dtype=%s, compute_dtype=%s)",
        model_path,
        device,
        encoder.model.encoder_dtype,
        encoder.model.compute_dtype,
    )
    return MossTTSNanoAudioTokenizer(
        encoder.model,
        device=device,
        encoder=encoder,
    )
