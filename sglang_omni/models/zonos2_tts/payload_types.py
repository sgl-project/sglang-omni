# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 TTS pipeline state and payload types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ZONOS2 model constants
ZONOS2_N_CODEBOOKS = 9
ZONOS2_CODEBOOK_SIZE = 1024
ZONOS2_EOA_ID = 1024  # End-of-audio token
ZONOS2_AUDIO_PAD_ID = 1025  # Audio padding token
ZONOS2_SAMPLE_RATE = 44100  # DAC 44kHz output
ZONOS2_LOSS_SOFTCAP = 15.0
ZONOS2_TEXT_VOCAB = 519

# Text tokenization constants (UTF-8 byte encoding)
ZONOS2_LEGACY_SYMBOL_VOCAB = 192
ZONOS2_BYTE_VOCAB = 256
ZONOS2_BYTE_TEXT_VOCAB = ZONOS2_LEGACY_SYMBOL_VOCAB + ZONOS2_BYTE_VOCAB  # 448
ZONOS2_BOS_ID = 2
ZONOS2_EOS_ID = 3


@dataclass
class Zonos2TTSState:
    """Per-request state for ZONOS2 TTS generation."""

    text: str = ""
    ref_audio: Any | None = None
    ref_text: str | None = None
    language: str | None = None
    text_normalization: bool = True
    speaker_embedding: Any | None = None

    # Generation parameters
    generation_kwargs: dict[str, Any] = field(default_factory=dict)

    # Model config (populated during preprocessing)
    n_codebooks: int = ZONOS2_N_CODEBOOKS
    codebook_size: int = ZONOS2_CODEBOOK_SIZE
    eoa_id: int = ZONOS2_EOA_ID
    audio_pad_id: int = ZONOS2_AUDIO_PAD_ID
    text_vocab: int = ZONOS2_TEXT_VOCAB
    speaking_rate_bucket: int | None = None
    quality_buckets: list[int | None] | None = None

    # Output state
    audio_codes: Any | None = None
    eos_frame: int | None = None
    sample_rate: int = ZONOS2_SAMPLE_RATE

    # Usage tracking
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    @staticmethod
    def _tensor_to_payload(value: Any) -> Any:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return value

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "text_normalization": bool(self.text_normalization),
            "generation_kwargs": dict(self.generation_kwargs),
            "sample_rate": int(self.sample_rate),
            "n_codebooks": int(self.n_codebooks),
            "codebook_size": int(self.codebook_size),
            "eoa_id": int(self.eoa_id),
            "audio_pad_id": int(self.audio_pad_id),
            "text_vocab": int(self.text_vocab),
        }
        if self.ref_audio is not None:
            data["ref_audio"] = self.ref_audio
        if self.ref_text is not None:
            data["ref_text"] = self.ref_text
        if self.language is not None:
            data["language"] = self.language
        if self.speaker_embedding is not None:
            data["speaker_embedding"] = self._tensor_to_payload(
                self.speaker_embedding
            )
        if self.speaking_rate_bucket is not None:
            data["speaking_rate_bucket"] = int(self.speaking_rate_bucket)
        if self.quality_buckets is not None:
            data["quality_buckets"] = list(self.quality_buckets)
        if self.audio_codes is not None:
            data["audio_codes"] = self._tensor_to_payload(self.audio_codes)
        if self.eos_frame is not None:
            data["eos_frame"] = int(self.eos_frame)
        if self.prompt_tokens:
            data["prompt_tokens"] = int(self.prompt_tokens)
        if self.completion_tokens:
            data["completion_tokens"] = int(self.completion_tokens)
        if self.engine_time_s:
            data["engine_time_s"] = float(self.engine_time_s)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "Zonos2TTSState":
        if not isinstance(data, dict):
            data = {}
        generation_kwargs = data.get("generation_kwargs")
        return cls(
            text=str(data.get("text", "")),
            ref_audio=data.get("ref_audio"),
            ref_text=data.get("ref_text"),
            language=data.get("language"),
            text_normalization=bool(data.get("text_normalization", True)),
            speaker_embedding=data.get("speaker_embedding"),
            generation_kwargs=(
                dict(generation_kwargs) if isinstance(generation_kwargs, dict) else {}
            ),
            n_codebooks=int(data.get("n_codebooks", ZONOS2_N_CODEBOOKS)),
            codebook_size=int(data.get("codebook_size", ZONOS2_CODEBOOK_SIZE)),
            eoa_id=int(data.get("eoa_id", ZONOS2_EOA_ID)),
            audio_pad_id=int(data.get("audio_pad_id", ZONOS2_AUDIO_PAD_ID)),
            text_vocab=int(data.get("text_vocab", ZONOS2_TEXT_VOCAB)),
            speaking_rate_bucket=data.get("speaking_rate_bucket"),
            quality_buckets=data.get("quality_buckets"),
            audio_codes=data.get("audio_codes"),
            eos_frame=data.get("eos_frame"),
            sample_rate=int(data.get("sample_rate", ZONOS2_SAMPLE_RATE)),
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            engine_time_s=float(data.get("engine_time_s", 0.0) or 0.0),
        )
