# SPDX-License-Identifier: Apache-2.0
"""Kimi-Audio prompt construction and audio feature extraction."""

from __future__ import annotations

import base64
import binascii
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from sglang_omni.utils.audio import load_audio

from .audio_tokenizer import Glm4SpeechTokenizer
from .checkpoint import resolve_glm4_audio_tokenizer
from .hf_config import KimiAudioConfig
from .text_tokenizer import KimiTextTokenizer

_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 30 * _SAMPLE_RATE


@dataclass(frozen=True)
class KimiSpecialTokens:
    msg_end: int
    media_begin: int
    media_end: int
    text_blank: int
    text_eos: int
    user_start: int
    assistant_start: int
    speech_continue_text: int


@dataclass
class KimiPrompt:
    audio_ids: list[int] = field(default_factory=list)
    text_ids: list[int] = field(default_factory=list)
    continuous_mask: list[bool] = field(default_factory=list)
    continuous_features: list[torch.Tensor] = field(default_factory=list)

    def extend(self, other: KimiPrompt) -> None:
        self.audio_ids.extend(other.audio_ids)
        self.text_ids.extend(other.text_ids)
        self.continuous_mask.extend(other.continuous_mask)
        self.continuous_features.extend(other.continuous_features)

    def validate(self) -> None:
        if not (len(self.audio_ids) == len(self.text_ids) == len(self.continuous_mask)):
            raise ValueError("Kimi-Audio prompt streams must have equal lengths")
        feature_rows = sum(int(item.shape[0]) for item in self.continuous_features)
        if feature_rows != sum(self.continuous_mask):
            raise ValueError(
                "Kimi-Audio continuous features do not match the audio token span "
                f"({feature_rows} feature rows, {sum(self.continuous_mask)} tokens)"
            )


def _load_whisper_encoder(checkpoint_dir: str) -> WhisperEncoder:
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=False)
    encoder = WhisperEncoder(config)
    state: dict[str, torch.Tensor] = {}
    for path in Path(checkpoint_dir).glob("model*.safetensors"):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                if name.startswith("model.encoder."):
                    state[name.removeprefix("model.encoder.")] = handle.get_tensor(name)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Kimi-Audio Whisper checkpoint does not match WhisperEncoder "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return encoder.eval()


class KimiAudioProcessor:
    """Owns the two frozen audio encoders and builds paired Kimi streams."""

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        device: str,
        dtype: torch.dtype = torch.bfloat16,
        audio_tokenizer_path: str = "THUDM/glm-4-voice-tokenizer",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device)
        self.dtype = dtype
        self.lock = threading.Lock()
        self.tokenizer = KimiTextTokenizer(checkpoint_dir)
        self.model_config = KimiAudioConfig.from_pretrained(checkpoint_dir)
        self.special = self._special_tokens()

        glm_checkpoint = resolve_glm4_audio_tokenizer(audio_tokenizer_path)
        # The official Kimi-Audio prompt manager keeps the discrete GLM
        # tokenizer in FP32. Its nearest-codebook lookup is precision-sensitive.
        self.speech_tokenizer = Glm4SpeechTokenizer(glm_checkpoint).to(
            device=self.device
        )

        whisper_dir = str(Path(checkpoint_dir) / "whisper-large-v3")
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
            whisper_dir
        )
        self.whisper_encoder = _load_whisper_encoder(whisper_dir).to(
            device=self.device, dtype=dtype
        )

    def _token_id(self, literal: str) -> int:
        token_id = int(self.tokenizer.convert_tokens_to_ids(literal))
        if token_id < 0:
            raise ValueError(f"Kimi-Audio tokenizer is missing {literal}")
        return token_id

    def _special_tokens(self) -> KimiSpecialTokens:
        return KimiSpecialTokens(
            msg_end=self._token_id("<|im_msg_end|>"),
            media_begin=self._token_id("<|im_media_begin|>"),
            media_end=self._token_id("<|im_media_end|>"),
            text_blank=self._token_id("<|im_kimia_text_blank|>"),
            text_eos=self._token_id("<|im_kimia_text_eos|>"),
            user_start=self._token_id("<|im_kimia_user_msg_start|>"),
            assistant_start=self._token_id("<|im_kimia_assistant_msg_start|>"),
            speech_continue_text=self._token_id("<|im_kimia_speech_ct_id|>"),
        )

    def _encode_text(self, text: str) -> list[int]:
        try:
            return list(self.tokenizer.encode(text, bos=False, eos=False))
        except TypeError:
            return list(self.tokenizer.encode(text, add_special_tokens=False))

    @torch.inference_mode()
    def _continuous_features(self, waveform: np.ndarray) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(waveform), _CHUNK_SAMPLES):
            segment = waveform[start : start + _CHUNK_SAMPLES]
            token_len = (len(segment) - 1) // 1280 + 1
            extracted = self.whisper_feature_extractor(
                segment,
                sampling_rate=_SAMPLE_RATE,
                return_tensors="pt",
                padding="max_length",
            )
            features = extracted.input_features.to(device=self.device, dtype=self.dtype)
            hidden = self.whisper_encoder(features).last_hidden_state
            hidden = hidden[:, : token_len * 4]
            chunks.append(hidden.reshape(token_len, hidden.shape[-1] * 4).cpu())
        if not chunks:
            return torch.empty((0, 5120), dtype=self.dtype)
        return torch.cat(chunks, dim=0)

    def _audio_fragment(self, source: Any) -> KimiPrompt:
        waveform = load_audio(
            source,
            source_name="Kimi-Audio",
            target_sample_rate=_SAMPLE_RATE,
        )
        wave_tensor = torch.from_numpy(np.ascontiguousarray(waveform)).to(torch.float32)
        with self.lock:
            speech_ids = self.speech_tokenizer.tokenize(wave_tensor).cpu().tolist()
            features = self._continuous_features(waveform)
        if len(speech_ids) != features.shape[0]:
            raise ValueError(
                "Kimi-Audio encoders produced different sequence lengths "
                f"({len(speech_ids)} discrete tokens, {features.shape[0]} feature rows)"
            )
        offset = int(self.model_config.kimia_token_offset)
        speech_ids = [int(token) + offset for token in speech_ids]
        count = len(speech_ids)
        return KimiPrompt(
            audio_ids=[self.special.media_begin, *speech_ids, self.special.media_end],
            text_ids=[self.special.text_blank] * (count + 2),
            continuous_mask=[False, *([True] * count), False],
            continuous_features=[features],
        )

    def _text_fragment(self, text: str) -> KimiPrompt:
        text_ids = self._encode_text(text)
        return KimiPrompt(
            audio_ids=[self.special.text_blank] * len(text_ids),
            text_ids=text_ids,
            continuous_mask=[False] * len(text_ids),
        )

    @staticmethod
    def _audio_content_value(item_type: str, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        url = value.get("url")
        if url is not None:
            return url
        data = value.get("data")
        if item_type != "input_audio" or not isinstance(data, str):
            return data
        if data.startswith("data:"):
            return data
        try:
            return base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Kimi-Audio input_audio.data is not valid base64") from exc

    @staticmethod
    def _content_parts(content: Any) -> list[tuple[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [("text", content)]
        if not isinstance(content, list):
            raise TypeError("Kimi-Audio message content must be a string or a list")
        parts: list[tuple[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                raise TypeError("Kimi-Audio content items must be objects")
            item_type = item.get("type")
            if item_type in ("text", "input_text"):
                parts.append(("text", item.get("text", "")))
            elif item_type in ("audio_url", "input_audio", "audio"):
                value = item.get(
                    "audio_url", item.get("input_audio", item.get("audio"))
                )
                value = KimiAudioProcessor._audio_content_value(item_type, value)
                parts.append(("audio", value))
            else:
                raise ValueError(
                    f"Kimi-Audio does not support content type {item_type!r}"
                )
        return parts

    def build_prompt(
        self, messages: list[dict[str, Any]], top_level_audios: list[Any] | None = None
    ) -> KimiPrompt:
        expanded: list[dict[str, Any]] = [dict(message) for message in messages]
        if top_level_audios:
            for index, message in enumerate(expanded):
                if message.get("role") != "user":
                    continue
                parts = self._content_parts(message.get("content"))
                parts.extend(("audio", audio) for audio in top_level_audios)
                message["_kimi_parts"] = parts
                expanded[index] = message
                break
            else:
                raise ValueError("Kimi-Audio audio input requires a user message")

        result = KimiPrompt()
        previous_role: str | None = None
        audio_count = 0
        for message_index, message in enumerate(expanded):
            role = str(message.get("role", ""))
            if role not in ("user", "assistant"):
                raise ValueError("Kimi-Audio supports only user and assistant roles")
            if role != previous_role:
                role_id = (
                    self.special.user_start
                    if role == "user"
                    else self.special.assistant_start
                )
                result.audio_ids.append(role_id)
                result.text_ids.append(self.special.text_blank)
                result.continuous_mask.append(False)
            parts = message.get("_kimi_parts") or self._content_parts(
                message.get("content")
            )
            role_boundary = (
                message_index == len(expanded) - 1
                or str(expanded[message_index + 1].get("role", "")) != role
            )
            has_text_part = False
            for part_index, (kind, value) in enumerate(parts):
                if kind == "text":
                    result.extend(self._text_fragment(str(value)))
                    has_text_part = True
                else:
                    if role == "assistant":
                        raise ValueError(
                            "Kimi-Audio assistant audio history is not supported "
                            "by the text-output integration"
                        )
                    if value is None:
                        raise ValueError(
                            "Kimi-Audio audio content is missing its URL or data"
                        )
                    result.extend(self._audio_fragment(value))
                    audio_count += 1
                    if part_index == len(parts) - 1 and role_boundary:
                        result.audio_ids.append(self.special.speech_continue_text)
                        result.text_ids.append(self.special.text_blank)
                        result.continuous_mask.append(False)
            if role == "assistant" and has_text_part:
                result.audio_ids.append(self.special.text_blank)
                result.text_ids.append(self.special.text_eos)
                result.continuous_mask.append(False)
            if role_boundary:
                result.audio_ids.append(self.special.msg_end)
                result.text_ids.append(self.special.text_blank)
                result.continuous_mask.append(False)
            previous_role = role

        if audio_count == 0:
            raise ValueError(
                "Kimi-Audio text output currently requires at least one audio input"
            )
        result.audio_ids.append(self.special.assistant_start)
        result.text_ids.append(self.special.text_blank)
        result.continuous_mask.append(False)
        result.validate()
        return result


__all__ = ["KimiAudioProcessor", "KimiPrompt", "KimiSpecialTokens"]
