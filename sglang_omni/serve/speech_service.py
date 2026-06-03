# SPDX-License-Identifier: Apache-2.0
"""Request validation and lowering for TTS speech API requests."""

from __future__ import annotations

import base64
import binascii
import importlib.util
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from sglang_omni.client import GenerateRequest, SamplingParams
from sglang_omni.preprocessing.base import MediaIO
from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector
from sglang_omni.serve.protocol import (
    SUPPORTED_TTS_LANGUAGES,
    SUPPORTED_TTS_RESPONSE_FORMATS,
    SUPPORTED_TTS_TASK_TYPES,
    TTS_SPEED_MAX,
    TTS_SPEED_MIN,
    CreateSpeechRequest,
    SpeechReference,
)
from sglang_omni.serve.speech_errors import bad_request, service_unavailable

_LANGUAGE_CANONICAL = {
    language.lower(): language for language in SUPPORTED_TTS_LANGUAGES
}
_TASK_TYPE_CANONICAL = {
    task_type.replace("_", "").replace("-", "").lower(): task_type
    for task_type in SUPPORTED_TTS_TASK_TYPES
}
MAX_SPEECH_INPUT_CHARS = 4096
MAX_REFERENCE_AUDIO_BYTES = 10 * 1024 * 1024
_REFERENCE_AUDIO_FIELDS = ("audio_path", "ref_audio", "audio")


class SpeechRequestValidator:
    """Validate and lower OpenAI-compatible TTS requests."""

    def __init__(
        self,
        *,
        default_model: str,
        allowed_local_media_path: str | Path | None = None,
    ) -> None:
        self.default_model = default_model
        self.allowed_local_media_path = (
            _resolve_allowed_local_media_path(allowed_local_media_path)
            if allowed_local_media_path is not None
            and str(allowed_local_media_path).strip()
            else None
        )
        self.reference_connector = MultiModalResourceConnector(
            allowed_local_media_path=(
                str(self.allowed_local_media_path)
                if self.allowed_local_media_path
                else ""
            )
        )
        self.encoder_dependency_errors = _build_encoder_dependency_errors()

    def parse_request(self, payload: Any) -> CreateSpeechRequest:
        """Parse and validate a raw HTTP payload."""

        if not isinstance(payload, dict):
            raise bad_request("speech request body must be a JSON object")
        self._validate_raw_payload(payload)
        try:
            request = CreateSpeechRequest.model_validate(payload)
        except ValidationError as exc:
            raise bad_request(_validation_error_message(exc)) from exc
        return self.prepare_request(request)

    def prepare_request(self, request: CreateSpeechRequest) -> CreateSpeechRequest:
        """Validate and normalize a request that was already parsed."""

        updates: dict[str, Any] = {}

        input_text = request.input
        if not isinstance(input_text, str) or not input_text.strip():
            raise bad_request("input must be a non-empty string", param="input")
        if len(input_text) > MAX_SPEECH_INPUT_CHARS:
            raise bad_request(
                f"input must be at most {MAX_SPEECH_INPUT_CHARS} characters",
                param="input",
            )

        response_format = _normalize_response_format(request.response_format)
        if request.stream and response_format != "pcm":
            raise bad_request(
                "stream=true requires response_format='pcm'",
                param="response_format",
            )
        if not request.stream:
            self._validate_encoder_dependency(response_format)
        updates["response_format"] = response_format

        if not TTS_SPEED_MIN <= float(request.speed) <= TTS_SPEED_MAX:
            raise bad_request(
                f"speed must be between {TTS_SPEED_MIN} and {TTS_SPEED_MAX}",
                param="speed",
            )

        if request.task_type is not None:
            updates["task_type"] = _normalize_task_type(request.task_type)
        if request.language is not None:
            updates["language"] = _normalize_language(request.language)

        for field_name in (
            "max_new_tokens",
            "initial_codec_chunk_frames",
            "token_count",
            "duration_tokens",
        ):
            _validate_positive_int(getattr(request, field_name), param=field_name)
        _validate_non_negative_int(request.seed, param="seed")

        ref_audio = request.ref_audio
        if ref_audio is not None:
            updates["ref_audio"] = self._normalize_media_reference(
                ref_audio, param="ref_audio"
            )

        if request.references:
            updates["references"] = [
                self._normalize_speech_reference(reference)
                for reference in request.references
            ]

        return request.model_copy(update=updates)

    def build_generate_request(
        self,
        request: CreateSpeechRequest,
        *,
        validate: bool = True,
    ) -> GenerateRequest:
        """Convert a validated speech request into a client GenerateRequest."""

        if validate:
            request = self.prepare_request(request)
        explicit_generation_params = sorted(
            field
            for field in (
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "seed",
            )
            if field in request.model_fields_set
        )

        tts_params: dict[str, Any] = {
            "voice": request.voice,
            "response_format": request.response_format,
            "speed": request.speed,
        }
        if explicit_generation_params:
            tts_params["explicit_generation_params"] = explicit_generation_params
        if request.task_type is not None:
            tts_params["task_type"] = request.task_type
        if request.language is not None:
            tts_params["language"] = request.language
        if request.instructions is not None:
            tts_params["instructions"] = request.instructions
        if request.ref_audio is not None:
            tts_params["ref_audio"] = request.ref_audio
        if request.ref_text is not None:
            tts_params["ref_text"] = request.ref_text
        if request.x_vector_only_mode is not None:
            tts_params["x_vector_only_mode"] = request.x_vector_only_mode
        if request.initial_codec_chunk_frames is not None:
            tts_params["initial_codec_chunk_frames"] = (
                request.initial_codec_chunk_frames
            )
        if request.token_count is not None:
            tts_params["token_count"] = request.token_count
        if request.duration_tokens is not None:
            tts_params["duration_tokens"] = request.duration_tokens
        if request.seed is not None:
            tts_params["seed"] = request.seed

        sampling = SamplingParams(
            temperature=0.8, top_p=0.8, top_k=30, repetition_penalty=1.1
        )
        if request.max_new_tokens is not None:
            sampling.max_new_tokens = request.max_new_tokens
        if request.temperature is not None:
            sampling.temperature = request.temperature
        if request.top_p is not None:
            sampling.top_p = request.top_p
        if request.top_k is not None:
            sampling.top_k = request.top_k
        if request.repetition_penalty is not None:
            sampling.repetition_penalty = request.repetition_penalty
        if request.seed is not None:
            sampling.seed = request.seed

        prompt: Any = request.input
        references: list[dict[str, Any]] = []
        if request.references:
            references.extend(
                reference.model_dump(exclude_none=True)
                for reference in request.references
            )
        if request.ref_audio is not None:
            ref = _reference_dict_from_media_reference(request.ref_audio)
            if request.ref_text is not None:
                ref["text"] = request.ref_text
            references.append(ref)
        if references:
            prompt = {"text": request.input, "references": references}

        return GenerateRequest(
            model=request.model or self.default_model,
            prompt=prompt,
            sampling=sampling,
            stage_params=request.stage_params,
            stream=request.stream,
            output_modalities=["audio"],
            metadata={
                "task": "tts",
                "tts_params": tts_params,
            },
        )

    def _validate_raw_payload(self, payload: dict[str, Any]) -> None:
        for field_name in (
            "model",
            "input",
            "voice",
            "speaker",
            "response_format",
            "task_type",
            "language",
            "instructions",
            "ref_audio",
            "ref_text",
        ):
            if field_name in payload and payload[field_name] is not None:
                if not isinstance(payload[field_name], str):
                    raise bad_request(
                        f"{field_name} must be a string", param=field_name
                    )
        for field_name in (
            "max_new_tokens",
            "initial_codec_chunk_frames",
            "token_count",
            "duration_tokens",
            "seed",
        ):
            if field_name in payload and payload[field_name] is not None:
                value = payload[field_name]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise bad_request(
                        f"{field_name} must be an integer", param=field_name
                    )
        if "top_k" in payload and payload["top_k"] is not None:
            value = payload["top_k"]
            if isinstance(value, bool) or not isinstance(value, int):
                raise bad_request("top_k must be an integer", param="top_k")
        for field_name in ("speed", "temperature", "top_p", "repetition_penalty"):
            if field_name in payload and payload[field_name] is not None:
                value = payload[field_name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise bad_request(
                        f"{field_name} must be a number", param=field_name
                    )
        for field_name in ("stream", "x_vector_only_mode"):
            if field_name in payload and payload[field_name] is not None:
                if not isinstance(payload[field_name], bool):
                    raise bad_request(
                        f"{field_name} must be a boolean", param=field_name
                    )

    def _normalize_speech_reference(
        self, reference: SpeechReference
    ) -> SpeechReference:
        updates: dict[str, Any] = {
            field_name: None for field_name in _REFERENCE_AUDIO_FIELDS
        }
        if reference.data is not None:
            updates.update(
                _SpeechReferenceMediaIO("references.data").load_base64(
                    reference.media_type or "audio/wav", reference.data
                )
            )
            return reference.model_copy(update=updates)

        for field_name in _REFERENCE_AUDIO_FIELDS:
            value = getattr(reference, field_name)
            if not isinstance(value, str):
                continue
            updates.update(
                self._load_media_reference_descriptor(
                    value, param=f"references.{field_name}"
                )
            )
            break
        return reference.model_copy(update=updates)

    def _normalize_media_reference(self, value: str, *, param: str) -> str:
        descriptor = self._load_media_reference_descriptor(value, param=param)
        return _media_reference_from_descriptor(descriptor)

    def _load_media_reference_descriptor(
        self, value: str, *, param: str
    ) -> dict[str, str]:
        url = urlparse(value)
        if url.scheme in {"http", "https"}:
            raise bad_request(
                "remote reference audio URLs are not supported by this endpoint",
                param=param,
            )
        if url.scheme not in {"data", "file"}:
            if Path(value).is_absolute():
                value = Path(value).expanduser().resolve().as_uri()
            else:
                raise bad_request(f"{param} must be a data or file:// URL", param=param)
        try:
            return self.reference_connector.load_resource(
                value, _SpeechReferenceMediaIO(param)
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise bad_request(str(exc), param=param) from exc

    def _validate_encoder_dependency(self, response_format: str) -> None:
        message = self.encoder_dependency_errors.get(response_format)
        if message is not None:
            raise service_unavailable(message, param="response_format")


class _SpeechReferenceMediaIO(MediaIO[dict[str, str]]):
    """Return backend reference descriptors after connector policy checks."""

    def __init__(self, param: str) -> None:
        self.param = param

    def load_bytes(self, data: bytes) -> dict[str, str]:
        raise ValueError(
            "remote reference audio URLs are not supported by this endpoint"
        )

    def load_base64(self, media_type: str, data: str) -> dict[str, str]:
        _validate_base64_media_data(data, media_type=media_type, param=self.param)
        return {"data": data, "media_type": media_type}

    def load_file(self, filepath: Path) -> dict[str, str]:
        if not filepath.is_file():
            raise ValueError(f"file:// {self.param} path must be a file: {filepath}")
        _validate_reference_size(filepath.stat().st_size, param=self.param)
        return {"audio_path": str(filepath)}


def _reference_dict_from_media_reference(value: str) -> dict[str, Any]:
    if value.startswith("data:"):
        media_type, encoded = _parse_data_url(value, param="ref_audio")
        return {"data": encoded, "media_type": media_type}
    return {"audio_path": value}


def _media_reference_from_descriptor(descriptor: dict[str, str]) -> str:
    audio_path = descriptor.get("audio_path")
    if audio_path is not None:
        return audio_path
    return f"data:{descriptor['media_type']};base64,{descriptor['data']}"


def _normalize_response_format(value: str) -> str:
    fmt = value.strip().lower()
    if fmt not in SUPPORTED_TTS_RESPONSE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_TTS_RESPONSE_FORMATS))
        raise bad_request(
            f"response_format must be one of: {supported}",
            param="response_format",
        )
    return fmt


def _validate_positive_int(value: int | None, *, param: str) -> None:
    if value is not None and value <= 0:
        raise bad_request(f"{param} must be greater than 0", param=param)


def _validate_non_negative_int(value: int | None, *, param: str) -> None:
    if value is not None and value < 0:
        raise bad_request(f"{param} must be greater than or equal to 0", param=param)


def _parse_data_url(value: str, *, param: str) -> tuple[str, str]:
    header, separator, encoded = value.partition(",")
    if not separator or ";base64" not in header.lower() or not encoded:
        raise bad_request(
            f"{param} data URL must include base64 media data",
            param=param,
        )
    media_type = header.removeprefix("data:").split(";", 1)[0] or "audio/wav"
    _validate_base64_media_data(encoded, media_type=media_type, param=param)
    return media_type, encoded


def _validate_base64_media_data(encoded: str, *, media_type: str, param: str) -> None:
    if not media_type.startswith("audio/"):
        raise bad_request(f"{param} data URL must use an audio media type", param=param)
    _validate_reference_size(_estimated_base64_decoded_size(encoded), param=param)
    try:
        base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise bad_request(
            f"{param} data URL must include valid base64 media data",
            param=param,
        ) from exc


def _estimated_base64_decoded_size(encoded: str) -> int:
    return (len(encoded.rstrip("=")) * 3) // 4


def _validate_reference_size(size_bytes: int, *, param: str) -> None:
    if size_bytes > MAX_REFERENCE_AUDIO_BYTES:
        raise bad_request(
            f"{param} must be at most {MAX_REFERENCE_AUDIO_BYTES} bytes",
            param=param,
        )


def _build_encoder_dependency_errors() -> dict[str, str]:
    errors: dict[str, str] = {}
    if importlib.util.find_spec("soundfile") is None:
        errors["flac"] = "soundfile is required for response_format='flac'"
    if importlib.util.find_spec("pydub") is None:
        for response_format in ("mp3", "aac", "opus"):
            errors[response_format] = (
                f"pydub is required for response_format={response_format!r}"
            )
    elif shutil.which("ffmpeg") is None and shutil.which("avconv") is None:
        for response_format in ("mp3", "aac", "opus"):
            errors[response_format] = (
                "ffmpeg or avconv is required for "
                f"response_format={response_format!r}"
            )
    return errors


def _normalize_language(value: str) -> str:
    normalized = _LANGUAGE_CANONICAL.get(value.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(SUPPORTED_TTS_LANGUAGES))
        raise bad_request(f"language must be one of: {supported}", param="language")
    return normalized


def _normalize_task_type(value: str) -> str:
    normalized = _TASK_TYPE_CANONICAL.get(
        value.strip().replace("_", "").replace("-", "").lower()
    )
    if normalized is None:
        supported = ", ".join(sorted(SUPPORTED_TTS_TASK_TYPES))
        raise bad_request(f"task_type must be one of: {supported}", param="task_type")
    return normalized


def _validation_error_message(exc: ValidationError) -> str:
    first_error = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(item) for item in first_error.get("loc", ()))
    message = first_error.get("msg") or "invalid speech request"
    return f"{location}: {message}" if location else str(message)


def _resolve_allowed_local_media_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"allowed local media path must be a directory: {path}")
    return resolved
