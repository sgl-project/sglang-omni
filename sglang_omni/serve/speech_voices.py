# SPDX-License-Identifier: Apache-2.0
"""Uploaded voice persistence for OpenAI-compatible TTS serving."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from sglang_omni.client.audio import DEFAULT_SAMPLE_RATE, encode_wav
from sglang_omni.scheduling.speaker_cache import (
    SpeakerArtifactCache,
    SpeakerCacheKey,
    get_speaker_artifact_cache,
)
from sglang_omni.serve.speech_errors import SpeechAPIError, bad_request, internal_error

logger = logging.getLogger(__name__)

DEFAULT_SPEAKER_SAMPLES_DIR = Path("~/.cache/sglang-omni/speakers").expanduser()
DEFAULT_SPEAKER_MAX_UPLOADED = 1000
MAX_VOICE_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_REFERENCE_AUDIO_SECONDS = 1.0
MAX_REFERENCE_AUDIO_SECONDS = 30.0
VOICE_SILENCE_THRESHOLD = 1e-5
VOICE_METADATA_INT_FIELDS = frozenset(
    {"created_at", "file_size", "sample_rate", "num_samples"}
)
VOICE_NAME_PATTERN = re.compile(r"^(?=.*[A-Za-z0-9])[A-Za-z0-9_.-]+$")
DEFAULT_VOICE_PRESETS = ("default",)
ACCEPTED_VOICE_UPLOAD_MIME_TYPES = frozenset(
    {
        "audio/aac",
        "audio/flac",
        "audio/mp4",
        "audio/mpeg",
        "audio/ogg",
        "audio/wav",
        "audio/webm",
        "audio/x-wav",
        "video/mp4",
        "video/webm",
    }
)
VOICE_UPLOAD_EXTENSION_MIME_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}


@dataclass(frozen=True)
class UploadedVoice:
    """Persisted uploaded voice metadata."""

    name: str
    normalized_name: str
    consent: str
    created_at: int
    file_size: int
    mime_type: str
    original_filename: str
    sample_rate: int
    num_samples: int
    fingerprint: str
    file_path: Path
    ref_text: str | None = None
    speaker_description: str | None = None

    def to_response_dict(self) -> dict[str, Any]:
        response: dict[str, Any] = {
            "name": self.name,
            "consent": self.consent,
            "created_at": self.created_at,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
        }
        if self.ref_text is not None:
            response["ref_text"] = self.ref_text
        if self.speaker_description is not None:
            response["speaker_description"] = self.speaker_description
        return response

    def to_safetensors_metadata(self) -> dict[str, str]:
        metadata = {
            "name": self.name,
            "normalized_name": self.normalized_name,
            "consent": self.consent,
            "created_at": str(self.created_at),
            "file_size": str(self.file_size),
            "mime_type": self.mime_type,
            "original_filename": self.original_filename,
            "sample_rate": str(self.sample_rate),
            "num_samples": str(self.num_samples),
            "fingerprint": self.fingerprint,
        }
        if self.ref_text is not None:
            metadata["ref_text"] = self.ref_text
        if self.speaker_description is not None:
            metadata["speaker_description"] = self.speaker_description
        return metadata


@dataclass(frozen=True)
class UploadedVoiceReference:
    """Reference audio resolved from an uploaded voice."""

    voice: UploadedVoice
    ref_audio: str


class SpeakerSampleStore:
    """Persistent uploaded voice registry backed by one safetensors file per voice."""

    def __init__(
        self,
        *,
        root_dir: str | Path | None = None,
        max_uploaded: int | None = None,
        cache: SpeakerArtifactCache | None = None,
    ) -> None:
        self.root_dir = _resolve_speaker_root(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_uploaded = (
            int(max_uploaded)
            if max_uploaded is not None
            else _speaker_max_uploaded_from_env()
        )
        if self.max_uploaded <= 0:
            raise ValueError("SPEAKER_MAX_UPLOADED must be positive")
        self.cache = cache or get_speaker_artifact_cache()
        self._voices: dict[str, UploadedVoice] = {}
        self._last_upload_timestamp = 0
        self._lock = RLock()
        self._restore()

    def list_response(self) -> dict[str, Any]:
        with self._lock:
            uploaded = sorted(self._voices.values(), key=lambda item: item.name.lower())
            voices = sorted(
                set(DEFAULT_VOICE_PRESETS) | {voice.name for voice in uploaded},
                key=str.lower,
            )
            return {
                "voices": voices,
                "uploaded_voices": [voice.to_response_dict() for voice in uploaded],
                "cache_stats": self.cache.stats(),
            }

    def get(self, name: str) -> UploadedVoice | None:
        normalized = normalize_voice_name(name)
        with self._lock:
            return self._voices.get(normalized)

    def upload(
        self,
        *,
        name: str,
        consent: str,
        audio_bytes: bytes,
        filename: str | None,
        content_type: str | None,
        ref_text: str | None = None,
        speaker_description: str | None = None,
    ) -> dict[str, Any]:
        normalized_name = normalize_voice_name(name)
        if normalized_name in DEFAULT_VOICE_PRESETS:
            raise bad_request("name is reserved for a preset voice", param="name")
        display_name = name.strip()
        consent = _normalize_required_text(consent, "consent")
        ref_text = _normalize_optional_text(ref_text)
        speaker_description = _normalize_optional_text(speaker_description)

        file_size = len(audio_bytes)
        if file_size == 0:
            raise bad_request("audio_sample must not be empty", param="audio_sample")
        if file_size > MAX_VOICE_UPLOAD_BYTES:
            raise bad_request(
                f"audio_sample must be at most {MAX_VOICE_UPLOAD_BYTES} bytes",
                param="audio_sample",
            )
        mime_type = _resolve_upload_mime_type(filename, content_type)
        samples, sample_rate = _decode_reference_audio(audio_bytes)
        _validate_reference_audio(samples, sample_rate)

        fingerprint = hashlib.sha256(audio_bytes).hexdigest()
        voice_path = self.root_dir / f"{normalized_name}.safetensors"
        voice = UploadedVoice(
            name=display_name,
            normalized_name=normalized_name,
            consent=consent,
            created_at=self._next_upload_timestamp(),
            file_size=file_size,
            mime_type=mime_type,
            original_filename=filename or "",
            sample_rate=sample_rate,
            num_samples=int(samples.shape[0]),
            fingerprint=fingerprint,
            file_path=voice_path,
            ref_text=ref_text,
            speaker_description=speaker_description,
        )
        temp_path: Path | None = None
        replaced = False
        try:
            temp_path = _write_voice_temp_file(
                self.root_dir,
                normalized_name,
                samples,
                voice.to_safetensors_metadata(),
            )
            with self._lock:
                replaced = normalized_name in self._voices
                if not replaced and len(self._voices) >= self.max_uploaded:
                    raise bad_request(
                        f"Uploaded voice limit reached ({self.max_uploaded})",
                        param="name",
                    )
                _replace_voice_file(temp_path, voice_path)
                temp_path = None
                if replaced:
                    self.cache.clear_voice(normalized_name)
                self._voices[normalized_name] = voice
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        response = voice.to_response_dict()
        if replaced:
            response["warning"] = f"Voice '{display_name}' overwritten"
        return response

    def delete(self, name: str) -> bool:
        normalized = normalize_voice_name(name)
        with self._lock:
            voice = self._voices.get(normalized)
            if voice is None:
                return False
            try:
                voice.file_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.exception(
                    "Failed to delete voice file %s: %s", voice.file_path, exc
                )
                raise internal_error(
                    f"Failed to delete voice '{voice.name}' from storage",
                ) from exc
            self._voices.pop(normalized)
            self.cache.clear_voice(normalized)
        return True

    def resolve_reference(self, name: str) -> UploadedVoiceReference | None:
        normalized = normalize_voice_name(name)
        while True:
            with self._lock:
                voice = self._voices.get(normalized)
                if voice is None:
                    return None
                cache_key = _voice_data_url_cache_key(voice)
                cached = self.cache.get(cache_key)
            if cached is not None:
                return UploadedVoiceReference(voice=voice, ref_audio=cached)

            samples, sample_rate = self._load_samples(voice)
            audio_b64 = base64.b64encode(encode_wav(samples, sample_rate)).decode(
                "ascii"
            )
            cached = f"data:audio/wav;base64,{audio_b64}"
            with self._lock:
                if self._voices.get(normalized) != voice:
                    continue
                self.cache.put(cache_key, cached)
                return UploadedVoiceReference(voice=voice, ref_audio=cached)

    def _restore(self) -> None:
        restored: dict[str, UploadedVoice] = {}
        last_timestamp = 0
        try:
            safe_open = _safetensors_safe_open()
        except SpeechAPIError as exc:
            logger.warning(f"{exc.message}; uploaded voices will not be restored")
            return
        candidates: list[UploadedVoice] = []
        for path in sorted(self.root_dir.glob("*.safetensors")):
            try:
                with safe_open(str(path), framework="np") as handle:
                    metadata = dict(handle.metadata() or {})
            except Exception as exc:
                logger.warning(f"Skipping unreadable voice file {path}: {exc}")
                continue
            try:
                voice = _voice_from_metadata(metadata, path)
            except SpeechAPIError as exc:
                logger.warning(f"Skipping invalid voice metadata {path}: {exc}")
                continue
            candidates.append(voice)
        candidates.sort(key=lambda voice: voice.created_at, reverse=True)
        for voice in candidates:
            if voice.normalized_name in restored:
                logger.warning(
                    f"Skipping duplicate restored voice {voice.file_path} for name {voice.normalized_name}",
                )
                continue
            if len(restored) >= self.max_uploaded:
                logger.warning(
                    f"Skipping restored voice {voice.file_path} because SPEAKER_MAX_UPLOADED={self.max_uploaded}",
                )
                continue
            restored[voice.normalized_name] = voice
            last_timestamp = max(last_timestamp, voice.created_at)
        with self._lock:
            self._voices = restored
            self._last_upload_timestamp = last_timestamp

    def _next_upload_timestamp(self) -> int:
        with self._lock:
            timestamp = max(int(time.time()), self._last_upload_timestamp + 1)
            self._last_upload_timestamp = timestamp
            return timestamp

    def _load_samples(self, voice: UploadedVoice) -> tuple[np.ndarray, int]:
        try:
            load_file = _safetensors_load_file()
            tensors = load_file(str(voice.file_path))
            samples = np.asarray(tensors["audio"], dtype=np.float32)
        except SpeechAPIError:
            raise
        except Exception as exc:
            raise bad_request(
                f"Uploaded voice '{voice.name}' is missing or corrupted",
                param="voice",
            ) from exc
        return samples, voice.sample_rate


def normalize_voice_name(name: str) -> str:
    if not isinstance(name, str):
        raise bad_request("name must be a string", param="name")
    value = name.strip()
    if not value:
        raise bad_request("name must be non-empty", param="name")
    if not VOICE_NAME_PATTERN.fullmatch(value):
        raise bad_request(
            "name must contain only letters, numbers, '.', '_', and '-'",
            param="name",
        )
    return value.lower()


def _resolve_speaker_root(root_dir: str | Path | None) -> Path:
    if root_dir is not None:
        return Path(root_dir).expanduser().resolve()
    env_root = os.environ.get("SPEAKER_SAMPLES_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return DEFAULT_SPEAKER_SAMPLES_DIR


def _speaker_max_uploaded_from_env() -> int:
    value = os.environ.get("SPEAKER_MAX_UPLOADED")
    if not value:
        return DEFAULT_SPEAKER_MAX_UPLOADED
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid SPEAKER_MAX_UPLOADED={value!r}; using default")
        return DEFAULT_SPEAKER_MAX_UPLOADED


def _normalize_required_text(value: str, param: str) -> str:
    if not isinstance(value, str):
        raise bad_request(f"{param} must be a string", param=param)
    normalized = value.strip()
    if not normalized:
        raise bad_request(f"{param} must be non-empty", param=param)
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_upload_mime_type(filename: str | None, content_type: str | None) -> str:
    content_type = (content_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(filename or "").suffix.lower()
    inferred = VOICE_UPLOAD_EXTENSION_MIME_TYPES.get(suffix)
    mime_type = (
        inferred if content_type in {"", "application/octet-stream"} else content_type
    )
    if mime_type not in ACCEPTED_VOICE_UPLOAD_MIME_TYPES:
        accepted = ", ".join(sorted(ACCEPTED_VOICE_UPLOAD_MIME_TYPES))
        raise bad_request(
            f"audio_sample MIME type must be one of: {accepted}",
            param="audio_sample",
        )
    return mime_type


def _decode_reference_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        from sglang_omni.preprocessing.audio import AudioMediaIO

        samples, sample_rate = AudioMediaIO(target_sr=DEFAULT_SAMPLE_RATE).load_bytes(
            audio_bytes
        )
    except Exception as exc:
        logger.debug("Failed to decode uploaded voice sample", exc_info=True)
        raise bad_request(
            "audio_sample could not be decoded as a supported audio format",
            param="audio_sample",
        ) from exc
    return np.asarray(samples, dtype=np.float32), int(sample_rate)


def _validate_reference_audio(samples: np.ndarray, sample_rate: int) -> None:
    if sample_rate <= 0 or samples.ndim != 1 or samples.size == 0:
        raise bad_request("audio_sample must contain mono audio", param="audio_sample")
    duration = samples.shape[0] / float(sample_rate)
    if duration < MIN_REFERENCE_AUDIO_SECONDS:
        raise bad_request(
            f"audio_sample must be at least {MIN_REFERENCE_AUDIO_SECONDS:.1f}s",
            param="audio_sample",
        )
    if duration > MAX_REFERENCE_AUDIO_SECONDS:
        raise bad_request(
            f"audio_sample must be at most {MAX_REFERENCE_AUDIO_SECONDS:.1f}s",
            param="audio_sample",
        )
    if float(np.max(np.abs(samples))) <= VOICE_SILENCE_THRESHOLD:
        raise bad_request(
            "audio_sample must contain non-silent speech reference audio",
            param="audio_sample",
        )


def _voice_data_url_cache_key(voice: UploadedVoice) -> SpeakerCacheKey:
    return SpeakerCacheKey(
        model_type="serve",
        voice_name=voice.normalized_name,
        voice_version=voice.created_at,
        artifact_kind="wav_data_url",
    )


def _write_voice_temp_file(
    directory: Path,
    stem: str,
    samples: np.ndarray,
    metadata: dict[str, str],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{stem}.",
        suffix=".tmp",
        dir=directory,
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        save_file = _safetensors_save_file()
        save_file(
            {"audio": samples.astype(np.float32, copy=False)},
            str(tmp_path),
            metadata=metadata,
        )
        return tmp_path
    except SpeechAPIError:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise internal_error("Failed to save uploaded voice") from exc


def _replace_voice_file(temp_path: Path, path: Path) -> None:
    try:
        os.replace(temp_path, path)
    except OSError as exc:
        raise internal_error("Failed to save uploaded voice") from exc


def _safetensors_safe_open() -> Any:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise internal_error("safetensors is required for uploaded voices") from exc
    return safe_open


def _safetensors_load_file() -> Any:
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise internal_error("safetensors is required for uploaded voices") from exc
    return load_file


def _safetensors_save_file() -> Any:
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise internal_error("safetensors is required for uploaded voices") from exc
    return save_file


def _voice_from_metadata(metadata: dict[str, str], path: Path) -> UploadedVoice:
    normalized_name = metadata.get("normalized_name") or metadata.get(
        "voice_name_lower"
    )
    name = metadata.get("name") or normalized_name
    if normalized_name is None or name is None:
        raise bad_request("voice metadata is missing name")
    values: dict[str, Any] = dict(metadata)
    try:
        for key in VOICE_METADATA_INT_FIELDS:
            if key in values:
                values[key] = int(values[key])
        return UploadedVoice(
            name=name,
            normalized_name=normalize_voice_name(normalized_name),
            consent=values.get("consent", ""),
            created_at=int(values.get("created_at", 0)),
            file_size=int(values.get("file_size", path.stat().st_size)),
            mime_type=values.get("mime_type", "audio/wav"),
            original_filename=values.get("original_filename", ""),
            sample_rate=int(values.get("sample_rate", DEFAULT_SAMPLE_RATE)),
            num_samples=int(values.get("num_samples", 0)),
            fingerprint=values.get("fingerprint", ""),
            file_path=path,
            ref_text=values.get("ref_text"),
            speaker_description=values.get("speaker_description"),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise bad_request(f"voice metadata is invalid: {path}") from exc
