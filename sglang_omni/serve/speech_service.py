# SPDX-License-Identifier: Apache-2.0
"""Request validation and lowering for TTS speech API requests."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic import ValidationError

from sglang_omni.client import ClientError, GenerateRequest, SamplingParams
from sglang_omni.client.audio import audio_encoding_unavailable_reason
from sglang_omni.models.tts_streaming import INITIAL_CODEC_CHUNK_FRAMES_PARAM
from sglang_omni.preprocessing.base import MediaIO
from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector
from sglang_omni.serve.protocol import (
    DEFAULT_TTS_BATCH_MAX_ITEMS,
    SUPPORTED_TTS_LANGUAGES,
    SUPPORTED_TTS_RESPONSE_FORMATS,
    SUPPORTED_TTS_TASK_TYPES,
    TTS_SPEED_MAX,
    TTS_SPEED_MIN,
    CreateSpeechBatchRequest,
    CreateSpeechRequest,
    SpeechBatchItem,
    SpeechBatchResponse,
    SpeechBatchResult,
    SpeechReference,
)
from sglang_omni.serve.speech_errors import (
    SpeechAPIError,
    bad_request,
    internal_error,
    openai_error_payload,
    service_unavailable,
)

if TYPE_CHECKING:
    from sglang_omni.client import Client
    from sglang_omni.serve.speech_voices import (
        SpeakerSampleStore,
        UploadedVoiceReference,
    )

logger = logging.getLogger(__name__)

_TTS_LANGUAGE_ALIASES = {
    language.lower(): language for language in SUPPORTED_TTS_LANGUAGES
}
_TTS_TASK_TYPE_ALIASES = {
    task_type.replace("_", "").replace("-", "").lower(): task_type
    for task_type in SUPPORTED_TTS_TASK_TYPES
}
MAX_SPEECH_INPUT_CHARS = 4096
MAX_REFERENCE_AUDIO_BYTES = 10 * 1024 * 1024
_REFERENCE_AUDIO_FIELDS = ("audio_path", "ref_audio", "audio")
RAW_PCM_DEFAULT_INITIAL_CODEC_CHUNK_FRAMES = 1
_ReferenceCacheKey = tuple[Any, ...]


@dataclass(frozen=True)
class PreparedSpeechRequest:
    request: CreateSpeechRequest
    reference_descriptors: list[dict[str, Any]]
    uploaded_voice: "UploadedVoiceReference | None" = None


@dataclass(frozen=True)
class PreparedSpeechReferences:
    request_updates: dict[str, Any]
    reference_descriptors: list[dict[str, Any]]
    uploaded_voice: "UploadedVoiceReference | None" = None


class SpeechRequestValidator:
    """Validate and lower OpenAI-compatible TTS requests."""

    def __init__(
        self,
        *,
        default_model: str,
        requires_uploaded_voice_for_named_voice: bool = False,
        supports_uploaded_voice_references: bool = True,
        allowed_local_media_path: str | Path | None = None,
        allowed_media_domains: list[str] | None = None,
        voice_store: "SpeakerSampleStore | None" = None,
        tts_batch_max_items: int = DEFAULT_TTS_BATCH_MAX_ITEMS,
    ) -> None:
        if tts_batch_max_items < 1:
            raise ValueError("tts_batch_max_items must be greater than 0")
        self.default_model = default_model
        self.requires_uploaded_voice_for_named_voice = (
            requires_uploaded_voice_for_named_voice
        )
        self.supports_uploaded_voice_references = (
            supports_uploaded_voice_references
            or requires_uploaded_voice_for_named_voice
        )
        self.voice_store = voice_store
        self.reference_connector = MultiModalResourceConnector(
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
            allow_remote_media_without_domains=True,
            reject_unsafe_remote_addresses=True,
        )
        self.tts_batch_max_items = int(tts_batch_max_items)

    def parse_request(self, payload: Any) -> CreateSpeechRequest:
        """Parse and validate a raw HTTP payload."""

        return self.prepare_request(self._parse_raw_request(payload))

    def parse_batch_request(self, payload: Any) -> CreateSpeechBatchRequest:
        """Parse and validate a raw batch speech payload."""

        if not isinstance(payload, dict):
            raise bad_request("speech batch request body must be a JSON object")
        default_payload = {
            key: value for key, value in payload.items() if key != "items"
        }
        self._validate_raw_payload(default_payload)
        try:
            request = CreateSpeechBatchRequest.model_validate(payload)
        except ValidationError as exc:
            raise bad_request(_validation_error_message(exc)) from exc
        if not request.items:
            raise bad_request("items must contain at least one request", param="items")
        if len(request.items) > self.tts_batch_max_items:
            raise bad_request(
                f"items must contain at most {self.tts_batch_max_items} requests",
                param="items",
            )
        if request.stream:
            raise bad_request(
                "stream is not supported for batch speech requests",
                param="stream",
            )
        self._validate_batch_defaults(request)
        return request

    def validate_raw_speech_fields(self, payload: dict[str, Any]) -> None:
        """Validate speech fields before Pydantic can coerce JSON values."""

        self._validate_raw_payload(payload)

    def parse_generation_request(self, payload: Any) -> PreparedSpeechRequest:
        """Parse and prepare a raw HTTP payload for GenerateRequest lowering."""

        return self.prepare_generation_request(self._parse_raw_request(payload))

    def _parse_raw_request(self, payload: Any) -> CreateSpeechRequest:
        if not isinstance(payload, dict):
            raise bad_request("speech request body must be a JSON object")
        self._validate_raw_payload(payload)
        try:
            request = CreateSpeechRequest.model_validate(payload)
        except ValidationError as exc:
            raise bad_request(_validation_error_message(exc)) from exc
        return request

    def prepare_request(self, request: CreateSpeechRequest) -> CreateSpeechRequest:
        """Validate and normalize a request that was already parsed."""

        return self.prepare_generation_request(request).request

    def prepare_generation_request(
        self, request: CreateSpeechRequest
    ) -> PreparedSpeechRequest:
        """Validate a parsed request and build backend reference descriptors."""

        updates = self._prepare_generation_updates(request)
        prepared_references = self._prepare_reference_fields(request)
        updates.update(prepared_references.request_updates)
        prepared_request = request.model_copy(update=updates)
        return PreparedSpeechRequest(
            request=prepared_request,
            reference_descriptors=prepared_references.reference_descriptors,
            uploaded_voice=prepared_references.uploaded_voice,
        )

    def validate_input_text(self, input_text: str) -> None:
        if not isinstance(input_text, str) or not input_text.strip():
            raise bad_request("input must be a non-empty string", param="input")
        if len(input_text) > MAX_SPEECH_INPUT_CHARS:
            raise bad_request(
                f"input must be at most {MAX_SPEECH_INPUT_CHARS} characters",
                param="input",
            )

    def _prepare_generation_updates(
        self, request: CreateSpeechRequest
    ) -> dict[str, Any]:
        self.validate_input_text(request.input)
        updates: dict[str, Any] = {}
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

        for field_name in ("max_new_tokens", "token_count", "duration_tokens"):
            _validate_positive_int(getattr(request, field_name), param=field_name)
        _validate_non_negative_int(
            request.initial_codec_chunk_frames,
            param=INITIAL_CODEC_CHUNK_FRAMES_PARAM,
        )
        _validate_non_negative_int(request.seed, param="seed")
        return updates

    def _prepare_reference_fields(
        self, request: CreateSpeechRequest
    ) -> PreparedSpeechReferences:
        updates: dict[str, Any] = {}
        reference_descriptors: list[dict[str, Any]] = []
        uploaded_voice = self._resolve_uploaded_voice_reference(request)

        ref_audio = request.ref_audio
        if ref_audio is not None:
            descriptor = self._load_media_reference_descriptor(
                ref_audio, param="ref_audio"
            )
            updates["ref_audio"] = _media_reference_from_descriptor(descriptor)

        if request.references:
            references: list[SpeechReference] = []
            for reference in request.references:
                normalized_reference = self._normalize_speech_reference(reference)
                references.append(normalized_reference)
                reference_descriptors.append(
                    normalized_reference.model_dump(exclude_none=True)
                )
            updates["references"] = references

        if ref_audio is not None:
            if request.ref_text is not None:
                descriptor = dict(descriptor)
                descriptor["text"] = request.ref_text
            reference_descriptors.append(descriptor)
        elif uploaded_voice is not None:
            descriptor = _uploaded_voice_reference_dict(uploaded_voice)
            if uploaded_voice.voice.ref_text is not None:
                descriptor["text"] = uploaded_voice.voice.ref_text
            reference_descriptors.append(descriptor)
            updates["task_type"] = "Base"

        return PreparedSpeechReferences(
            request_updates=updates,
            reference_descriptors=reference_descriptors,
            uploaded_voice=uploaded_voice,
        )

    def build_generate_request(
        self,
        request: CreateSpeechRequest,
        *,
        validate: bool = True,
        reference_descriptors: list[dict[str, Any]] | None = None,
        uploaded_voice: "UploadedVoiceReference | None" = None,
    ) -> GenerateRequest:
        """Convert a validated speech request into a client GenerateRequest."""

        if validate:
            prepared = self.prepare_generation_request(request)
            request = prepared.request
            reference_descriptors = prepared.reference_descriptors
            uploaded_voice = prepared.uploaded_voice
        elif uploaded_voice is None and reference_descriptors is None:
            uploaded_voice = self._resolve_uploaded_voice_reference(request)

        return GenerateRequest(
            model=request.model or self.default_model,
            prompt=_build_speech_prompt(request, reference_descriptors),
            sampling=_build_sampling_params(request),
            stage_params=request.stage_params,
            extra_params=_build_extra_params(request),
            stream=request.stream,
            output_modalities=["audio"],
            metadata={
                "task": "tts",
                "tts_params": _build_tts_params(
                    request,
                    uploaded_voice=uploaded_voice,
                ),
            },
        )

    async def create_speech_batch(
        self,
        client: "Client",
        batch: CreateSpeechBatchRequest,
        *,
        request_id: str,
    ) -> SpeechBatchResponse:
        """Run a batch speech request through the normal speech path."""

        results: list[SpeechBatchResult | None] = [None] * len(batch.items)
        tasks: list[asyncio.Task[SpeechBatchResult]] = []
        task_indexes: list[int] = []
        task_request_ids: list[str] = []
        reference_tasks: dict[
            _ReferenceCacheKey, asyncio.Task[PreparedSpeechReferences]
        ] = {}
        reference_task_lock = asyncio.Lock()

        for index, item in enumerate(batch.items):
            try:
                request = self._build_batch_item_request(batch, item, index=index)
            except SpeechAPIError as exc:
                results[index] = _batch_error_result(
                    index, _batch_item_error(exc, index=index)
                )
                continue
            item_request_id = f"{request_id}-{index}"
            task = asyncio.create_task(
                self._run_batch_item(
                    client,
                    request,
                    request_id=item_request_id,
                    index=index,
                    reference_tasks=reference_tasks,
                    reference_task_lock=reference_task_lock,
                )
            )
            tasks.append(task)
            task_indexes.append(index)
            task_request_ids.append(item_request_id)

        if tasks:
            try:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                await self._cancel_batch_items(client, tasks, task_request_ids)
                raise
            for index, task_result in zip(task_indexes, task_results, strict=True):
                if isinstance(task_result, SpeechBatchResult):
                    results[index] = task_result
                elif isinstance(task_result, SpeechAPIError):
                    results[index] = _batch_error_result(
                        index, _batch_item_error(task_result, index=index)
                    )
                else:
                    results[index] = _batch_error_result(
                        index, internal_error(str(task_result))
                    )

        final_results = [result for result in results if result is not None]
        succeeded = sum(1 for result in final_results if result.status == "success")
        failed = len(final_results) - succeeded
        return SpeechBatchResponse(
            id=request_id,
            results=final_results,
            total=len(final_results),
            succeeded=succeeded,
            failed=failed,
        )

    async def _run_batch_item(
        self,
        client: "Client",
        request: CreateSpeechRequest,
        *,
        request_id: str,
        index: int,
        reference_tasks: dict[
            _ReferenceCacheKey, asyncio.Task[PreparedSpeechReferences]
        ],
        reference_task_lock: asyncio.Lock,
    ) -> SpeechBatchResult:
        prepared = await self._prepare_batch_item_request(
            request,
            reference_tasks=reference_tasks,
            reference_task_lock=reference_task_lock,
        )
        gen_req = self.build_generate_request(
            prepared.request,
            validate=False,
            reference_descriptors=prepared.reference_descriptors,
            uploaded_voice=prepared.uploaded_voice,
        )
        try:
            result = await client.speech(
                gen_req,
                request_id=request_id,
                response_format=prepared.request.response_format,
                speed=prepared.request.speed,
                allow_format_fallback=False,
            )
        except ClientError as exc:
            raise internal_error(str(exc)) from exc
        return SpeechBatchResult(
            index=index,
            status="success",
            audio_data=base64.b64encode(result.audio_bytes).decode("ascii"),
            format=result.format,
            media_type=result.mime_type,
        )

    async def _prepare_batch_item_request(
        self,
        request: CreateSpeechRequest,
        *,
        reference_tasks: dict[
            _ReferenceCacheKey, asyncio.Task[PreparedSpeechReferences]
        ],
        reference_task_lock: asyncio.Lock,
    ) -> PreparedSpeechRequest:
        updates = self._prepare_generation_updates(request)
        cache_key = _batch_reference_cache_key(request)
        async with reference_task_lock:
            task = reference_tasks.get(cache_key)
            if task is None:
                task = asyncio.create_task(
                    asyncio.to_thread(self._prepare_reference_fields, request)
                )
                reference_tasks[cache_key] = task

        prepared_references = await task
        updates.update(prepared_references.request_updates)
        return PreparedSpeechRequest(
            request=request.model_copy(update=updates),
            reference_descriptors=prepared_references.reference_descriptors,
            uploaded_voice=prepared_references.uploaded_voice,
        )

    async def _cancel_batch_items(
        self,
        client: "Client",
        tasks: list[asyncio.Task[SpeechBatchResult]],
        request_ids: list[str],
    ) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for request_id in request_ids:
            try:
                await client.abort(request_id)
            except Exception:
                logger.warning(
                    "Failed to abort speech batch item request %s",
                    request_id,
                    exc_info=True,
                )

    def _build_batch_item_request(
        self,
        batch: CreateSpeechBatchRequest,
        item: SpeechBatchItem,
        *,
        index: int,
    ) -> CreateSpeechRequest:
        payload = batch.model_dump(exclude={"items"}, exclude_none=True)
        item_payload = item.model_dump(exclude_none=True)
        payload.update(item_payload)
        self._validate_raw_payload(payload)
        if item_payload.get("stream"):
            raise bad_request(
                "stream is not supported for batch speech requests",
                param=f"items.{index}.stream",
            )
        payload.pop("stream", None)
        try:
            return CreateSpeechRequest.model_validate(payload)
        except ValidationError as exc:
            raise bad_request(
                _validation_error_message(exc),
                param=_validation_error_param(exc, prefix=f"items.{index}"),
            ) from exc

    def _validate_batch_defaults(self, batch: CreateSpeechBatchRequest) -> None:
        response_format = _normalize_response_format(batch.response_format)
        self._validate_encoder_dependency(response_format)
        if not TTS_SPEED_MIN <= float(batch.speed) <= TTS_SPEED_MAX:
            raise bad_request(
                f"speed must be between {TTS_SPEED_MIN} and {TTS_SPEED_MAX}",
                param="speed",
            )
        task_type = (
            _normalize_task_type(batch.task_type)
            if batch.task_type is not None
            else None
        )
        if batch.language is not None:
            _normalize_language(batch.language)
        for field_name in ("max_new_tokens", "token_count", "duration_tokens"):
            _validate_positive_int(getattr(batch, field_name), param=field_name)
        _validate_non_negative_int(
            batch.initial_codec_chunk_frames,
            param=INITIAL_CODEC_CHUNK_FRAMES_PARAM,
        )
        _validate_non_negative_int(batch.seed, param="seed")
        if (
            self.voice_store is not None
            and self.supports_uploaded_voice_references
            and batch.ref_audio is None
            and not batch.references
            and batch.voice
            and batch.voice.lower() != "default"
        ):
            uploaded_voice = self.voice_store.resolve_reference(batch.voice)
            if uploaded_voice is None and self.requires_uploaded_voice_for_named_voice:
                raise bad_request(
                    f"Unknown voice '{batch.voice}'. Upload a voice first via "
                    "POST /v1/audio/voices, or use ref_audio + ref_text.",
                    param="voice",
                )
            if (
                uploaded_voice is not None
                and task_type is not None
                and task_type != "Base"
            ):
                raise bad_request(
                    "uploaded voice requests require task_type='Base'",
                    param="task_type",
                )

    def _resolve_uploaded_voice_reference(
        self, request: CreateSpeechRequest
    ) -> "UploadedVoiceReference | None":
        if (
            self.voice_store is None
            or not self.supports_uploaded_voice_references
            or request.ref_audio is not None
            or request.references
        ):
            return None
        if not request.voice or request.voice.lower() == "default":
            return None
        uploaded_voice = self.voice_store.resolve_reference(request.voice)
        if uploaded_voice is None and self.requires_uploaded_voice_for_named_voice:
            raise bad_request(
                f"Unknown voice '{request.voice}'. Upload a voice first via "
                "POST /v1/audio/voices, or use ref_audio + ref_text.",
                param="voice",
            )
        if uploaded_voice is not None:
            task_type = request.task_type
            if task_type is not None and _normalize_task_type(task_type) != "Base":
                raise bad_request(
                    "uploaded voice requests require task_type='Base'",
                    param="task_type",
                )
        return uploaded_voice

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

    def _load_media_reference_descriptor(
        self, value: str, *, param: str
    ) -> dict[str, str]:
        url = urlparse(value)
        if url.scheme not in {"http", "https", "data", "file"}:
            if url.scheme:
                raise bad_request(
                    f"{param} must be an http, https, data, file:// URL, or local path",
                    param=param,
                )
            return {"audio_path": str(Path(value).expanduser().resolve())}
        try:
            return self.reference_connector.load_resource(
                value,
                _SpeechReferenceMediaIO(param),
                max_bytes=MAX_REFERENCE_AUDIO_BYTES,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise bad_request(str(exc), param=param) from exc

    def _validate_encoder_dependency(self, response_format: str) -> None:
        message = audio_encoding_unavailable_reason(response_format)
        if message is not None:
            raise service_unavailable(message, param="response_format")


def _explicit_generation_params(request: CreateSpeechRequest) -> list[str]:
    return sorted(
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


def _build_tts_params(
    request: CreateSpeechRequest,
    *,
    uploaded_voice: "UploadedVoiceReference | None" = None,
) -> dict[str, Any]:
    tts_params: dict[str, Any] = {
        "voice": request.voice,
        "response_format": request.response_format,
        "speed": request.speed,
    }
    explicit_generation_params = _explicit_generation_params(request)
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
    if uploaded_voice is not None:
        tts_params["task_type"] = "Base"
        tts_params["ref_audio"] = uploaded_voice.ref_audio
        if uploaded_voice.voice.ref_text is not None:
            tts_params["ref_text"] = uploaded_voice.voice.ref_text
        tts_params["uploaded_voice_name"] = uploaded_voice.voice.normalized_name
        tts_params["uploaded_voice_created_at"] = uploaded_voice.voice.created_at
    if request.x_vector_only_mode is not None:
        tts_params["x_vector_only_mode"] = request.x_vector_only_mode
    if request.initial_codec_chunk_frames is not None:
        tts_params[INITIAL_CODEC_CHUNK_FRAMES_PARAM] = (
            request.initial_codec_chunk_frames
        )
    if request.token_count is not None:
        tts_params["token_count"] = request.token_count
    if request.duration_tokens is not None:
        tts_params["duration_tokens"] = request.duration_tokens
    if request.seed is not None:
        tts_params["seed"] = request.seed
    return tts_params


def _build_sampling_params(request: CreateSpeechRequest) -> SamplingParams:
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
    return sampling


def _build_speech_prompt(
    request: CreateSpeechRequest,
    reference_descriptors: list[dict[str, Any]] | None,
) -> Any:
    if reference_descriptors is None:
        reference_descriptors = _reference_descriptors_from_request(request)
    if reference_descriptors:
        return {"text": request.input, "references": reference_descriptors}
    return request.input


def _build_extra_params(request: CreateSpeechRequest) -> dict[str, Any]:
    extra_params: dict[str, Any] = {}
    initial_codec_chunk_frames = request.initial_codec_chunk_frames
    if initial_codec_chunk_frames is None and request.stream:
        initial_codec_chunk_frames = RAW_PCM_DEFAULT_INITIAL_CODEC_CHUNK_FRAMES
    if initial_codec_chunk_frames is not None:
        extra_params[INITIAL_CODEC_CHUNK_FRAMES_PARAM] = initial_codec_chunk_frames
    return extra_params


class _SpeechReferenceMediaIO(MediaIO[dict[str, str]]):
    """Return backend reference descriptors after connector policy checks."""

    def __init__(self, param: str) -> None:
        self.param = param

    def load_bytes(self, data: bytes) -> dict[str, str]:
        return {
            "data": base64.b64encode(data).decode("ascii"),
            "media_type": "audio/wav",
        }

    def load_http_bytes(self, data: bytes, media_type: str | None) -> dict[str, str]:
        if media_type is not None and not (
            media_type.startswith("audio/") or media_type == "application/octet-stream"
        ):
            raise ValueError(f"{self.param} URL must return an audio media type")
        descriptor = self.load_bytes(data)
        if media_type is not None and media_type.startswith("audio/"):
            descriptor["media_type"] = media_type
        return descriptor

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


def _reference_descriptors_from_request(
    request: CreateSpeechRequest,
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    if request.references:
        references.extend(
            reference.model_dump(exclude_none=True) for reference in request.references
        )
    if request.ref_audio is not None:
        ref = _reference_dict_from_media_reference(request.ref_audio)
        if request.ref_text is not None:
            ref["text"] = request.ref_text
        references.append(ref)
    return references


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


def _uploaded_voice_reference_dict(
    uploaded_voice: "UploadedVoiceReference",
) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "audio_path": uploaded_voice.ref_audio,
        "uploaded_voice_name": uploaded_voice.voice.normalized_name,
        "uploaded_voice_created_at": uploaded_voice.voice.created_at,
    }
    parsed = urlparse(uploaded_voice.ref_audio)
    if parsed.scheme == "data" and "," in parsed.path:
        header, data = parsed.path.split(",", 1)
        media_type = header.split(";", 1)[0] or "audio/wav"
        ref["media_type"] = media_type
        ref["data"] = data
    return ref


def _batch_error_result(index: int, error: SpeechAPIError) -> SpeechBatchResult:
    return SpeechBatchResult(
        index=index,
        status="error",
        error=openai_error_payload(
            error.message,
            error_type=error.error_type,
            param=error.param,
            code=error.code,
        )["error"],
    )


def _batch_item_error(error: SpeechAPIError, *, index: int) -> SpeechAPIError:
    if error.param is None or error.param.startswith("items."):
        return error
    return SpeechAPIError(
        message=error.message,
        status_code=error.status_code,
        error_type=error.error_type,
        param=f"items.{index}.{error.param}",
        code=error.code,
    )


def _batch_reference_cache_key(request: CreateSpeechRequest) -> _ReferenceCacheKey:
    references = tuple(
        _freeze_reference_value(reference.model_dump(mode="json", exclude_none=True))
        for reference in request.references or ()
    )
    return (
        request.voice,
        request.task_type,
        request.ref_audio,
        request.ref_text,
        references,
    )


def _freeze_reference_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (key, _freeze_reference_value(item)) for key, item in sorted(value.items())
        )
    if isinstance(value, list):
        return tuple(_freeze_reference_value(item) for item in value)
    return value


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


def _normalize_language(value: str) -> str:
    normalized = _TTS_LANGUAGE_ALIASES.get(value.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(SUPPORTED_TTS_LANGUAGES))
        raise bad_request(f"language must be one of: {supported}", param="language")
    return normalized


def _normalize_task_type(value: str) -> str:
    normalized = _TTS_TASK_TYPE_ALIASES.get(
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


def _validation_error_param(
    exc: ValidationError, *, prefix: str | None = None
) -> str | None:
    first_error = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(item) for item in first_error.get("loc", ()))
    if not location:
        return prefix
    return f"{prefix}.{location}" if prefix else location
