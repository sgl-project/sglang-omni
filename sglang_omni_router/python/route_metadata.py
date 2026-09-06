# SPDX-License-Identifier: Apache-2.0
"""Route metadata extraction for Omni router worker selection."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from fastapi import Request

from sglang_omni_router.python.config import DEFAULT_CAPABILITIES, Capability
from sglang_omni_router.python.worker import ServiceClass

ROUTE_METADATA_JSON_LIMIT_BYTES = 1024 * 1024
MULTIPART_PART_HEADER_LIMIT_BYTES = 8 * 1024
MULTIPART_FIELD_VALUE_LIMIT_BYTES = 4 * 1024
ROUTE_MODEL_HEADER = "x-sglang-omni-route-model"
ROUTE_STREAM_HEADER = "x-sglang-omni-route-stream"
ROUTE_CAPABILITIES_HEADER = "x-sglang-omni-route-capabilities"
ROUTE_HEADER_NAMES = {
    ROUTE_MODEL_HEADER,
    ROUTE_STREAM_HEADER,
    ROUTE_CAPABILITIES_HEADER,
}

INPUT_FIELD_CAPABILITIES: dict[str, Capability] = {
    "image": "image_input",
    "images": "image_input",
    "audio_inputs": "audio_input",
    "audios": "audio_input",
    "video": "video_input",
    "videos": "video_input",
}
MESSAGE_TYPE_CAPABILITIES: dict[str, Capability] = {
    "image": "image_input",
    "image_url": "image_input",
    "input_image": "image_input",
    "audio": "audio_input",
    "audio_url": "audio_input",
    "input_audio": "audio_input",
    "video": "video_input",
    "video_url": "video_input",
    "input_video": "video_input",
}
OUTPUT_MODALITY_FIELDS = ("modalities", "output_modalities")


class RouteMetadataError(ValueError):
    pass


class RouteKind(str, Enum):
    GENERATION = "generation"
    SPEECH = "speech"
    SPEECH_BATCH = "speech_batch"
    VOICE_CONTROL = "voice_control"
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"


def classify_route(path: str) -> RouteKind:
    if path == "/v1/audio/speech":
        return RouteKind.SPEECH
    if path == "/v1/audio/speech/batch":
        return RouteKind.SPEECH_BATCH
    if path.startswith("/v1/audio/voices"):
        return RouteKind.VOICE_CONTROL
    if path == "/v1/audio/transcriptions":
        return RouteKind.TRANSCRIPTION
    if path == "/v1/audio/translations":
        return RouteKind.TRANSLATION
    return RouteKind.GENERATION


@dataclass
class RouteMetadata:
    request_id: str
    model: str | None
    stream: bool
    required_capabilities: set[Capability]
    is_body_over_metadata_limit: bool
    has_route_model_header: bool
    has_route_capabilities_header: bool
    route_kind: RouteKind
    service_class: ServiceClass
    voice_names_requiring_registry: set[str]


@dataclass(frozen=True)
class SpeechRouteFacts:
    model: str | None
    voice_names_requiring_registry: frozenset[str]
    has_reference_audio: bool


@dataclass
class LargeJsonMetadata:
    request_id: str | None = None
    model: str | None = None
    stream: bool | None = None


def extract_route_metadata(
    request: Request,
    route_kind: RouteKind,
    body: bytes,
) -> RouteMetadata:
    request_id = _request_id_from_request(request)
    route_model, has_route_model_header = _route_model_from_header(request)
    route_stream, has_route_stream_header = _route_stream_from_header(request)
    route_capabilities, has_route_capabilities_header = _route_capabilities_from_header(
        request
    )
    has_json_body = route_kind in {
        RouteKind.SPEECH,
        RouteKind.SPEECH_BATCH,
    } or _is_json_request(request)
    is_body_over_metadata_limit = has_json_body and (
        len(body) > ROUTE_METADATA_JSON_LIMIT_BYTES
    )

    payload: dict[str, Any] | None = None
    large_json_metadata: LargeJsonMetadata | None = None
    if has_json_body and body and not is_body_over_metadata_limit:
        payload = _parse_json_object(body)
    elif is_body_over_metadata_limit:
        large_json_metadata = _scan_large_json_metadata(body)

    speech_facts = (
        extract_speech_route_facts(payload, route_kind)
        if payload is not None
        and route_kind in {RouteKind.SPEECH, RouteKind.SPEECH_BATCH}
        else None
    )
    if payload is not None:
        request_id = request_id or _string_or_none(payload.get("request_id"))
        model = (
            speech_facts.model
            if speech_facts is not None
            else _string_or_none(payload.get("model"))
        )
        stream = payload.get("stream") is True
        required_capabilities = _required_capabilities(
            route_kind,
            payload,
            stream=stream,
            route_capabilities=set(),
            speech_facts=speech_facts,
        )
        _validate_body_route_headers(
            model=model,
            stream=stream,
            required_capabilities=required_capabilities,
            route_model=route_model,
            has_route_model_header=has_route_model_header,
            route_stream=route_stream,
            has_route_stream_header=has_route_stream_header,
            route_capabilities=route_capabilities,
            has_route_capabilities_header=has_route_capabilities_header,
        )
    elif large_json_metadata is not None:
        request_id = request_id or large_json_metadata.request_id
        model = large_json_metadata.model
        stream = large_json_metadata.stream is True
        required_capabilities = _required_capabilities(
            route_kind,
            payload,
            stream=stream,
            route_capabilities=route_capabilities,
            speech_facts=None,
        )
        _validate_body_route_headers(
            model=model,
            stream=stream,
            required_capabilities=required_capabilities,
            route_model=route_model,
            has_route_model_header=has_route_model_header,
            route_stream=route_stream,
            has_route_stream_header=has_route_stream_header,
            route_capabilities=set(),
            has_route_capabilities_header=False,
        )
        if model is None:
            model = route_model
    else:
        model = route_model
        stream = route_stream
        if route_kind in {RouteKind.TRANSCRIPTION, RouteKind.TRANSLATION}:
            form = _multipart_form_facts(request, body)
            if form.model is not None:
                if has_route_model_header and route_model != form.model:
                    raise RouteMetadataError(
                        f"{ROUTE_MODEL_HEADER} conflicts with the multipart "
                        "form model"
                    )
                model = form.model
            if form.stream is not None:
                if has_route_stream_header and route_stream != form.stream:
                    raise RouteMetadataError(
                        f"{ROUTE_STREAM_HEADER} conflicts with the multipart "
                        "form stream"
                    )
                stream = form.stream
        required_capabilities = _required_capabilities(
            route_kind,
            payload,
            stream=stream,
            route_capabilities=route_capabilities,
            speech_facts=None,
        )

    if route_kind is RouteKind.SPEECH_BATCH and stream:
        raise RouteMetadataError("stream is not supported for batch speech requests")

    return RouteMetadata(
        request_id=request_id or str(uuid.uuid4()),
        model=model,
        stream=stream,
        required_capabilities=required_capabilities,
        is_body_over_metadata_limit=is_body_over_metadata_limit,
        has_route_model_header=has_route_model_header,
        has_route_capabilities_header=has_route_capabilities_header,
        route_kind=route_kind,
        service_class=_service_class_for_route(route_kind),
        voice_names_requiring_registry=(
            set(speech_facts.voice_names_requiring_registry)
            if speech_facts is not None
            else set()
        ),
    )


def _request_id_from_request(request: Request) -> str | None:
    return (
        request.headers.get("x-sglang-omni-request-id")
        or request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
    )


def _route_model_from_header(request: Request) -> tuple[str | None, bool]:
    value = request.headers.get(ROUTE_MODEL_HEADER)
    if value is None:
        return None, False
    model = value.strip()
    if not model:
        raise RouteMetadataError(f"{ROUTE_MODEL_HEADER} must not be empty")
    return model, True


def _route_stream_from_header(request: Request) -> tuple[bool, bool]:
    value = request.headers.get(ROUTE_STREAM_HEADER)
    if value is None:
        return False, False
    normalized = value.strip().lower()
    if normalized == "true":
        return True, True
    if normalized == "false":
        return False, True
    raise RouteMetadataError(f"{ROUTE_STREAM_HEADER} must be true or false")


def _route_capabilities_from_header(request: Request) -> tuple[set[Capability], bool]:
    value = request.headers.get(ROUTE_CAPABILITIES_HEADER)
    if value is None:
        return set(), False

    capabilities: set[Capability] = set()
    for item in value.split(","):
        capability = item.strip()
        if not capability:
            continue
        if capability not in DEFAULT_CAPABILITIES:
            raise RouteMetadataError(
                f"{ROUTE_CAPABILITIES_HEADER} contains unsupported capability "
                f"{capability!r}"
            )
        capabilities.add(cast(Capability, capability))
    if not capabilities:
        raise RouteMetadataError(f"{ROUTE_CAPABILITIES_HEADER} must not be empty")
    return capabilities, True


def _validate_body_route_headers(
    *,
    model: str | None,
    stream: bool,
    required_capabilities: set[Capability],
    route_model: str | None,
    has_route_model_header: bool,
    route_stream: bool,
    has_route_stream_header: bool,
    route_capabilities: set[Capability],
    has_route_capabilities_header: bool,
) -> None:
    if has_route_model_header and model is not None and route_model != model:
        raise RouteMetadataError(f"{ROUTE_MODEL_HEADER} conflicts with JSON body model")
    if has_route_stream_header and route_stream != stream:
        raise RouteMetadataError(
            f"{ROUTE_STREAM_HEADER} conflicts with JSON body stream"
        )
    if has_route_capabilities_header and not route_capabilities.issubset(
        required_capabilities
    ):
        raise RouteMetadataError(
            f"{ROUTE_CAPABILITIES_HEADER} conflicts with JSON body"
        )


def _is_json_request(request: Request) -> bool:
    return "json" in request.headers.get("content-type", "").lower()


def _parse_json_object(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except Exception:
        raise RouteMetadataError("invalid JSON body") from None
    if not isinstance(payload, dict):
        raise RouteMetadataError("JSON request body must be an object")
    return payload


def _scan_large_json_metadata(body: bytes) -> LargeJsonMetadata:
    scanner = _JsonTopLevelScanner(body)
    try:
        return scanner.scan_metadata()
    except (IndexError, UnicodeDecodeError, ValueError):
        raise RouteMetadataError("invalid JSON body") from None


class _JsonTopLevelScanner:
    _METADATA_KEYS = {"model", "request_id", "stream"}

    def __init__(self, body: bytes):
        self._body = body
        self._length = len(body)

    def scan_metadata(self) -> LargeJsonMetadata:
        metadata = LargeJsonMetadata()
        index = self._skip_ws(0)
        if index >= self._length or self._body[index] != ord("{"):
            raise ValueError("JSON request body must be an object")
        index += 1

        index = self._skip_ws(index)
        if index < self._length and self._body[index] == ord("}"):
            index = self._skip_ws(index + 1)
            if index != self._length:
                raise ValueError("trailing data")
            return metadata

        while True:
            index = self._skip_ws(index)
            key, index = self._parse_string(index)
            index = self._skip_ws(index)
            if index >= self._length or self._body[index] != ord(":"):
                raise ValueError("missing object separator")
            index = self._skip_ws(index + 1)

            if key in self._METADATA_KEYS:
                index = self._read_metadata_value(metadata, key, index)
            else:
                index = self._skip_value(index)

            index = self._skip_ws(index)
            if index >= self._length:
                raise ValueError("unterminated object")
            byte = self._body[index]
            if byte == ord("}"):
                index = self._skip_ws(index + 1)
                if index != self._length:
                    raise ValueError("trailing data")
                return metadata
            if byte != ord(","):
                raise ValueError("invalid object separator")
            index += 1

    def _read_metadata_value(
        self,
        metadata: LargeJsonMetadata,
        key: str,
        index: int,
    ) -> int:
        if key == "stream":
            if self._body.startswith(b"true", index):
                metadata.stream = True
                return index + 4
            if self._body.startswith(b"false", index):
                metadata.stream = False
                return index + 5
            return self._skip_value(index)

        if index < self._length and self._body[index] == ord('"'):
            value, next_index = self._parse_string(index)
            if value:
                if key == "model":
                    metadata.model = value
                else:
                    metadata.request_id = value
            return next_index
        return self._skip_value(index)

    def _skip_ws(self, index: int) -> int:
        while index < self._length and self._body[index] in b" \t\r\n":
            index += 1
        return index

    def _parse_string(self, index: int) -> tuple[str, int]:
        start = index
        end = self._skip_string(index)
        value = json.loads(self._body[start:end])
        if not isinstance(value, str):
            raise ValueError("expected string")
        return value, end

    def _skip_string(self, index: int) -> int:
        if index >= self._length or self._body[index] != ord('"'):
            raise ValueError("expected string")
        index += 1
        while index < self._length:
            byte = self._body[index]
            if byte == ord('"'):
                return index + 1
            if byte == ord("\\"):
                index += 2
            else:
                if byte < 0x20:
                    raise ValueError("invalid string control character")
                index += 1
        raise ValueError("unterminated string")

    def _skip_value(self, index: int) -> int:
        index = self._skip_ws(index)
        if index >= self._length:
            raise ValueError("missing value")
        byte = self._body[index]
        if byte == ord('"'):
            return self._skip_string(index)
        if byte == ord("{"):
            return self._skip_object(index)
        if byte == ord("["):
            return self._skip_array(index)
        if byte == ord("t") and self._body.startswith(b"true", index):
            return index + 4
        if byte == ord("f") and self._body.startswith(b"false", index):
            return index + 5
        if byte == ord("n") and self._body.startswith(b"null", index):
            return index + 4
        return self._skip_number(index)

    def _skip_object(self, index: int) -> int:
        index += 1
        index = self._skip_ws(index)
        if index < self._length and self._body[index] == ord("}"):
            return index + 1
        while True:
            index = self._skip_string(self._skip_ws(index))
            index = self._skip_ws(index)
            if index >= self._length or self._body[index] != ord(":"):
                raise ValueError("missing object separator")
            index = self._skip_value(index + 1)
            index = self._skip_ws(index)
            if index >= self._length:
                raise ValueError("unterminated object")
            byte = self._body[index]
            if byte == ord("}"):
                return index + 1
            if byte != ord(","):
                raise ValueError("invalid object separator")
            index += 1

    def _skip_array(self, index: int) -> int:
        index += 1
        index = self._skip_ws(index)
        if index < self._length and self._body[index] == ord("]"):
            return index + 1
        while True:
            index = self._skip_value(index)
            index = self._skip_ws(index)
            if index >= self._length:
                raise ValueError("unterminated array")
            byte = self._body[index]
            if byte == ord("]"):
                return index + 1
            if byte != ord(","):
                raise ValueError("invalid array separator")
            index += 1

    def _skip_number(self, index: int) -> int:
        decoder = json.JSONDecoder()
        text = self._body[index : min(self._length, index + 128)].decode("utf-8")
        value, consumed = decoder.raw_decode(text)
        if not isinstance(value, (int, float)):
            raise ValueError("invalid JSON value")
        return index + consumed


@dataclass(frozen=True)
class MultipartFormFacts:
    model: str | None = None
    stream: bool | None = None


_MULTIPART_FORM_FIELDS = frozenset({"model", "stream"})
_FORM_TRUE_VALUES = {"true", "1", "yes", "on"}
_FORM_FALSE_VALUES = {"false", "0", "no", "off"}


def _multipart_form_facts(request: Request, body: bytes) -> MultipartFormFacts:
    if not body:
        return MultipartFormFacts()
    boundary = _multipart_boundary(request)
    if boundary is None:
        return MultipartFormFacts()
    values = _scan_multipart_form_fields(body, boundary)
    return MultipartFormFacts(
        model=values.get("model") or None,
        stream=_form_bool(values.get("stream")),
    )


def _form_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _FORM_TRUE_VALUES:
        return True
    if normalized in _FORM_FALSE_VALUES:
        return False
    return None


def _multipart_boundary(request: Request) -> bytes | None:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type.lower():
        return None
    for param in content_type.split(";"):
        key, _, value = param.strip().partition("=")
        if key.strip().lower() != "boundary":
            continue
        value = value.strip().strip('"')
        try:
            return value.encode("ascii") if value else None
        except UnicodeEncodeError:
            return None
    return None


def _scan_multipart_form_fields(body: bytes, boundary: bytes) -> dict[str, str]:
    delimiter = b"--" + boundary
    values: dict[str, str] = {}
    position = body.find(delimiter)
    if position < 0:
        return values
    position += len(delimiter)
    while True:
        if body.startswith(b"--", position):
            return values  # closing delimiter: the form has no more parts
        if not body.startswith(b"\r\n", position):
            return values
        position += 2
        headers_end = body.find(
            b"\r\n\r\n", position, position + MULTIPART_PART_HEADER_LIMIT_BYTES
        )
        if headers_end < 0:
            return values
        name, has_filename = _content_disposition_name(body[position:headers_end])
        value_start = headers_end + 4
        value_end = body.find(b"\r\n" + delimiter, value_start)
        if value_end < 0:
            return values
        if (
            name in _MULTIPART_FORM_FIELDS
            and name not in values
            and not has_filename
            and value_end - value_start <= MULTIPART_FIELD_VALUE_LIMIT_BYTES
        ):
            try:
                values[name] = body[value_start:value_end].decode("utf-8").strip()
            except UnicodeDecodeError:
                pass
            if _MULTIPART_FORM_FIELDS <= values.keys():
                return values
        position = value_end + 2 + len(delimiter)


def _content_disposition_name(header_block: bytes) -> tuple[str | None, bool]:
    for raw_line in header_block.split(b"\r\n"):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not line.lower().startswith("content-disposition:"):
            continue
        name: str | None = None
        has_filename = False
        for param in line.split(";")[1:]:
            key, _, value = param.strip().partition("=")
            key = key.strip().lower()
            if key == "name":
                name = value.strip().strip('"')
            elif key == "filename":
                has_filename = True
        return name, has_filename
    return None, False


def _required_capabilities(
    route_kind: RouteKind,
    payload: dict[str, Any] | None,
    *,
    stream: bool,
    route_capabilities: set[Capability],
    speech_facts: SpeechRouteFacts | None,
) -> set[Capability]:
    if route_kind in {RouteKind.SPEECH, RouteKind.SPEECH_BATCH}:
        capabilities: set[Capability] = {"speech"}
    elif route_kind is RouteKind.VOICE_CONTROL:
        capabilities = {"speech"}
    elif route_kind in {RouteKind.TRANSCRIPTION, RouteKind.TRANSLATION}:
        capabilities = {"audio_input"}
    else:
        capabilities = {"chat"}

    if stream:
        capabilities.add("streaming")
    capabilities.update(route_capabilities)
    if payload is not None:
        capabilities.update(
            _infer_payload_capabilities(
                route_kind,
                payload,
                speech_facts=speech_facts,
            )
        )
    return capabilities


def _infer_payload_capabilities(
    route_kind: RouteKind,
    payload: dict[str, Any],
    *,
    speech_facts: SpeechRouteFacts | None,
) -> set[Capability]:
    capabilities: set[Capability] = set()
    capabilities.update(_infer_input_field_capabilities(payload))
    if speech_facts is not None and speech_facts.has_reference_audio:
        capabilities.add("audio_input")
    if _modalities_include_audio(payload) or _has_non_empty(payload.get("audio")):
        capabilities.add("audio_output")
    capabilities.update(_infer_message_part_capabilities(payload.get("messages")))
    return capabilities


def _service_class_for_route(route_kind: RouteKind) -> ServiceClass:
    if route_kind is RouteKind.SPEECH:
        return "speech_http"
    if route_kind is RouteKind.SPEECH_BATCH:
        return "speech_batch"
    if route_kind is RouteKind.VOICE_CONTROL:
        return "voice_control"
    if route_kind in {RouteKind.TRANSCRIPTION, RouteKind.TRANSLATION}:
        return "transcription"
    return "generation"


def extract_speech_route_facts(
    payload: dict[str, Any],
    route_kind: RouteKind,
) -> SpeechRouteFacts:
    if route_kind is RouteKind.SPEECH:
        return _speech_route_facts(payload)
    if route_kind is RouteKind.SPEECH_BATCH:
        return _speech_batch_route_facts(payload)
    raise ValueError(f"{route_kind.value} is not a speech route")


def _speech_route_facts(payload: dict[str, Any]) -> SpeechRouteFacts:
    voice_name = _voice_name(payload)
    has_explicit_reference = _has_explicit_speech_reference(payload)
    return SpeechRouteFacts(
        model=_string_or_none(payload.get("model")),
        voice_names_requiring_registry=(
            frozenset({voice_name})
            if voice_name is not None and not has_explicit_reference
            else frozenset()
        ),
        has_reference_audio=_speech_has_reference_audio(payload),
    )


def _speech_batch_route_facts(payload: dict[str, Any]) -> SpeechRouteFacts:
    items = payload.get("items")
    if not isinstance(items, list):
        return _speech_route_facts(payload)

    default_facts = _speech_route_facts(payload)
    models: set[str] = set()
    # The worker validates a named batch default before constructing effective
    # items, so an item-level reference does not suppress default voice lookup.
    voice_names = set(default_facts.voice_names_requiring_registry)
    has_reference_audio = False
    defaults = {key: value for key, value in payload.items() if key != "items"}
    for item in items:
        if not isinstance(item, dict):
            continue
        effective = dict(defaults)
        effective.update(
            {key: value for key, value in item.items() if value is not None}
        )
        item_voice = _voice_name(item)
        if item_voice is not None:
            effective["voice"] = item_voice
        facts = _speech_route_facts(effective)
        if facts.model is not None:
            models.add(facts.model)
        voice_names.update(facts.voice_names_requiring_registry)
        has_reference_audio = has_reference_audio or facts.has_reference_audio

    if len(models) > 1:
        raise RouteMetadataError(
            "speech batch items must resolve to one model for router forwarding"
        )
    model = next(iter(models), _string_or_none(payload.get("model")))
    return SpeechRouteFacts(
        model=model,
        voice_names_requiring_registry=frozenset(voice_names),
        has_reference_audio=has_reference_audio,
    )


def _voice_name(payload: dict[str, Any]) -> str | None:
    value = payload.get("voice", payload.get("speaker"))
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _infer_input_field_capabilities(payload: dict[str, Any]) -> set[Capability]:
    capabilities: set[Capability] = set()
    for field, capability in INPUT_FIELD_CAPABILITIES.items():
        if _has_non_empty(payload.get(field)):
            capabilities.add(capability)
    return capabilities


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _has_non_empty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, list, dict)):
        return bool(value)
    return True


def _modalities_include_audio(payload: dict[str, Any]) -> bool:
    for field in OUTPUT_MODALITY_FIELDS:
        modalities = payload.get(field)
        if isinstance(modalities, list) and any(item == "audio" for item in modalities):
            return True
    return False


def _speech_has_reference_audio(payload: dict[str, Any]) -> bool:
    reference_fields = ("audio_path", "ref_audio", "audio", "data")
    if _has_non_empty(payload.get("ref_audio")):
        return True
    references = payload.get("references")
    if not isinstance(references, list):
        return False
    return any(
        isinstance(reference, dict)
        and any(_has_non_empty(reference.get(field)) for field in reference_fields)
        for reference in references
    )


def _has_explicit_speech_reference(payload: dict[str, Any]) -> bool:
    return payload.get("ref_audio") is not None or bool(payload.get("references"))


def _infer_message_part_capabilities(messages: Any) -> set[Capability]:
    capabilities: set[Capability] = set()
    if not isinstance(messages, list):
        return capabilities
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            capability = MESSAGE_TYPE_CAPABILITIES.get(part_type)
            if capability is not None:
                capabilities.add(capability)
    return capabilities
