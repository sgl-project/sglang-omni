# SPDX-License-Identifier: Apache-2.0
"""Route metadata extraction for Omni router worker selection."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, cast

from fastapi import Request

from sglang_omni_router.config import DEFAULT_CAPABILITIES, Capability

ROUTE_METADATA_JSON_LIMIT_BYTES = 1024 * 1024
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


class RouteMetadataError(ValueError):
    pass


@dataclass
class RouteMetadata:
    request_id: str
    model: str | None
    stream: bool
    required_capabilities: set[Capability]
    body_exceeds_metadata_limit: bool
    route_capabilities_header_present: bool


@dataclass
class LargeJsonMetadata:
    request_id: str | None = None
    model: str | None = None
    stream: bool | None = None


def extract_route_metadata(request: Request, path: str, body: bytes) -> RouteMetadata:
    request_id = _request_id_from_request(request)
    route_model, route_model_header_present = _route_model_from_header(request)
    route_stream, route_stream_header_present = _route_stream_from_header(request)
    route_capabilities, route_capabilities_header_present = (
        _route_capabilities_from_header(request)
    )
    body_exceeds_metadata_limit = _is_json_request(request) and (
        len(body) > ROUTE_METADATA_JSON_LIMIT_BYTES
    )

    payload: dict[str, Any] | None = None
    large_json_metadata: LargeJsonMetadata | None = None
    if _is_json_request(request) and body and not body_exceeds_metadata_limit:
        payload = _parse_json_object(body)
    elif body_exceeds_metadata_limit:
        large_json_metadata = _scan_large_json_metadata(body)

    if payload is not None:
        request_id = request_id or _string_or_none(payload.get("request_id"))
        model = _string_or_none(payload.get("model"))
        stream = payload.get("stream") is True
        required_capabilities = _required_capabilities(
            path,
            payload,
            stream=stream,
            route_capabilities=set(),
        )
        _validate_body_route_headers(
            model=model,
            stream=stream,
            required_capabilities=required_capabilities,
            route_model=route_model,
            route_model_header_present=route_model_header_present,
            route_stream=route_stream,
            route_stream_header_present=route_stream_header_present,
            route_capabilities=route_capabilities,
            route_capabilities_header_present=route_capabilities_header_present,
        )
    elif large_json_metadata is not None:
        request_id = request_id or large_json_metadata.request_id
        model = large_json_metadata.model
        stream = large_json_metadata.stream is True
        required_capabilities = _required_capabilities(
            path,
            payload,
            stream=stream,
            route_capabilities=route_capabilities,
        )
        _validate_body_route_headers(
            model=model,
            stream=stream,
            required_capabilities=required_capabilities,
            route_model=route_model,
            route_model_header_present=route_model_header_present,
            route_stream=route_stream,
            route_stream_header_present=route_stream_header_present,
            route_capabilities=set(),
            route_capabilities_header_present=False,
        )
        if model is None:
            model = route_model
    else:
        model = route_model
        stream = route_stream
        required_capabilities = _required_capabilities(
            path,
            payload,
            stream=stream,
            route_capabilities=route_capabilities,
        )

    return RouteMetadata(
        request_id=request_id or str(uuid.uuid4()),
        model=model,
        stream=stream,
        required_capabilities=required_capabilities,
        body_exceeds_metadata_limit=body_exceeds_metadata_limit,
        route_capabilities_header_present=route_capabilities_header_present,
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
    route_model_header_present: bool,
    route_stream: bool,
    route_stream_header_present: bool,
    route_capabilities: set[Capability],
    route_capabilities_header_present: bool,
) -> None:
    if route_model_header_present and model is not None and route_model != model:
        raise RouteMetadataError(f"{ROUTE_MODEL_HEADER} conflicts with JSON body model")
    if route_stream_header_present and route_stream != stream:
        raise RouteMetadataError(
            f"{ROUTE_STREAM_HEADER} conflicts with JSON body stream"
        )
    if route_capabilities_header_present and not route_capabilities.issubset(
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


def _required_capabilities(
    path: str,
    payload: dict[str, Any] | None,
    *,
    stream: bool,
    route_capabilities: set[Capability],
) -> set[Capability]:
    if path == "/v1/audio/speech":
        capabilities: set[Capability] = {"speech"}
    else:
        capabilities = {"chat"}

    if stream:
        capabilities.add("streaming")
    capabilities.update(route_capabilities)
    if payload is not None:
        capabilities.update(_infer_payload_capabilities(path, payload))
    return capabilities


def _infer_payload_capabilities(
    path: str,
    payload: dict[str, Any],
) -> set[Capability]:
    capabilities: set[Capability] = set()
    for field, capability in INPUT_FIELD_CAPABILITIES.items():
        if _has_non_empty(payload.get(field)):
            capabilities.add(capability)
    if path == "/v1/audio/speech" and _speech_uses_reference_audio(payload):
        capabilities.add("audio_input")
    if _modalities_include_audio(payload) or _has_non_empty(payload.get("audio")):
        capabilities.add("audio_output")
    capabilities.update(_infer_message_part_capabilities(payload.get("messages")))
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
    modalities = payload.get("modalities")
    if not isinstance(modalities, list):
        return False
    return any(item == "audio" for item in modalities)


def _speech_uses_reference_audio(payload: dict[str, Any]) -> bool:
    if _has_non_empty(payload.get("ref_audio")):
        return True
    references = payload.get("references")
    if not isinstance(references, list):
        return False
    return any(
        isinstance(reference, dict) and _has_non_empty(reference.get("audio_path"))
        for reference in references
    )


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
