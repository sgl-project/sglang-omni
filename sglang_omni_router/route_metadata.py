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
    idempotency_key_present: bool
    body_exceeds_metadata_limit: bool
    route_model_header_present: bool
    route_capabilities_header_present: bool
    route_stream_header_present: bool


def extract_route_metadata(request: Request, path: str, body: bytes) -> RouteMetadata:
    request_id = _request_id_from_request(request)
    model, route_model_header_present = _route_model_from_header(request)
    stream, route_stream_header_present = _route_stream_from_header(request)
    route_capabilities, route_capabilities_header_present = (
        _route_capabilities_from_header(request)
    )
    body_exceeds_metadata_limit = _is_json_request(request) and (
        len(body) > ROUTE_METADATA_JSON_LIMIT_BYTES
    )

    payload: dict[str, Any] | None = None
    if _is_json_request(request) and body and not body_exceeds_metadata_limit:
        payload = _parse_json_object(body)

    if payload is not None:
        request_id = request_id or _string_or_none(payload.get("request_id"))
        if not route_model_header_present:
            model = _string_or_none(payload.get("model"))
        if not route_stream_header_present:
            stream = payload.get("stream") is True

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
        idempotency_key_present=bool(request.headers.get("idempotency-key")),
        body_exceeds_metadata_limit=body_exceeds_metadata_limit,
        route_model_header_present=route_model_header_present,
        route_capabilities_header_present=route_capabilities_header_present,
        route_stream_header_present=route_stream_header_present,
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
        capabilities.update(_infer_payload_capabilities(payload))
    return capabilities


def _infer_payload_capabilities(payload: dict[str, Any]) -> set[Capability]:
    capabilities: set[Capability] = set()
    for field, capability in INPUT_FIELD_CAPABILITIES.items():
        if _has_non_empty(payload.get(field)):
            capabilities.add(capability)
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
