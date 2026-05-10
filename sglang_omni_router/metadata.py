# SPDX-License-Identifier: Apache-2.0
"""Minimal route metadata extraction for raw Omni request bodies."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from fastapi import Request

from sglang_omni_router.config import Capability

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


@dataclass
class _RouteFields:
    request_id: str | None = None
    model: str | None = None
    stream: bool = False
    capabilities: set[Capability] | None = None

    def __post_init__(self) -> None:
        if self.capabilities is None:
            self.capabilities = set()


def extract_route_metadata(request: Request, path: str, body: bytes) -> RouteMetadata:
    request_id = (
        request.headers.get("x-sglang-omni-request-id")
        or request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
    )
    fields = _RouteFields(request_id=request_id)

    content_type = request.headers.get("content-type", "").lower()
    if "json" in content_type and body:
        try:
            fields = _JsonRouteScanner(body, fields).scan()
        except _JsonRouteSyntaxError as exc:
            raise RouteMetadataError(str(exc)) from None

    return RouteMetadata(
        request_id=fields.request_id or str(uuid.uuid4()),
        model=fields.model,
        stream=fields.stream,
        required_capabilities=_required_capabilities(path, fields),
        idempotency_key_present=bool(request.headers.get("idempotency-key")),
    )


def _required_capabilities(path: str, fields: _RouteFields) -> set[Capability]:
    if path == "/v1/audio/speech":
        capabilities: set[Capability] = {"speech"}
        if fields.stream:
            capabilities.add("streaming")
        return capabilities

    capabilities = {"chat"}
    if fields.stream:
        capabilities.add("streaming")
    capabilities.update(fields.capabilities or set())
    return capabilities


class _JsonRouteSyntaxError(ValueError):
    pass


class _JsonRouteScanner:
    def __init__(self, body: bytes, fields: _RouteFields) -> None:
        self._body = body
        self._length = len(body)
        self._fields = fields
        self._pos = 0

    def scan(self) -> _RouteFields:
        self._skip_ws()
        if self._peek() != ord("{"):
            raise _JsonRouteSyntaxError("JSON request body must be an object")
        self._scan_top_level_object()
        self._skip_ws()
        if self._pos != self._length:
            raise _JsonRouteSyntaxError("invalid JSON body")
        return self._fields

    def _scan_top_level_object(self) -> None:
        self._expect(ord("{"))
        self._skip_ws()
        if self._consume_if(ord("}")):
            return

        while True:
            key = self._read_string()
            self._skip_ws()
            self._expect(ord(":"))
            self._skip_ws()
            self._scan_top_level_value(key)
            self._skip_ws()
            if self._consume_if(ord("}")):
                return
            self._expect(ord(","))
            self._skip_ws()

    def _scan_top_level_value(self, key: str) -> None:
        if key == "request_id":
            value = self._read_optional_string()
            if value:
                self._fields.request_id = self._fields.request_id or value
            return
        if key == "model":
            self._fields.model = self._read_optional_string()
            return
        if key == "stream":
            self._fields.stream = self._read_optional_true()
            return
        if key in INPUT_FIELD_CAPABILITIES:
            if self._value_is_non_empty():
                self._fields.capabilities.add(INPUT_FIELD_CAPABILITIES[key])
            return
        if key == "audio":
            if self._value_is_non_empty():
                self._fields.capabilities.add("audio_output")
            return
        if key == "modalities":
            if self._array_includes_string("audio"):
                self._fields.capabilities.add("audio_output")
            return
        if key == "messages":
            self._scan_messages()
            return
        self._skip_value()

    def _scan_messages(self) -> None:
        if self._peek() != ord("["):
            self._skip_value()
            return
        self._expect(ord("["))
        self._skip_ws()
        if self._consume_if(ord("]")):
            return

        while True:
            if self._peek() == ord("{"):
                self._scan_message_object()
            else:
                self._skip_value()
            self._skip_ws()
            if self._consume_if(ord("]")):
                return
            self._expect(ord(","))
            self._skip_ws()

    def _scan_message_object(self) -> None:
        self._expect(ord("{"))
        self._skip_ws()
        if self._consume_if(ord("}")):
            return

        while True:
            key = self._read_string()
            self._skip_ws()
            self._expect(ord(":"))
            self._skip_ws()
            if key == "content":
                self._scan_message_content()
            else:
                self._skip_value()
            self._skip_ws()
            if self._consume_if(ord("}")):
                return
            self._expect(ord(","))
            self._skip_ws()

    def _scan_message_content(self) -> None:
        if self._peek() != ord("["):
            self._skip_value()
            return
        self._expect(ord("["))
        self._skip_ws()
        if self._consume_if(ord("]")):
            return

        while True:
            if self._peek() == ord("{"):
                self._scan_message_part()
            else:
                self._skip_value()
            self._skip_ws()
            if self._consume_if(ord("]")):
                return
            self._expect(ord(","))
            self._skip_ws()

    def _scan_message_part(self) -> None:
        self._expect(ord("{"))
        self._skip_ws()
        if self._consume_if(ord("}")):
            return

        while True:
            key = self._read_string()
            self._skip_ws()
            self._expect(ord(":"))
            self._skip_ws()
            if key == "type":
                part_type = self._read_optional_string()
                capability = MESSAGE_TYPE_CAPABILITIES.get(part_type or "")
                if capability is not None:
                    self._fields.capabilities.add(capability)
            else:
                self._skip_value()
            self._skip_ws()
            if self._consume_if(ord("}")):
                return
            self._expect(ord(","))
            self._skip_ws()

    def _array_includes_string(self, target: str) -> bool:
        if self._peek() != ord("["):
            self._skip_value()
            return False
        found = False
        self._expect(ord("["))
        self._skip_ws()
        if self._consume_if(ord("]")):
            return False

        while True:
            if self._peek() == ord('"'):
                found = self._read_string() == target or found
            else:
                self._skip_value()
            self._skip_ws()
            if self._consume_if(ord("]")):
                return found
            self._expect(ord(","))
            self._skip_ws()

    def _value_is_non_empty(self) -> bool:
        token = self._peek()
        if token == ord('"'):
            start = self._pos
            end = self._scan_string_end()
            self._pos = end
            return end > start + 2
        if token == ord("["):
            return self._container_is_non_empty_and_skip(ord("["), ord("]"))
        if token == ord("{"):
            return self._container_is_non_empty_and_skip(ord("{"), ord("}"))
        if self._match_literal(b"false") or self._match_literal(b"null"):
            return False
        self._skip_value()
        return True

    def _container_is_non_empty_and_skip(self, open_char: int, close_char: int) -> bool:
        start = self._pos
        self._expect(open_char)
        self._skip_ws()
        non_empty = self._peek() != close_char
        self._pos = start
        self._skip_value()
        return non_empty

    def _read_optional_string(self) -> str | None:
        if self._peek() != ord('"'):
            self._skip_value()
            return None
        return self._read_string()

    def _read_optional_true(self) -> bool:
        if self._match_literal(b"true"):
            return True
        self._skip_value()
        return False

    def _skip_value(self) -> None:
        token = self._peek()
        if token == ord('"'):
            self._pos = self._scan_string_end()
            return
        if token == ord("{"):
            self._skip_object()
            return
        if token == ord("["):
            self._skip_array()
            return
        if self._match_literal(b"true"):
            return
        if self._match_literal(b"false"):
            return
        if self._match_literal(b"null"):
            return
        self._skip_number()

    def _skip_object(self) -> None:
        self._expect(ord("{"))
        self._skip_ws()
        if self._consume_if(ord("}")):
            return

        while True:
            self._read_string()
            self._skip_ws()
            self._expect(ord(":"))
            self._skip_ws()
            self._skip_value()
            self._skip_ws()
            if self._consume_if(ord("}")):
                return
            self._expect(ord(","))
            self._skip_ws()

    def _skip_array(self) -> None:
        self._expect(ord("["))
        self._skip_ws()
        if self._consume_if(ord("]")):
            return

        while True:
            self._skip_value()
            self._skip_ws()
            if self._consume_if(ord("]")):
                return
            self._expect(ord(","))
            self._skip_ws()

    def _skip_number(self) -> None:
        start = self._pos

        if self._consume_if(ord("-")):
            pass

        if self._consume_if(ord("0")):
            pass
        elif self._pos < self._length and ord("1") <= self._body[self._pos] <= ord("9"):
            self._pos += 1
            while self._pos < self._length and ord("0") <= self._body[self._pos] <= ord(
                "9"
            ):
                self._pos += 1
        else:
            raise _JsonRouteSyntaxError("invalid JSON body")

        if self._consume_if(ord(".")):
            if not (
                self._pos < self._length
                and ord("0") <= self._body[self._pos] <= ord("9")
            ):
                raise _JsonRouteSyntaxError("invalid JSON body")
            while self._pos < self._length and ord("0") <= self._body[self._pos] <= ord(
                "9"
            ):
                self._pos += 1

        if self._pos < self._length and self._body[self._pos] in b"eE":
            self._pos += 1
            if self._pos < self._length and self._body[self._pos] in b"+-":
                self._pos += 1
            if not (
                self._pos < self._length
                and ord("0") <= self._body[self._pos] <= ord("9")
            ):
                raise _JsonRouteSyntaxError("invalid JSON body")
            while self._pos < self._length and ord("0") <= self._body[self._pos] <= ord(
                "9"
            ):
                self._pos += 1

        if self._pos == start:
            raise _JsonRouteSyntaxError("invalid JSON body")

    def _read_string(self) -> str:
        start = self._pos
        end = self._scan_string_end()
        raw = self._body[start:end]
        self._pos = end
        try:
            value = json.loads(raw)
        except Exception:
            raise _JsonRouteSyntaxError("invalid JSON body") from None
        if not isinstance(value, str):
            raise _JsonRouteSyntaxError("invalid JSON body")
        return value

    def _scan_string_end(self) -> int:
        if self._peek() != ord('"'):
            raise _JsonRouteSyntaxError("invalid JSON body")
        pos = self._pos + 1
        while pos < self._length:
            char = self._body[pos]
            if char == ord("\\"):
                pos += 1
                if pos >= self._length:
                    raise _JsonRouteSyntaxError("invalid JSON body")
                escape = self._body[pos]
                if escape in b'"\\/bfnrt':
                    pos += 1
                    continue
                if escape == ord("u"):
                    end = pos + 5
                    if end > self._length or not all(
                        _is_hex_digit(self._body[index])
                        for index in range(pos + 1, end)
                    ):
                        raise _JsonRouteSyntaxError("invalid JSON body")
                    pos = end
                    continue
                raise _JsonRouteSyntaxError("invalid JSON body")
            if char == ord('"'):
                return pos + 1
            if char < 0x20:
                raise _JsonRouteSyntaxError("invalid JSON body")
            pos += 1
        raise _JsonRouteSyntaxError("invalid JSON body")

    def _match_literal(self, literal: bytes) -> bool:
        end = self._pos + len(literal)
        if self._body[self._pos : end] == literal:
            self._pos = end
            return True
        return False

    def _skip_ws(self) -> None:
        while self._pos < self._length and self._body[self._pos] in b" \t\r\n":
            self._pos += 1

    def _peek(self) -> int:
        if self._pos >= self._length:
            raise _JsonRouteSyntaxError("invalid JSON body")
        return self._body[self._pos]

    def _expect(self, char: int) -> None:
        if self._peek() != char:
            raise _JsonRouteSyntaxError("invalid JSON body")
        self._pos += 1

    def _consume_if(self, char: int) -> bool:
        if self._pos < self._length and self._body[self._pos] == char:
            self._pos += 1
            return True
        return False


def _is_hex_digit(char: int) -> bool:
    return (
        ord("0") <= char <= ord("9")
        or ord("a") <= char <= ord("f")
        or ord("A") <= char <= ord("F")
    )
