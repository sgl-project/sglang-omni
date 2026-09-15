# SPDX-License-Identifier: Apache-2.0
"""Shared OpenAI-compatible API error classification helpers."""

from __future__ import annotations

import re

from sglang_omni.admission import InvalidRequestError, QueueFullError

_BAD_REQUEST_MARKERS = (
    "Unsupported language:",
    "longer than the model's context length",
    "Requested token count exceeds the model's maximum context length",
    "Request requires more tokens than the thinker KV cache can hold",
    "accepts audio up to",
    "could not decode the uploaded audio",
    "max_new_tokens must be",
    "exceeds the maximum allowed length",
    "sequence exceeds max_length",
    "multimodal_train_inputs",
    "disallowed special token",
    "stop strings are allowed",
    "stop_regex patterns are allowed",
    "AuK speech requires",
    "AuK gen_seconds must be",
    "AuK seed must be",
    "AuK requires a natural-language",
    "AuK accepts at most one",
    "AuK expected a",
    "AuK references must be",
)
_BAD_REQUEST_PATTERNS = (
    re.compile(
        r"\bAuK (?:nfe|cfg_strength|sway_sampling_coef|max_seconds) is a server-level setting"
    ),
    re.compile(r"^Request\s+\S+\s+exceeds the maximum number of tokens:"),
    re.compile(r"^Request\s+\S+\s+requires too many SWA KV tokens for"),
    re.compile(r"^stop_regex is \d+ bytes, over the \d+-byte limit"),
)


def is_bad_request_error(exc: BaseException) -> bool:
    if InvalidRequestError.matches(exc):
        return True
    message = str(exc)
    return any(marker in message for marker in _BAD_REQUEST_MARKERS) or any(
        pattern.search(message) is not None for pattern in _BAD_REQUEST_PATTERNS
    )


def http_status_from_error(exc: BaseException) -> int:
    """Preserve admission errors across the string-only coordinator IPC."""
    if QueueFullError.matches(exc):
        return 503
    if re.fullmatch(r"Request .+ already exists", str(exc), flags=re.DOTALL):
        return 409
    return 400 if is_bad_request_error(exc) else 500
