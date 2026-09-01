# SPDX-License-Identifier: Apache-2.0
"""Shared OpenAI-compatible API error classification helpers."""

from __future__ import annotations

import re

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
)
_BAD_REQUEST_PATTERNS = (
    re.compile(r"^Request\s+\S+\s+exceeds the maximum number of tokens:"),
    re.compile(r"^Request\s+\S+\s+requires too many SWA KV tokens for"),
)


def is_bad_request_error(exc: BaseException) -> bool:
    message = str(exc)
    return any(marker in message for marker in _BAD_REQUEST_MARKERS) or any(
        pattern.search(message) is not None for pattern in _BAD_REQUEST_PATTERNS
    )
