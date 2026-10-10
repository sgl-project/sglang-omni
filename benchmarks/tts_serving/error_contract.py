# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible error response validation."""

from __future__ import annotations

import json

EXPECTED_ERROR_TYPES = {
    400: "BadRequestError",
    404: "NotFoundError",
}


def is_openai_error_response(
    body: str,
    *,
    expected_status: int,
    expected_error_type: str | None = None,
    expected_error_code: int | str | None = None,
) -> bool:
    if not body.strip() or body.lstrip().startswith("<"):
        return False
    expected_error_type = expected_error_type or EXPECTED_ERROR_TYPES.get(
        expected_status
    )
    if expected_error_type is None:
        return False
    if expected_error_code is None:
        expected_error_code = expected_status
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    message = error.get("message")
    if not isinstance(message, str) or not message.strip():
        return False
    if error.get("type") != expected_error_type:
        return False
    if error.get("code") != expected_error_code:
        return False
    if "param" not in error:
        return False
    param = error.get("param")
    return param is None or isinstance(param, str)
