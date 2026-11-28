# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""HTTP endpoint validation, authorization header construction, and retry client for C3."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import JsonValue

RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
HTTP_ATTEMPTS = 4
BACKOFF_SECONDS = 0.75
SAFE_ENV_VAR_NAME_PATTERN = r"^[A-Z_][A-Z0-9_]{0,127}$"
SAFE_ENV_VAR_NAME_RE = re.compile(SAFE_ENV_VAR_NAME_PATTERN)


def validate_http_endpoint_url(base_url: str) -> str:
    try:
        parsed_endpoint_url = urlsplit(base_url)
        is_valid_url = (
            parsed_endpoint_url.scheme in ("http", "https")
            and bool(parsed_endpoint_url.hostname)
            and (
                parsed_endpoint_url.port is None
                or 1 <= parsed_endpoint_url.port <= 65535
            )
            and parsed_endpoint_url.username is None
            and parsed_endpoint_url.password is None
            and not parsed_endpoint_url.query
            and not parsed_endpoint_url.fragment
        )
    except ValueError:
        is_valid_url = False
    if not is_valid_url:
        raise ValueError(
            "Endpoint URL must be an HTTP(S) URL with a valid host/port and no userinfo, query, or fragment"
        )
    else:
        return base_url.rstrip("/")


def sanitize_endpoint_origin(base_url: str) -> str:
    validate_http_endpoint_url(base_url)
    parsed_endpoint_url = urlsplit(base_url)
    endpoint_host = parsed_endpoint_url.hostname or ""
    if ":" in endpoint_host and not endpoint_host.startswith("["):
        endpoint_host = f"[{endpoint_host}]"
    else:
        pass
    if parsed_endpoint_url.port is not None:
        endpoint_host = f"{endpoint_host}:{parsed_endpoint_url.port}"
    else:
        pass
    return urlunsplit((parsed_endpoint_url.scheme, endpoint_host, "", "", ""))


def build_chat_completions_url(base_url: str) -> str:
    cleaned_endpoint_url = validate_http_endpoint_url(base_url)
    if cleaned_endpoint_url.endswith("/chat/completions"):
        return cleaned_endpoint_url
    elif cleaned_endpoint_url.endswith("/v1"):
        return f"{cleaned_endpoint_url}/chat/completions"
    else:
        return f"{cleaned_endpoint_url}/v1/chat/completions"


def compute_endpoint_identity_sha256(base_url: str) -> str:
    canonical_completions_url = build_chat_completions_url(base_url)
    parsed_completions_url = urlsplit(canonical_completions_url)
    normalized_host = (parsed_completions_url.hostname or "").lower()
    if ":" in normalized_host and not normalized_host.startswith("["):
        normalized_host = f"[{normalized_host}]"
    else:
        pass
    normalized_netloc = (
        f"{normalized_host}:{parsed_completions_url.port}"
        if parsed_completions_url.port is not None
        else normalized_host
    )
    canonical_target_url = urlunsplit(
        (
            parsed_completions_url.scheme.lower(),
            normalized_netloc,
            parsed_completions_url.path,
            "",
            "",
        )
    )
    return hashlib.sha256(canonical_target_url.encode("utf-8")).hexdigest()


def resolve_bearer_headers(api_key_env: str | None) -> dict[str, str]:
    request_headers = {"Content-Type": "application/json"}
    if api_key_env is None:
        return request_headers
    elif SAFE_ENV_VAR_NAME_RE.fullmatch(api_key_env) is None:
        raise ValueError(
            "api_key_env must be a valid uppercase environment variable name"
        )
    else:
        raw_api_key = os.environ.get(api_key_env)
        if not raw_api_key:
            raise ValueError(
                f"Required API key environment variable {api_key_env} is not set"
            )
        elif any(
            ord(character) < 33 or ord(character) > 126 or character.isspace()
            for character in raw_api_key
        ):
            raise ValueError(
                f"Environment variable {api_key_env} contains whitespace, non-ASCII, or control characters"
            )
        else:
            request_headers["Authorization"] = f"Bearer {raw_api_key}"
            return request_headers


async def post_chat_completion_with_retries(
    http_client: httpx.AsyncClient,
    *,
    api_url: str,
    request_headers: dict[str, str],
    request_payload: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    last_failure_reason = "unknown transport failure"
    for attempt_index in range(HTTP_ATTEMPTS):
        try:
            http_response = await http_client.post(
                api_url, headers=request_headers, json=request_payload
            )
            if http_response.status_code in RETRYABLE_HTTP_STATUS:
                last_failure_reason = f"HTTP {http_response.status_code}"
            elif http_response.status_code >= 400:
                raise RuntimeError(f"HTTP {http_response.status_code}")
            else:
                try:
                    decoded_json_body = http_response.json()
                except ValueError:
                    raise RuntimeError("Invalid JSON response body") from None
                if not isinstance(decoded_json_body, dict):
                    raise RuntimeError("Non-object JSON response")
                else:
                    return decoded_json_body
        except (httpx.TimeoutException, httpx.TransportError) as transport_error:
            last_failure_reason = (
                "HTTP timeout"
                if isinstance(transport_error, httpx.TimeoutException)
                else type(transport_error).__name__
            )
        if attempt_index + 1 < HTTP_ATTEMPTS:
            await asyncio.sleep(BACKOFF_SECONDS * (attempt_index + 1))
        else:
            pass
    raise RuntimeError(last_failure_reason)
