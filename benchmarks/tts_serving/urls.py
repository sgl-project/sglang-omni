# SPDX-License-Identifier: Apache-2.0
"""URL helpers for benchmark API paths."""

from __future__ import annotations


def api_url(base_url: str, path: str) -> str:
    """Join a benchmark API path with either a root or /v1 base URL."""
    base = base_url.rstrip("/")
    normalized_path = "/" + path.lstrip("/")
    if base.endswith("/v1") and normalized_path.startswith("/v1/"):
        normalized_path = normalized_path[len("/v1") :]
    return f"{base}{normalized_path}"


def websocket_url(base_url: str, path: str) -> str:
    url = api_url(base_url, path)
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url
