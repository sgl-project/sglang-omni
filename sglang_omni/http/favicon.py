# SPDX-License-Identifier: Apache-2.0
"""Serve the project favicon for FastAPI apps."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAVICON_PATH = _REPO_ROOT / "docs" / "_static" / "image" / "logo.ico"
_FAVICON_MEDIA_TYPE = "image/vnd.microsoft.icon"


def resolve_favicon_path(path: Path | None = None) -> Path | None:
    """Return an on-disk favicon path, or None if none is available."""
    candidate = path or DEFAULT_FAVICON_PATH
    return candidate if candidate.is_file() else None


def register_favicon(app: FastAPI, *, favicon_path: Path | None = None) -> None:
    """Register GET /favicon.ico using the SGLang-Omni logo when present."""
    resolved = resolve_favicon_path(favicon_path)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        if resolved is None:
            return Response(status_code=404)
        return FileResponse(resolved, media_type=_FAVICON_MEDIA_TYPE)
