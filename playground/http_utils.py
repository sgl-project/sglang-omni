# SPDX-License-Identifier: Apache-2.0
"""Shared HTTP helpers for FastAPI playground apps."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from sglang_omni.http.favicon import register_favicon


def register_playground_favicon(
    app: FastAPI,
    *,
    frontend_dir: Path,
) -> None:
    """Register GET /favicon.ico from ``frontend_dir/favicon.ico`` when present."""
    local = frontend_dir / "favicon.ico"
    register_favicon(app, favicon_path=local if local.is_file() else None)
