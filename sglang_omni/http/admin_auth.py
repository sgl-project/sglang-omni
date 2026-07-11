# SPDX-License-Identifier: Apache-2.0
"""Shared FastAPI authentication helpers for Omni admin endpoints."""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException

ADMIN_API_KEY_ENV = "SGLANG_OMNI_ADMIN_KEY"


def resolve_admin_api_key(admin_api_key: str | None = None) -> str | None:
    return admin_api_key or os.environ.get(ADMIN_API_KEY_ENV) or None


def make_admin_auth_dependency(admin_api_key: str | None):
    if not admin_api_key:

        async def _no_auth() -> None:
            return

        return _no_auth

    async def _check_admin_key(
        authorization: str | None = Header(default=None),
    ) -> None:
        token = _extract_bearer_token(authorization)
        if token is None:
            raise HTTPException(
                status_code=401,
                detail="Admin API key required: Authorization: Bearer <key>",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not hmac.compare_digest(
            token.encode("utf-8"),
            admin_api_key.encode("utf-8"),
        ):
            raise HTTPException(status_code=403, detail="Invalid admin API key")

    return _check_admin_key


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        return None
    return parts[1]
