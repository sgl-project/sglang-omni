# SPDX-License-Identifier: Apache-2.0
"""Factory entry point for multi-process serving.

Rebuilds the router app in a child process from a RouterConfig serialized by
the supervisor: the config JSON path travels via SGLANG_OMNI_ROUTER_CONFIG_FILE
and the admin key via SGLANG_OMNI_ADMIN_KEY (the existing
resolve_admin_api_key fallback).
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from sglang_omni_router.app import create_app
from sglang_omni_router.config import RouterConfig

CONFIG_FILE_ENV = "SGLANG_OMNI_ROUTER_CONFIG_FILE"


def load_config_from_env() -> RouterConfig:
    path = os.environ.get(CONFIG_FILE_ENV)
    if not path:
        raise RuntimeError(
            f"{CONFIG_FILE_ENV} is not set; this factory entry point is only "
            "meant to be spawned by the router supervisor"
        )
    with open(path, encoding="utf-8") as f:
        return RouterConfig.model_validate_json(f.read())


def create_app_from_env() -> FastAPI:
    return create_app(load_config_from_env())
