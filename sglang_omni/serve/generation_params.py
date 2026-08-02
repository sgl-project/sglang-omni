# SPDX-License-Identifier: Apache-2.0
"""Helpers for preserving explicit generation parameter metadata."""

from __future__ import annotations

from typing import Any

from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY


def record_explicit_generation_params(
    metadata: dict[str, Any],
    explicit_fields: list[str],
) -> None:
    if explicit_fields:
        metadata[EXPLICIT_GENERATION_PARAMS_KEY] = explicit_fields
