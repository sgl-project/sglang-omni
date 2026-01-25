# SPDX-License-Identifier: Apache-2.0
"""Routing helpers for the Qwen3-Omni split pipeline."""

from __future__ import annotations

from typing import Any


def frontend_next(request_id: str, output: Any) -> list[str]:
    del request_id, output
    return ["image_encoder", "audio_encoder", "mm_aggregate"]


def image_next(request_id: str, output: Any) -> str:
    del request_id, output
    return "mm_aggregate"


def audio_next(request_id: str, output: Any) -> str:
    del request_id, output
    return "mm_aggregate"


def aggregate_next(request_id: str, output: Any) -> str:
    del request_id, output
    return "thinker"


def thinker_next(request_id: str, output: Any) -> str:
    del request_id, output
    return "decode"


def decode_next(request_id: str, output: Any) -> None:
    del request_id, output
    return None

