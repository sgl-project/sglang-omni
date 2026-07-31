# SPDX-License-Identifier: Apache-2.0
"""Scheduler environment variable parsing helpers."""

from __future__ import annotations

import os


def env_float(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None
