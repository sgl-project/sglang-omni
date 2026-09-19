# SPDX-License-Identifier: Apache-2.0
"""Shared NVML lifecycle and device-query helpers."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def nvml_session(pynvml: Any) -> Generator[Any, None, None]:
    """Balance each successful initialization; cleanup must not mask query errors."""
    pynvml.nvmlInit()
    try:
        yield pynvml
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            logger.debug("NVML shutdown failed", exc_info=True)


def try_import_pynvml() -> Any | None:
    try:
        return importlib.import_module("pynvml")
    except ModuleNotFoundError:
        return None


def get_device_handle(pynvml: Any, device_id: int | str) -> Any:
    if isinstance(device_id, int):
        return pynvml.nvmlDeviceGetHandleByIndex(device_id)

    get_by_uuid = pynvml.nvmlDeviceGetHandleByUUID
    try:
        return get_by_uuid(device_id)
    except TypeError:
        return get_by_uuid(device_id.encode("utf-8"))


def decode_nvml_string(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
