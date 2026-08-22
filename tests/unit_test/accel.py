# SPDX-License-Identifier: Apache-2.0
"""Shared accelerator detection for unit tests.

``sglang.srt.utils.get_device()`` raises when the host exposes no accelerator,
which must not break test collection on CPU-only machines.
"""

from sglang.srt.utils import get_device


def get_device_or_cpu() -> str:
    try:
        return get_device()
    except RuntimeError:
        return "cpu"


def has_accelerator() -> bool:
    return get_device_or_cpu().partition(":")[0] in ("cuda", "xpu")
