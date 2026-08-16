# SPDX-License-Identifier: Apache-2.0
"""Resolve the device spec a pipeline stage runs on. Never initializes a device."""

from __future__ import annotations


def _with_index(dev_type: str, raw_index: str, index: int | None) -> str:
    if index is not None:
        idx: int | None = int(index)
    else:
        idx = int(raw_index) if raw_index else None
    if dev_type == "cpu" or idx is None:
        return dev_type
    return f"{dev_type}:{idx}"


def resolve_device_spec(device: str | None, index: int | None = None) -> str:
    """Return the concrete device spec for a stage."""
    from sglang_omni.platforms import current_platform

    platform_type = current_platform.device_type

    if device is None:
        return _with_index(platform_type, "", index)

    dev_type, _, raw_index = str(device).strip().partition(":")
    dev_type = dev_type.lower()
    if dev_type not in ("cpu", platform_type):
        raise ValueError(
            f"device={device!r} names {dev_type!r}, but this host resolved to "
            f"{platform_type!r}. Pass device=None to run on whatever the host "
            f"provides, or 'cpu'/'{platform_type}' explicitly."
        )
    return _with_index(dev_type, raw_index, index)


def place_device_spec(device: str, index: int | None = None) -> str:
    """Apply a placement index to an explicit device spec, keeping its own type.

    The caller named the platform, so honor it rather than validate it: a CUDA-only
    stage keeps CUDA on any host, and an explicit xpu or cpu is never rewritten.
    """
    dev_type, _, raw_index = str(device).strip().partition(":")
    return _with_index(dev_type.lower(), raw_index, index)


__all__ = ["place_device_spec", "resolve_device_spec"]
