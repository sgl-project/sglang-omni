# SPDX-License-Identifier: Apache-2.0
"""Retarget hardcoded device specs at the resolved platform.

Device selection belongs to ``sglang_omni.platforms.current_platform``, which
auto-detects the accelerator. This module only rewrites the literal ``"cuda"``
specs the pipeline configs carry so they land on whatever platform resolved.
Import-safe: never initializes a device.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_ACCEL_TYPES = ("cuda", "xpu")


def _accel_available(dev_type: str) -> bool:
    """Is the accelerator type importable and reporting a device? Never raises."""
    mod = getattr(torch, dev_type, None)
    if mod is None:
        return False
    try:
        return bool(mod.is_available())
    except Exception:  # noqa: BLE001 — probes must not crash callers
        return False


def remap_accelerator_spec(spec: str, index: int | None = None) -> str:
    """Retarget an accelerator spec at the live backend, honoring the caller.

    Stage factories default ``device`` to a literal ``"cuda"``/``"cuda:0"`` (and
    the pipeline configs pass that literal), which does not exist on an XPU-only
    build. Rewrite only that case -- an accelerator type this host does not have
    -- so an explicitly requested accelerator, index included, is preserved.
    ``index`` overrides the index when the caller supplies one (e.g. a stage's
    ``gpu_id``). A ``"cpu"`` request stays on cpu and drops any index, since a
    cpu stage is not bound to a card.

    The spec is split by hand rather than through ``torch.device``, which rejects
    a type this torch build does not know at all (e.g. ``"npu:0"``) -- the very
    case this needs to retarget.
    """
    from sglang_omni.platforms import current_platform

    dev_type, _, raw_index = str(spec).strip().partition(":")
    dev_type = dev_type.lower()
    if dev_type != "cpu" and (
        dev_type not in _ACCEL_TYPES or not _accel_available(dev_type)
    ):
        # Either an accelerator this build has no concept of, or one it knows but
        # cannot reach; only the resolved platform can say what to use instead.
        dev_type = current_platform.device_type
    if index is not None:
        idx: int | None = int(index)
    else:
        idx = int(raw_index) if raw_index else None
    if dev_type == "cpu" or idx is None:
        return dev_type
    return f"{dev_type}:{idx}"


__all__ = ["remap_accelerator_spec"]
