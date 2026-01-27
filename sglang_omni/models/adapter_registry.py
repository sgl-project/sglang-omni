# SPDX-License-Identifier: Apache-2.0
"""Process-local registry for omni adapters."""

from __future__ import annotations

from threading import Lock
from typing import Any

from sglang_omni.models.omni_adapter import OmniAdapter
from sglang_omni.proto import StagePayload

_ADAPTERS: dict[str, OmniAdapter] = {}
_LOCK = Lock()


def register_adapter(adapter: OmniAdapter) -> OmniAdapter:
    """Register an adapter once per process."""

    with _LOCK:
        cached = _ADAPTERS.get(adapter.name)
        if cached is not None:
            return cached
        _ADAPTERS[adapter.name] = adapter
        return adapter


def get_adapter(name: str) -> OmniAdapter:
    """Fetch a previously registered adapter."""

    with _LOCK:
        adapter = _ADAPTERS.get(name)
    if adapter is None:
        raise KeyError(f"Adapter not registered: {name}")
    return adapter


def get_adapter_from_payload(payload: StagePayload) -> OmniAdapter:
    """Resolve adapter using payload metadata."""

    data = payload.data if isinstance(payload.data, dict) else {}
    adapter_name = data.get("adapter_name")
    if not isinstance(adapter_name, str) or not adapter_name:
        raise KeyError("Payload missing adapter_name")
    return get_adapter(adapter_name)


def describe_adapters() -> dict[str, Any]:
    with _LOCK:
        names = sorted(_ADAPTERS.keys())
    return {"count": len(names), "names": names}
