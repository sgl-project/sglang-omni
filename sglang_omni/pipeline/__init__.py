# SPDX-License-Identifier: Apache-2.0
"""Pipeline package exports with lazy loading.

This avoids importing ``pipeline.stage.runtime`` (and therefore the scheduler
stack) when callers only need lightweight helpers such as ``Coordinator`` or
input-handler types.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "Coordinator": ("sglang_omni.pipeline.coordinator", "Coordinator"),
    "AggregatedInput": ("sglang_omni.pipeline.stage.input", "AggregatedInput"),
    "DirectInput": ("sglang_omni.pipeline.stage.input", "DirectInput"),
    "InputHandler": ("sglang_omni.pipeline.stage.input", "InputHandler"),
    "Stage": ("sglang_omni.pipeline.stage.runtime", "Stage"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
