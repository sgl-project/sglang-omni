"""Signature-aware helpers for SGLang compatibility call sites."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def supported_kwargs(factory: Callable, **kwargs: Any) -> dict[str, Any]:
    """Return only keyword arguments declared by ``factory``."""
    parameters = inspect.signature(factory).parameters
    return {name: value for name, value in kwargs.items() if name in parameters}


__all__ = ["supported_kwargs"]
