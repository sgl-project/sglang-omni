# SPDX-License-Identifier: Apache-2.0
"""Model adapter base class."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelAdapter:
    """Thin model configuration. Cases use ``name`` and ``api_endpoint`` to
    construct requests; ``default_params`` supplies model-specific defaults."""

    name: str = ""
    api_endpoint: str = ""
    default_params: dict = field(default_factory=dict)
