# SPDX-License-Identifier: Apache-2.0
"""Lightweight scheduler message types shared across scheduling backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class IncomingMessage:
    request_id: str
    type: Literal["new_request", "stream_chunk", "stream_done"]
    data: Any = None


@dataclass
class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream", "error"]
    data: Any = None
    target: str | None = None
    metadata: dict[str, Any] | None = None
