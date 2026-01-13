# SPDX-License-Identifier: Apache-2.0
"""Shared memory metadata."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SHMMetadata:
    """Metadata for shared memory segment."""

    name: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SHMMetadata":
        return cls(name=d["name"], size=d["size"])
