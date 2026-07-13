# SPDX-License-Identifier: Apache-2.0
"""Runtime policy for externalizing tensors on one pipeline edge."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorRefPolicy:
    threshold_bytes: int
    consumer_stage: str
    paths: tuple[str, ...]

    def should_externalize(self, path: str, nbytes: int) -> bool:
        matches_path = any(
            path == configured_path or path.startswith(f"{configured_path}[")
            for configured_path in self.paths
        )
        return matches_path and nbytes >= self.threshold_bytes
