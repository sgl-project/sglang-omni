# SPDX-License-Identifier: Apache-2.0
"""Encoder runner - pure compute."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Request:
    request_id: str
    data: Any


class EncoderRunner(ABC):
    """Pure compute. No async, no queue."""

    def __init__(self, device: str = "cuda:0", dtype: torch.dtype = torch.float16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model: torch.nn.Module | None = None

    @abstractmethod
    def load(self) -> None:
        """Load model. Must set self.model."""
        ...

    @abstractmethod
    def prepare(self, requests: list[Request]) -> Any:
        """Prepare batch input from requests."""
        ...

    @abstractmethod
    @torch.inference_mode()
    def forward(self, batch: Any) -> Any:
        """Run forward pass."""
        ...

    @abstractmethod
    def unbatch(self, requests: list[Request], output: Any) -> list[Any]:
        """Split output to individual results."""
        ...
