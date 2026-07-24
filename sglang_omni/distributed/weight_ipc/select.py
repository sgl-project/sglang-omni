# SPDX-License-Identifier: Apache-2.0
"""Share policies for which tensors are exported via weight IPC."""

from __future__ import annotations

from typing import Protocol

import torch


class SharePolicy(Protocol):
    def select(self, model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
        pass


class ArParametersPolicy:
    """Share all CUDA ``named_parameters()``; buffers stay private by default."""

    def __init__(self, buffer_whitelist: frozenset[str] | None = None) -> None:
        self.buffer_whitelist = buffer_whitelist or frozenset()

    def select(self, model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
        selected: list[tuple[str, torch.Tensor]] = []
        for name, param in model.named_parameters():
            if param is None or not param.is_cuda:
                continue
            selected.append((name, param))
        if self.buffer_whitelist:
            for name, buf in model.named_buffers():
                if name not in self.buffer_whitelist:
                    continue
                if buf is None or not buf.is_cuda:
                    raise ValueError(
                        f"whitelisted buffer {name!r} is missing or not on CUDA"
                    )
                selected.append((name, buf))
        return selected
