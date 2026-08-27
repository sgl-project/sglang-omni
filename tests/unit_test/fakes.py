# SPDX-License-Identifier: Apache-2.0
"""Shared test doubles."""

from __future__ import annotations

import contextlib


class FakeExecutionBridge:
    """SGLangExecutionBridge double for scheduler-owned ModelRunner tests."""

    def __init__(self, device: object | None = None) -> None:
        import torch

        self.published: list[tuple[object, object]] = []
        self.isolate_sampling_calls: list[bool] = []
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.device_module = torch.get_device_module(self.device)

    @contextlib.contextmanager
    def forward_context(self, batch: object, *, isolate_sampling: bool = False):
        del batch
        self.isolate_sampling_calls.append(isolate_sampling)
        yield

    def publish_next_tokens(self, batch: object, next_token_ids: object) -> None:
        self.published.append((batch, next_token_ids))

    def record_completion(self):
        return self.device_module.Event()
