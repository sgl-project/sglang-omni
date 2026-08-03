# SPDX-License-Identifier: Apache-2.0
"""Shared test doubles."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace


class FakeExecutionBridge:
    """SGLangExecutionBridge double for scheduler-owned ModelRunner tests."""

    def __init__(self) -> None:
        self.published: list[tuple[object, object]] = []
        self.isolate_sampling_calls: list[bool] = []

    @contextlib.contextmanager
    def forward_context(self, batch: object, *, isolate_sampling: bool = False):
        del batch
        self.isolate_sampling_calls.append(isolate_sampling)
        yield

    def publish_next_tokens(self, batch: object, next_token_ids: object) -> None:
        self.published.append((batch, next_token_ids))

    def record_completion(self):
        import torch

        return torch.cuda.Event()


class FakeServerArgs(SimpleNamespace):
    """ServerArgs double exposing the 0.5.16 override() mutation entry point."""

    def override(self, source: str, **fields: object) -> None:
        del source
        for name, value in fields.items():
            setattr(self, name, value)
