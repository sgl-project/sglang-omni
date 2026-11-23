# SPDX-License-Identifier: Apache-2.0
"""A CUDA prompt hidden store whose pinned rows are released when its block ends."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace

import torch

from sglang_omni.scheduling.sglang_backend.prompt_hidden import PromptHiddenStore


@contextlib.contextmanager
def cuda_prompt_hidden_store(
    *, slot_count: int, req_to_token: torch.Tensor
) -> Iterator[PromptHiddenStore]:
    store = PromptHiddenStore(
        slot_count=slot_count,
        hidden_size=2,
        dtype=torch.float32,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        device="cuda",
    )
    try:
        yield store
    finally:
        store.destroy()
