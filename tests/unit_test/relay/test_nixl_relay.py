# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from sglang_omni.relay.nixl import GetOperation, PutOperation


class _FakeNixl:
    def __init__(self, *, states: list[str] | None = None) -> None:
        self.notifications: list[dict[str, list[bytes]]] = []
        self.states = list(states or [])
        self.released_handles: list[int] = []

    def get_new_notifs(self):
        return self.notifications.pop(0) if self.notifications else {}

    def check_xfer_state(self, handle: int) -> str:
        assert handle == 7
        return self.states.pop(0) if self.states else "PROC"

    def release_xfer_handle(self, handle: int) -> None:
        self.released_handles.append(handle)


def test_nixl_put_releases_credit_only_after_matching_notification() -> None:
    api = _FakeNixl()
    api.notifications = [{"peer": [b"other"]}, {"peer": [b"request-1"]}]
    released: list[bool] = []
    operation = PutOperation(
        SimpleNamespace(_nixl=api),
        metadata={},
        expected_notification=b"request-1",
        on_completion_cb=lambda: released.append(True),
    )

    asyncio.run(operation.wait_for_completion(timeout=1))

    assert released == [True]


def test_nixl_put_timeout_keeps_sender_memory_pinned() -> None:
    released: list[bool] = []
    operation = PutOperation(
        SimpleNamespace(_nixl=_FakeNixl()),
        metadata={},
        expected_notification=b"request-1",
        on_completion_cb=lambda: released.append(True),
    )

    with pytest.raises(TimeoutError, match="remains pinned"):
        asyncio.run(operation.wait_for_completion(timeout=0))

    assert released == []


def test_nixl_get_timeout_keeps_handle_and_credit_pinned() -> None:
    import torch

    api = _FakeNixl(states=["PROC"])
    released: list[bool] = []
    operation = GetOperation(
        SimpleNamespace(_nixl=api),
        handle=7,
        src_pool_tensor=torch.zeros(4, dtype=torch.uint8),
        dest_tensor=torch.zeros(4, dtype=torch.uint8),
        copy_size=4,
        on_completion_cb=lambda: released.append(True),
    )

    with pytest.raises(TimeoutError, match="remains pinned"):
        asyncio.run(operation.wait_for_completion(timeout=0))

    assert api.released_handles == []
    assert released == []


def test_nixl_get_success_copies_and_releases() -> None:
    import torch

    api = _FakeNixl(states=["PROC", "DONE"])
    released: list[bool] = []
    destination = torch.zeros(4, dtype=torch.uint8)
    operation = GetOperation(
        SimpleNamespace(_nixl=api),
        handle=7,
        src_pool_tensor=torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
        dest_tensor=destination,
        copy_size=4,
        on_completion_cb=lambda: released.append(True),
    )

    asyncio.run(operation.wait_for_completion(timeout=1))

    assert destination.tolist() == [1, 2, 3, 4]
    assert api.released_handles == [7]
    assert released == [True]
