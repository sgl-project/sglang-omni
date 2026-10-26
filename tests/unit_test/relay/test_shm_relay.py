# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from contextlib import closing, suppress
from multiprocessing.shared_memory import SharedMemory
from typing import Literal

import pytest
import torch

from sglang_omni.relay.shm import ShmPutOperation, ShmRelay


def assert_shared_memory_unlinked(shared_memory_name: str) -> None:
    with (
        pytest.raises(FileNotFoundError),
        closing(SharedMemory(name=shared_memory_name)),
    ):
        pass


def test_shm_put_timeout_unlinks_block_and_releases_credit() -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="sender", device="cpu", credits=1)
        tensor = torch.arange(16, dtype=torch.uint8)
        op = await relay.put_async(tensor, request_id="r0")
        shm_name = op.metadata["transfer_info"]["shm_name"]
        with closing(SharedMemory(name=shm_name)) as shared_memory_block:
            try:
                with pytest.raises(TimeoutError, match="was not consumed in time"):
                    await op.wait_for_completion(timeout=0.0)
                assert_shared_memory_unlinked(shm_name)
            finally:
                with suppress(FileNotFoundError):
                    shared_memory_block.unlink()

        # The timeout path released the semaphore credit; another put should not
        # block even though the first transfer failed.
        op2 = await asyncio.wait_for(
            relay.put_async(tensor, request_id="r1"),
            timeout=1.0,
        )
        shm_name2 = op2.metadata["transfer_info"]["shm_name"]
        with closing(SharedMemory(name=shm_name2)) as shared_memory_block:
            try:
                with pytest.raises(TimeoutError):
                    await op2.wait_for_completion(timeout=0.0)
                assert_shared_memory_unlinked(shm_name2)
            finally:
                with suppress(FileNotFoundError):
                    shared_memory_block.unlink()

    asyncio.run(run())


@pytest.mark.parametrize("receiver_cleanup", ["pending", "consumed", "during_unlink"])
def test_shm_put_cancellation_unlinks_block_and_releases_credit_once(
    receiver_cleanup: Literal["pending", "consumed", "during_unlink"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        released_credits = 0

        def release_credit() -> None:
            nonlocal released_credits
            released_credits += 1

        original_unlink = SharedMemory.unlink

        def receiver_unlinks_first(shared_memory_block: SharedMemory) -> None:
            original_unlink(shared_memory_block)
            original_unlink(shared_memory_block)

        payload_bytes = bytes(range(16))
        sender_block = SharedMemory(create=True, size=len(payload_bytes))
        with closing(SharedMemory(name=sender_block.name)) as receiver_block:
            try:
                sender_block.buf[: len(payload_bytes)] = payload_bytes
                operation = ShmPutOperation(
                    metadata={},
                    shm_obj=sender_block,
                    shm_name=sender_block.name,
                    release_cb=release_credit,
                )
                if receiver_cleanup == "consumed":
                    receiver_block.unlink()
                elif receiver_cleanup == "during_unlink":
                    monkeypatch.setattr(SharedMemory, "unlink", receiver_unlinks_first)
                else:
                    pass

                completion_task = asyncio.create_task(operation.wait_for_completion())
                await asyncio.sleep(0)
                completion_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await completion_task

                assert_shared_memory_unlinked(sender_block.name)
                assert bytes(receiver_block.buf[: len(payload_bytes)]) == payload_bytes
                assert released_credits == 1

                operation.mark_receiver_done()
                await operation.wait_for_completion()
                assert released_credits == 1
            finally:
                sender_block.close()
                with suppress(FileNotFoundError):
                    original_unlink(receiver_block)

    asyncio.run(run())


def test_shm_round_trips_zero_byte_tensor() -> None:
    async def scenario() -> None:
        relay = ShmRelay(engine_id="e", device="cpu")
        put = await relay.put_async(torch.empty(0, dtype=torch.long), request_id="r0")
        assert put.metadata["transfer_info"]["size"] == 0
        dest = torch.empty(0, dtype=torch.long)
        get = await relay.get_async(put.metadata, dest)
        await get.wait_for_completion(timeout=1.0)
        assert dest.numel() == 0

    asyncio.run(scenario())
