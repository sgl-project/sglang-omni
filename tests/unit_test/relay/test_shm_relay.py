# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from multiprocessing import shared_memory

import pytest
import torch

from sglang_omni.relay.shm import ShmRelay


def _shm_exists(name: str) -> bool:
    try:
        handle = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    handle.close()
    return True


def test_shm_put_timeout_unlinks_block_and_releases_credit() -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="sender", device="cpu", credits=1)
        tensor = torch.arange(16, dtype=torch.uint8)
        op = await relay.put_async(tensor, request_id="r0")
        shm_name = op.metadata["transfer_info"]["shm_name"]
        assert _shm_exists(shm_name)
        with pytest.raises(TimeoutError, match="was not consumed in time"):
            await op.wait_for_completion(timeout=0.0)
        assert not _shm_exists(shm_name)

        # The timeout path released the semaphore credit; another put should not
        # block even though the first transfer failed.
        op2 = await asyncio.wait_for(
            relay.put_async(tensor, request_id="r1"),
            timeout=1.0,
        )
        shm_name2 = op2.metadata["transfer_info"]["shm_name"]
        try:
            assert _shm_exists(shm_name2)
        finally:
            with pytest.raises(TimeoutError):
                await op2.wait_for_completion(timeout=0.0)
            assert not _shm_exists(shm_name2)

    asyncio.run(run())


def test_shm_raw_tensor_roundtrip_ignores_allocation_padding(monkeypatch) -> None:
    from sglang_omni.comm.stage_io import read_tensor, write_tensor

    original = shared_memory.SharedMemory

    def page_rounded_memory(*args, **kwargs):
        if kwargs.get("create"):
            kwargs["size"] = ((kwargs["size"] + 16383) // 16384) * 16384
        return original(*args, **kwargs)

    monkeypatch.setattr(shared_memory, "SharedMemory", page_rounded_memory)

    async def run() -> None:
        relay = ShmRelay(engine_id="prompt-capture", device="cpu", credits=1)
        tensor = torch.arange(73 * 2048, dtype=torch.float32).reshape(73, 2048)
        data_ref, put = await write_tensor(
            relay, tensor=tensor, object_id="prompt", transport="shm"
        )
        try:
            received = await read_tensor(relay, data_ref)
            assert received.shape == tensor.shape
            assert torch.equal(received, tensor)
            assert (
                put.metadata["transfer_info"]["size"]
                == tensor.numel() * tensor.element_size()
            )
        finally:
            put.mark_receiver_done()
            await put.wait_for_completion()

    asyncio.run(run())
