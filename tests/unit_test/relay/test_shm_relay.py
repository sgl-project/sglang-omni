# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
from multiprocessing import shared_memory

import pytest
import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.relay.shm import ShmRelay
from tests.unit_test.fixtures.pipeline_fakes import make_stage_payload


def test_shm_put_timeout_unlinks_block_and_releases_credit() -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="sender", device="cpu", credits=1)
        tensor = torch.arange(16, dtype=torch.uint8)
        op = await relay.put_async(tensor, request_id="r0")
        shm_name = op.metadata["transfer_info"]["shm_name"]
        shm_path = f"/dev/shm/{shm_name}"

        assert os.path.exists(shm_path)
        with pytest.raises(TimeoutError, match="was not consumed in time"):
            await op.wait_for_completion(timeout=0.0)
        assert not os.path.exists(shm_path)

        # The timeout path released the semaphore credit; another put should not
        # block even though the first transfer failed.
        op2 = await asyncio.wait_for(
            relay.put_async(tensor, request_id="r1"),
            timeout=1.0,
        )
        shm_name2 = op2.metadata["transfer_info"]["shm_name"]
        shm_path2 = f"/dev/shm/{shm_name2}"
        try:
            assert os.path.exists(shm_path2)
        finally:
            with pytest.raises(TimeoutError):
                await op2.wait_for_completion(timeout=0.0)
            assert not os.path.exists(shm_path2)

    asyncio.run(run())


@pytest.mark.parametrize("chunk_bytes", [(0,), (3, 5, 1), (40000, 1, 30000)])
def test_shm_put_chunks_round_trips_their_bytes_in_order(
    chunk_bytes: tuple[int, ...],
) -> None:
    async def scenario() -> None:
        relay = ShmRelay(engine_id="e", device="cpu")
        generator = torch.Generator().manual_seed(0)
        chunks = [
            torch.randint(0, 256, (byte_count,), dtype=torch.uint8, generator=generator)
            for byte_count in chunk_bytes
        ]
        put_operation = await relay.put_chunks_async(chunks, request_id="r0")
        destination = torch.empty(sum(chunk_bytes), dtype=torch.uint8)
        get_operation = await relay.get_async(put_operation.metadata, destination)
        await get_operation.wait_for_completion(timeout=1.0)
        assert torch.equal(destination, torch.cat(chunks))

    asyncio.run(scenario())


def test_shm_put_chunks_that_fail_to_copy_unlink_block_and_release_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_names: list[str] = []

    class RecordingSharedMemory(shared_memory.SharedMemory):
        def __init__(
            self, name: str | None = None, create: bool = False, size: int = 0
        ) -> None:
            super().__init__(name=name, create=create, size=size)
            created_names.append(self.name)

    def failing_cat(chunks: list[torch.Tensor], *, out: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("copy failed")

    async def scenario() -> None:
        relay = ShmRelay(engine_id="sender", device="cpu", credits=1)
        chunks = [torch.arange(16, dtype=torch.uint8)]
        with monkeypatch.context() as patch:
            patch.setattr(shared_memory, "SharedMemory", RecordingSharedMemory)
            patch.setattr(torch, "cat", failing_cat)
            with pytest.raises(RuntimeError, match="copy failed"):
                await relay.put_chunks_async(chunks, request_id="r0")
        assert created_names and not os.path.exists(f"/dev/shm/{created_names[0]}")
        put_operation = await asyncio.wait_for(
            relay.put_chunks_async(chunks, request_id="r1"), 1.0
        )
        with pytest.raises(TimeoutError):
            await put_operation.wait_for_completion(timeout=0.0)

    asyncio.run(scenario())


def test_shm_payload_round_trips_tensors_of_mixed_dtypes() -> None:
    async def scenario() -> None:
        relay = ShmRelay(engine_id="e", device="cpu")
        generator = torch.Generator().manual_seed(0)
        data = {
            "flags": torch.tensor([True, False, True]),
            "features": torch.randn(5, dtype=torch.float32, generator=generator),
            "ids": torch.arange(6, dtype=torch.int64).view(2, 3),
            "embeds": torch.randn(4, 7, dtype=torch.bfloat16, generator=generator).t(),
        }
        payload = make_stage_payload(data=data, request_id="r0")
        data_ref, put_operation = await stage_io.write_payload(
            relay, "r0", payload, transport=TransportKind.SHM
        )
        restored = await stage_io.read_payload(relay, "r0", data_ref)
        for name, tensor in data.items():
            assert torch.equal(restored.data[name], tensor)
        put_operation.mark_receiver_done()
        await put_operation.wait_for_completion(timeout=1.0)

    asyncio.run(scenario())


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
