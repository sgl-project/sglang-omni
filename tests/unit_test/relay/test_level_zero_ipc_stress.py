# SPDX-License-Identifier: Apache-2.0
"""Many concurrent Level-Zero IPC transfers through one small pool, digest-checked."""
from __future__ import annotations

import asyncio
import functools
import hashlib
import multiprocessing as mp
import queue
import random
import time
import traceback

import pytest
import torch

# Before the imports below, which can map a foreign loader; workers run it too.
if torch.version.xpu is not None:
    try:
        torch.xpu.init()
    except RuntimeError:
        # Collection still has to succeed; the tests below skip themselves.
        pass
else:
    pass

from sglang_omni.relay.level_zero_ipc import (  # noqa: E402 - must follow the XPU init
    LevelZeroIpcRelay,
)
from sglang_omni.relay.level_zero_ipc_metadata import (  # noqa: E402 - must follow the XPU init
    LevelZeroIpcMetadata,
    PeerPoolReference,
)
from sglang_omni.relay.level_zero_ipc_operations import (  # noqa: E402 - must follow the XPU init
    LevelZeroIpcPutOperation,
)
from sglang_omni.relay.level_zero_ipc_peers import (  # noqa: E402 - must follow the XPU init
    PeerPool,
)
from tests.unit_test._util.level_zero_pair import (  # noqa: E402 - must follow the XPU init
    run_pair,
    skip_without_level_zero_ipc,
)

TRANSFER_COUNT = 300
SLOT_SIZE_KB = 64
# Holds only a handful of transfers, so slots wrap around and free out of order.
POOL_SIZE_MB = 4
LARGEST_TRANSFER_BYTES = 3 << 19
SENDS_IN_FLIGHT = 16
RECEIVE_BATCH_LIMIT = 32
# Long enough for several transfers to be published before the pool is first mapped.
FIRST_BATCH_WAIT_S = 0.5
# Lets later batches gather a few transfers, so their ACKs can go out of order.
LATER_BATCH_WAIT_S = 0.02
MESSAGE_TIMEOUT_S = 60
STRESS_TIMEOUT_S = 300


def transfer_sizes() -> tuple[int, ...]:
    sizes = random.Random(2802)
    slot_bytes = SLOT_SIZE_KB * 1024
    edges = (1, slot_bytes - 1, slot_bytes, slot_bytes + 1, LARGEST_TRANSFER_BYTES)
    return edges + tuple(
        sizes.randint(1, LARGEST_TRANSFER_BYTES)
        for _ in range(TRANSFER_COUNT - len(edges))
    )


def payload_for(index: int, num_bytes: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(index)
    return torch.randint(0, 256, (num_bytes,), dtype=torch.uint8, generator=generator)


def digest_of(host_bytes: torch.Tensor) -> str:
    return hashlib.sha256(host_bytes.numpy().tobytes()).hexdigest()


def stress_sender(
    source_gpu: int,
    payload_sizes: tuple[int, ...],
    pool_size_mb: int,
    metadata_queue: mp.Queue,
    ack_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    relay = None
    try:
        torch.xpu.set_device(source_gpu)
        relay = LevelZeroIpcRelay(
            engine_id="sender",
            device=f"xpu:{source_gpu}",
            slot_size_kb=SLOT_SIZE_KB,
            pool_size_mb=pool_size_mb,
        )

        async def run() -> None:
            loop = asyncio.get_running_loop()
            in_flight = asyncio.Semaphore(SENDS_IN_FLIGHT)
            operations: dict[int, LevelZeroIpcPutOperation] = {}

            async def send(index: int, num_bytes: int) -> None:
                async with in_flight:
                    payload = payload_for(index, num_bytes)
                    operation = await relay.put_async(
                        payload.to(f"xpu:{source_gpu}"),
                        request_id=f"r{index}",
                        receiver_id="receiver",
                    )
                    operations[index] = operation
                    metadata_queue.put((index, digest_of(payload), operation.metadata))

            async def settle_in_ack_order() -> None:
                next_ack = functools.partial(ack_queue.get, True, MESSAGE_TIMEOUT_S)
                for _ in payload_sizes:
                    index, status = await loop.run_in_executor(None, next_ack)
                    operation = operations.pop(index)
                    if status == "ok":
                        operation.mark_receiver_done()
                    else:
                        operation.mark_receiver_failed(RuntimeError(status))
                    await operation.wait_for_completion()

            settling = asyncio.create_task(settle_in_ack_order())
            await asyncio.gather(
                *(send(index, size) for index, size in enumerate(payload_sizes))
            )
            await settling
            if relay.allocator.free_slots != relay.allocator.slot_count:
                raise AssertionError(
                    f"{relay.allocator.slot_count - relay.allocator.free_slots} "
                    f"slots stayed reserved after every ACK"
                )
            else:
                pass

        asyncio.run(run())
        result_queue.put(("sender", "ok", True))
    except Exception:
        result_queue.put(("sender", "err", traceback.format_exc()))
    finally:
        if relay is not None:
            relay.close()
        else:
            pass


def stress_receiver(
    destination_gpu: int,
    payload_sizes: tuple[int, ...],
    pool_size_mb: int,
    metadata_queue: mp.Queue,
    ack_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    relay = None
    try:
        torch.xpu.set_device(destination_gpu)
        relay = LevelZeroIpcRelay(
            engine_id="receiver",
            device=f"xpu:{destination_gpu}",
            slot_size_kb=SLOT_SIZE_KB,
            pool_size_mb=pool_size_mb,
        )
        ack_order = random.Random(15)
        map_peer_pool = relay.peer_pools.map_peer_pool
        pools_mapped: list[str] = []

        def count_mapping(
            reference: PeerPoolReference, destination_device_index: int
        ) -> PeerPool:
            pools_mapped.append(reference.pool_id)
            return map_peer_pool(reference, destination_device_index)

        relay.peer_pools.map_peer_pool = count_mapping

        async def receive(
            index: int, digest: str, metadata: LevelZeroIpcMetadata
        ) -> int:
            num_bytes = metadata["transfer_info"]["size"]
            destination = torch.empty(
                num_bytes, dtype=torch.uint8, device=f"xpu:{destination_gpu}"
            )
            operation = await relay.get_async(
                metadata, destination, request_id=f"r{index}"
            )
            await operation.wait_for_completion()
            if digest_of(destination.cpu()) != digest:
                raise AssertionError(
                    f"transfer {index} of {num_bytes} bytes is corrupt"
                )
            else:
                pass
            return index

        async def run() -> None:
            received = 0
            batch_wait_s = FIRST_BATCH_WAIT_S
            while received < len(payload_sizes):
                batch = [metadata_queue.get(timeout=MESSAGE_TIMEOUT_S)]
                deadline = time.monotonic() + batch_wait_s
                while len(batch) < RECEIVE_BATCH_LIMIT and received + len(batch) < len(
                    payload_sizes
                ):
                    try:
                        batch.append(
                            metadata_queue.get(
                                timeout=max(0.0, deadline - time.monotonic())
                            )
                        )
                    except queue.Empty:
                        break
                batch_wait_s = LATER_BATCH_WAIT_S
                done = list(await asyncio.gather(*(receive(*item) for item in batch)))
                ack_order.shuffle(done)
                for index in done:
                    ack_queue.put((index, "ok"))
                received += len(batch)
            # Reads that raced for the first mapping must have shared one.
            if len(pools_mapped) != 1:
                raise AssertionError(
                    f"one sender pool was mapped {len(pools_mapped)} times"
                )
            else:
                pass

        asyncio.run(run())
        result_queue.put(("receiver", "ok", True))
    except Exception:
        result_queue.put(("receiver", "err", traceback.format_exc()))
    finally:
        if relay is not None:
            relay.close()
        else:
            pass


@pytest.mark.accelerator
@pytest.mark.parametrize(
    ("source_gpu", "destination_gpu"), [(0, 0), (0, 1)], ids=["same-card", "cross-card"]
)
def test_concurrent_transfers_wrap_the_pool_and_all_arrive_intact(
    source_gpu: int, destination_gpu: int
) -> None:
    """Concurrent puts, out-of-order ACKs and racing first maps, as at concurrency 4."""
    skip_without_level_zero_ipc()
    if destination_gpu >= torch.xpu.device_count():
        pytest.skip("requires two Intel GPUs")
    else:
        pass
    run_pair(
        stress_sender,
        stress_receiver,
        source_gpu,
        destination_gpu,
        payload_sizes=transfer_sizes(),
        pool_size_mb=POOL_SIZE_MB,
        result_timeout_s=STRESS_TIMEOUT_S,
    )
