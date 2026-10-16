# SPDX-License-Identifier: Apache-2.0
"""Two-process device-to-device round trips for the Level-Zero IPC relay."""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import traceback
from types import SimpleNamespace

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

from sglang_omni.comm.data_ref import (  # noqa: E402 - must follow the XPU init
    TransportKind,
)
from sglang_omni.comm.router import CommRouter  # noqa: E402 - must follow the XPU init
from sglang_omni.config.schema import (  # noqa: E402 - must follow the XPU init
    CommConfig,
)
from sglang_omni.pipeline.runtime_config import (  # noqa: E402 - must follow the XPU init
    build_comm_config,
)
from sglang_omni.relay.level_zero_ipc import (  # noqa: E402 - must follow the XPU init
    LevelZeroIpcRelay,
)
from tests.unit_test._util.level_zero_pair import (  # noqa: E402 - must follow the XPU init
    run_pair,
    skip_without_level_zero_ipc,
)

PAYLOAD_SIZES = (1024 * 1024, 64 * 1024, 3 * 1024 * 1024)
SLOT_SIZE_KB = 64
POOL_SIZE_MB = 8
MESSAGE_TIMEOUT_S = 60
RESULT_TIMEOUT_S = 120


def make_expected(num_bytes: int) -> torch.Tensor:
    return (torch.arange(num_bytes, dtype=torch.int64) % 251).to(torch.uint8)


def level_zero_sender(
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
            for num_bytes in payload_sizes:
                payload = make_expected(num_bytes).to(f"xpu:{source_gpu}")
                operation = await relay.put_async(
                    payload,
                    request_id=f"r{num_bytes}",
                    receiver_id="receiver",
                )
                metadata_queue.put(operation.metadata)
                ack_status, ack_value = ack_queue.get(timeout=MESSAGE_TIMEOUT_S)
                if ack_status == "ok":
                    operation.mark_receiver_done()
                elif ack_status == "err":
                    operation.mark_receiver_failed(RuntimeError(ack_value))
                else:
                    raise RuntimeError(
                        f"receiver returned invalid ACK status {ack_status!r}"
                    )
                await operation.wait_for_completion()

        asyncio.run(run())
        result_queue.put(("sender", "ok", True))
    except Exception:
        result_queue.put(("sender", "err", traceback.format_exc()))
    finally:
        if relay is not None:
            relay.close()
        else:
            pass


def level_zero_receiver(
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
        mapped_pools: set[int] = set()

        async def receive(num_bytes: int) -> None:
            metadata = metadata_queue.get(timeout=MESSAGE_TIMEOUT_S)
            size = metadata["transfer_info"]["size"]
            if size != num_bytes:
                raise AssertionError(f"expected {num_bytes} bytes, got {size}")
            else:
                pass
            destination = torch.zeros(
                size, dtype=torch.uint8, device=f"xpu:{destination_gpu}"
            )
            operation = await relay.get_async(
                metadata, destination, request_id=f"r{num_bytes}"
            )
            await operation.wait_for_completion()
            if not torch.equal(destination.cpu(), make_expected(size)):
                raise AssertionError(
                    f"received {size} Level-Zero IPC bytes do not match the source"
                )
            else:
                pass
            mapped_pools.add(len(relay.peer_pools.mappings))

        async def run() -> None:
            for num_bytes in payload_sizes:
                try:
                    await receive(num_bytes)
                except Exception as error:
                    ack_queue.put(("err", repr(error)))
                    raise
                ack_queue.put(("ok", None))

        asyncio.run(run())
        # Every payload came from one sender pool, so the mapping is opened once.
        if mapped_pools != {1}:
            raise AssertionError(f"expected one cached peer pool, saw {mapped_pools}")
        else:
            pass
        result_queue.put(("receiver", "ok", True))
    except Exception:
        result_queue.put(("receiver", "err", traceback.format_exc()))
    finally:
        if relay is not None:
            relay.close()
        else:
            pass


def run_case(
    source_gpu: int,
    destination_gpu: int,
    *,
    payload_sizes: tuple[int, ...] = PAYLOAD_SIZES,
    pool_size_mb: int = POOL_SIZE_MB,
    sender_env: dict[str, str] | None = None,
) -> None:
    run_pair(
        level_zero_sender,
        level_zero_receiver,
        source_gpu,
        destination_gpu,
        payload_sizes=payload_sizes,
        pool_size_mb=pool_size_mb,
        result_timeout_s=RESULT_TIMEOUT_S,
        sender_env=sender_env,
    )


@pytest.mark.accelerator
def test_level_zero_ipc_same_gpu_round_trip() -> None:
    skip_without_level_zero_ipc()
    run_case(0, 0)


@pytest.mark.accelerator
def test_level_zero_ipc_cross_gpu_round_trip() -> None:
    skip_without_level_zero_ipc()
    if torch.xpu.device_count() < 2:
        pytest.skip("requires two Intel GPUs")
    else:
        pass
    run_case(0, 1)


@pytest.mark.accelerator
def test_level_zero_ipc_pool_does_not_follow_the_senders_allocator_config() -> None:
    """Expandable segments map torch memory in 20 MiB chunks. A pool carved from
    torch exported only the first chunk, so a payload reaching past it crashed the
    receiver."""
    skip_without_level_zero_ipc()
    run_case(
        0,
        0,
        payload_sizes=(24 << 20,),
        pool_size_mb=32,
        sender_env={"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
    )


@pytest.mark.accelerator
def test_level_zero_ipc_reaches_another_card_from_a_sender_that_sees_one() -> None:
    """A process that sees one card gets compressed device memory unless the
    allocation asks for export, and another card cannot read compressed memory."""
    skip_without_level_zero_ipc()
    if torch.xpu.device_count() < 2:
        pytest.skip("requires two Intel GPUs")
    else:
        pass
    # A mask lists its cards ascending, so the lowest is xpu:0 here and in the sender.
    mask = os.environ.get("ZE_AFFINITY_MASK", "").strip()
    first_card = (
        min(int(entry.split(".")[0]) for entry in mask.split(",")) if mask else 0
    )
    run_case(0, 1, sender_env={"ZE_AFFINITY_MASK": str(first_card)})


@pytest.mark.accelerator
def test_a_configured_pool_size_reaches_the_relay() -> None:
    skip_without_level_zero_ipc()
    stage = SimpleNamespace(
        comm=CommConfig(level_zero_ipc_slot_size_kb=32, level_zero_ipc_pool_size_mb=4)
    )
    router = CommRouter(
        stage_name="talker",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"vocoder"},
        comm_config=build_comm_config(stage),
    )
    try:
        relay = router.relay(TransportKind.LEVEL_ZERO_IPC)
        assert relay.slot_size == 32 * 1024
        assert relay.pool_size == 4 * 1024 * 1024
    finally:
        router.close()
