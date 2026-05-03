# SPDX-License-Identifier: Apache-2.0
"""Multiprocess tests for v1 unified relay implementations."""

import asyncio
import multiprocessing
import os
import pickle
import time
import traceback
from queue import Empty

import numpy as np
import pytest
import torch

if torch.cuda.is_available():
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import sglang_omni_v1.relay.nccl  # noqa: F401
import sglang_omni_v1.relay.nixl  # noqa: F401
import sglang_omni_v1.relay.shm  # noqa: F401
from sglang_omni_v1.relay.base import create_relay
from sglang_omni_v1.utils import find_available_port


def sender_process(
    config, meta_queue, num_transfers, data_size, results, init_barrier=None
):
    relay_type = config.get("relay_type", "nixl")
    gpu_id = config.get("gpu_id")
    worker_id = config.get("worker_id", "sender_worker")
    device = (
        f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu"
    )

    try:
        if relay_type == "nccl":
            if init_barrier is not None:
                init_barrier.wait(timeout=60)
            os.environ.setdefault("MASTER_ADDR", config.get("master_addr", "127.0.0.1"))
            os.environ.setdefault("MASTER_PORT", config.get("master_port", "29500"))

        relay_kwargs = {
            "engine_id": config.get("worker_id", "sender_worker"),
            "device": device,
            "credits": 4,
            "slot_size_mb": 1,
            **config,
        }
        relay_kwargs.pop("relay_type", None)
        connector = create_relay(relay_type, **relay_kwargs)
    except Exception as e:
        results["sender_error"] = f"Init failed: {e}\n{traceback.format_exc()}"
        return

    tensor_device = connector.device if hasattr(connector, "device") else device

    async def _async_sender():
        try:
            test_tensor = torch.randn(
                data_size, dtype=torch.bfloat16, device=tensor_device
            )
            max_buffer_size = len(pickle.dumps(test_tensor)) + 4096
            transport_tensor = torch.zeros(
                max_buffer_size, dtype=torch.uint8, device=tensor_device
            )

            for i in range(num_transfers):
                data_tensor = torch.randn(
                    data_size, dtype=torch.bfloat16, device=tensor_device
                )
                original = data_tensor.cpu().clone()
                serialized_data = pickle.dumps(data_tensor)
                data_len = len(serialized_data)
                if data_len > max_buffer_size:
                    raise ValueError(
                        f"Data size {data_len} exceeds buffer {max_buffer_size}"
                    )

                data_np = np.frombuffer(serialized_data, dtype=np.uint8).copy()
                transport_tensor[:data_len].copy_(torch.from_numpy(data_np))
                req_id = f"{worker_id}:req_{i}"
                readable_op = await connector.put_async(
                    transport_tensor[:data_len],
                    request_id=req_id,
                    dst_rank=config.get("dst_rank", 1),
                )
                metadata = readable_op.metadata
                if callable(metadata):
                    metadata = metadata()
                if not isinstance(metadata, dict):
                    metadata = {
                        "engine_id": getattr(metadata, "engine_id", None),
                        "agent_meta": getattr(metadata, "agent_meta", None),
                        "transfer_info": getattr(metadata, "transfer_info", None),
                        "descriptors": getattr(metadata, "descriptors", None),
                    }
                meta_queue.put(
                    {"metadata": metadata, "original": pickle.dumps(original)}
                )
                await readable_op.wait_for_completion()
                if hasattr(connector, "cleanup"):
                    connector.cleanup(req_id)
            meta_queue.put(None)
        except Exception as e:
            results["sender_error"] = str(e)
            results["sender_traceback"] = traceback.format_exc()

    try:
        asyncio.run(_async_sender())
    finally:
        if "connector" in locals() and hasattr(connector, "close"):
            connector.close()


def receiver_process(
    config,
    meta_queue,
    num_transfers,
    results,
    init_barrier=None,
    expected_senders=1,
):
    relay_type = config.get("relay_type", "nixl")
    gpu_id = config.get("gpu_id")
    device = (
        f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu"
    )

    try:
        if relay_type == "nccl":
            if init_barrier is not None:
                init_barrier.wait(timeout=60)
            os.environ.setdefault("MASTER_ADDR", config.get("master_addr", "127.0.0.1"))
            os.environ.setdefault("MASTER_PORT", config.get("master_port", "29500"))

        relay_kwargs = {
            "engine_id": config.get("worker_id", "receiver_worker"),
            "device": device,
            "credits": 4,
            "slot_size_mb": 1,
            **config,
        }
        relay_kwargs.pop("relay_type", None)
        connector = create_relay(relay_type, **relay_kwargs)
    except Exception as e:
        results["receiver_error"] = f"Init failed: {e}\n{traceback.format_exc()}"
        return

    tensor_device = connector.device if hasattr(connector, "device") else device

    async def _async_receiver():
        try:
            count = 0
            completed_senders = 0
            while count < num_transfers and completed_senders < expected_senders:
                try:
                    item = meta_queue.get(timeout=60)
                    if item is None:
                        completed_senders += 1
                        continue
                    remote_meta = item["metadata"]
                    if "transfer_info" in remote_meta and remote_meta["transfer_info"]:
                        data_size = remote_meta["transfer_info"]["size"]
                    elif "descriptors" in remote_meta:
                        descs = remote_meta["descriptors"]
                        data_size = (
                            descs[0]["size"]
                            if isinstance(descs, list)
                            else descs["size"]
                        )
                    else:
                        raise ValueError(
                            f"Unknown metadata format: {remote_meta.keys()}"
                        )

                    recv_tensor = torch.zeros(
                        data_size, dtype=torch.uint8, device=tensor_device
                    )
                    req_id = f"req_{count}"
                    op = await connector.get_async(
                        remote_meta, recv_tensor, request_id=req_id
                    )
                    await op.wait_for_completion()

                    received_data = pickle.loads(recv_tensor.cpu().numpy().tobytes())
                    received = (
                        received_data.cpu()
                        if isinstance(received_data, torch.Tensor)
                        else torch.tensor(received_data).cpu()
                    )
                    original = pickle.loads(item["original"])
                    assert original.shape == received.shape, "Shape mismatch"
                    assert torch.allclose(
                        original, received, rtol=1e-5, atol=1e-5
                    ), "Data mismatch"
                    if hasattr(connector, "cleanup"):
                        connector.cleanup(req_id)
                    count += 1
                except Empty:
                    results["receiver_error"] = "Queue timeout"
                    break
                except Exception as e:
                    results["receiver_error"] = str(e)
                    results["receiver_traceback"] = traceback.format_exc()
                    break
            results["transfers_completed"] = count
        except Exception as e:
            results["receiver_error"] = str(e)
            results["receiver_traceback"] = traceback.format_exc()

    try:
        asyncio.run(_async_receiver())
    finally:
        if "connector" in locals() and hasattr(connector, "close"):
            connector.close()


@pytest.mark.parametrize("relay_type", ["nixl", "shm", "nccl"])
def test_multiprocess_transfer(relay_type):
    if relay_type in ("nixl", "nccl"):
        if torch.cuda.is_available() and torch.cuda.device_count() < 2:
            pytest.skip(f"{relay_type.upper()} requires at least 2 GPUs")

    master_port = str(find_available_port())
    master_addr = "127.0.0.1"
    config0 = {
        "worker_id": "worker0",
        "relay_type": relay_type,
        "master_addr": master_addr,
        "master_port": master_port,
        "gpu_id": 0,
    }
    config1 = {
        "worker_id": "worker1",
        "relay_type": relay_type,
        "master_addr": master_addr,
        "master_port": master_port,
        "gpu_id": 1 if torch.cuda.device_count() > 1 else 0,
    }

    if relay_type == "nccl":
        config0.update(
            {
                "rank": 0,
                "world_size": 2,
                "send_to_ranks": [1],
                "recv_from_ranks": [],
                "dst_rank": 1,
            }
        )
        config1.update(
            {
                "rank": 1,
                "world_size": 2,
                "send_to_ranks": [],
                "recv_from_ranks": [0],
            }
        )

    meta_queue = multiprocessing.Queue()
    results = multiprocessing.Manager().dict()
    num_transfers = 5
    data_size = 100000
    init_barrier = (
        multiprocessing.Barrier(2, timeout=60) if relay_type == "nccl" else None
    )

    sender = multiprocessing.Process(
        target=sender_process,
        args=(config0, meta_queue, num_transfers, data_size, results, init_barrier),
    )
    receiver = multiprocessing.Process(
        target=receiver_process,
        args=(config1, meta_queue, num_transfers, results, init_barrier),
    )

    try:
        if relay_type == "nccl":
            sender.start()
            receiver.start()
        else:
            sender.start()
            time.sleep(2)
            receiver.start()

        sender.join(timeout=300)
        receiver.join(timeout=300)

        if sender.exitcode != 0 or receiver.exitcode != 0:
            error_msg = (
                f"Process failed ({relay_type}): "
                f"sender={sender.exitcode}, receiver={receiver.exitcode}"
            )
            if "sender_error" in results:
                error_msg += f"\n[Sender Error]: {results['sender_error']}"
            if "receiver_error" in results:
                error_msg += f"\n[Receiver Error]: {results['receiver_error']}"
            pytest.fail(error_msg)
        if "sender_error" in results:
            pytest.fail(f"Sender logical error:\n{results['sender_error']}")
        if "receiver_error" in results:
            pytest.fail(f"Receiver logical error:\n{results['receiver_error']}")
        assert results.get("transfers_completed", 0) == num_transfers
    finally:
        for process in (sender, receiver):
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_multiprocess_two_senders_one_receiver_nixl():
    if torch.cuda.is_available() and torch.cuda.device_count() < 3:
        pytest.skip("NIXL two-sender transfer requires at least 3 GPUs")

    master_port = str(find_available_port())
    master_addr = "127.0.0.1"
    sender_configs = [
        {
            "worker_id": "worker0",
            "relay_type": "nixl",
            "master_addr": master_addr,
            "master_port": master_port,
            "gpu_id": 0,
        },
        {
            "worker_id": "worker1",
            "relay_type": "nixl",
            "master_addr": master_addr,
            "master_port": master_port,
            "gpu_id": 1 if torch.cuda.device_count() > 1 else 0,
        },
    ]
    receiver_config = {
        "worker_id": "worker2",
        "relay_type": "nixl",
        "master_addr": master_addr,
        "master_port": master_port,
        "gpu_id": 2 if torch.cuda.device_count() > 2 else 0,
    }

    meta_queue = multiprocessing.Queue()
    results = multiprocessing.Manager().dict()
    transfers_per_sender = 3
    total_transfers = transfers_per_sender * len(sender_configs)
    data_size = 50000

    senders = [
        multiprocessing.Process(
            target=sender_process,
            args=(config, meta_queue, transfers_per_sender, data_size, results),
        )
        for config in sender_configs
    ]
    receiver = multiprocessing.Process(
        target=receiver_process,
        args=(
            receiver_config,
            meta_queue,
            total_transfers,
            results,
            None,
            len(sender_configs),
        ),
    )
    processes = [*senders, receiver]

    try:
        for sender in senders:
            sender.start()
        time.sleep(2)
        receiver.start()

        for process in processes:
            process.join(timeout=300)

        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            error_msg = "Process failed (nixl two-sender): " + ", ".join(
                f"pid={process.pid}, exitcode={process.exitcode}" for process in failed
            )
            if "sender_error" in results:
                error_msg += f"\n[Sender Error]: {results['sender_error']}"
            if "receiver_error" in results:
                error_msg += f"\n[Receiver Error]: {results['receiver_error']}"
            pytest.fail(error_msg)
        if "sender_error" in results:
            pytest.fail(f"Sender logical error:\n{results['sender_error']}")
        if "receiver_error" in results:
            pytest.fail(f"Receiver logical error:\n{results['receiver_error']}")
        assert results.get("transfers_completed", 0) == total_transfers
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
