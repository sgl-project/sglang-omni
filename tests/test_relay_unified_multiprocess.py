# SPDX-License-Identifier: Apache-2.0
"""Multiprocess tests for unified relay implementations.

Pattern:
- Sender: wraps serialized payload in a Tensor, uses `put_async`
- Receiver: allocates a Tensor, uses `get_async`, extracts payload
"""

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

# Set multiprocessing start method to 'spawn' (required for CUDA)
if torch.cuda.is_available():
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import sglang_omni.relay.nccl  # noqa: F401
import sglang_omni.relay.nixl  # noqa: F401 (Trigger @register_relay)
import sglang_omni.relay.shm  # noqa: F401
from sglang_omni.relay.base import create_relay
from tests.utils import find_free_port


def sender_process(
    config, meta_queue, num_transfers, data_size, results, init_barrier=None
):
    """Sender process: creates data, wraps in Tensor, and sends via put."""
    relay_type = config.get("relay_type", "nixl")

    # Construct device string
    gpu_id = config.get("gpu_id")
    device = (
        f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu"
    )

    try:
        # [Modification 2] Environment Setup for NCCL
        # Note: We still need this specific setup block for Environment Variables & Barrier
        if relay_type == "nccl":
            if init_barrier is not None:
                init_barrier.wait(timeout=60)

            os.environ.setdefault("MASTER_ADDR", config.get("master_addr", "127.0.0.1"))
            os.environ.setdefault("MASTER_PORT", config.get("master_port", "29500"))

        # [Modification 3] Unified Initialization via Factory
        # We prepare a kwargs dict that contains EVERYTHING.
        # The factory (create_relay) will filter out what's needed for the specific class.
        relay_kwargs = {
            "engine_id": config.get("worker_id", "sender_worker"),
            "device": device,
            "credits": 4,
            "slot_size_mb": 1,  # Default slot size
            # Pass through all config items (includes rank, world_size, topology for NCCL)
            **config,
        }
        relay_kwargs.pop("relay_type", None)

        # ONE line to rule them all
        connector = create_relay(relay_type, **relay_kwargs)

    except Exception as e:
        results["sender_error"] = f"Init failed: {e}\n{traceback.format_exc()}"
        return

    tensor_device = connector.device if hasattr(connector, "device") else device

    async def _async_sender():
        try:
            print(
                f"[Sender] Starting {num_transfers} transfers via {relay_type.upper()}..."
            )

            # Estimate maximum buffer size
            test_tensor = torch.randn(
                data_size, dtype=torch.bfloat16, device=tensor_device
            )
            test_serialized = pickle.dumps(test_tensor)
            max_buffer_size = len(test_serialized) + 4096
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

                # Fill transport tensor
                data_np = np.frombuffer(serialized_data, dtype=np.uint8).copy()
                transport_tensor[:data_len].copy_(torch.from_numpy(data_np))
                tensor_to_send = transport_tensor[:data_len]
                req_id = f"req_{i}"

                # [Common Interface] put_async
                # NCCL needs dst_rank, others ignore it. We can safely pass it if it's in config.
                dst_rank = config.get("dst_rank", 1)
                readable_op = await connector.put_async(
                    tensor_to_send, request_id=req_id, dst_rank=dst_rank
                )

                # Extract metadata
                metadata = readable_op.metadata
                if callable(metadata):
                    metadata = metadata()

                # Normalize metadata to dict
                if not isinstance(metadata, dict):
                    meta_dict = {
                        "engine_id": getattr(metadata, "engine_id", None),
                        "agent_meta": getattr(metadata, "agent_meta", None),
                        "transfer_info": getattr(metadata, "transfer_info", None),
                        # Add legacy support if needed
                        "descriptors": getattr(metadata, "descriptors", None),
                    }
                else:
                    meta_dict = metadata

                meta_queue.put(
                    {
                        "metadata": meta_dict,
                        "original": pickle.dumps(original),
                    }
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


def receiver_process(config, meta_queue, num_transfers, results, init_barrier=None):
    """Receiver process: receives data into Tensor using get."""
    relay_type = config.get("relay_type", "nixl")

    gpu_id = config.get("gpu_id")
    device = (
        f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu"
    )

    try:
        # [Modification 2] Environment Setup for NCCL
        if relay_type == "nccl":
            if init_barrier is not None:
                init_barrier.wait(timeout=60)

            os.environ.setdefault("MASTER_ADDR", config.get("master_addr", "127.0.0.1"))
            os.environ.setdefault("MASTER_PORT", config.get("master_port", "29500"))

        # [Modification 3] Unified Initialization via Factory
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
            print(
                f"[Receiver] Ready to receive {num_transfers} transfers via {relay_type.upper()}..."
            )
            count = 0

            while count < num_transfers:
                try:
                    item = meta_queue.get(timeout=60)
                    if item is None:
                        break

                    remote_meta = item["metadata"]

                    # Robust size extraction
                    if "transfer_info" in remote_meta and remote_meta["transfer_info"]:
                        data_size = remote_meta["transfer_info"]["size"]
                    elif "descriptors" in remote_meta:
                        # Legacy fallback
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

                    # Verify
                    buffer_bytes = recv_tensor.cpu().numpy().tobytes()
                    received_data = pickle.loads(buffer_bytes)

                    if isinstance(received_data, torch.Tensor):
                        received = received_data.cpu()
                    else:
                        received = torch.tensor(received_data).cpu()

                    original = pickle.loads(item["original"])

                    assert original.shape == received.shape, "Shape mismatch"
                    assert torch.allclose(
                        original, received, rtol=1e-5, atol=1e-5
                    ), "Data mismatch"

                    if hasattr(connector, "cleanup"):
                        connector.cleanup(req_id)

                    print(f"[Receiver] Transfer {count+1}: Verified")
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
    """Test data transfer between two processes using configured Relay."""

    if relay_type == "nixl" or relay_type == "nccl":
        if torch.cuda.is_available() and torch.cuda.device_count() < 2:
            pytest.skip(f"{relay_type.upper()} requires at least 2 GPUs")

    # [Modification 4] Dynamic Port Generation
    master_port = str(find_free_port())
    master_addr = "127.0.0.1"

    # Base configuration
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

    # NCCL Specific Configuration (will be filtered out by Factory for SHM/NIXL)
    if relay_type == "nccl":
        config0.update(
            {
                "rank": 0,
                "world_size": 2,
                "send_to_ranks": [1],
                "recv_from_ranks": [],
                "dst_rank": 1,  # Used by put_async call
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

    init_barrier = None
    if relay_type == "nccl":
        init_barrier = multiprocessing.Barrier(2, timeout=60)

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
            time.sleep(2)  # Wait for pool initialization (SHM/NIXL specific)
            receiver.start()

        sender.join(timeout=300)
        receiver.join(timeout=300)

        # Check for non-zero exit codes
        if sender.exitcode != 0 or receiver.exitcode != 0:
            error_msg = f"Process failed ({relay_type}): sender={sender.exitcode}, receiver={receiver.exitcode}"
            if "sender_error" in results:
                error_msg += f"\n[Sender Error]: {results['sender_error']}"
            if "receiver_error" in results:
                error_msg += f"\n[Receiver Error]: {results['receiver_error']}"
            pytest.fail(error_msg)

        # Check for logical errors caught inside processes
        if "sender_error" in results:
            pytest.fail(f"Sender logical error:\n{results['sender_error']}")

        if "receiver_error" in results:
            pytest.fail(f"Receiver logical error:\n{results['receiver_error']}")

        assert results.get("transfers_completed", 0) == num_transfers

    finally:
        for p in [sender, receiver]:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
