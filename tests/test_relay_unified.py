# SPDX-License-Identifier: Apache-2.0
"""Multiprocess tests for relay implementations (NIXLRelay and SHMRelay).

This test follows the same pattern as stage.py and worker.py:
- Sender: serializes data, creates descriptors, uses put_async (like worker)
- Receiver: creates local descriptors, uses get_async, extracts data directly (like stage)
"""

import asyncio
import multiprocessing
import pickle
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

from sglang_omni.relay.descriptor import Descriptor


def sender_process(
    relay_type, config, queue, done_event, num_transfers, data_size, results
):
    """Sender process: creates data and sends via put_async (following worker.py pattern)."""

    async def run():
        if relay_type == "nixl":
            from sglang_omni.relay.relays.nixl import NIXLRelay

            connector = NIXLRelay(config)
            device = f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else "cpu"
        else:
            from sglang_omni.relay.relays.shm import SHMRelay

            connector = SHMRelay(config)
            device = "cpu"

        try:
            for i in range(num_transfers):
                # Create test data
                tensor = torch.randn(data_size, dtype=torch.bfloat16, device=device)
                original = tensor.cpu().clone()

                # Follow worker.py pattern: serialize data first
                # This allows both SHMRelay and NIXLRelay to use the same descriptor format
                serialized_data = pickle.dumps(tensor)
                data_size_bytes = len(serialized_data)

                # Create a numpy buffer to hold the serialized data (like worker.py)
                buffer = np.frombuffer(serialized_data, dtype=np.uint8).copy()

                # Create descriptor with serialized data buffer (like worker.py)
                descriptor = Descriptor(
                    (buffer.ctypes.data, data_size_bytes, "cpu", buffer)
                )

                # Put data and get metadata (like worker.py)
                readable_op = await connector.put_async([descriptor])
                metadata = readable_op.metadata()

                # Serialize metadata for inter-process communication
                try:
                    meta_bytes = pickle.dumps(metadata)
                except Exception:
                    # Fallback: convert to dict if direct pickle fails
                    meta_dict = (
                        metadata.model_dump()
                        if hasattr(metadata, "model_dump")
                        else metadata.dict()
                    )
                    meta_bytes = pickle.dumps(meta_dict)

                # Send metadata and original data for verification
                queue.put(
                    {
                        "metadata": meta_bytes,
                        "original": pickle.dumps(original),
                    }
                )

                # Wait for completion if needed (like worker.py)
                if hasattr(readable_op, "wait_for_completion"):
                    await readable_op.wait_for_completion()

            queue.put(None)  # Signal completion
            # Wait for receiver to finish processing
            if not done_event.wait(timeout=300):
                results["sender_error"] = "Timeout waiting for receiver to complete"
        except Exception as e:
            results["sender_error"] = str(e)
            import traceback

            results["sender_traceback"] = traceback.format_exc()
        finally:
            connector.close()
            # Ensure done_event is set even if sender fails
            try:
                done_event.set()
            except:
                pass

    asyncio.run(run())


def receiver_process(relay_type, config, queue, done_event, num_transfers, results):
    """Receiver process: receives data via get_async (following stage.py pattern)."""

    async def run():
        if relay_type == "nixl":
            from sglang_omni.relay.nixl import RdmaMetadata
            from sglang_omni.relay.relays.nixl import NIXLRelay

            connector = NIXLRelay(config)
        else:
            from sglang_omni.relay.nixl import SHMMetadata
            from sglang_omni.relay.relays.shm import SHMRelay

            connector = SHMRelay(config)

        try:
            count = 0
            while count < num_transfers:
                try:
                    item = queue.get(timeout=60)
                    if item is None:
                        break

                    # Deserialize metadata
                    meta_obj = pickle.loads(item["metadata"])

                    # Reconstruct metadata object (like stage.py)
                    if relay_type == "nixl":
                        metadata = (
                            RdmaMetadata(**meta_obj)
                            if isinstance(meta_obj, dict)
                            else meta_obj
                        )
                    else:
                        metadata = (
                            SHMMetadata(**meta_obj)
                            if isinstance(meta_obj, dict)
                            else meta_obj
                        )

                    # Follow stage.py pattern: extract remote descriptors from metadata
                    remote_descriptors = metadata.to_descriptors()

                    # Handle both single Descriptor and list[Descriptor] cases (like stage.py)
                    if isinstance(remote_descriptors, list):
                        # Multiple descriptors - create buffers for each
                        local_descriptors = []
                        for remote_desc in remote_descriptors:
                            # Create a buffer of the same size (like stage.py)
                            buffer = np.empty(remote_desc.size, dtype=np.uint8)
                            local_desc = Descriptor(
                                (buffer.ctypes.data, remote_desc.size, "cpu", buffer)
                            )
                            local_descriptors.append(local_desc)
                    else:
                        # Single descriptor (like stage.py)
                        buffer = np.empty(remote_descriptors.size, dtype=np.uint8)
                        local_desc = Descriptor(
                            (buffer.ctypes.data, remote_descriptors.size, "cpu", buffer)
                        )
                        local_descriptors = [local_desc]

                    # Unified interface: both SHMRelay and NIXLRelay use descriptors (like stage.py)
                    read_op = await connector.get_async(
                        metadata=metadata, descriptors=local_descriptors
                    )

                    # Wait for data transfer to complete (like stage.py)
                    await read_op.wait_for_completion()

                    # Extract and deserialize data directly from local_descriptors buffers (like stage.py)
                    if len(local_descriptors) == 1:
                        # Single descriptor: extract directly
                        buffer = local_descriptors[0]._data_ref
                        buffer_bytes = buffer.tobytes()
                    else:
                        # Multiple descriptors: concatenate data from all descriptors
                        buffer_parts = []
                        for desc in local_descriptors:
                            buffer_parts.append(desc._data_ref.tobytes())
                        buffer_bytes = b"".join(buffer_parts)

                    # Deserialize the data (like stage.py)
                    received_data = pickle.loads(buffer_bytes)

                    # Convert to tensor for verification
                    if isinstance(received_data, torch.Tensor):
                        received = received_data.cpu()
                    else:
                        received = torch.tensor(received_data).cpu()

                    # Verify data
                    original = pickle.loads(item["original"])

                    assert (
                        original.shape == received.shape
                    ), f"Shape mismatch in transfer {count + 1}: {original.shape} vs {received.shape}"
                    assert (
                        original.dtype == received.dtype
                    ), f"Dtype mismatch in transfer {count + 1}: {original.dtype} vs {received.dtype}"
                    assert torch.allclose(
                        original, received, rtol=1e-5, atol=1e-5
                    ), f"Data mismatch in transfer {count + 1}: max diff = {torch.max(torch.abs(original - received)).item()}"
                    assert not torch.isnan(
                        received
                    ).any(), f"Received data contains NaN in transfer {count + 1}"
                    assert not torch.isinf(
                        received
                    ).any(), f"Received data contains Inf in transfer {count + 1}"

                    count += 1
                except Empty:
                    # Queue timeout - sender may have failed or finished
                    results["receiver_error"] = (
                        "Queue timeout: no data received within 60 seconds"
                    )
                    break
                except Exception as e:
                    results["receiver_error"] = str(e)
                    import traceback

                    results["receiver_traceback"] = traceback.format_exc()
                    break

            results["transfers_completed"] = count
        except Exception as e:
            results["receiver_error"] = str(e)
            import traceback

            results["receiver_traceback"] = traceback.format_exc()
        finally:
            # Always set done_event to unblock sender, even on error
            done_event.set()
            connector.close()

    asyncio.run(run())


@pytest.mark.parametrize("relay_type", ["nixl", "shm"])
def test_multiprocess_transfer(relay_type):
    """Test data transfer between two processes using different relay implementations.

    This test follows the same pattern as stage.py and worker.py:
    - Sender uses put_async with serialized data descriptors (like worker)
    - Receiver uses get_async with local descriptors and extracts data directly (like stage)
    """

    if relay_type == "nixl":
        if torch.cuda.is_available() and torch.cuda.device_count() < 2:
            pytest.skip("NIXLRelay requires at least 2 GPUs")

        config0 = {
            "host": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "device_name": "",
            "gpu_id": 0,
            "worker_id": "worker0",
        }
        config1 = {
            "host": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "device_name": "",
            "gpu_id": 1 if torch.cuda.is_available() else 0,
            "worker_id": "worker1",
        }
    else:  # shm
        config0 = {}
        config1 = {}

    queue = multiprocessing.Queue()
    done_event = multiprocessing.Event()
    results = multiprocessing.Manager().dict()

    num_transfers = 5
    data_size = 100000

    sender = multiprocessing.Process(
        target=sender_process,
        args=(
            relay_type,
            config0,
            queue,
            done_event,
            num_transfers,
            data_size,
            results,
        ),
    )

    receiver = multiprocessing.Process(
        target=receiver_process,
        args=(relay_type, config1, queue, done_event, num_transfers, results),
    )

    try:
        sender.start()
        receiver.start()

        sender.join(timeout=300)
        receiver.join(timeout=300)

        if sender.exitcode != 0 or receiver.exitcode != 0:
            error_msg = f"Process failed: sender={sender.exitcode}, receiver={receiver.exitcode}"
            if "sender_error" in results:
                error_msg += f"\nSender error: {results['sender_error']}"
                if "sender_traceback" in results:
                    error_msg += f"\n{results['sender_traceback']}"
            if "receiver_error" in results:
                error_msg += f"\nReceiver error: {results['receiver_error']}"
                if "receiver_traceback" in results:
                    error_msg += f"\n{results['receiver_traceback']}"
            pytest.fail(error_msg)

        if "sender_error" in results:
            error_msg = f"Sender error: {results['sender_error']}"
            if "sender_traceback" in results:
                error_msg += f"\n{results['sender_traceback']}"
            pytest.fail(error_msg)

        if "receiver_error" in results:
            error_msg = f"Receiver error: {results['receiver_error']}"
            if "receiver_traceback" in results:
                error_msg += f"\n{results['receiver_traceback']}"
            pytest.fail(error_msg)

        assert (
            results.get("transfers_completed", 0) == num_transfers
        ), f"Not all transfers completed: {results.get('transfers_completed', 0)}/{num_transfers}"

    finally:
        for p in [sender, receiver]:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
