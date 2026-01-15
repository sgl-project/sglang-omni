# SPDX-License-Identifier: Apache-2.0
"""Multiprocess tests for relay implementations (NixlRelay) with Tensor interface.

This test follows the same pattern as stage.py and worker.py:
- Sender: wraps serialized data in Tensor, uses put
- Receiver: allocates Tensor, uses get, extracts data
"""

import multiprocessing
import pickle
import time
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

from sglang_omni.relay.nixl import NixlRelay


def sender_process(config, meta_queue, num_transfers, data_size, results):
    """Sender process: creates data, wraps in Tensor, and sends via put."""
    worker_id = config.get("worker_id", "test_worker")
    device = "cuda" if config.get("gpu_id") is not None else "cpu"

    try:
        connector = NixlRelay(engine_id=worker_id, device=device)
    except Exception as e:
        results["sender_error"] = f"Init failed: {e}"
        return

    tensor_device = (
        f'cuda:{config["gpu_id"]}'
        if torch.cuda.is_available() and config.get("gpu_id") is not None
        else "cpu"
    )

    try:
        print(f"[Sender] Starting {num_transfers} transfers...")

        # Estimate maximum buffer size
        test_tensor = torch.randn(data_size, dtype=torch.bfloat16, device=tensor_device)
        test_serialized = pickle.dumps(test_tensor)
        max_buffer_size = len(test_serialized) + 4096

        # Create a reusable ByteTensor as transport container
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

            # Fill transport tensor with serialized data
            data_np = np.frombuffer(serialized_data, dtype=np.uint8)
            transport_tensor[:data_len].copy_(torch.from_numpy(data_np))

            tensor_to_send = transport_tensor[:data_len]
            req_id = f"req_{i}"

            readable_op = connector.put(tensor_to_send, request_id=req_id)
            metadata = readable_op.metadata()

            # Handle metadata format compatibility
            if not isinstance(metadata, dict):
                meta_dict = {
                    "engine_id": getattr(metadata, "engine_id", None),
                    "agent_meta": getattr(metadata, "agent_meta", None),
                    "descriptors": getattr(metadata, "descriptors", None),
                    "transfer_info": getattr(
                        metadata,
                        "transfer_info",
                        (
                            metadata.get("transfer_info")
                            if hasattr(metadata, "get")
                            else None
                        ),
                    ),
                }
            else:
                meta_dict = metadata

            meta_queue.put(
                {
                    "metadata": meta_dict,
                    "original": pickle.dumps(original),
                }
            )

            # Wait for receiver notification
            readable_op.wait_for_completion()

            # Reset pool for reuse
            if hasattr(connector, "reset_pool"):
                connector.reset_pool()

            connector.cleanup(req_id)

        meta_queue.put(None)  # Signal completion

    except Exception as e:
        results["sender_error"] = str(e)
        import traceback

        results["sender_traceback"] = traceback.format_exc()
    finally:
        if "connector" in locals():
            connector.close()


def receiver_process(config, meta_queue, num_transfers, results):
    """Receiver process: receives data into Tensor using get."""
    worker_id = config.get("worker_id", "test_worker")
    device = "cuda" if config.get("gpu_id") is not None else "cpu"

    try:
        connector = NixlRelay(engine_id=worker_id, device=device)
    except Exception as e:
        results["receiver_error"] = f"Init failed: {e}"
        return

    try:
        print(f"[Receiver] Ready to receive {num_transfers} transfers...")
        count = 0

        while count < num_transfers:
            try:
                item = meta_queue.get(timeout=60)
                if item is None:
                    break

                remote_meta = item["metadata"]

                # Extract data size from metadata
                remote_descs_data = remote_meta.get("descriptors", [])
                if not remote_descs_data:
                    # Compatible with transfer_info format
                    data_size = remote_meta["transfer_info"]["size"]
                else:
                    if not isinstance(remote_descs_data, list):
                        remote_descs_data = [remote_descs_data]
                    data_size = remote_descs_data[0]["size"]

                recv_tensor = torch.zeros(data_size, dtype=torch.uint8, device=device)
                req_id = f"req_{count}"

                # Get data (handles pool allocation, RDMA read, D2D copy, and notification)
                op = connector.get(remote_meta, recv_tensor, request_id=req_id)
                op.wait_for_completion()

                # Deserialize data
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

                if hasattr(connector, "reset_pool"):
                    connector.reset_pool()

                connector.cleanup(req_id)
                print(f"[Receiver] Transfer {count+1}: Verified")
                count += 1

            except Empty:
                results["receiver_error"] = "Queue timeout"
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
        if "connector" in locals():
            connector.close()


def test_multiprocess_transfer():
    """Test data transfer between two processes using NixlRelay."""
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        # pytest.skip("NixlRelay requires at least 2 GPUs")
        pass

    config0 = {"gpu_id": 0, "worker_id": "worker0"}
    config1 = {
        "gpu_id": 1 if torch.cuda.device_count() > 1 else 0,
        "worker_id": "worker1",
    }

    meta_queue = multiprocessing.Queue()
    results = multiprocessing.Manager().dict()

    num_transfers = 5
    data_size = 100000

    sender = multiprocessing.Process(
        target=sender_process,
        args=(config0, meta_queue, num_transfers, data_size, results),
    )

    receiver = multiprocessing.Process(
        target=receiver_process,
        args=(config1, meta_queue, num_transfers, results),
    )

    try:
        sender.start()
        time.sleep(2)  # Wait for pool initialization
        receiver.start()

        sender.join(timeout=300)
        receiver.join(timeout=300)

        if sender.exitcode != 0 or receiver.exitcode != 0:
            error_msg = f"Process failed: sender={sender.exitcode}, receiver={receiver.exitcode}"
            if "sender_error" in results:
                error_msg += f"\nSender error: {results['sender_error']}\n{results.get('sender_traceback', '')}"
            if "receiver_error" in results:
                error_msg += f"\nReceiver error: {results['receiver_error']}\n{results.get('receiver_traceback', '')}"
            pytest.fail(error_msg)

        if "sender_error" in results:
            pytest.fail(
                f"Sender error: {results['sender_error']}\n{results.get('sender_traceback', '')}"
            )

        if "receiver_error" in results:
            pytest.fail(
                f"Receiver error: {results['receiver_error']}\n{results.get('receiver_traceback', '')}"
            )

        assert results.get("transfers_completed", 0) == num_transfers

    finally:
        for p in [sender, receiver]:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
