# SPDX-License-Identifier: Apache-2.0
"""Multiprocess tests for relay implementations (NIXLRelay and SHMRelay)."""

import asyncio
import multiprocessing
import pickle
from queue import Empty

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
    """Sender process: creates data and sends via put_async."""

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
                tensor = torch.randn(data_size, dtype=torch.bfloat16, device=device)
                original = tensor.cpu().clone()

                readable_op = await connector.put_async([Descriptor(tensor)])
                metadata = readable_op.metadata()

                # Serialize metadata
                try:
                    meta_bytes = pickle.dumps(metadata)
                except Exception:
                    meta_dict = (
                        metadata.model_dump()
                        if hasattr(metadata, "model_dump")
                        else metadata.dict()
                    )
                    meta_bytes = pickle.dumps(meta_dict)

                queue.put(
                    {
                        "metadata": meta_bytes,
                        "size": data_size,
                        "dtype": tensor.dtype,
                        "original": pickle.dumps(original),
                        "relay_type": relay_type,
                    }
                )

                if hasattr(readable_op, "wait_for_completion"):
                    await readable_op.wait_for_completion()

            queue.put(None)  # Signal completion
            done_event.wait(timeout=300)
        except Exception as e:
            results["sender_error"] = str(e)
            import traceback

            results["sender_traceback"] = traceback.format_exc()
        finally:
            connector.close()

    asyncio.run(run())


def receiver_process(relay_type, config, queue, done_event, num_transfers, results):
    """Receiver process: receives data via get_async."""

    async def run():
        if relay_type == "nixl":
            from sglang_omni.relay.nixl import RdmaMetadata
            from sglang_omni.relay.relays.nixl import NIXLRelay

            connector = NIXLRelay(config)
            device = f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else "cpu"
        else:
            from sglang_omni.relay.relays.shm import SHMRelay

            connector = SHMRelay(config)
            device = "cpu"

        try:
            count = 0
            while count < num_transfers:
                try:
                    item = queue.get(timeout=60)
                    if item is None:
                        break

                    # Deserialize metadata
                    meta_obj = pickle.loads(item["metadata"])

                    if relay_type == "nixl":
                        metadata = (
                            RdmaMetadata(**meta_obj)
                            if isinstance(meta_obj, dict)
                            else meta_obj
                        )
                        # Receive data into buffer for NIXL
                        buffer = torch.empty(
                            item["size"], dtype=item["dtype"], device=device
                        )
                        read_op = await connector.get_async(
                            metadata, [Descriptor(buffer)]
                        )
                        if hasattr(read_op, "wait_for_completion"):
                            await read_op.wait_for_completion()
                        received = buffer.cpu()
                    else:
                        # For SHM, metadata reconstruction depends on implementation
                        if isinstance(meta_obj, dict):
                            from sglang_omni.core.types import SHMMetadata

                            metadata = SHMMetadata(**meta_obj)
                        else:
                            metadata = meta_obj

                        # SHM returns data directly
                        read_op = await connector.get_async(metadata, [])
                        received_data = read_op.data
                        received = (
                            received_data.cpu()
                            if isinstance(received_data, torch.Tensor)
                            else torch.tensor(received_data).cpu()
                        )

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
                    break
                except Exception as e:
                    results["receiver_error"] = str(e)
                    import traceback

                    results["receiver_traceback"] = traceback.format_exc()
                    break

            results["transfers_completed"] = count
            done_event.set()
        except Exception as e:
            results["receiver_error"] = str(e)
            import traceback

            results["receiver_traceback"] = traceback.format_exc()
        finally:
            connector.close()

    asyncio.run(run())


@pytest.mark.parametrize("relay_type", ["nixl", "shm"])
def test_multiprocess_transfer(relay_type):
    """Test data transfer between two processes using different relay implementations."""

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
