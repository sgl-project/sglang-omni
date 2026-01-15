# SPDX-License-Identifier: Apache-2.0
"""Unified tests for relay implementations (NixlRelay) with Tensor interface.

This test follows the same pattern as stage.py and worker.py:
- Sender: wraps serialized data in Tensor, uses put
- Receiver: allocates Tensor, uses get, extracts data
"""

import pickle

import numpy as np
import pytest
import torch

from sglang_omni.relay.nixl import NixlRelay


@pytest.fixture(params=["nixl"])
def relay_class(request):
    if request.param == "nixl":
        return NixlRelay


@pytest.fixture
def relay_configs(relay_class):
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        pytest.skip("NixlRelay requires at least 2 GPUs")
    # Return configs as tuples: (engine_id, device)
    return [
        ("worker0", "cuda:0"),
        ("worker1", "cuda:1"),
    ]


@pytest.fixture
def relay_configs_three(relay_class):
    """Fixture for three connectors (two senders, one receiver)."""
    if torch.cuda.is_available() and torch.cuda.device_count() < 3:
        pytest.skip("NixlRelay requires at least 3 GPUs for this test")
    # Return configs as tuples: (engine_id, device)
    return [
        ("worker0", "cuda:0"),
        ("worker1", "cuda:1"),
        ("worker2", "cuda:2"),
    ]


def _create_connectors(relay_class, configs):
    try:
        # NixlRelay(engine_id, device)
        return relay_class(configs[0][0], device=configs[0][1]), relay_class(
            configs[1][0], device=configs[1][1]
        )
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Failed to initialize {relay_class.__name__}: {e}")


class TestRelayUnified:
    def test_transfer(self, relay_class, relay_configs):
        """Test synchronous data transfer using Tensor interface."""
        connector0, connector1 = _create_connectors(relay_class, relay_configs)
        try:
            # 1. Create test data
            # Use a smaller size for unit testing to be fast
            test_tensor = torch.randn(1024, dtype=torch.bfloat16, device="cpu")
            original = test_tensor.cpu().clone()

            # 2. Serialize data (Sender side)
            serialized_data = pickle.dumps(test_tensor)
            data_size = len(serialized_data)

            # 3. Create Transport Tensor (Sender side)
            # Create a uint8 tensor on the sender's device
            sender_device = connector0.device

            # Use numpy copy to ensure writable memory and handle H2D copy if needed
            data_np = np.frombuffer(serialized_data, dtype=np.uint8).copy()
            src_tensor = torch.tensor(data_np, dtype=torch.uint8, device=sender_device)

            # 4. Put data (Sender side)
            readable_op = connector0.put(src_tensor)
            metadata = readable_op.metadata()

            # 5. Prepare Receiver Tensor (Receiver side)
            # Extract size from metadata
            # Support both descriptor list and transfer_info formats
            if hasattr(metadata, "descriptors") and metadata.descriptors:
                # Assuming list[dict] or list[object]
                descs = metadata.descriptors
                if isinstance(descs, list):
                    recv_size = descs[0]["size"]
                else:
                    recv_size = descs["size"]
            elif isinstance(metadata, dict) and "transfer_info" in metadata:
                recv_size = metadata["transfer_info"]["size"]
            elif isinstance(metadata, dict) and "descriptors" in metadata:
                recv_size = metadata["descriptors"][0]["size"]
            else:
                # Fallback or error
                recv_size = data_size

            receiver_device = connector1.device
            dest_tensor = torch.zeros(
                recv_size, dtype=torch.uint8, device=receiver_device
            )

            # 6. Get data (Receiver side)
            # This triggers RDMA Read -> D2D Copy
            read_op = connector1.get(metadata=metadata, dest_tensor=dest_tensor)

            # Wait for completion (if async)
            if hasattr(read_op, "wait_for_completion"):
                read_op.wait_for_completion()

            # 7. Sync & Cleanup (Crucial for Pool Mode)
            # Sender waits for Receiver to finish reading (notification)
            # Note: In a single-thread test like this, the get() above has already finished
            # the transfer and sent the notification.
            connector0.wait_for_notification(b"done")

            # Reset pools for reuse
            if hasattr(connector0, "reset_pool"):
                connector0.reset_pool()
            if hasattr(connector1, "reset_pool"):
                connector1.reset_pool()

            # 8. Verify Data
            # Convert back to bytes
            if dest_tensor.is_cuda:
                buffer_bytes = dest_tensor.cpu().numpy().tobytes()
            else:
                buffer_bytes = dest_tensor.numpy().tobytes()

            received_data = pickle.loads(buffer_bytes)

            # Convert to tensor for verification
            if isinstance(received_data, torch.Tensor):
                received = received_data.cpu()
            else:
                received = torch.tensor(received_data).cpu()

            # Verify
            assert (
                original.shape == received.shape
            ), f"Shape mismatch: {original.shape} vs {received.shape}"
            assert (
                original.dtype == received.dtype
            ), f"Dtype mismatch: {original.dtype} vs {received.dtype}"
            assert torch.allclose(
                original, received, rtol=1e-5, atol=1e-5
            ), f"Data mismatch: max diff = {torch.max(torch.abs(original - received)).item()}"

        finally:
            if hasattr(connector0, "close"):
                connector0.close()
            if hasattr(connector1, "close"):
                connector1.close()

    def test_health(self, relay_class, relay_configs):
        """Test relay health check."""
        connector = _create_connectors(relay_class, relay_configs)[0]
        try:
            # NixlRelay might not implement health yet, or return basic info
            if hasattr(connector, "health"):
                health = connector.health()
                # Just check it returns a dict, content varies
                assert isinstance(health, dict)
        finally:
            if hasattr(connector, "close"):
                connector.close()

    def test_cleanup(self, relay_class, relay_configs):
        """Test relay cleanup."""
        connector = _create_connectors(relay_class, relay_configs)[0]
        try:
            # Cleanup is mostly no-op in Pool mode but shouldn't crash
            connector.cleanup("test_request_id")
        finally:
            if hasattr(connector, "close"):
                connector.close()

    def test_two_senders_one_receiver(self, relay_class, relay_configs_three):
        """Test two senders sending to one receiver using Tensor interface."""
        configs = relay_configs_three
        try:
            connector0 = relay_class(configs[0][0], device=configs[0][1])
            connector1 = relay_class(configs[1][0], device=configs[1][1])
            connector2 = relay_class(configs[2][0], device=configs[2][1])  # Receiver
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Failed to initialize {relay_class.__name__}: {e}")

        try:
            # --- Prepare Data ---
            tensor0 = torch.randn(1000, dtype=torch.bfloat16, device="cpu")
            tensor1 = torch.randn(1000, dtype=torch.bfloat16, device="cpu")
            original0 = tensor0.cpu().clone()
            original1 = tensor1.cpu().clone()

            # --- Sender 0 -> Receiver ---
            # 1. Serialize & Wrap
            data0_np = np.frombuffer(pickle.dumps(tensor0), dtype=np.uint8).copy()
            src_tensor0 = torch.tensor(
                data0_np, dtype=torch.uint8, device=connector0.device
            )

            # 2. Put
            op0 = connector0.put(src_tensor0)
            meta0 = op0.metadata()

            # 3. Receiver Get
            # Extract size
            if isinstance(meta0, dict) and "transfer_info" in meta0:
                size0 = meta0["transfer_info"]["size"]
            else:
                size0 = meta0["descriptors"][0]["size"]

            dest_tensor0 = torch.zeros(
                size0, dtype=torch.uint8, device=connector2.device
            )
            connector2.get(meta0, dest_tensor0)

            # 4. Sync
            op0.wait_for_completion()
            if hasattr(connector0, "reset_pool"):
                connector0.reset_pool()
            if hasattr(connector2, "reset_pool"):
                connector2.reset_pool()

            # --- Sender 1 -> Receiver ---
            # 1. Serialize & Wrap
            data1_np = np.frombuffer(pickle.dumps(tensor1), dtype=np.uint8).copy()
            src_tensor1 = torch.tensor(
                data1_np, dtype=torch.uint8, device=connector1.device
            )

            # 2. Put
            op1 = connector1.put(src_tensor1)
            meta1 = op1.metadata()

            # 3. Receiver Get
            if isinstance(meta1, dict) and "transfer_info" in meta1:
                size1 = meta1["transfer_info"]["size"]
            else:
                size1 = meta1["descriptors"][0]["size"]

            dest_tensor1 = torch.zeros(
                size1, dtype=torch.uint8, device=connector2.device
            )
            connector2.get(meta1, dest_tensor1)

            # 4. Sync
            op1.wait_for_completion()
            if hasattr(connector1, "reset_pool"):
                connector1.reset_pool()
            if hasattr(connector2, "reset_pool"):
                connector2.reset_pool()

            # --- Verify Data ---
            # Verify 0
            bytes0 = (
                dest_tensor0.cpu().numpy().tobytes()
                if dest_tensor0.is_cuda
                else dest_tensor0.numpy().tobytes()
            )
            rec0 = pickle.loads(bytes0)
            if not isinstance(rec0, torch.Tensor):
                rec0 = torch.tensor(rec0)
            assert torch.equal(original0, rec0.cpu()), "Transfer 0 mismatch"

            # Verify 1
            bytes1 = (
                dest_tensor1.cpu().numpy().tobytes()
                if dest_tensor1.is_cuda
                else dest_tensor1.numpy().tobytes()
            )
            rec1 = pickle.loads(bytes1)
            if not isinstance(rec1, torch.Tensor):
                rec1 = torch.tensor(rec1)
            assert torch.equal(original1, rec1.cpu()), "Transfer 1 mismatch"

        finally:
            if hasattr(connector0, "close"):
                connector0.close()
            if hasattr(connector1, "close"):
                connector1.close()
            if hasattr(connector2, "close"):
                connector2.close()
