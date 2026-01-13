# SPDX-License-Identifier: Apache-2.0
"""Unified tests for relay implementations (NIXLRelay and SHMRelay).

This test follows the same pattern as stage.py and worker.py:
- Sender: serializes data, creates descriptors, uses put/put_async (like worker)
- Receiver: creates local descriptors, uses get/get_async, extracts data directly (like stage)
"""

import pickle

import numpy as np
import pytest
import torch

from sglang_omni.relay.descriptor import Descriptor


@pytest.fixture(params=["nixl", "shm"])
def relay_class(request):
    if request.param == "nixl":
        from sglang_omni.relay.relays.nixl import NIXLRelay

        return NIXLRelay
    else:
        from sglang_omni.relay.relays.shm import SHMRelay

        return SHMRelay


@pytest.fixture
def relay_configs(relay_class):
    if relay_class.__name__ == "NIXLRelay":
        if torch.cuda.is_available() and torch.cuda.device_count() < 2:
            pytest.skip("NIXLRelay requires at least 2 GPUs")
        return [
            {
                "host": "127.0.0.1",
                "metadata_server": "http://127.0.0.1:8080/metadata",
                "device_name": "",
                "gpu_id": 0,
                "worker_id": "worker0",
            },
            {
                "host": "127.0.0.1",
                "metadata_server": "http://127.0.0.1:8080/metadata",
                "device_name": "",
                "gpu_id": 1 if torch.cuda.is_available() else 0,
                "worker_id": "worker1",
            },
        ]
    return [{}, {}]


def _create_connectors(relay_class, configs):
    try:
        return relay_class(configs[0]), relay_class(configs[1])
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Failed to initialize {relay_class.__name__}: {e}")


class TestRelayUnified:
    def test_transfer(self, relay_class, relay_configs):
        """Test synchronous data transfer (following worker.py and stage.py patterns)."""
        connector0, connector1 = _create_connectors(relay_class, relay_configs)
        try:
            # Create test data
            test_tensor = torch.randn(100000, dtype=torch.bfloat16, device="cpu")
            original = test_tensor.cpu().clone()

            # Follow worker.py pattern: serialize data first
            # This allows both SHMRelay and NIXLRelay to use the same descriptor format
            serialized_data = pickle.dumps(test_tensor)
            data_size = len(serialized_data)

            # Create a numpy buffer to hold the serialized data (like worker.py)
            buffer = np.frombuffer(serialized_data, dtype=np.uint8).copy()

            # Create descriptor with serialized data buffer (like worker.py)
            descriptor = Descriptor((
                buffer.ctypes.data,
                data_size,
                "cpu",
                buffer
            ))

            # Put data and get metadata (like worker.py)
            readable_op = connector0.put([descriptor])
            metadata = readable_op.metadata()

            # Follow stage.py pattern: extract remote descriptors from metadata
            remote_descriptors = metadata.to_descriptors()

            # Handle both single Descriptor and list[Descriptor] cases (like stage.py)
            if isinstance(remote_descriptors, list):
                # Multiple descriptors - create buffers for each
                local_descriptors = []
                for remote_desc in remote_descriptors:
                    # Create a buffer of the same size (like stage.py)
                    recv_buffer = np.empty(remote_desc.size, dtype=np.uint8)
                    local_desc = Descriptor((
                        recv_buffer.ctypes.data,
                        remote_desc.size,
                        "cpu",
                        recv_buffer
                    ))
                    local_descriptors.append(local_desc)
            else:
                # Single descriptor (like stage.py)
                recv_buffer = np.empty(remote_descriptors.size, dtype=np.uint8)
                local_desc = Descriptor((
                    recv_buffer.ctypes.data,
                    remote_descriptors.size,
                    "cpu",
                    recv_buffer
                ))
                local_descriptors = [local_desc]

            # Unified interface: both SHMRelay and NIXLRelay use descriptors (like stage.py)
            read_op = connector1.get(metadata=metadata, descriptors=local_descriptors)

            # Wait for data transfer to complete (like stage.py)
            if hasattr(read_op, "wait_for_completion"):
                coro = read_op.wait_for_completion()
                if coro:
                    if hasattr(connector1, "_run_maybe_async"):
                        connector1._run_maybe_async(coro)
                    else:
                        import asyncio
                        asyncio.run(coro)

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
            assert (
                original.shape == received.shape
            ), f"Shape mismatch: {original.shape} vs {received.shape}"
            assert (
                original.dtype == received.dtype
            ), f"Dtype mismatch: {original.dtype} vs {received.dtype}"
            assert torch.allclose(
                original, received, rtol=1e-5, atol=1e-5
            ), f"Data mismatch: max diff = {torch.max(torch.abs(original - received)).item()}"

            assert not torch.isnan(received).any(), "Received data contains NaN"
            assert not torch.isinf(received).any(), "Received data contains Inf"

            assert connector0._metrics["puts"] >= 1
            assert connector1._metrics["gets"] >= 1
        finally:
            connector0.close()
            connector1.close()

    @pytest.mark.asyncio
    async def test_transfer_async(self, relay_class, relay_configs):
        """Test asynchronous data transfer (following worker.py and stage.py patterns)."""
        connector0, connector1 = _create_connectors(relay_class, relay_configs)
        try:
            # Create test data
            test_tensor = torch.randn(100000, dtype=torch.bfloat16, device="cpu")
            original = test_tensor.cpu().clone()

            # Follow worker.py pattern: serialize data first
            # This allows both SHMRelay and NIXLRelay to use the same descriptor format
            serialized_data = pickle.dumps(test_tensor)
            data_size = len(serialized_data)

            # Create a numpy buffer to hold the serialized data (like worker.py)
            buffer = np.frombuffer(serialized_data, dtype=np.uint8).copy()

            # Create descriptor with serialized data buffer (like worker.py)
            descriptor = Descriptor((
                buffer.ctypes.data,
                data_size,
                "cpu",
                buffer
            ))

            # Put data and get metadata (like worker.py)
            readable_op = await connector0.put_async([descriptor])
            metadata = readable_op.metadata()

            # Follow stage.py pattern: extract remote descriptors from metadata
            remote_descriptors = metadata.to_descriptors()

            # Handle both single Descriptor and list[Descriptor] cases (like stage.py)
            if isinstance(remote_descriptors, list):
                # Multiple descriptors - create buffers for each
                local_descriptors = []
                for remote_desc in remote_descriptors:
                    # Create a buffer of the same size (like stage.py)
                    recv_buffer = np.empty(remote_desc.size, dtype=np.uint8)
                    local_desc = Descriptor((
                        recv_buffer.ctypes.data,
                        remote_desc.size,
                        "cpu",
                        recv_buffer
                    ))
                    local_descriptors.append(local_desc)
            else:
                # Single descriptor (like stage.py)
                recv_buffer = np.empty(remote_descriptors.size, dtype=np.uint8)
                local_desc = Descriptor((
                    recv_buffer.ctypes.data,
                    remote_descriptors.size,
                    "cpu",
                    recv_buffer
                ))
                local_descriptors = [local_desc]

            # Unified interface: both SHMRelay and NIXLRelay use descriptors (like stage.py)
            read_op = await connector1.get_async(metadata=metadata, descriptors=local_descriptors)

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
            assert (
                original.shape == received.shape
            ), f"Shape mismatch: {original.shape} vs {received.shape}"
            assert (
                original.dtype == received.dtype
            ), f"Dtype mismatch: {original.dtype} vs {received.dtype}"
            assert torch.allclose(
                original, received, rtol=1e-5, atol=1e-5
            ), f"Data mismatch: max diff = {torch.max(torch.abs(original - received)).item()}"

            assert not torch.isnan(received).any(), "Received data contains NaN"
            assert not torch.isinf(received).any(), "Received data contains Inf"
        finally:
            connector0.close()
            connector1.close()

    def test_health(self, relay_class, relay_configs):
        """Test relay health check."""
        connector = _create_connectors(relay_class, relay_configs)[0]
        try:
            health = connector.health()
            assert health["status"] == "healthy"
        finally:
            connector.close()

    def test_cleanup(self, relay_class, relay_configs):
        """Test relay cleanup."""
        connector = _create_connectors(relay_class, relay_configs)[0]
        try:
            connector.cleanup("test_request_id")
        finally:
            connector.close()
