# SPDX-License-Identifier: Apache-2.0
"""Unified tests for v1 relay implementations with Tensor interface."""

import pickle

import numpy as np
import pytest
import torch

from sglang_omni_v1.relay.nixl import NixlRelay


@pytest.fixture(params=["nixl"])
def relay_class(request):
    if request.param == "nixl":
        return NixlRelay


@pytest.fixture
def relay_configs(relay_class):
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        pytest.skip("NixlRelay requires at least 2 GPUs")
    return [
        ("worker0", "cuda:0"),
        ("worker1", "cuda:1"),
    ]


@pytest.fixture
def relay_configs_three(relay_class):
    if torch.cuda.is_available() and torch.cuda.device_count() < 3:
        pytest.skip("NixlRelay requires at least 3 GPUs for this test")
    return [
        ("worker0", "cuda:0"),
        ("worker1", "cuda:1"),
        ("worker2", "cuda:2"),
    ]


def _create_connectors(relay_class, configs):
    try:
        return relay_class(configs[0][0], device=configs[0][1]), relay_class(
            configs[1][0], device=configs[1][1]
        )
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Failed to initialize {relay_class.__name__}: {e}")


class TestRelayUnified:
    @pytest.mark.asyncio
    async def test_transfer(self, relay_class, relay_configs):
        connector0, connector1 = _create_connectors(relay_class, relay_configs)
        try:
            test_tensor = torch.randn(1024, dtype=torch.bfloat16, device="cpu")
            original = test_tensor.cpu().clone()

            serialized_data = pickle.dumps(test_tensor)
            data_size = len(serialized_data)
            sender_device = connector0.device
            data_np = np.frombuffer(serialized_data, dtype=np.uint8).copy()
            src_tensor = torch.tensor(data_np, dtype=torch.uint8, device=sender_device)

            put_op = await connector0.put_async(src_tensor)
            metadata = put_op.metadata

            if isinstance(metadata, dict) and "transfer_info" in metadata:
                recv_size = metadata["transfer_info"]["size"]
            else:
                recv_size = data_size

            receiver_device = connector1.device
            dest_tensor = torch.zeros(
                recv_size, dtype=torch.uint8, device=receiver_device
            )

            get_op = await connector1.get_async(
                metadata=metadata, dest_tensor=dest_tensor
            )
            await get_op.wait_for_completion()
            await put_op.wait_for_completion()

            buffer_bytes = (
                dest_tensor.cpu().numpy().tobytes()
                if dest_tensor.is_cuda
                else dest_tensor.numpy().tobytes()
            )
            received_data = pickle.loads(buffer_bytes)
            received = (
                received_data.cpu()
                if isinstance(received_data, torch.Tensor)
                else torch.tensor(received_data).cpu()
            )

            assert original.shape == received.shape
            assert original.dtype == received.dtype
            assert torch.allclose(original, received, rtol=1e-5, atol=1e-5)
        finally:
            if hasattr(connector0, "close"):
                connector0.close()
            if hasattr(connector1, "close"):
                connector1.close()

    @pytest.mark.asyncio
    async def test_two_senders_one_receiver(self, relay_class, relay_configs_three):
        configs = relay_configs_three
        try:
            connector0 = relay_class(configs[0][0], device=configs[0][1])
            connector1 = relay_class(configs[1][0], device=configs[1][1])
            connector2 = relay_class(configs[2][0], device=configs[2][1])
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Failed to initialize {relay_class.__name__}: {e}")

        try:
            tensor0 = torch.randn(1000, dtype=torch.bfloat16, device="cpu")
            tensor1 = torch.randn(1000, dtype=torch.bfloat16, device="cpu")
            original0 = tensor0.cpu().clone()
            original1 = tensor1.cpu().clone()

            data0_np = np.frombuffer(pickle.dumps(tensor0), dtype=np.uint8).copy()
            src_tensor0 = torch.tensor(
                data0_np, dtype=torch.uint8, device=connector0.device
            )
            op0 = await connector0.put_async(src_tensor0)
            meta0 = op0.metadata
            dest_tensor0 = torch.zeros(
                meta0["transfer_info"]["size"],
                dtype=torch.uint8,
                device=connector2.device,
            )
            get_op0 = await connector2.get_async(meta0, dest_tensor0)

            data1_np = np.frombuffer(pickle.dumps(tensor1), dtype=np.uint8).copy()
            src_tensor1 = torch.tensor(
                data1_np, dtype=torch.uint8, device=connector1.device
            )
            op1 = await connector1.put_async(src_tensor1)
            meta1 = op1.metadata
            dest_tensor1 = torch.zeros(
                meta1["transfer_info"]["size"],
                dtype=torch.uint8,
                device=connector2.device,
            )
            get_op1 = await connector2.get_async(meta1, dest_tensor1)

            await get_op0.wait_for_completion()
            await get_op1.wait_for_completion()
            await op0.wait_for_completion()
            await op1.wait_for_completion()

            rec0 = pickle.loads(
                dest_tensor0.cpu().numpy().tobytes()
                if dest_tensor0.is_cuda
                else dest_tensor0.numpy().tobytes()
            )
            if not isinstance(rec0, torch.Tensor):
                rec0 = torch.tensor(rec0)
            assert torch.equal(original0, rec0.cpu()), "Transfer 0 mismatch"

            rec1 = pickle.loads(
                dest_tensor1.cpu().numpy().tobytes()
                if dest_tensor1.is_cuda
                else dest_tensor1.numpy().tobytes()
            )
            if not isinstance(rec1, torch.Tensor):
                rec1 = torch.tensor(rec1)
            assert torch.equal(original1, rec1.cpu()), "Transfer 1 mismatch"
        finally:
            for connector in (connector0, connector1, connector2):
                if hasattr(connector, "close"):
                    connector.close()

    def test_health(self, relay_class, relay_configs):
        connector = _create_connectors(relay_class, relay_configs)[0]
        try:
            if hasattr(connector, "health"):
                health = connector.health()
                assert isinstance(health, dict)
        finally:
            if hasattr(connector, "close"):
                connector.close()

    def test_cleanup(self, relay_class, relay_configs):
        connector = _create_connectors(relay_class, relay_configs)[0]
        try:
            if hasattr(connector, "cleanup"):
                connector.cleanup("test_request_id")
        finally:
            if hasattr(connector, "close"):
                connector.close()
