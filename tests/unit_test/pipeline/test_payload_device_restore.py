# SPDX-License-Identifier: Apache-2.0
"""What stage_io fixes up when a payload crosses a process boundary.

An accelerator-origin tensor relayed over host shm must land on the receiver's
card: only the sender's device string travels with the payload, so the receiver
has to supply its own device, and using the sender's index would target the wrong
card. Device events must not travel at all; they only order readers inside the
process that recorded them.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.stage_io import (
    read_payload,
    restore_tensor_device,
    strip_process_local_metadata,
    write_payload,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.relay.shm import ShmRelay


class BackendEvent(torch.Event):
    """A vendor event class, as torch.xpu.Event and torch.npu.Event are."""


def test_host_origin_tensor_stays_on_the_host() -> None:
    restored = restore_tensor_device(torch.empty(2), "cpu", "xpu:1")

    assert restored.device.type == "cpu"


def test_already_resident_tensor_is_left_alone() -> None:
    """A cuda_ipc payload arrives on the accelerator, so it must not be copied."""
    tensor = torch.empty(2, device="meta")

    restored = restore_tensor_device(tensor, "meta:0", "meta:1")

    assert restored is tensor


def test_accelerator_origin_tensor_moves_to_the_receiver_device() -> None:
    """The receiver's index wins over the sender's."""
    restored = restore_tensor_device(torch.empty(2), "meta:3", "meta:1")

    assert restored.device.type == "meta"


def test_host_only_stage_keeps_an_accelerator_origin_tensor_on_the_host() -> None:
    """A stage with no card assigned consumes the host copy."""
    restored = restore_tensor_device(torch.empty(2), "meta:3", None)

    assert restored.device.type == "cpu"


def test_a_raw_stream_ref_carries_its_source_device() -> None:
    """A raw DataRef had nowhere to record residency, so a host-shm hop lost it.
    Qwen3-TTS emits device-resident codec chunks, and a process-isolated vocoder
    (--vocoder.process vocoder) would otherwise feed CPU codes to an accelerator decoder.
    """
    from sglang_omni.comm.data_ref import (
        BackendRef,
        DataKind,
        DataLayout,
        DataRef,
        TransportKind,
    )

    ref = DataRef(
        version=1,
        kind=DataKind.STREAM_CHUNK,
        object_id="req:stream:talker:vocoder:0",
        transport=TransportKind.SHM,
        layout=DataLayout.RAW_TENSOR,
        buffer=BackendRef(transport=TransportKind.SHM, info={}, length=16),
        shape=(2,),
        dtype="torch.int64",
        device="meta:3",
        offset=0,
    )

    assert DataRef.from_dict(ref.to_dict()).device == "meta:3"


def test_a_stream_ref_without_a_device_stays_wire_compatible() -> None:
    """An older ref records no device, so from_dict yields None and nothing moves."""
    from sglang_omni.comm.data_ref import (
        BackendRef,
        DataKind,
        DataLayout,
        DataRef,
        TransportKind,
    )

    payload = DataRef(
        version=1,
        kind=DataKind.STREAM_CHUNK,
        object_id="req:stream:talker:vocoder:0",
        transport=TransportKind.SHM,
        layout=DataLayout.RAW_TENSOR,
        buffer=BackendRef(transport=TransportKind.SHM, info={}, length=16),
        shape=(2,),
        dtype="torch.int64",
        offset=0,
    ).to_dict()

    assert "device" not in payload
    assert DataRef.from_dict(payload).device is None


async def stream_round_trip(local_device: str | None, *, with_metadata: bool):
    """Write a chunk through a host-shm relay and read it back."""
    from sglang_omni.comm import stage_io
    from sglang_omni.comm.data_ref import TransportKind
    from tests.unit_test.fixtures.pipeline_fakes import FakeRelay

    relay = FakeRelay(device="cpu")
    codes = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    metadata = (
        {"ref_code": torch.tensor([7, 8], dtype=torch.long)} if with_metadata else None
    )

    data_ref, _ = await stage_io.write_stream_chunk(
        relay,
        request_id="req-1",
        data=codes,
        target_stage="vocoder",
        from_stage="talker",
        chunk_id=0,
        metadata=metadata,
        transport=TransportKind.SHM,
    )
    return data_ref, await stage_io.read_stream_chunk(relay, data_ref, local_device)


def test_a_metadata_bearing_chunk_keeps_its_source_device() -> None:
    """Qwen3-TTS attaches metadata to every codec chunk, and the metadata rebuild
    used to drop the outer device, so an isolated vocoder got the shm CPU tensor.
    """
    import asyncio

    data_ref, (data, metadata) = asyncio.run(
        stream_round_trip(None, with_metadata=True)
    )

    assert data_ref.device == "cpu"
    assert data.tolist() == [1, 2, 3, 4]
    assert metadata is not None and metadata["ref_code"].tolist() == [7, 8]


def test_a_metadata_tensor_records_its_own_source_device() -> None:
    """Nested refs are restored from their own recorded device, not the outer one."""
    import asyncio

    data_ref, _ = asyncio.run(stream_round_trip(None, with_metadata=True))

    assert data_ref.metadata_tensors
    assert all(ref.ref.device == "cpu" for ref in data_ref.metadata_tensors)


def test_a_chunk_without_metadata_still_records_its_device() -> None:
    import asyncio

    data_ref, _ = asyncio.run(stream_round_trip(None, with_metadata=False))

    assert data_ref.device == "cpu"


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
        torch.complex128,
    ],
)
def test_mixed_host_relay_payload_preserves_strided_tensor_bytes(
    dtype: torch.dtype,
) -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="sender", credits=1)
        tensor = torch.arange(24, device="cuda").reshape(4, 6).to(dtype).t()
        tensors = {
            "prefix": torch.tensor([7], dtype=torch.uint8),
            "values": tensor,
            "empty": tensor[:0],
            "tail": torch.tensor([0.125], dtype=torch.float64),
        }
        payload = StagePayload(
            request_id="mixed", request=OmniRequest(inputs="relay"), data=tensors
        )
        reference, operation = await write_payload(
            relay, "mixed", payload, transport=TransportKind.SHM
        )
        restored = await read_payload(relay, "mixed", reference, local_device="cuda:0")
        operation.mark_receiver_done()
        await operation.wait_for_completion()
        for name, source in tensors.items():
            actual = restored.data[name]
            assert actual.dtype == source.dtype
            assert actual.shape == source.shape
            assert actual.device == source.device
            assert torch.equal(
                actual.contiguous().reshape(-1).view(torch.uint8),
                source.contiguous().reshape(-1).view(torch.uint8),
            )
        relay.close()

    asyncio.run(run())


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_host_relay_snapshot_survives_producer_reuse_under_backpressure() -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="sender", credits=1)
        occupied = await relay.put_async(
            torch.ones(1, dtype=torch.uint8), request_id="occupied"
        )
        tensors = {
            "video": torch.ones(1024, device="cuda", dtype=torch.bfloat16),
            "audio": torch.full((1024,), 2.0, device="cuda"),
        }
        payload = StagePayload(
            request_id="next", request=OmniRequest(inputs="relay"), data=tensors
        )
        pending = asyncio.create_task(
            write_payload(relay, "next", payload, transport=TransportKind.SHM)
        )
        await asyncio.sleep(0)
        assert not pending.done()
        tensors["video"].fill_(9)
        tensors["audio"].fill_(9)
        receive = await relay.get_async(
            occupied.metadata, torch.empty(1, dtype=torch.uint8)
        )
        await receive.wait_for_completion()
        occupied.mark_receiver_done()
        await occupied.wait_for_completion()
        reference, operation = await asyncio.wait_for(pending, timeout=5)
        restored = await read_payload(relay, "next", reference, local_device="cuda:0")
        operation.mark_receiver_done()
        await operation.wait_for_completion()
        assert torch.equal(restored.data["video"], torch.ones_like(tensors["video"]))
        assert torch.equal(
            restored.data["audio"], torch.full_like(tensors["audio"], 2.0)
        )
        relay.close()

    asyncio.run(run())


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cancelled_host_payload_does_not_consume_relay_credit() -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="sender", credits=1)
        occupied = await relay.put_async(
            torch.ones(1, dtype=torch.uint8), request_id="occupied"
        )
        payload = StagePayload(
            request_id="next",
            request=OmniRequest(inputs="relay"),
            data={
                "video": torch.ones(32, device="cuda"),
                "audio": torch.ones(8, device="cuda"),
            },
        )
        pending = asyncio.create_task(
            write_payload(relay, "next", payload, transport=TransportKind.SHM)
        )
        await asyncio.sleep(0)
        assert not pending.done()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        receive = await relay.get_async(
            occupied.metadata, torch.empty(1, dtype=torch.uint8)
        )
        await receive.wait_for_completion()
        occupied.mark_receiver_done()
        await occupied.wait_for_completion()
        reference, operation = await asyncio.wait_for(
            write_payload(relay, "next", payload, transport=TransportKind.SHM),
            timeout=5,
        )
        restored = await read_payload(relay, "next", reference, local_device="cuda:0")
        operation.mark_receiver_done()
        await operation.wait_for_completion()
        assert torch.equal(restored.data["video"], payload.data["video"])
        relay.close()

    asyncio.run(run())


def test_a_backend_event_is_stripped_and_the_rest_survives() -> None:
    """A device event means nothing in the receiving process, and the transport
    establishes readiness itself once the payload crosses over."""
    metadata = {
        "codes_ready_event": BackendEvent(),
        "ref_code_len": 7,
        "sample_rate": 24000,
    }

    stripped = strip_process_local_metadata(metadata)

    assert stripped == {"ref_code_len": 7, "sample_rate": 24000}


def test_metadata_without_events_is_returned_whole() -> None:
    assert strip_process_local_metadata({"ref_code_len": 0}) == {"ref_code_len": 0}
    assert strip_process_local_metadata(None) is None
