# SPDX-License-Identifier: Apache-2.0
"""An accelerator-origin tensor relayed over host shm must land on the receiver's card.

Only the sender's device string travels with the payload, so the receiver has to
supply its own device; using the sender's index would target the wrong card.
"""

from __future__ import annotations

import torch

from sglang_omni.comm.stage_io import _restore_tensor_device


def test_host_origin_tensor_stays_on_the_host() -> None:
    restored = _restore_tensor_device(torch.empty(2), "cpu", "xpu:1")

    assert restored.device.type == "cpu"


def test_already_resident_tensor_is_left_alone() -> None:
    """A cuda_ipc payload arrives on the accelerator, so it must not be copied."""
    tensor = torch.empty(2, device="meta")

    restored = _restore_tensor_device(tensor, "meta:0", "meta:1")

    assert restored is tensor


def test_accelerator_origin_tensor_moves_to_the_receiver_device() -> None:
    """The receiver's index wins over the sender's."""
    restored = _restore_tensor_device(torch.empty(2), "meta:3", "meta:1")

    assert restored.device.type == "meta"


def test_host_only_stage_keeps_an_accelerator_origin_tensor_on_the_host() -> None:
    """A stage with no card assigned consumes the host copy."""
    restored = _restore_tensor_device(torch.empty(2), "meta:3", None)

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


async def _stream_round_trip(local_device: str | None, *, with_metadata: bool):
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
        _stream_round_trip(None, with_metadata=True)
    )

    assert data_ref.device == "cpu"
    assert data.tolist() == [1, 2, 3, 4]
    assert metadata is not None and metadata["ref_code"].tolist() == [7, 8]


def test_a_metadata_tensor_records_its_own_source_device() -> None:
    """Nested refs are restored from their own recorded device, not the outer one."""
    import asyncio

    data_ref, _ = asyncio.run(_stream_round_trip(None, with_metadata=True))

    assert data_ref.metadata_tensors
    assert all(ref.ref.device == "cpu" for ref in data_ref.metadata_tensors)


def test_a_chunk_without_metadata_still_records_its_device() -> None:
    import asyncio

    data_ref, _ = asyncio.run(_stream_round_trip(None, with_metadata=False))

    assert data_ref.device == "cpu"
