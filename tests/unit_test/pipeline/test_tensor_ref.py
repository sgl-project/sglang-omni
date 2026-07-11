# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest
import torch

from sglang_omni.pipeline import relay_io
from sglang_omni.pipeline.tensor_ref import (
    DEFAULT_TENSOR_REF_PATHS,
    DEFAULT_TENSOR_REF_THRESHOLD_MB,
    TensorRef,
    TensorRefPolicy,
    is_tensor_ref_dict,
    tensor_ref_numel,
    tensor_refs_enabled,
)
from tests.unit_test.fixtures.pipeline_fakes import (
    DestructiveFakeRelay,
    FakeRelay,
    make_stage_payload,
)


def _make_ref(**overrides) -> TensorRef:
    defaults = dict(
        ref_id="req-1:tensor_ref:image_encoder:mm_aggregate:abc:video_embeds",
        request_id="req-1",
        producer_stage="image_encoder",
        consumer_stage="thinker",
        path="encoder_outs.image_encoder.video_embeds",
        shape=(2, 3),
        dtype="torch.float32",
        nbytes=24,
        blob_key="req-1:tensor_ref:image_encoder:mm_aggregate:abc:video_embeds",
        blob_metadata={"relay_info": {}, "tensor_shape": [2, 3]},
    )
    defaults.update(overrides)
    return TensorRef(**defaults)


def test_tensor_ref_to_dict_from_dict_round_trip() -> None:
    ref = _make_ref()
    restored = TensorRef.from_dict(ref.to_dict())
    assert restored == ref


def test_is_tensor_ref_dict_and_numel() -> None:
    ref = _make_ref()
    ref_dict = ref.to_dict()
    assert is_tensor_ref_dict(ref_dict)
    assert not is_tensor_ref_dict({"shape": [1]})
    assert not is_tensor_ref_dict(torch.zeros(1))
    assert tensor_ref_numel(ref_dict) == 6
    assert tensor_ref_numel(ref) == 6


def test_policy_should_externalize_respects_allowlist_and_threshold() -> None:
    policy = TensorRefPolicy(
        threshold_bytes=64,
        from_stage="image_encoder",
        to_stage="mm_aggregate",
        consumer_stage="thinker",
        path_allowlist=("video_embeds",),
    )
    big = torch.zeros(32, dtype=torch.float32)
    small = torch.zeros(4, dtype=torch.float32)

    assert policy.should_externalize("encoder_outs.image_encoder.video_embeds", big)
    assert not policy.should_externalize(
        "encoder_outs.image_encoder.video_embeds", small
    )
    assert not policy.should_externalize("encoder_outs.image_encoder.image_embeds", big)


def test_policy_from_env_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_OMNI_ENABLE_TENSOR_REFS", raising=False)
    assert not tensor_refs_enabled()
    assert (
        TensorRefPolicy.from_env(from_stage="image_encoder", to_stage="mm_aggregate")
        is None
    )


def test_policy_from_env_requires_declared_edge(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_ENABLE_TENSOR_REFS", "1")
    monkeypatch.setenv(
        "SGLANG_OMNI_TENSOR_REF_EDGES", "image_encoder:mm_aggregate:thinker"
    )
    assert tensor_refs_enabled()
    assert (
        TensorRefPolicy.from_env(from_stage="mm_aggregate", to_stage="thinker") is None
    )
    policy = TensorRefPolicy.from_env(
        from_stage="image_encoder", to_stage="mm_aggregate"
    )
    assert policy is not None
    assert policy.consumer_stage == "thinker"
    assert DEFAULT_TENSOR_REF_THRESHOLD_MB == 2.0
    assert policy.threshold_bytes == 2 * 1024 * 1024
    assert policy.path_allowlist == DEFAULT_TENSOR_REF_PATHS


async def _write_and_read_mid_payload(
    relay: FakeRelay, tensor: torch.Tensor
) -> tuple[relay_io.StagePayload, torch.Tensor]:
    payload = make_stage_payload(
        request_id="req-1",
        data={"encoder_outs": {"image_encoder": {"video_embeds": tensor}}},
    )
    policy = TensorRefPolicy(
        threshold_bytes=1,
        from_stage="image_encoder",
        to_stage="mm_aggregate",
        consumer_stage="thinker",
        path_allowlist=("video_embeds",),
    )

    metadata, op = await relay_io.write_payload(
        relay,
        payload.request_id,
        payload,
        from_stage="image_encoder",
        to_stage="mm_aggregate",
        tensor_ref_policy=policy,
    )
    await op.wait_for_completion()
    assert metadata["tensor_ref_stats"]["ref_count"] == 1
    assert metadata["tensor_ref_stats"]["ref_bytes"] == tensor.numel() * 4

    mid_payload = await relay_io.read_payload(relay, payload.request_id, metadata)
    video_embeds = mid_payload.data["encoder_outs"]["image_encoder"]["video_embeds"]
    assert is_tensor_ref_dict(video_embeds)

    for task in list(relay_io._BACKGROUND_REF_TASKS):
        await task

    return mid_payload, tensor


@pytest.mark.parametrize(
    "current_stage,expect_resolved", [("thinker", True), ("mm_aggregate", False)]
)
def test_materialize_payload_tensor_refs_resolves_only_for_consumer_stage(
    current_stage, expect_resolved
) -> None:
    async def _run() -> None:
        relay = FakeRelay()
        tensor = torch.arange(8, dtype=torch.float32)
        mid_payload, original = await _write_and_read_mid_payload(relay, tensor)

        resolved_payload = await relay_io.materialize_payload_tensor_refs(
            relay, mid_payload, current_stage=current_stage
        )
        value = resolved_payload.data["encoder_outs"]["image_encoder"]["video_embeds"]

        if expect_resolved:
            assert torch.equal(value, original)
        else:
            assert is_tensor_ref_dict(value)

    asyncio.run(_run())


def test_materialize_tensor_refs_materialize_all_overrides_consumer_stage() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        tensor = torch.arange(8, dtype=torch.float32)
        mid_payload, original = await _write_and_read_mid_payload(relay, tensor)

        resolved_data = await relay_io.materialize_tensor_refs(
            relay,
            mid_payload.data,
            current_stage="mm_aggregate",
            materialize_all=True,
        )
        value = resolved_data["encoder_outs"]["image_encoder"]["video_embeds"]
        assert torch.equal(value, original)

    asyncio.run(_run())


def test_write_payload_without_policy_is_unchanged() -> None:
    """Default call sites (no tensor_ref_policy) keep today's exact wire format."""

    async def _run() -> None:
        relay = FakeRelay()
        payload = make_stage_payload(
            request_id="req-1", data={"x": torch.arange(4, dtype=torch.float32)}
        )
        metadata, op = await relay_io.write_payload(relay, payload.request_id, payload)
        await op.wait_for_completion()
        assert "tensor_ref_stats" not in metadata
        restored = await relay_io.read_payload(relay, payload.request_id, metadata)
        assert torch.equal(restored.data["x"], payload.data["x"])

    asyncio.run(_run())


def test_write_payload_keeps_small_allowlisted_tensor_inline_with_policy() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        tensor = torch.arange(4, dtype=torch.float32)
        payload = make_stage_payload(
            request_id="req-1",
            data={"encoder_outs": {"image_encoder": {"video_embeds": tensor}}},
        )
        policy = TensorRefPolicy(
            threshold_bytes=tensor.numel() * tensor.element_size() + 1,
            from_stage="image_encoder",
            to_stage="mm_aggregate",
            consumer_stage="thinker",
            path_allowlist=("video_embeds",),
        )

        metadata, op = await relay_io.write_payload(
            relay,
            payload.request_id,
            payload,
            from_stage="image_encoder",
            to_stage="mm_aggregate",
            tensor_ref_policy=policy,
        )
        await op.wait_for_completion()

        stats = metadata["tensor_ref_stats"]
        assert stats["ref_count"] == 0
        assert stats["ref_bytes"] == 0
        assert stats["inline_tensor_bytes"] == tensor.numel() * tensor.element_size()

        restored = await relay_io.read_payload(relay, payload.request_id, metadata)
        value = restored.data["encoder_outs"]["image_encoder"]["video_embeds"]
        assert torch.equal(value, tensor)
        assert not is_tensor_ref_dict(value)

    asyncio.run(_run())


def test_release_tensor_ref_blobs_from_metadata_drops_unconsumed_blob() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        tensor = torch.arange(8, dtype=torch.float32)
        payload = make_stage_payload(
            request_id="req-1",
            data={"encoder_outs": {"image_encoder": {"video_embeds": tensor}}},
        )
        policy = TensorRefPolicy(
            threshold_bytes=1,
            from_stage="image_encoder",
            to_stage="mm_aggregate",
            consumer_stage="thinker",
            path_allowlist=("video_embeds",),
        )

        metadata, op = await relay_io.write_payload(
            relay,
            payload.request_id,
            payload,
            from_stage="image_encoder",
            to_stage="mm_aggregate",
            tensor_ref_policy=policy,
        )
        await op.wait_for_completion()

        ref_blob = metadata["tensor_ref_blobs"][0]
        blob_key = ref_blob["blob_key"]
        assert blob_key in relay.storage

        released = relay_io.release_tensor_ref_blobs_from_metadata(relay, metadata)
        assert released == 1
        assert blob_key not in relay.storage

        for task in list(relay_io._BACKGROUND_REF_TASKS):
            await task

    asyncio.run(_run())


def test_destructive_relay_blob_read_is_single_use() -> None:
    """A blob read is destructive on the SHM backend (unlink-on-read): the first
    resolve succeeds, a second independent resolve of the same blob fails. This
    is why a TensorRef consumer MUST resolve exactly once -- the thinker
    (OmniScheduler, requires_tp_work_fanout=False) resolves once on the leader;
    a requires_tp_work_fanout=True consumer would double-unlink and crash.
    """

    async def _run() -> None:
        relay = DestructiveFakeRelay()
        tensor = torch.arange(8, dtype=torch.float32)
        metadata, op = await relay_io.write_blob(relay, "req-1:blob", tensor)
        await op.wait_for_completion()

        first = await relay_io.read_blob(relay, "req-1:blob", metadata)
        assert torch.equal(first, tensor)

        with pytest.raises(RuntimeError):
            await relay_io.read_blob(relay, "req-1:blob", metadata)

    asyncio.run(_run())
