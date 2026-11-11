# SPDX-License-Identifier: Apache-2.0
"""DFlash speculative decoding across the PD handoff, without weights or GPU kernels."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_scheduler import (
    OmniDecodeScheduler,
    OmniPrefillScheduler,
    validate_pd_runtime,
)
from sglang_omni.scheduling.pd_utils import DecodeAdmission, build_kv_pool
from tests.unit_test.pipeline.test_pd_lifecycle import decode_scheduler
from tests.unit_test.pipeline.test_pd_utils import (
    KVAllocator,
    ReqPool,
    make_allocation,
    make_continuation,
)


class FakeKVCache:
    def __init__(self, *, layers: int, slots: int, width: int) -> None:
        self.layer_num = layers
        self.k = [torch.zeros(slots, width) for _ in range(layers)]
        self.v = [torch.zeros(slots, width) for _ in range(layers)]

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v[layer_id]

    def get_contiguous_buf_infos(self):
        buffers = [t for pair in zip(self.k, self.v) for t in pair]
        return (
            [t.data_ptr() for t in buffers],
            [t.numel() * t.element_size() for t in buffers],
            [t[0].numel() * t.element_size() for t in buffers],
        )


class FakeDraftKVCache(FakeKVCache):
    pass


def test_kv_pool_appends_the_draft_buffers_after_the_target() -> None:
    target = FakeKVCache(layers=2, slots=8, width=4)
    draft = FakeDraftKVCache(layers=1, slots=8, width=2)

    plain = build_kv_pool(target, pool_id="decode:kv")
    pool = build_kv_pool(target, pool_id="decode:kv", draft_kv_pool=draft)

    assert [buffer.name for buffer in pool.buffers] == [
        "kv_buffer.0",
        "kv_buffer.1",
        "kv_buffer.2",
        "kv_buffer.3",
        "draft_kv_buffer.0",
        "draft_kv_buffer.1",
    ]
    assert pool.buffers[4].tensor is draft.k[0]
    assert pool.buffers[4].bytes_per_page == 2 * 4
    assert all(
        ours.tensor is theirs.tensor and ours.name == theirs.name
        for ours, theirs in zip(pool.buffers[:4], plain.buffers, strict=True)
    )
    # A Prefill half with a draft and a Decode half without one must not pair.
    assert pool.layout != plain.layout
    assert "+draft=" in pool.layout_id
    pool.validate_page_indices((0, 7))


def pd_scheduler(
    scheduler_type: type[OmniScheduler],
    algorithm: SpeculativeAlgorithm,
    *,
    plain_fallback: bool = False,
    draft_pool: object | None = None,
) -> OmniScheduler:
    scheduler = object.__new__(scheduler_type)
    scheduler.session_adapter = None
    scheduler.req_to_token_pool = ReqPool()
    scheduler.token_to_kv_pool_allocator = KVAllocator()
    scheduler.tp_size = 1
    scheduler.page_size = 1
    scheduler.server_args = SimpleNamespace(disable_radix_cache=True)
    scheduler.spec_algorithm = algorithm
    scheduler.enable_plain_fallback = plain_fallback
    scheduler.draft_worker = (
        None
        if algorithm.is_none()
        else SimpleNamespace(primary_draft_kv_pool=draft_pool)
    )
    return scheduler


@pytest.mark.parametrize("scheduler_type", [OmniPrefillScheduler, OmniDecodeScheduler])
def test_pd_accepts_dflash_with_a_device_draft_pool(scheduler_type) -> None:
    validate_pd_runtime(
        pd_scheduler(scheduler_type, SpeculativeAlgorithm.DFLASH, draft_pool=object())
    )
    validate_pd_runtime(pd_scheduler(scheduler_type, SpeculativeAlgorithm.NONE))


@pytest.mark.parametrize("scheduler_type", [OmniPrefillScheduler, OmniDecodeScheduler])
@pytest.mark.parametrize(
    "algorithm, plain_fallback, draft_pool, message",
    [
        (
            SpeculativeAlgorithm.EAGLE,
            False,
            object(),
            "DFLASH speculative decoding only",
        ),
        (SpeculativeAlgorithm.DFLASH, True, object(), "plain-fallback"),
        (SpeculativeAlgorithm.DFLASH, False, None, "device-resident DFLASH draft"),
    ],
)
def test_pd_rejects_speculative_state_it_does_not_carry(
    scheduler_type, algorithm, plain_fallback, draft_pool, message
) -> None:
    scheduler = pd_scheduler(
        scheduler_type,
        algorithm,
        plain_fallback=plain_fallback,
        draft_pool=draft_pool,
    )
    with pytest.raises(NotImplementedError, match=message):
        validate_pd_runtime(scheduler)


@pytest.mark.parametrize("speculative", [False, True])
def test_decode_admission_keeps_the_speculative_path(speculative: bool) -> None:
    scheduler = decode_scheduler()
    scheduler.draft_worker = object() if speculative else None
    scheduler.pd_admissions.put(DecodeAdmission(make_continuation(), make_allocation()))

    scheduler.drain_decode_admissions()

    [request] = scheduler.waiting_queue
    assert request.omni_data.decode_path == ("spec" if speculative else "plain")


def test_speculative_decode_leaves_the_first_token_to_committed_emission(
    monkeypatch,
) -> None:
    scheduler = decode_scheduler()
    scheduler.draft_worker = object()
    scheduler.emit_transferred_token = True
    scheduler.stream_output_builder = Mock()
    calls = []
    monkeypatch.setattr(
        OmniScheduler,
        "stream_output",
        lambda self, reqs, return_logprob=False, skip_req=None: calls.append(reqs),
    )
    reqs = [SimpleNamespace(rid="request-1", output_ids=[42])]

    scheduler.stream_output(reqs)

    # Committed-token emission starts counting at the transferred token, so
    # staging it here as well would stream it twice.
    assert calls == [reqs]
    scheduler.stream_output_builder.assert_not_called()
    assert scheduler.outbox.empty()
