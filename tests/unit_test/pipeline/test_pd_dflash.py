# SPDX-License-Identifier: Apache-2.0
"""DFlash handoff tests for the shared slot mapping and PREBUILT admission."""

import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from huggingface_hub import constants as hub_constants
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from transformers import PretrainedConfig

from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_scheduler import OmniDecodeScheduler, validate_pd_runtime
from sglang_omni.scheduling.pd_utils import (
    build_kv_pool,
    dflash_checkpoint_identity,
    kv_buffer_regions,
)
from tests.unit_test.pipeline.test_pd_lifecycle import decode_scheduler
from tests.unit_test.pipeline.test_pd_utils import make_continuation, make_message


class RowKVCache:
    def __init__(self, size: int, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        self.size = size
        self.page_size = 1
        self.kv_rows = [torch.zeros((size + 1, *shape), dtype=dtype) for _ in range(2)]

    def _pd_registerable_tensors(
        self,
    ) -> list[torch.Tensor]:  # noqa: leading-underscore -- SGLang KVCache contract
        return self.kv_rows

    def get_contiguous_buf_infos(self) -> tuple[list[int], list[int], list[int]]:
        return (
            [tensor.data_ptr() for tensor in self.kv_rows],
            [tensor.numel() * tensor.element_size() for tensor in self.kv_rows],
            [tensor[0].numel() * tensor.element_size() for tensor in self.kv_rows],
        )


def test_joint_kv_layout_keeps_row_order_and_supports_different_stage_capacities() -> (
    None
):
    source_target = RowKVCache(8, (2, 4), torch.bfloat16)
    source_draft = RowKVCache(8, (1, 4), torch.bfloat16)
    destination_target = RowKVCache(12, (2, 4), torch.bfloat16)
    destination_draft = RowKVCache(12, (1, 4), torch.bfloat16)
    source = build_kv_pool(
        source_target,
        pool_id="prefill:kv",
        draft_kv_pool=source_draft,
        dflash_model_id="model",
    )
    destination = build_kv_pool(
        destination_target,
        pool_id="decode:kv",
        draft_kv_pool=destination_draft,
        dflash_model_id="model",
    )
    assert source.layout.compatible_with(destination.layout)
    assert [buffer.name for buffer in source.buffers] == [
        "kv_buffer.0",
        "kv_buffer.1",
        "draft_kv_buffer.0",
        "draft_kv_buffer.1",
    ]
    assert [buffer.bytes_per_page for buffer in source.buffers] == [16, 16, 8, 8]
    for buffer, tensor in zip(
        source.buffers, source_target.kv_rows + source_draft.kv_rows, strict=True
    ):
        assert buffer.tensor is tensor
    destination.validate_page_indices((9, 12))

    for wrong_draft, model_id in [
        (RowKVCache(12, (1, 4), torch.float16), "model"),
        (RowKVCache(12, (2, 2), torch.bfloat16), "model"),
        (destination_draft, "other-checkpoint"),
    ]:
        incompatible = build_kv_pool(
            destination_target,
            pool_id="decode:kv",
            draft_kv_pool=wrong_draft,
            dflash_model_id=model_id,
        )
        assert not source.layout.compatible_with(incompatible.layout)
    plain = build_kv_pool(destination_target, pool_id="decode:kv")
    assert not source.layout.compatible_with(plain.layout)


def test_joint_kv_rejects_wrong_row_capacity_or_registration_order() -> None:
    target = RowKVCache(8, (2, 4), torch.bfloat16)
    draft = RowKVCache(7, (1, 4), torch.bfloat16)
    with pytest.raises(ValueError, match="slot capacity"):
        build_kv_pool(
            target, pool_id="decode:kv", draft_kv_pool=draft, dflash_model_id="model"
        )
    draft = RowKVCache(8, (1, 4), torch.bfloat16)
    pointers, lengths, strides = draft.get_contiguous_buf_infos()
    draft.get_contiguous_buf_infos = lambda: (pointers[::-1], lengths, strides)
    with pytest.raises(ValueError, match="contiguous KV slot rows"):
        build_kv_pool(
            target, pool_id="decode:kv", draft_kv_pool=draft, dflash_model_id="model"
        )


def test_kv_region_validation_uses_full_pages() -> None:
    pool = MHATokenToKVPool(
        size=8,
        page_size=4,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=4,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    regions = kv_buffer_regions(pool, prefix="kv", validate_slot_layout=True)

    assert [region.bytes_per_page for region in regions] == [64, 64]
    assert [region.page_count for region in regions] == [3, 3]
    with pytest.raises(ValueError, match="page_size"):
        build_kv_pool(
            pool, pool_id="decode:kv", draft_kv_pool=pool, dflash_model_id="model"
        )


@pytest.mark.parametrize(
    "variant, error",
    [
        ("valid", None),
        ("missing_opt_in", "explicitly configured DFlash"),
        ("private_table", "shared, direct KV slot mapping"),
        ("private_allocator", "shared, direct KV slot mapping"),
        ("translated_slots", "shared, direct KV slot mapping"),
    ],
)
def test_pd_dflash_validates_the_initialized_worker(
    variant: str, error: str | None
) -> None:
    scheduler = decode_scheduler()
    scheduler.session_adapter = None
    scheduler.server_args = SimpleNamespace(disable_radix_cache=True)
    scheduler.spec_algorithm = SpeculativeAlgorithm.DFLASH
    scheduler.tp_size = 1
    scheduler.page_size = 1
    scheduler.device = "cuda:0"
    scheduler.enable_plain_fallback = False
    scheduler.token_to_kv_pool_allocator.get_kvcache = Mock(
        return_value=RowKVCache(8, (2, 4), torch.bfloat16)
    )
    target_runner = SimpleNamespace(
        token_to_kv_pool=scheduler.token_to_kv_pool_allocator.get_kvcache(),
        req_to_token_pool=scheduler.req_to_token_pool,
        kv_index_translator=SimpleNamespace(
            space=lambda kind: SimpleNamespace(write=None)
        ),
    )
    draft_runner = SimpleNamespace(
        token_to_kv_pool=RowKVCache(8, (1, 4), torch.bfloat16),
        req_to_token_pool=(
            object() if variant == "private_table" else scheduler.req_to_token_pool
        ),
        token_to_kv_pool_allocator=(
            object()
            if variant == "private_allocator"
            else scheduler.token_to_kv_pool_allocator
        ),
        kv_index_translator=SimpleNamespace(
            space=lambda kind: SimpleNamespace(
                write=int if variant == "translated_slots" else None
            )
        ),
    )
    scheduler.tp_worker = SimpleNamespace(model_runner=target_runner)
    scheduler.draft_worker = SimpleNamespace(
        draft_worker=SimpleNamespace(model_runner=draft_runner), selector=None
    )
    enable_dflash = variant != "missing_opt_in"
    if error is None:
        assert (
            validate_pd_runtime(scheduler, enable_dflash=enable_dflash)
            is draft_runner.token_to_kv_pool
        )
    else:
        with pytest.raises((ValueError, NotImplementedError), match=error):
            validate_pd_runtime(scheduler, enable_dflash=enable_dflash)


def test_dflash_identity_uses_the_loaded_target_revision(tmp_path: Path) -> None:
    weight = tmp_path / "draft.safetensors"
    weight.touch()
    loader = LoadConfig(load_format="auto")
    target = SimpleNamespace(
        model_path="target/checkpoint",
        revision="first",
        hf_config=SimpleNamespace(),
        dtype=torch.bfloat16,
        quantization=None,
        model_override_args={},
    )
    identities = []
    for stage in ("prefill", "decode"):
        directory = tmp_path / stage
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").symlink_to(weight)
        draft = SimpleNamespace(
            model_path=str(directory),
            dtype=torch.bfloat16,
            quantization=None,
            model_override_args={},
        )
        identities.append(dflash_checkpoint_identity(target, loader, draft, loader))
    assert identities[0] == identities[1]
    safetensors_loader = LoadConfig(load_format="safetensors")
    assert (
        dflash_checkpoint_identity(target, loader, draft, safetensors_loader)
        != identities[0]
    )
    target.revision = "second"
    assert dflash_checkpoint_identity(target, loader, draft, loader) != identities[0]


def test_dflash_identity_distinguishes_cached_hub_draft_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = LoadConfig(load_format="auto")
    target = SimpleNamespace(
        model_path="target/checkpoint",
        revision="main",
        hf_config=PretrainedConfig(),
        dtype=torch.bfloat16,
        quantization=None,
        model_override_args={},
    )
    draft = SimpleNamespace(**vars(target))
    draft.model_path = "draft/checkpoint"
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--draft--checkpoint"
    reference = repo / "refs" / "main"
    reference.parent.mkdir(parents=True)
    first_commit, second_commit = "a" * 40, "b" * 40
    for commit in (first_commit, second_commit):
        cached_config = repo / "snapshots" / commit / "config.json"
        cached_config.parent.mkdir(parents=True)
        cached_config.write_text("{}")
    reference.write_text(first_commit)
    identity = dflash_checkpoint_identity(target, loader, draft, loader)
    assert dflash_checkpoint_identity(target, loader, draft, loader) == identity
    reference.write_text(second_commit)
    assert dflash_checkpoint_identity(target, loader, draft, loader) != identity


@pytest.mark.parametrize("overlap", [False, True])
def test_committed_dflash_request_seeds_native_prebuilt(overlap: bool) -> None:
    scheduler = decode_scheduler()
    scheduler.draft_worker = Mock()
    scheduler.req_to_token_pool = ReqToTokenPool(
        4, 32, "cpu", enable_memory_saver=False
    )
    prepare = make_message()
    receiver = scheduler.pd_receiver
    destination = receiver.reserve(prepare)
    scheduler.drain_decode_admissions()
    assert scheduler.waiting_queue == []

    receiver.commit(prepare, destination)
    scheduler.drain_decode_admissions()
    [request] = scheduler.waiting_queue
    assert request.omni_data.emitted_token_count == 0
    assert list(request.output_ids) == [42]
    assert request.sampling_params.sampling_seed == 17
    assert request.sampling_params.temperature == pytest.approx(0.7)
    assert request.sampling_params.top_p == pytest.approx(0.9)
    assert request.sampling_params.stop_token_ids == {2}
    assert scheduler.req_to_token_pool.req_to_token[
        request.kv.req_pool_idx, :3
    ].tolist() == list(destination.page_indices)

    future_map = Mock(spec=FutureMap)
    with get_context().override_server_args(speculative_algorithm="DFLASH"):
        batch = ScheduleBatch.init_new(
            [request],
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
            Mock(),
            SimpleNamespace(vocab_size=128),
            overlap,
            SpeculativeAlgorithm.DFLASH,
        )
        batch.prepare_for_prebuilt()
        batch.process_prebuilt(future_map)
    assert batch.forward_mode.is_prebuilt()
    assert isinstance(batch.spec_info, DFlashDraftInputV2)
    assert batch.input_ids is None
    assert batch.spec_info.bonus_tokens.tolist() == [42]
    assert batch.spec_info.new_seq_lens.tolist() == [3]
    if overlap:
        assert batch.spec_info.future_indices.tolist() == [request.kv.req_pool_idx]
        assert future_map.publish.call_count == future_map.stash.call_count == 1
        assert future_map.stash.call_args.args[1].bonus_tokens.tolist() == [42]
    else:
        future_map.publish.assert_not_called()
        future_map.stash.assert_not_called()


def test_rejected_dflash_admission_frees_slots_and_retry_admits() -> None:
    scheduler = decode_scheduler()
    scheduler.is_entry_rank = True
    scheduler.draft_worker = Mock()
    receiver = scheduler.pd_receiver
    prepare = make_message()
    continuation = replace(make_continuation(), output_ids=[42, 43])
    prepare = replace(prepare, metadata={"decode_continuation": continuation.encode()})
    destination = receiver.reserve(prepare)
    receiver.commit(prepare, destination)
    scheduler.drain_decode_admissions()

    assert scheduler.waiting_queue == []
    assert scheduler.outbox.get_nowait().type == "admitted"
    assert scheduler.outbox.get_nowait().type == "error"
    assert [slots.tolist() for slots in scheduler.token_to_kv_pool_allocator.freed] == [
        list(destination.page_indices)
    ]
    valid = make_message(transfer_id="retry")
    receiver.commit(valid, receiver.reserve(valid))
    scheduler.drain_decode_admissions()
    assert [request.rid for request in scheduler.waiting_queue] == [valid.request_id]


@pytest.mark.parametrize("enable_overlap", [False, True])
def test_off_thread_decode_abort_blocks_admission_and_defers_row_release(
    monkeypatch: pytest.MonkeyPatch, enable_overlap: bool
) -> None:
    scheduler: OmniDecodeScheduler = decode_scheduler()
    prepare = make_message()
    scheduler.pd_receiver.commit(prepare, scheduler.pd_receiver.reserve(prepare))
    scheduler.drain_decode_admissions()
    pending_continuation = replace(
        make_continuation(), request_id="pending", transfer_id="pending-transfer"
    )
    pending = replace(
        prepare,
        request_id=pending_continuation.request_id,
        transfer_id=pending_continuation.transfer_id,
        metadata={"decode_continuation": pending_continuation.encode()},
    )
    allocation = scheduler.pd_receiver.reserve(pending)
    scheduler.pd_receiver.commit(pending, allocation)
    scheduler.enable_overlap = enable_overlap
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.is_entry_rank = True
    scheduler.inbox = queue.SimpleQueue()
    scheduler.aborted_request_id_order = deque()
    parent_abort = OmniScheduler.abort

    def abort(
        self: OmniScheduler, request_id: str, *, defer_running_cleanup: bool = True
    ) -> None:
        if threading.get_ident() != self.scheduler_thread_id:
            parent_abort(self, request_id, defer_running_cleanup=defer_running_cleanup)
        else:
            self.waiting_queue = [
                request for request in self.waiting_queue if request.rid != request_id
            ]

    monkeypatch.setattr(OmniScheduler, "abort", abort)
    with ThreadPoolExecutor(1) as worker:
        worker.submit(scheduler.abort, prepare.request_id).result(timeout=5)
        worker.submit(scheduler.abort, pending.request_id).result(timeout=5)
    assert scheduler.pd_due_releases.empty()
    [request] = scheduler.waiting_queue
    scheduler.drain_decode_admissions()
    assert scheduler.waiting_queue == [request]
    assert [slots.tolist() for slots in scheduler.token_to_kv_pool_allocator.freed] == [
        list(allocation.page_indices)
    ]

    events = [scheduler.inbox.get_nowait(), scheduler.inbox.get_nowait()]
    assert [(event.type, event.request_id) for event in events] == [
        ("abort", prepare.request_id),
        ("abort", pending.request_id),
    ]
    for event in events:
        scheduler.abort(event.request_id)
        scheduler.abort(event.request_id)
    assert scheduler.pd_due_releases.get_nowait() is request
    assert scheduler.pd_due_releases.empty()
