# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import sglang.srt.managers.schedule_policy as schedule_policy
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from sglang.srt.runtime_context import get_context

from sglang_omni.models.qwen3_omni.config import TALKER_MAX_NEW_TOKENS_ESTIMATION

_TALKER_KV_TOKENS = 33_377
_CONCURRENCY = 64
_PROMPT_TOKENS = 128
_TALKER_MAX_NEW_TOKENS = 4096
_PAGE_SIZE = 1


@pytest.fixture(autouse=True)
def _scheduler_server_args() -> Iterator[None]:
    # Restore the process-wide context so later bootstrap tests can publish it.
    with get_context().override_server_args():
        yield


def _talker_request(rid: int) -> MagicMock:
    req = MagicMock(spec=Req)
    req.rid = str(rid)
    req.prefix_indices = []
    req.full_untruncated_fill_ids = list(range(_PROMPT_TOKENS))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(
        max_new_tokens=_TALKER_MAX_NEW_TOKENS, ignore_eos=False
    )
    req.host_hit_length = 0
    req.storage_hit_length = 0
    req.retracted_stain = False
    req.last_node = MagicMock()
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    return req


def _admitted_from_empty_pool(clip: int, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(schedule_policy, "CLIP_MAX_NEW_TOKENS", clip)
    tree_cache = MagicMock()
    tree_cache.disable = False
    tree_cache.evictable_size.return_value = 0
    tree_cache.supports_mamba.return_value = False
    tree_cache.is_tree_cache.return_value = False
    tree_cache.inc_lock_ref.return_value = IncLockRefResult()
    tree_cache.dec_lock_ref.return_value = DecLockRefResult()
    allocator = MagicMock()
    allocator.available_size.return_value = _TALKER_KV_TOKENS
    running_batch = MagicMock()
    running_batch.reqs = []

    adder = PrefillAdder(
        page_size=_PAGE_SIZE,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=running_batch,
        new_token_ratio=1.0,
        rem_input_tokens=1_000_000,
        rem_chunk_tokens=None,
    )
    for rid in range(_CONCURRENCY):
        result = adder.add_one_req(
            _talker_request(rid), has_chunked_req=False, truncation_align_size=None
        )
        if result == AddReqResult.NO_TOKEN:
            break
    return len(adder.can_run_list)


def test_sglang_default_clip_admits_a_fraction_of_the_talker_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _admitted_from_empty_pool(4096, monkeypatch)

    assert admitted == _TALKER_KV_TOKENS // (
        _PROMPT_TOKENS + _TALKER_MAX_NEW_TOKENS + _PAGE_SIZE
    )
    assert admitted < _CONCURRENCY


def test_speech_length_clip_admits_the_whole_talker_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = int(TALKER_MAX_NEW_TOKENS_ESTIMATION)

    assert _admitted_from_empty_pool(clip, monkeypatch) == _CONCURRENCY
