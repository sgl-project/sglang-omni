# SPDX-License-Identifier: Apache-2.0
"""Speculative AR sessions: startup admission and history edits on real CPU KV pools, and tiny draft/target decoding on Hopper."""

import json
import subprocess
import sys
from array import array
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH, Req
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, SessionOperation, TimedChunk
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionBridge
from sglang_omni.scheduling.sglang_backend.history_cut import DraftSessionKV, HistoryCut
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.fixtures.tiny_speculative_session import (
    TinyTextAdapter,
    create_tiny_checkpoints,
)

HEAD_DIM = 8
CONTEXT_TOKENS = 64
ENGINE_RUN_TIMEOUT_SECONDS = 240


@pytest.fixture(autouse=True)
def published_session_settings() -> Iterator[None]:
    with get_context().override_server_args(page_size=1, strip_thinking_cache=False):
        yield


def rotary_table(theta: float) -> torch.Tensor:
    angles = torch.outer(
        torch.arange(CONTEXT_TOKENS).float(),
        theta ** (-torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM),
    )
    return torch.cat((angles.cos(), angles.sin()), -1)


class SessionPools:
    """Drive the bridge's host lifecycle while allocating and freeing real token slots."""

    def __init__(
        self, size: int = 64, reserve_tokens: int = 8, speculative: bool = True
    ) -> None:
        self.target = MHATokenToKVPool(
            size, 1, torch.float32, 1, HEAD_DIM, 2, "cpu", False
        )
        self.draft = MHATokenToKVPool(
            size, 1, torch.float32, 1, HEAD_DIM, 1, "cpu", False
        )
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size, torch.float32, "cpu", self.target, False
        )
        self.req_to_token_pool = ReqToTokenPool(4, CONTEXT_TOKENS, "cpu", False)
        self.tree_cache = UnifiedRadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=1,
                enable_kv_cache_events=False,
                eviction_policy="lru",
                tree_components=(ComponentType.FULL,),
            )
        )
        self.session_controller = SessionController(self.tree_cache)
        self.model_config = SimpleNamespace(vocab_size=128)
        self.max_req_len = CONTEXT_TOKENS
        self.max_running_requests = 4
        self.waiting_queue: list[Req] = []
        self.chunked_req = None
        self.adapter = TinyTextAdapter(
            DraftSessionKV(self.target, rotary_table(10000.0))
        )
        self.bridge = ARSessionBridge(
            self,
            self.adapter,
            speculative_reserve_tokens=reserve_tokens,
            draft_kv=(
                DraftSessionKV(self.draft, rotary_table(1000.0))
                if speculative
                else None
            ),
        )

    def synchronize_launched_decode(self) -> None:
        pass

    def resolve_pending_async(self) -> None:
        pass

    def run_abort_callback(self, request_id: str) -> None:
        pass

    def abort(self, request_id: str) -> None:
        request = self.bridge.units_by_request_id[request_id].session_request
        request.to_finish = FINISH_ABORT()

    def release_request_kv_cache(self, request: Req) -> None:
        release_kv_cache(request, self.tree_cache, checkpoint=True)

    def open(self, session_id: str) -> None:
        self.bridge.apply_operation(
            StagePayload("open", OmniRequest(None), {}),
            SessionOperation("open", SessionIdentity(session_id), ("text",)),
        )

    def append(
        self,
        session_id: str,
        request_id: str,
        token_ids: list[int],
        max_new_tokens: int,
    ) -> Req:
        payload = StagePayload(request_id, OmniRequest(None), {})
        self.bridge.accept(
            payload,
            SessionOperation(
                "append",
                SessionIdentity(session_id),
                ("text",),
                TimedChunk("text", 0, 0, 0, {}, format="text_delta"),
            ),
        )
        request = Req(
            request_id,
            "",
            array("q", token_ids),
            SamplingParams(max_new_tokens=max_new_tokens),
            vocab_size=128,
        )
        request.tokenizer = None
        request_data = SGLangARRequestData(req=request)
        self.bridge.create_session_request(payload, request_data)
        session_request = self.bridge.units_by_request_id[request_id].session_request
        session_request.omni_data = request_data
        return session_request

    def allocate(self, request: Req, token_count: int) -> torch.Tensor:
        request.refresh_fill_ids()
        self.tree_cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(request.full_untruncated_fill_ids[:-1]),
                req=request,
            )
        )
        assert self.req_to_token_pool.alloc([request]) is not None
        self.tree_cache.maybe_hand_to_session(request)
        previous_count = request.kv.kv_allocated_len
        slots = self.token_to_kv_pool_allocator.alloc(token_count - previous_count)
        assert slots is not None
        row = self.req_to_token_pool.req_to_token[request.kv.req_pool_idx]
        row[previous_count:token_count] = slots.to(row.dtype)
        request.kv.kv_allocated_len = token_count
        request.kv.kv_committed_len = token_count
        self.bridge.units_by_request_id[request.rid].is_enqueued = True
        return row[:token_count].long().clone()

    def finish(self, request: Req, output_ids: list[int], finished_tokens: int) -> None:
        request.output_ids = output_ids
        request.refresh_fill_ids()
        request.finished_reason = FINISH_LENGTH(finished_tokens)
        request.finished_len = finished_tokens
        self.release_request_kv_cache(request)
        self.bridge.complete(request.rid)


@pytest.mark.parametrize("cancel_append", [False, True])
def test_finished_blocks_resume_and_close_without_leaking_slots(
    cancel_append: bool,
) -> None:
    """A finished speculative turn keeps only the KV prefix plain decoding keeps; the next append reuses it, and cancel or close returns every slot."""
    pools = SessionPools()
    initial_free_slot_count = pools.token_to_kv_pool_allocator.available_size()
    pools.open("s")
    first_request = pools.append("s", "first", [1, 2, 3, 4, 5], 3)
    first_slots = pools.allocate(first_request, 12)
    pools.finish(first_request, [6, 7, 8, 9, 10], 3)
    record = pools.tree_cache.session_records()["s"]
    assert first_request.output_ids == [6, 7, 8]
    assert record.kv_allocated_len == 8
    assert record.kv_committed_len == 7
    assert (
        pools.token_to_kv_pool_allocator.available_size() == initial_free_slot_count - 8
    )

    second_request = pools.append("s", "second", [11, 12], 2)
    assert list(second_request.origin_input_ids) == [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    second_slots = pools.allocate(second_request, 16)
    assert torch.equal(first_slots[:7], second_slots[:7])
    if cancel_append:
        pools.bridge.cancel("second")
        assert not pools.tree_cache.session_records()
        third_request = pools.append("s", "third", [13], 1)
        assert list(third_request.origin_input_ids) == [1, 2, 3, 4, 5, 6, 7, 8, 13]
        pools.allocate(third_request, 10)
        pools.finish(third_request, [14], 1)
    else:
        pools.finish(second_request, [13, 14, 15, 16], 2)
    pools.bridge.close_session(pools.bridge.sessions["s"])
    assert pools.token_to_kv_pool_allocator.available_size() == initial_free_slot_count
    assert pools.req_to_token_pool.available_size() == 4
    assert not pools.tree_cache.session_records()


@pytest.mark.parametrize("speculative", [False, True])
def test_history_cut_moves_both_models_before_freeing_the_shared_tail(
    speculative: bool,
) -> None:
    """A history cut moves and re-rotates the retained keys of the target and, with a draft, the draft before their shared tail slots are freed."""
    pools = SessionPools(speculative=speculative)
    initial_free_slot_count = pools.token_to_kv_pool_allocator.available_size()
    pools.open("s")
    request = pools.append("s", "first", list(range(1, 9)), 3)
    cached_token_count = 11 if speculative else 10
    slots = pools.allocate(request, cached_token_count)
    original_keys = torch.randn(
        cached_token_count, 1, HEAD_DIM, generator=torch.Generator().manual_seed(4)
    )
    original_values = torch.randn(
        cached_token_count, 1, HEAD_DIM, generator=torch.Generator().manual_seed(5)
    )
    buffers_before_cut: list[tuple[torch.Tensor, torch.Tensor]] = []
    pools_with_rope_theta = [(pools.target, 10000.0)]
    if speculative:
        pools_with_rope_theta.append((pools.draft, 1000.0))
    else:
        pass
    for kv_pool, _ in pools_with_rope_theta:
        for layer_id in range(kv_pool.layer_num):
            kv_pool.get_key_buffer(layer_id)[slots] = original_keys + layer_id
            kv_pool.get_value_buffer(layer_id)[slots] = original_values + layer_id
            buffers_before_cut.append(
                (
                    kv_pool.get_key_buffer(layer_id).clone(),
                    kv_pool.get_value_buffer(layer_id).clone(),
                )
            )
    pools.finish(request, [20, 21, 22], 3)
    assert pools.tree_cache.session_records()["s"].kv_committed_len == 10
    free_slot_count_before_cut = pools.token_to_kv_pool_allocator.available_size()
    pools.bridge.drop_history(
        pools.bridge.sessions["s"],
        pools.session_controller.get("s"),
        HistoryCut(((2, 4),)),
    )
    kept_positions = torch.tensor([0, 1, 4, 5, 6, 7, 8, 9])
    distances = torch.full((6,), 2)
    for kv_pool, theta in pools_with_rope_theta:
        for layer_id in range(kv_pool.layer_num):
            keys, values = buffers_before_cut.pop(0)
            expected_keys = keys[slots[kept_positions]].clone()
            complex_keys = torch.complex(
                expected_keys[2:, :, : HEAD_DIM // 2],
                expected_keys[2:, :, HEAD_DIM // 2 :],
            )
            angles = -distances[:, None, None] * theta ** (
                -torch.arange(0, HEAD_DIM, 2).float()[None, None, :] / HEAD_DIM
            )
            rotated_keys = complex_keys * torch.polar(torch.ones_like(angles), angles)
            expected_keys[2:] = torch.cat((rotated_keys.real, rotated_keys.imag), -1)
            torch.testing.assert_close(
                kv_pool.get_key_buffer(layer_id)[slots[:8]], expected_keys
            )
            assert torch.equal(
                kv_pool.get_value_buffer(layer_id)[slots[:8]],
                values[slots[kept_positions]],
            )
    assert (
        pools.token_to_kv_pool_allocator.available_size()
        == free_slot_count_before_cut + cached_token_count - 8
    )
    assert pools.tree_cache.session_records()["s"].kv_allocated_len == 8
    next_request = pools.append("s", "next", [30], 1)
    assert list(next_request.origin_input_ids) == [1, 2, 5, 6, 7, 8, 20, 21, 22, 30]
    pools.bridge.cancel("next")
    pools.bridge.close_session(pools.bridge.sessions["s"])
    assert pools.token_to_kv_pool_allocator.available_size() == initial_free_slot_count


@pytest.mark.parametrize("reserve_tokens", [0, 8])
def test_session_capacity_reserves_a_draft_block_for_each_pending_append(
    reserve_tokens: int,
) -> None:
    """Session admission and the history capacity each reserve a draft block for every pending append."""
    pools = SessionPools(size=24, reserve_tokens=reserve_tokens)
    for session_id in ("a", "b"):
        pools.open(session_id)
    pools.append("a", "a1", [1, 2, 3], 4)
    assert pools.bridge.check_session_capacity("a1") is None
    pools.append("b", "b1", [4, 5, 6], 4)
    expected_refusal = "session KV capacity exhausted" if reserve_tokens else None
    assert pools.bridge.check_session_capacity("b1") == expected_refusal
    assert pools.adapter.history_capacities == [56 - reserve_tokens] * 2


@pytest.mark.parametrize(
    (
        "algorithm",
        "enable_overlap",
        "enable_multi_layer_eagle",
        "speculative_draft_window_size",
        "message",
    ),
    [
        (
            SpeculativeAlgorithm.DSPARK,
            False,
            False,
            None,
            "DFLASH and EAGLE3 drafts, not DSPARK$",
        ),
        (SpeculativeAlgorithm.EAGLE3, True, False, None, "synchronous"),
        (
            SpeculativeAlgorithm.EAGLE3,
            False,
            True,
            None,
            "single-runner EAGLE3 draft",
        ),
        (
            SpeculativeAlgorithm.EAGLE3,
            False,
            False,
            16,
            "without draft windowing",
        ),
    ],
)
def test_a_session_stage_refuses_a_draft_it_cannot_keep(
    monkeypatch: pytest.MonkeyPatch,
    algorithm: SpeculativeAlgorithm,
    enable_overlap: bool,
    enable_multi_layer_eagle: bool,
    speculative_draft_window_size: int | None,
    message: str,
) -> None:
    """Other algorithms, overlap, multi-runner EAGLE and windowed drafts are refused at startup, each naming its cause."""
    pools = SessionPools()
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler.get_spec",
        lambda: SimpleNamespace(
            enable_multi_layer_eagle=enable_multi_layer_eagle,
            speculative_draft_window_size=speculative_draft_window_size,
        ),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.draft_worker = SimpleNamespace(
        draft_worker=SimpleNamespace(
            draft_runner=SimpleNamespace(
                token_to_kv_pool=pools.draft,
                req_to_token_pool=pools.req_to_token_pool,
                model=SimpleNamespace(model=SimpleNamespace(layers=[])),
            )
        )
    )
    scheduler.spec_algorithm = algorithm
    scheduler.enable_overlap = enable_overlap

    with pytest.raises(ValueError, match=message):
        scheduler.speculative_session_draft_kv(
            pools.req_to_token_pool, pools.token_to_kv_pool_allocator
        )


@pytest.mark.parametrize(
    ("draft_page_size", "shares_token_table", "is_neox_style", "message"),
    [
        (1, True, True, None),
        (4, True, True, "static NHD token pools"),
        (1, False, True, "static NHD token pools"),
        (1, True, False, "shared full Neox RoPE"),
    ],
)
def test_a_session_stage_keeps_only_a_draft_its_history_cut_can_move(
    monkeypatch: pytest.MonkeyPatch,
    draft_page_size: int,
    shares_token_table: bool,
    is_neox_style: bool,
    message: str | None,
) -> None:
    """A DFlash draft on a static token pool with Neox RoPE is kept; a paged pool, its own token table or GPT-J RoPE is refused."""
    pools = SessionPools()
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler.get_spec",
        lambda: SimpleNamespace(speculative_draft_window_size=None),
    )
    draft_pool = MHATokenToKVPool(
        64, draft_page_size, torch.float32, 1, HEAD_DIM, 1, "cpu", False
    )
    # note (evanderfan): RotaryEmbedding's constructor needs vllm off CUDA; the check reads only these fields.
    rotary_embedding = object.__new__(RotaryEmbedding)
    torch.nn.Module.__init__(rotary_embedding)
    rotary_embedding.head_size = HEAD_DIM
    rotary_embedding.rotary_dim = HEAD_DIM
    rotary_embedding.is_neox_style = is_neox_style
    rotary_embedding.position_start = 0
    rotary_embedding.cos_sin_cache = rotary_table(1000.0)
    draft_layer = SimpleNamespace(
        self_attn=SimpleNamespace(rotary_emb=rotary_embedding)
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.spec_algorithm = SpeculativeAlgorithm.DFLASH
    scheduler.enable_overlap = False
    scheduler.draft_worker = SimpleNamespace(
        draft_model_runner=SimpleNamespace(
            token_to_kv_pool=draft_pool,
            req_to_token_pool=(
                pools.req_to_token_pool
                if shares_token_table
                else ReqToTokenPool(4, CONTEXT_TOKENS, "cpu", False)
            ),
        ),
        draft_model=SimpleNamespace(layers=[draft_layer, draft_layer]),
    )

    if message is None:
        draft_kv = scheduler.speculative_session_draft_kv(
            pools.req_to_token_pool, pools.token_to_kv_pool_allocator
        )
        assert draft_kv.cache is draft_pool
        assert draft_kv.cos_sin_cache is rotary_embedding.cos_sin_cache
    else:
        with pytest.raises(ValueError, match=message):
            scheduler.speculative_session_draft_kv(
                pools.req_to_token_pool, pools.token_to_kv_pool_allocator
            )


@pytest.mark.accelerator
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("speculative_algorithm", ["DFLASH", "EAGLE3"])
def test_tiny_speculative_sessions_match_plain_decode(
    tmp_path: Path, speculative_algorithm: str, captured: bool
) -> None:
    """Real draft/target forwards must preserve turn outputs through cache reuse, cuts and aborts."""
    require_cuda()
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the FA3 session integration test requires Hopper")
    else:
        pass
    target_dir, draft_dirs = create_tiny_checkpoints(tmp_path)
    output_ids_by_mode: list[dict[str, list[int]]] = []
    for algorithm, draft_path in (
        ("", ""),
        (speculative_algorithm, str(draft_dirs[speculative_algorithm])),
    ):
        # note (evanderfan): a process publishes one SGLang runtime context, so each decode mode builds its engine in a fresh interpreter.
        script = (
            "import json, sys; "
            "from tests.unit_test.fixtures.tiny_speculative_session import run_tiny_sessions; "
            "result = run_tiny_sessions("
            'sys.argv[1], sys.argv[2] or None, sys.argv[3] or None, sys.argv[4] == "True"); '
            'open(sys.argv[5], "w").write(json.dumps(result))'
        )
        result_path = tmp_path / f"result-{len(output_ids_by_mode)}.json"
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(target_dir),
                algorithm,
                draft_path,
                str(captured),
                str(result_path),
            ],
            check=True,
            timeout=ENGINE_RUN_TIMEOUT_SECONDS,
        )
        output_ids_by_mode.append(json.loads(result_path.read_text()))
    assert output_ids_by_mode[0] == output_ids_by_mode[1]
    assert len(output_ids_by_mode[0]) == 8
