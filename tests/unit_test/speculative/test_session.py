# SPDX-License-Identifier: Apache-2.0
"""CPU session ownership and history edits on real SGLang KV pools, without inference."""

import json
import subprocess
import sys
from array import array
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
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
from transformers import Qwen3Config, Qwen3ForCausalLM

from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, SessionOperation, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionBridge,
)
from sglang_omni.scheduling.sglang_backend.history_cut import DraftSessionKV, HistoryCut
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from tests.unit_test.fixtures.accelerator import require_cuda

HEAD_DIM = 8
CONTEXT_TOKENS = 64


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


class TextSessionAdapter(ARSessionAdapter):
    def __init__(self, target_kv: DraftSessionKV) -> None:
        self.target_kv = target_kv
        self.history_capacities: list[int] = []

    def plan_history_cut(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        history_length: int,
        history_capacity: int,
    ) -> HistoryCut | None:
        self.history_capacities.append(history_capacity)
        return None

    def shift_key_positions(
        self, keys: torch.Tensor, distances: torch.Tensor
    ) -> torch.Tensor:
        return self.target_kv.shift_key_positions(keys, distances)


class SessionPools:
    """Drive the bridge's host lifecycle while allocating and freeing real token slots."""

    def __init__(self, size: int = 64, reserve_tokens: int = 8) -> None:
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
        self.adapter = TextSessionAdapter(
            DraftSessionKV(self.target, rotary_table(10000.0))
        )
        self.bridge = ARSessionBridge(
            self,
            self.adapter,
            speculative_reserve_tokens=reserve_tokens,
            draft_kv=DraftSessionKV(self.draft, rotary_table(1000.0)),
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
        self, session_id: str, request_id: str, token_ids: list[int], output_tokens: int
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
            SamplingParams(max_new_tokens=output_tokens),
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
    pools = SessionPools()
    initial_free = pools.token_to_kv_pool_allocator.available_size()
    pools.open("s")
    first = pools.append("s", "first", [1, 2, 3, 4, 5], 3)
    first_slots = pools.allocate(first, 12)
    pools.finish(first, [6, 7, 8, 9, 10], 3)
    record = pools.tree_cache.session_records()["s"]
    assert first.output_ids == [6, 7, 8]
    assert record.kv_allocated_len == record.kv_committed_len == 8
    assert pools.token_to_kv_pool_allocator.available_size() == initial_free - 8

    second = pools.append("s", "second", [11, 12], 2)
    assert list(second.origin_input_ids) == [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    second_slots = pools.allocate(second, 16)
    assert torch.equal(first_slots[:8], second_slots[:8])
    if cancel_append:
        pools.bridge.cancel("second")
        assert not pools.tree_cache.session_records()
        third = pools.append("s", "third", [13], 1)
        assert list(third.origin_input_ids) == [1, 2, 3, 4, 5, 6, 7, 8, 13]
        pools.allocate(third, 10)
        pools.finish(third, [14], 1)
    else:
        pools.finish(second, [13, 14, 15, 16], 2)
    pools.bridge.close_session(pools.bridge.sessions["s"])
    assert pools.token_to_kv_pool_allocator.available_size() == initial_free
    assert pools.req_to_token_pool.available_size() == 4
    assert not pools.tree_cache.session_records()


def test_history_cut_moves_both_models_before_freeing_the_shared_tail() -> None:
    pools = SessionPools()
    pools.open("s")
    request = pools.append("s", "first", list(range(1, 9)), 3)
    slots = pools.allocate(request, 11)
    original_keys = torch.randn(
        11, 1, HEAD_DIM, generator=torch.Generator().manual_seed(4)
    )
    original_values = torch.randn(
        11, 1, HEAD_DIM, generator=torch.Generator().manual_seed(5)
    )
    before: list[tuple[torch.Tensor, torch.Tensor]] = []
    for cache in (pools.target, pools.draft):
        for layer in range(cache.layer_num):
            cache.get_key_buffer(layer)[slots] = original_keys + layer
            cache.get_value_buffer(layer)[slots] = original_values + layer
            before.append(
                (
                    cache.get_key_buffer(layer).clone(),
                    cache.get_value_buffer(layer).clone(),
                )
            )
    pools.finish(request, [20, 21, 22], 3)
    free_before = pools.token_to_kv_pool_allocator.available_size()
    pools.bridge.drop_history(
        pools.bridge.sessions["s"],
        pools.session_controller.get("s"),
        HistoryCut(((2, 4),)),
    )
    kept = torch.tensor([0, 1, 4, 5, 6, 7, 8, 9, 10])
    distances = torch.full((7,), 2)
    for cache, theta in ((pools.target, 10000.0), (pools.draft, 1000.0)):
        for layer in range(cache.layer_num):
            keys, values = before.pop(0)
            expected_keys = keys[slots[kept]].clone()
            complex_keys = torch.complex(
                expected_keys[2:, :, : HEAD_DIM // 2],
                expected_keys[2:, :, HEAD_DIM // 2 :],
            )
            angles = -distances[:, None, None] * theta ** (
                -torch.arange(0, HEAD_DIM, 2).float()[None, None, :] / HEAD_DIM
            )
            rotated = complex_keys * torch.polar(torch.ones_like(angles), angles)
            expected_keys[2:] = torch.cat((rotated.real, rotated.imag), -1)
            torch.testing.assert_close(
                cache.get_key_buffer(layer)[slots[:9]], expected_keys
            )
            assert torch.equal(
                cache.get_value_buffer(layer)[slots[:9]], values[slots[kept]]
            )
    assert pools.token_to_kv_pool_allocator.available_size() == free_before + 2
    next_request = pools.append("s", "next", [30], 1)
    assert list(next_request.origin_input_ids) == [1, 2, 5, 6, 7, 8, 20, 21, 22, 30]
    pools.bridge.cancel("next")
    pools.bridge.close_session(pools.bridge.sessions["s"])
    assert pools.token_to_kv_pool_allocator.available_size() == 64


@pytest.mark.parametrize("reserve_tokens", [0, 8])
def test_session_capacity_reserves_a_draft_block_for_each_pending_append(
    reserve_tokens: int,
) -> None:
    pools = SessionPools(size=24, reserve_tokens=reserve_tokens)
    for session_id in ("a", "b"):
        pools.open(session_id)
    pools.append("a", "a1", [1, 2, 3], 4)
    assert pools.bridge.check_session_capacity("a1") is None
    pools.append("b", "b1", [4, 5, 6], 4)
    expected = "session KV capacity exhausted" if reserve_tokens else None
    assert pools.bridge.check_session_capacity("b1") == expected
    assert pools.adapter.history_capacities == [56 - reserve_tokens] * 2


def create_tiny_checkpoints(directory: Path) -> tuple[Path, Path]:
    """Build deterministic random weights on CPU, with no model downloads."""
    torch.manual_seed(23)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    target = Qwen3ForCausalLM(config)
    target.save_pretrained(directory / "target")
    config.num_hidden_layers = 1
    draft = Qwen3ForCausalLM(config)
    weights = dict(draft.state_dict())
    weights["fc.weight"] = torch.eye(128)
    weights["hidden_norm.weight"] = torch.ones(128)
    draft_dir = directory / "draft"
    draft_dir.mkdir(parents=True)
    save_file(weights, draft_dir / "model.safetensors")
    draft_config = config.to_dict()
    draft_config.update(
        architectures=["DFlashDraftModel"],
        dflash_config={
            "target_layer_ids": [0],
            "mask_token_id": 31,
            "block_size": 4,
            "sample_from_anchor": False,
        },
    )
    (draft_dir / "config.json").write_text(json.dumps(draft_config))
    return directory / "target", draft_dir


@pytest.mark.accelerator
@pytest.mark.parametrize("captured", [False, True])
def test_tiny_dflash_sessions_match_plain_decode(
    tmp_path: Path, captured: bool
) -> None:
    """Real draft/target forwards must preserve turn outputs through cache reuse, cuts and aborts."""
    require_cuda()
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the FA3 session integration test requires Hopper")
    else:
        pass
    target_dir, draft_dir = create_tiny_checkpoints(tmp_path)
    results = []
    for draft_path in (None, str(draft_dir)):
        script = (
            "import json, sys; "
            "from tests.unit_test.fixtures.tiny_speculative_session import run_tiny_sessions; "
            'result = run_tiny_sessions(sys.argv[1], sys.argv[2] or None, sys.argv[3] == "True"); '
            'open(sys.argv[4], "w").write(json.dumps(result))'
        )
        result_path = tmp_path / f"result-{len(results)}.json"
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(target_dir),
                draft_path or "",
                str(captured),
                str(result_path),
            ],
            check=True,
            timeout=240,
        )
        result = json.loads(result_path.read_text())
        result.pop("graph_replays")
        results.append(result)
    assert results[0] == results[1]
    assert len(results[0]) == 8
