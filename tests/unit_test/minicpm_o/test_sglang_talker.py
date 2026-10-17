# SPDX-License-Identifier: Apache-2.0
"""Public talker contract: condition embeddings match the reference math, and a turn past the talker context rolls its session history over."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.session.session_controller import SessionController
from sglang.srt.session.streaming_session import StreamingSession
from torch import nn

from sglang_omni.models.minicpm_o.components.sglang_talker import (
    MiniCPMOTalkerForCausalLM,
    MiniCPMTTSProjector,
)
from sglang_omni.models.minicpm_o.talker_model_runner import MiniCPMOTalkerModelRunner
from sglang_omni.models.minicpm_o.talker_session import (
    TALKER_TOKENS_PER_UNIT,
    TalkerAdapter,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, SessionOperation, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionBridge

HIDDEN = 8
LLM_DIM = 16
NUM_TEXT = 20
TEXT_EOS = 5
AUDIO_BOS = 6
NUM_AUDIO_TOKENS = 64
CODEC_EOS = NUM_AUDIO_TOKENS - 1
IDENTITY = SessionIdentity("talker")
CONDITIONS_PER_UNIT = 5
UNIT_COUNT = 12
TURN_END_UNIT = 10
TALKER_MAX_REQUEST_LENGTH = 200
TALKER_CONTEXT_LENGTH = 256
KV_POOL_TOKENS = 1024


def bare_model() -> MiniCPMOTalkerForCausalLM:
    model = object.__new__(MiniCPMOTalkerForCausalLM)
    nn.Module.__init__(model)
    model.text_eos_token_id = TEXT_EOS
    model.audio_bos_token_id = AUDIO_BOS
    model.normalize_projected_hidden = True
    model.emb_text = nn.Embedding(NUM_TEXT, HIDDEN)
    model.projector_semantic = MiniCPMTTSProjector(LLM_DIM, HIDDEN)
    return model


def test_condition_matches_reference_math():
    model = bare_model()
    tokens = torch.tensor([3, 7, 1], dtype=torch.long)
    hidden = torch.randn(3, LLM_DIM)

    condition = model.build_condition_embeddings(tokens, hidden)

    # reference: emb_text(t) + l2norm(projector(h)), then [text_eos, audio_bos]
    ref = model.emb_text(tokens) + F.normalize(
        model.projector_semantic(hidden), p=2, dim=-1
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, torch.cat([ref, boundary], dim=0))
    assert condition.shape == (5, HIDDEN)


def test_condition_empty_span_is_boundary_only():
    model = bare_model()
    condition = model.build_condition_embeddings(
        torch.empty(0, dtype=torch.long), torch.empty(0, LLM_DIM)
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, boundary)


def test_condition_length_mismatch_raises():
    model = bare_model()
    with pytest.raises(ValueError, match="length mismatch"):
        model.build_condition_embeddings(torch.tensor([1, 2]), torch.randn(3, LLM_DIM))


class TalkerSessionScheduler:
    """The bridge's scheduler surface over real SGLang session, request-row and KV pools."""

    def __init__(self) -> None:
        self.req_to_token_pool = ReqToTokenPool(
            size=1,
            max_context_len=TALKER_CONTEXT_LENGTH,
            device="cpu",
            enable_memory_saver=False,
        )
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=KV_POOL_TOKENS,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=MHATokenToKVPool(
                size=KV_POOL_TOKENS,
                page_size=1,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=HIDDEN,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
            ),
            need_sort=False,
        )
        self.tree_cache = StreamingSession(
            ChunkCache(
                CacheInitParams(
                    disable=True,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    page_size=1,
                )
            )
        )
        self.session_controller = SessionController(self.tree_cache)
        self.model_config = SimpleNamespace(vocab_size=NUM_AUDIO_TOKENS)
        self.waiting_queue: list[Req] = []
        self.chunked_req: Req | None = None
        self.max_running_requests = 1

    def synchronize_launched_decode(self) -> None:
        pass

    def resolve_pending_async(self) -> None:
        pass

    def run(self, request: Req, codec_tokens: list[int]) -> None:
        """Prefill and decode one request on the pools as SGLang does, then release it."""
        request.init_next_round_input(self.tree_cache)
        if request.kv.holds_kv:
            pass
        else:
            self.req_to_token_pool.alloc([request])
        reused_length = len(request.prefix_indices)
        kv_length = len(request.full_untruncated_fill_ids) + len(codec_tokens) - 1
        slots = self.token_to_kv_pool_allocator.alloc(kv_length - reused_length)
        self.req_to_token_pool.req_to_token[
            request.kv.req_pool_idx, reused_length:kv_length
        ] = slots.to(torch.int32)
        request.kv.kv_allocated_len = kv_length
        request.kv.kv_committed_len = kv_length
        request.output_ids.extend(codec_tokens)
        request.finished_reason = FINISH_LENGTH(length=len(codec_tokens))
        release_kv_cache(request, self.tree_cache)


def talker_unit_payload(unit_index: int) -> StagePayload:
    generator = torch.Generator().manual_seed(unit_index)
    return StagePayload(
        f"unit-{unit_index}",
        OmniRequest(
            None,
            params={"talker_temperature": 0.8, "talker_repetition_penalty": 1.05},
        ),
        {
            "talker_conditions": [
                (int(token_id), torch.randn(LLM_DIM, generator=generator), False)
                for token_id in torch.randint(
                    0, NUM_TEXT, (CONDITIONS_PER_UNIT,), generator=generator
                )
            ],
            "text": "",
            "is_listen": False,
            "end_of_turn": unit_index == TURN_END_UNIT,
        },
    )


def test_a_turn_past_the_talker_context_keeps_its_newest_units() -> None:
    """Every unit fits the talker request limit: a unit that would not restarts the turn's history from its newest finished units that fit with it in half the limit, and a new turn starts empty."""
    model = bare_model()
    model.num_audio_tokens = NUM_AUDIO_TOKENS
    model.codec_eos_id = CODEC_EOS
    adapter = TalkerAdapter(model, max_request_length=TALKER_MAX_REQUEST_LENGTH)
    scheduler = TalkerSessionScheduler()
    bridge = ARSessionBridge(scheduler, adapter)
    bridge.apply_operation(
        talker_unit_payload(0), SessionOperation("open", IDENTITY, ("talker",))
    )
    runner = object.__new__(MiniCPMOTalkerModelRunner)

    units = []
    for unit_index in range(UNIT_COUNT):
        payload = talker_unit_payload(unit_index)
        chunk = TimedChunk("audio", 1000 * unit_index, 1000, unit_index, b"")
        unit = bridge.accept(
            payload, SessionOperation("append", IDENTITY, ("talker",), chunk)
        )
        is_bypassed = bridge.prepare_unit(unit, payload)
        holds_history = IDENTITY.id in scheduler.tree_cache.slots
        request_data = adapter.build(IDENTITY, chunk, payload)
        bridge.create_session_request(payload, request_data)
        request_token_ids = list(request_data.req.origin_input_ids)
        scheduler.run(
            request_data.req,
            [
                (unit_index + offset) % CODEC_EOS
                for offset in range(TALKER_TOKENS_PER_UNIT)
            ],
        )
        runner.on_request_finished(request_data.req.rid, request_data)
        request_data.output_ids = list(request_data.req.output_ids)
        adapter.result(IDENTITY, request_data)
        bridge.complete(request_data.req.rid)
        units.append(
            (is_bypassed, holds_history, request_token_ids, payload, request_data)
        )

    condition_rows = CONDITIONS_PER_UNIT + 1
    finished_unit_length = condition_rows + len(units[0][4].output_ids)
    carried_count = (
        TALKER_MAX_REQUEST_LENGTH // 2 - condition_rows - TALKER_TOKENS_PER_UNIT
    ) // finished_unit_length
    restarted_units = []
    for unit_index, (
        is_bypassed,
        holds_history,
        request_token_ids,
        payload,
        request_data,
    ) in enumerate(units):
        is_turn_start = unit_index in (0, TURN_END_UNIT + 1)
        assert not is_bypassed, f"unit {unit_index} runs the talker"
        assert (
            len(request_token_ids) + TALKER_TOKENS_PER_UNIT <= TALKER_MAX_REQUEST_LENGTH
        ), f"unit {unit_index} fits the talker request limit"
        assert (
            payload.data["speech_turn_start"] is is_turn_start
        ), f"unit {unit_index} starts a turn only after a turn end"
        assert request_data.req.sampling_params.min_new_tokens == (
            0
            if is_turn_start or unit_index == TURN_END_UNIT
            else TALKER_TOKENS_PER_UNIT
        ), f"unit {unit_index} keeps the mid-turn token floor"
        if unit_index == 0 or holds_history:
            pass
        elif is_turn_start:
            restarted_units.append(unit_index)
            assert (
                request_token_ids == [CODEC_EOS] * condition_rows
            ), "a new turn starts from its own condition only"
        else:
            _, _, previous_token_ids, _, previous_request_data = units[unit_index - 1]
            restarted_units.append(unit_index)
            assert (
                len(previous_token_ids)
                + len(previous_request_data.output_ids)
                + condition_rows
                + TALKER_TOKENS_PER_UNIT
                > TALKER_MAX_REQUEST_LENGTH
            ), f"unit {unit_index} restarts only when appending would not fit"
            carried = [
                carried_request_data
                for _, _, _, _, carried_request_data in units[
                    unit_index - carried_count : unit_index
                ]
            ]
            assert request_token_ids == [
                *[
                    token_id
                    for carried_request_data in carried
                    for token_id in [
                        *[CODEC_EOS] * condition_rows,
                        *carried_request_data.output_ids,
                    ]
                ],
                *[CODEC_EOS] * condition_rows,
            ], "the rollover keeps the newest units' conditions and codes"
            *carried_conditions, condition = request_data.unit_embedding_spans
            offset = 0
            for carried_condition, carried_request_data in zip(
                carried_conditions, carried, strict=True
            ):
                assert (carried_condition.start, carried_condition.end) == (
                    offset,
                    offset + condition_rows,
                )
                assert (
                    carried_condition.input_embeds
                    is carried_request_data.unit_embedding_spans[-1].input_embeds
                ), "the rollover carries each kept unit's own condition"
                offset += condition_rows + len(carried_request_data.output_ids)
            assert (condition.start, condition.end) == (
                len(request_token_ids) - condition_rows,
                len(request_token_ids),
            )
    assert restarted_units == [
        6,
        TURN_END_UNIT,
        TURN_END_UNIT + 1,
    ], "31-token units pass 200 tokens at units 6 and 10, each rollover keeps two units, and the next turn starts empty"
