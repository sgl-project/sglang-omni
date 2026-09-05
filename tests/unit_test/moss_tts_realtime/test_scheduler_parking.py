# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sglang.srt.managers.schedule_batch import NextBatchPlan, ReqKvInfo

from sglang_omni.models.moss_tts_realtime import scheduler as scheduler_module
from sglang_omni.models.moss_tts_realtime.config import MossTTSRealtimeResourceLimits
from sglang_omni.models.moss_tts_realtime.payload_types import MossTTSRealtimeState
from sglang_omni.models.moss_tts_realtime.request_builders import (
    build_moss_tts_realtime_prefill_rows,
    build_moss_tts_realtime_row_cache_key,
    build_moss_tts_realtime_row_cache_key_ids,
)
from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimeInputUpdate,
    MossTTSRealtimePendingInput,
    MossTTSRealtimeRequestData,
    MossTTSRealtimeTurnLedger,
    MossTTSRealtimeTurnPhase,
    MossTTSRealtimeTurnState,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_BOS_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_BOS_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_EOS_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_PAD_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_VOCAB_SIZE,
    MODEL_CONFIG,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    REFERENCE_AUDIO_PAD_TOKEN_ID as MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    TEXT_PAD_TOKEN_ID as MOSS_TTS_REALTIME_TEXT_PAD_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.scheduler_test_utils import (
    _AlignedDecodeBatch,
    _base_rows,
    _payload,
    _request_data,
    _scheduler,
    _wire_update,
)


def _decode_request(
    *,
    token_ids: tuple[int, ...],
    input_done: bool,
    frame: tuple[int, ...] = tuple(range(1, 17)),
    request_id: str = "request-1",
    session_id: str = "session-1",
    turn_id: str = "turn-1",
    req_pool_idx: int = 1,
) -> tuple[Any, Any, MossTTSRealtimeRequestData]:
    limits = MossTTSRealtimeResourceLimits()
    turn = MossTTSRealtimeTurnState(
        session_id=session_id,
        turn_id=turn_id,
        request_id=request_id,
        pending_input=MossTTSRealtimePendingInput.from_limits(limits),
        ledger=MossTTSRealtimeTurnLedger(model_config=MODEL_CONFIG),
    )
    turn.seed_initial_input(token_ids, input_done=input_done)
    prefill = turn.take_prefill_tokens()
    rows = torch.cat(
        [
            _base_rows(),
            build_moss_tts_realtime_prefill_rows(
                prefill,
                model_config=MODEL_CONFIG,
            ),
        ]
    )
    turn.ledger.extend_rows(rows.tolist())
    turn.observe_audio_frame(frame, generation_step=0)
    data = MossTTSRealtimeRequestData(
        req=None,
        state=MossTTSRealtimeState(
            session_id=session_id,
            turn_id=turn_id,
            generation_kwargs={"max_new_tokens": 32},
        ),
        turn_state=turn,
        prompt_rows=rows,
        model_config=SimpleNamespace(
            vocab_size=200_000,
            rvq=16,
            delay_tokens_len=12,
            audio_pad_token=MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID,
            audio_bos_token=MOSS_TTS_REALTIME_AUDIO_BOS_TOKEN_ID,
            audio_eos_token=MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID,
            audio_vocab_size=AUDIO_VOCAB_SIZE,
            reference_audio_pad=MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID,
            text_pad=MOSS_TTS_REALTIME_TEXT_PAD_TOKEN_ID,
        ),
        provisional_output_id=MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID,
    )
    seq_len = int(rows.shape[0])
    req = SimpleNamespace(
        rid=request_id,
        output_ids=[MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID],
        origin_input_ids=build_moss_tts_realtime_row_cache_key_ids(rows),
        req_pool_idx=req_pool_idx,
        kv_committed_len=seq_len,
        kv_allocated_len=seq_len,
        mamba_pool_idx=None,
        is_retracted=False,
        finished=lambda: False,
        _omni_data=data,
    )
    data.req = req
    batch = _AlignedDecodeBatch([req])
    return req, batch, data


def test_materialization_consumes_one_token_and_replaces_both_scalar_ids(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    req, batch, data = _decode_request(
        token_ids=tuple(range(13)),
        input_done=False,
    )
    hook_calls: list[tuple[str, tuple[int, ...]]] = []
    scheduler._model_runner = SimpleNamespace(
        on_realtime_row_materialized=lambda request, materialized: hook_calls.append(
            (request.request_id, materialized.row)
        )
    )
    relayed: list[tuple[list[int], list[int]]] = []
    scheduler.future_map = SimpleNamespace(
        stash=lambda indices, payload: relayed.append(
            (indices.tolist(), payload.bonus_tokens.tolist())
        )
    )
    batch.input_ids = batch.output_ids.clone()
    upstream_calls: list[Any] = []

    def _upstream(owner, candidate):
        assert owner is scheduler
        upstream_calls.append(candidate)
        return candidate

    monkeypatch.setattr(scheduler_module._Upstream, "update_running_batch", _upstream)

    assert scheduler.update_running_batch(batch) is batch

    expected_row = (12, *range(1, 17))
    expected_key = build_moss_tts_realtime_row_cache_key(expected_row)
    assert req.output_ids[-1] == expected_key
    assert batch.output_ids.tolist() == [expected_key]
    assert batch.input_ids.tolist() == [expected_key]
    assert relayed == [([req.req_pool_idx], [expected_key])]
    assert data.turn_state.provisional_frame is None
    assert data.turn_state.last_materialized_row.row == expected_row
    assert tuple(data.prompt_rows[-1].tolist()) == expected_row
    assert data.turn_state.ledger.rows[-1] == expected_row
    assert hook_calls == [("request-1", expected_row)]
    assert upstream_calls == [batch]


def test_mixed_rate_batch_parks_only_starved_request_before_upstream(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    slow_req, _, slow_data = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        request_id="slow",
        turn_id="turn-slow",
        req_pool_idx=3,
    )
    fast_req, _, fast_data = _decode_request(
        token_ids=tuple(range(13)),
        input_done=False,
        request_id="fast",
        turn_id="turn-fast",
        req_pool_idx=4,
    )
    batch = _AlignedDecodeBatch([slow_req, fast_req])
    upstream_calls: list[Any] = []

    def _upstream(owner, candidate):
        upstream_calls.append((owner, candidate))
        return candidate

    monkeypatch.setattr(scheduler_module._Upstream, "update_running_batch", _upstream)

    assert scheduler.update_running_batch(batch) is batch

    assert [req.rid for req in batch.reqs] == ["fast"]
    assert upstream_calls == [(scheduler, batch)]
    assert list(scheduler._parked_input) == ["slow"]
    assert slow_data.turn_state.phase is MossTTSRealtimeTurnPhase.PARKED_INPUT
    assert slow_data.turn_state.provisional_frame is not None
    assert slow_req.output_ids[-1] == MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID
    assert slow_req.kv_committed_len == len(slow_data.turn_state.ledger.rows)
    assert fast_data.turn_state.last_materialized_row.row[0] == 12
    assert fast_data.turn_state.provisional_frame is None
    assert scheduler._park_total == 1


def test_update_running_batch_resyncs_output_ids_after_merge(monkeypatch) -> None:
    """Regression: a post-prefill merge realigns batch.reqs without touching the
    scheduler-owned per-row output-id tensor, so a request merged into the
    running batch read a misaligned id and failed row materialization. The
    tensor must be resynced from request state before parking/materializing."""
    scheduler = _scheduler()
    req_a, _, _ = _decode_request(
        token_ids=tuple(range(13)),
        input_done=False,
        request_id="request-a",
        turn_id="turn-a",
        req_pool_idx=5,
    )
    req_b, _, _ = _decode_request(
        token_ids=tuple(range(13)),
        input_done=False,
        request_id="request-b",
        turn_id="turn-b",
        req_pool_idx=6,
    )
    batch = _AlignedDecodeBatch([req_a, req_b])
    # Stale merge window: only the first row's id is present in the tensor.
    batch.output_ids = batch.output_ids[:1].clone()

    hook_calls: list[tuple[str, tuple[int, ...]]] = []
    scheduler._model_runner = SimpleNamespace(
        on_realtime_row_materialized=lambda request, materialized: hook_calls.append(
            (request.request_id, materialized.row)
        )
    )
    relayed: list[tuple[list[int], list[int]]] = []
    scheduler.future_map = SimpleNamespace(
        stash=lambda indices, payload: relayed.append(
            (indices.tolist(), payload.bonus_tokens.tolist())
        )
    )
    upstream_calls: list[Any] = []

    def _upstream(owner, candidate):
        upstream_calls.append(candidate)
        return candidate

    monkeypatch.setattr(scheduler_module._Upstream, "update_running_batch", _upstream)

    assert scheduler.update_running_batch(batch) is batch

    expected_keys = [
        build_moss_tts_realtime_row_cache_key((12, *range(1, 17)))
        for _ in (req_a, req_b)
    ]
    assert batch.output_ids.tolist() == expected_keys
    assert [int(req.output_ids[-1]) for req in (req_a, req_b)] == expected_keys
    assert [call[0] for call in hook_calls] == ["request-a", "request-b"]
    assert len(relayed) == 2
    assert upstream_calls == [batch]


def test_materialization_failure_detaches_only_bad_runnable_request(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    bad_req, _, bad_data = _decode_request(
        token_ids=tuple(range(13)),
        input_done=False,
        request_id="bad",
        turn_id="turn-bad",
        req_pool_idx=18,
    )
    good_req, _, good_data = _decode_request(
        token_ids=tuple(range(13)),
        input_done=False,
        request_id="good",
        turn_id="turn-good",
        req_pool_idx=19,
    )
    bad_data.prompt_rows = None
    batch = _AlignedDecodeBatch([bad_req, good_req])
    scheduler.running_batch = batch
    released: list[Any] = []
    scheduler._release_request_kv_cache = released.append
    monkeypatch.setattr(
        scheduler_module._Upstream,
        "get_next_batch_to_run",
        lambda owner, running_batch, last_batch: NextBatchPlan(
            batch_to_run=owner.update_running_batch(running_batch),
            running_batch=running_batch,
        ),
    )
    monkeypatch.setattr(
        scheduler_module._Upstream,
        "update_running_batch",
        lambda owner, candidate: candidate,
    )

    assert scheduler.get_next_batch_to_run() is None

    assert [req.rid for req in scheduler.running_batch.reqs] == ["good"]
    assert scheduler.running_batch.output_ids.shape == (1,)
    assert released == [bad_req]
    assert bad_data.turn_state.phase is MossTTSRealtimeTurnPhase.FAILED
    assert good_data.turn_state.provisional_frame is not None
    output = scheduler.outbox.get_nowait()
    assert output.request_id == "bad"
    assert output.type == "error"


def test_parked_input_update_wakes_same_live_request_without_prefill(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    req, batch, data = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        req_pool_idx=5,
    )
    scheduler.running_batch = batch
    upstream_calls: list[Any] = []
    monkeypatch.setattr(
        scheduler_module._Upstream,
        "update_running_batch",
        lambda owner, candidate: upstream_calls.append(candidate) or candidate,
    )

    assert scheduler.update_running_batch(batch) is batch
    assert batch.reqs == []
    assert upstream_calls == []
    parked_record = scheduler._parked_input[req.rid]
    original_seq_len = parked_record.seq_len
    original_req = parked_record.req

    scheduler._on_input_update(
        req.rid,
        _wire_update(seq_no=0, token_ids=(99,)),
    )
    scheduler._build_parked_decode_batch = lambda records: _AlignedDecodeBatch(
        [record.req for record in records]
    )

    assert scheduler._wake_parked_requests() == 1
    assert scheduler.running_batch.reqs == [original_req]
    assert scheduler.running_batch.reqs[0] is req
    assert scheduler.running_batch.seq_lens.tolist() == [original_seq_len]
    assert scheduler.waiting_queue == []
    assert scheduler._parked_input == {}
    assert scheduler._wake_total == 1

    woken_batch = scheduler.running_batch
    assert scheduler.update_running_batch(woken_batch) is woken_batch
    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.RUNNING
    assert data.turn_state.last_materialized_row.row[0] == 99
    assert upstream_calls == [woken_batch]


def test_input_done_wakes_parked_request_into_text_pad_drain(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    req, batch, data = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        req_pool_idx=6,
    )
    scheduler.running_batch = batch
    monkeypatch.setattr(
        scheduler_module._Upstream,
        "update_running_batch",
        lambda owner, candidate: candidate,
    )

    scheduler.update_running_batch(batch)
    scheduler._on_input_update(
        req.rid,
        _wire_update(seq_no=0, input_done=True),
    )
    scheduler._build_parked_decode_batch = lambda records: _AlignedDecodeBatch(
        [record.req for record in records]
    )

    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.DRAINING
    assert scheduler._wake_parked_requests() == 1
    scheduler.update_running_batch(scheduler.running_batch)

    assert data.turn_state.last_materialized_row.row[0] == (
        MOSS_TTS_REALTIME_TEXT_PAD_TOKEN_ID
    )
    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.DRAINING


def test_multiple_parked_requests_wake_independently_in_park_order(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    req1, _, data1 = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        request_id="request-1",
        turn_id="turn-1",
        req_pool_idx=7,
    )
    req2, _, data2 = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        request_id="request-2",
        turn_id="turn-2",
        req_pool_idx=8,
    )
    batch = _AlignedDecodeBatch([req1, req2])
    scheduler.running_batch = batch
    monkeypatch.setattr(
        scheduler_module._Upstream,
        "update_running_batch",
        lambda owner, candidate: candidate,
    )

    scheduler.update_running_batch(batch)
    assert list(scheduler._parked_input) == ["request-1", "request-2"]
    scheduler._build_parked_decode_batch = lambda records: _AlignedDecodeBatch(
        [record.req for record in records]
    )

    scheduler._on_input_update(
        "request-2",
        _wire_update(
            request_id="request-2",
            turn_id="turn-2",
            seq_no=0,
            token_ids=(202,),
        ),
    )
    assert scheduler._wake_parked_requests() == 1
    assert [req.rid for req in scheduler.running_batch.reqs] == ["request-2"]
    assert list(scheduler._parked_input) == ["request-1"]

    scheduler._on_input_update(
        "request-1",
        _wire_update(seq_no=0, token_ids=(101,)),
    )
    assert scheduler._wake_parked_requests() == 1
    assert [req.rid for req in scheduler.running_batch.reqs] == [
        "request-2",
        "request-1",
    ]
    assert scheduler._parked_input == {}
    assert data1.turn_state.phase is MossTTSRealtimeTurnPhase.RUNNING
    assert data2.turn_state.phase is MossTTSRealtimeTurnPhase.RUNNING
    assert scheduler._wake_total == 2


def test_wake_batch_rebuild_uses_existing_req_pool_and_kv_lengths(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    req, batch, _ = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        req_pool_idx=9,
    )
    scheduler._park_starved_requests(batch)
    record = scheduler._parked_input[req.rid]
    req._omni_data.turn_state.append_input_update(
        MossTTSRealtimeInputUpdate(
            seq_no=0,
            token_ids=(77,),
        )
    )
    captured: dict[str, Any] = {}
    sampling_info = object()

    def _init_new(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            reqs=list(kwargs["reqs"]),
            device=torch.device("cpu"),
            return_logprob=False,
        )

    monkeypatch.setattr(
        scheduler_module.ScheduleBatch,
        "init_new",
        staticmethod(_init_new),
    )
    monkeypatch.setattr(
        scheduler_module.SamplingBatchInfo,
        "from_schedule_batch",
        staticmethod(lambda candidate, vocab_size: sampling_info),
    )

    rebuilt = scheduler._build_parked_decode_batch([record])

    assert captured["reqs"] == [req]
    assert rebuilt.req_pool_indices.tolist() == [record.req_pool_idx]
    assert rebuilt.seq_lens.tolist() == [record.seq_len]
    assert rebuilt.seq_lens_cpu.tolist() == [record.seq_len]
    assert rebuilt.orig_seq_lens.tolist() == [record.orig_seq_len]
    assert rebuilt.output_ids.tolist() == [record.provisional_output_id]
    assert rebuilt.seq_lens_sum == record.seq_len
    assert rebuilt.sampling_info is sampling_info


def test_wake_batch_rebuild_matches_installed_schedule_batch_contract(
    monkeypatch,
) -> None:
    from sglang.srt.managers import schedule_batch as schedule_batch_module
    from sglang.srt.sampling import sampling_batch_info as sampling_batch_info_module
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    # sglang 0.5.18 reads deterministic/logit-processor toggles off the exec
    # config bag, and Req.is_prefill_only reads the spec bag; init_new takes
    # spec_algorithm as a parameter.
    exec_bag = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
        features=SimpleNamespace(enable_custom_logit_processor=False),
    )
    monkeypatch.setattr(
        sampling_batch_info_module,
        "get_exec",
        lambda: exec_bag,
    )
    monkeypatch.setattr(
        schedule_batch_module,
        "get_spec",
        lambda: SimpleNamespace(speculative_algorithm=None),
    )

    scheduler = _scheduler()
    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
    data = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )
    req = data.req
    assert req is not None
    req._omni_data = data
    data.turn_state.observe_audio_frame(tuple(range(1, 17)), generation_step=0)
    req.output_ids.append(MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID)
    req.req_pool_idx = 15
    req.kv_committed_len = len(req.origin_input_ids)
    req.kv = ReqKvInfo(
        kv_allocated_len=req.kv_committed_len,
        swa_evicted_seqlen=0,
    )
    batch = _AlignedDecodeBatch([req])
    scheduler._park_starved_requests(batch)
    record = scheduler._parked_input[req.rid]
    data.turn_state.append_input_update(
        MossTTSRealtimeInputUpdate(seq_no=0, token_ids=(88,))
    )

    rebuilt = scheduler._build_parked_decode_batch([record])

    assert isinstance(rebuilt, scheduler_module.ScheduleBatch)
    assert rebuilt.reqs == [req]
    assert rebuilt.req_pool_indices.tolist() == [15]
    assert rebuilt.seq_lens.tolist() == [record.seq_len]
    assert rebuilt.output_ids.tolist() == [record.provisional_output_id]
    assert rebuilt.sampling_info is not None


def test_parked_requests_count_toward_active_turn_and_prefill_capacity(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    scheduler._max_active_turns = 3
    req1, _, _ = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        request_id="request-1",
        turn_id="turn-1",
        req_pool_idx=10,
    )
    req2, _, _ = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        request_id="request-2",
        turn_id="turn-2",
        req_pool_idx=11,
    )
    scheduler._park_starved_requests(_AlignedDecodeBatch([req1, req2]))

    observed_running_bs: list[int] = []
    monkeypatch.setattr(
        scheduler_module._Upstream,
        "get_num_allocatable_reqs",
        lambda owner, running_bs: observed_running_bs.append(running_bs) or 8,
    )

    assert scheduler.get_num_allocatable_reqs(0) == 1
    assert observed_running_bs == [2]

    scheduler._max_active_turns = 2
    new_data = _request_data(tuple(range(12)), input_done=False)
    with pytest.raises(RuntimeError, match="active-turn limit exceeded"):
        scheduler._finalize_built_request(
            _payload(tuple(range(12)), request_id="request-3"),
            False,
            new_data,
        )
    assert new_data.req is None


def test_parked_request_is_visible_to_lookup_admin_active_and_idle(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    req, batch, data = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        req_pool_idx=12,
    )
    scheduler._park_starved_requests(batch)
    scheduler.running_batch = batch

    assert scheduler._find_request_data(req.rid) is data
    assert req.rid in scheduler._active_request_ids()
    assert scheduler.is_fully_idle() is False

    monkeypatch.setattr(
        scheduler_module._Upstream,
        "is_fully_idle",
        lambda owner, for_health_check=False: True,
    )
    assert scheduler.is_fully_idle(for_health_check=True) is True
    monkeypatch.setattr(
        scheduler_module.OmniScheduler,
        "_admin_model_info",
        lambda owner: {"success": True, "data": {}},
    )
    info = scheduler._admin_model_info()["data"]
    assert info["runnable_batch_size"] == 0
    assert info["parked_input_size"] == 1
    assert info["active_turn_count"] == 1
    assert info["park_total"] == 1


def test_abort_parked_request_releases_turn_and_kv_once() -> None:
    scheduler = _scheduler()
    req, batch, data = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        req_pool_idx=13,
    )
    scheduler._park_starved_requests(batch)
    released: list[Any] = []
    scheduler._release_request_kv_cache = released.append

    scheduler.abort(req.rid)

    assert scheduler._parked_input == {}
    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.CANCELLED
    assert data.turn_state.terminal_reason == "aborted"
    assert released == [req]
    assert scheduler.is_input_update_terminal(req.rid)


def test_stop_cleans_all_parked_requests() -> None:
    scheduler = _scheduler()
    req1, _, data1 = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        request_id="request-1",
        turn_id="turn-1",
        req_pool_idx=16,
    )
    req2, _, data2 = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        request_id="request-2",
        turn_id="turn-2",
        req_pool_idx=17,
    )
    scheduler._park_starved_requests(_AlignedDecodeBatch([req1, req2]))
    released: list[Any] = []
    scheduler._release_request_kv_cache = released.append
    scheduler._running = True

    scheduler.stop()

    assert scheduler._running is False
    assert scheduler._parked_input == {}
    assert released == [req1, req2]
    assert data1.turn_state.phase is MossTTSRealtimeTurnPhase.CANCELLED
    assert data2.turn_state.phase is MossTTSRealtimeTurnPhase.CANCELLED


def test_parked_input_idle_timeout_fails_request_locally() -> None:
    scheduler = _scheduler()
    scheduler._input_idle_timeout_s = 1.0
    req, batch, data = _decode_request(
        token_ids=tuple(range(12)),
        input_done=False,
        req_pool_idx=14,
    )
    scheduler._park_starved_requests(batch)
    scheduler._parked_input[req.rid].last_input_at = time.monotonic() - 2.0
    released: list[Any] = []
    scheduler._release_request_kv_cache = released.append

    scheduler._expire_parked_requests()

    assert scheduler._parked_input == {}
    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.FAILED
    assert data.turn_state.terminal_reason == "input_idle_timeout"
    assert released == [req]
    assert scheduler._park_timeout_total == 1
    output = scheduler.outbox.get_nowait()
    assert output.request_id == req.rid
    assert output.type == "error"
    assert "idle timeout" in str(output.data)


def test_done_consumes_queued_token_before_text_pad_drain(monkeypatch) -> None:
    scheduler = _scheduler()
    req, batch, data = _decode_request(
        token_ids=tuple(range(13)),
        input_done=True,
    )
    scheduler._model_runner = SimpleNamespace(
        on_realtime_row_materialized=lambda request, materialized: None
    )
    monkeypatch.setattr(
        scheduler_module._Upstream,
        "update_running_batch",
        lambda owner, candidate: candidate,
    )

    scheduler.update_running_batch(batch)
    assert data.turn_state.last_materialized_row.row[0] == 12
    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.RUNNING

    data.turn_state.observe_audio_frame(tuple(range(17, 33)), generation_step=1)
    req.output_ids.append(MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID)
    batch.output_ids = torch.tensor(
        [MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID],
        dtype=torch.long,
    )
    scheduler.update_running_batch(batch)

    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.DRAINING
    assert data.turn_state.last_materialized_row.row[0] == (
        MOSS_TTS_REALTIME_TEXT_PAD_TOKEN_ID
    )
