# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from array import array
from collections import deque
from queue import Queue
from types import SimpleNamespace
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN, ReqKvInfo
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.session.session_controller import SessionController
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession

from sglang_omni.models.moss_tts_realtime.config import MossTTSRealtimeResourceLimits
from sglang_omni.models.moss_tts_realtime.payload_types import MossTTSRealtimeState
from sglang_omni.models.moss_tts_realtime.request_builders import (
    build_moss_tts_realtime_row_cache_key,
    build_moss_tts_realtime_row_cache_key_ids,
)
from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimeRequestData,
)
from sglang_omni.models.moss_tts_realtime.scheduler import MossTTSRealtimeScheduler
from sglang_omni.proto.messages import InputUpdateMessage
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_BOS_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_BOS_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_EOS_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_PAD_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import AUDIO_VOCAB_SIZE
from tests.unit_test.moss_tts_realtime.runtime_config import (
    REFERENCE_AUDIO_PAD_TOKEN_ID as MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    TEXT_PAD_TOKEN_ID as MOSS_TTS_REALTIME_TEXT_PAD_TOKEN_ID,
)


class _AlignedDecodeBatch:
    def __init__(self, reqs: list[Any]) -> None:
        self.reqs = list(reqs)
        self.batch_is_full = False
        self.req_pool_indices = torch.tensor(
            [int(req.req_pool_idx) for req in reqs],
            dtype=torch.long,
        )
        self.seq_lens = torch.tensor(
            [int(req.kv_committed_len) for req in reqs],
            dtype=torch.long,
        )
        self.seq_lens_cpu = self.seq_lens.clone()
        self.orig_seq_lens = self.seq_lens.to(dtype=torch.int32)
        self.seq_lens_sum = int(self.seq_lens.sum().item())
        self.output_ids = torch.tensor(
            [int(req.output_ids[-1]) for req in reqs],
            dtype=torch.long,
        )

    def filter_batch(
        self,
        chunked_req_to_exclude: Any | None = None,
        keep_indices: list[int] | None = None,
    ) -> None:
        if keep_indices is None:
            if chunked_req_to_exclude is None:
                excluded: list[Any] = []
            elif isinstance(chunked_req_to_exclude, list):
                excluded = chunked_req_to_exclude
            else:
                excluded = [chunked_req_to_exclude]
            keep_indices = [
                index
                for index, req in enumerate(self.reqs)
                if not req.finished() and req not in excluded
            ]
        self.reqs = [self.reqs[index] for index in keep_indices]
        index_t = torch.tensor(keep_indices, dtype=torch.long)
        self.req_pool_indices = self.req_pool_indices[index_t]
        self.seq_lens = self.seq_lens[index_t]
        self.seq_lens_cpu = self.seq_lens_cpu[index_t]
        self.orig_seq_lens = self.orig_seq_lens[index_t]
        # output_ids intentionally not sliced: upstream filter_batch knows
        # nothing about this scheduler-owned extension; update_running_batch
        # resyncs it from request state after filtering.
        self.seq_lens_sum = int(self.seq_lens.sum().item())

    def merge_batch(self, other: _AlignedDecodeBatch) -> None:
        self.reqs.extend(other.reqs)
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        # See filter_batch: merging upstream batches leaves output_ids stale by
        # design; the scheduler resyncs it on the next update_running_batch.
        self.seq_lens_sum += other.seq_lens_sum

    def is_empty(self) -> bool:
        return not self.reqs


def _scheduler() -> MossTTSRealtimeScheduler:
    scheduler = object.__new__(MossTTSRealtimeScheduler)
    scheduler._moss_tts_realtime_limits = MossTTSRealtimeResourceLimits()
    scheduler._moss_tts_realtime_model_config = SimpleNamespace(
        delay_tokens_len=12,
    )
    scheduler._max_active_turns = 16
    scheduler._max_session_rows = 4096
    scheduler._max_held_kv_tokens = 64 * 4096
    scheduler._codec_slots = 16
    scheduler._input_idle_timeout_s = 30.0
    scheduler._turn_timeout_s = 600.0
    scheduler._parked_input = {}
    scheduler._park_sequence = 0
    scheduler._park_total = 0
    scheduler._wake_total = 0
    scheduler._park_timeout_total = 0
    scheduler._moss_tts_realtime_sessions = {}
    scheduler._moss_tts_realtime_requests = {}
    scheduler._session_idle_ttl_s = 300.0
    scheduler._session_reap_interval_s = 1.0
    scheduler._last_session_reap_at = 0.0
    scheduler._scheduler_thread_id = threading.get_ident()
    scheduler._request_admission_lock = threading.RLock()
    scheduler._terminal_tombstone_limit = 32
    scheduler._buffered_input_updates = {}
    scheduler._input_update_terminal_ids = set()
    scheduler._input_update_terminal_order = deque()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._completed_request_ids = {}
    scheduler._pending_request_builds = {}
    scheduler._pending_request_admissions = {}
    scheduler._backlogged_request_build_payloads = deque()
    scheduler._request_build_executor = None
    scheduler.request_build_max_pending = 0
    scheduler._request_build_max_pending_observed = 0
    scheduler.waiting_queue = []
    scheduler.running_batch = _AlignedDecodeBatch([])
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._async_pending = None
    scheduler.chunked_req = None
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler._abort_callback = None
    scheduler._stream_output_builder = None
    scheduler._request_finished_callback = None
    scheduler._shutdown_callback = None
    scheduler._shutdown_lock = threading.Lock()
    scheduler._mark_running_request_aborted = lambda request_id: False
    scheduler._release_immediate_request_resources = lambda request_id: None
    scheduler._model_runner = None
    scheduler.future_map = SimpleNamespace(stash=lambda indices, payload: None)
    req_to_token_pool = SimpleNamespace(
        device=torch.device("cpu"),
        req_to_token=torch.arange(64 * 4096, dtype=torch.int64).reshape(64, 4096),
        free_slots=[],
    )
    token_to_kv_pool_allocator = SimpleNamespace(
        device=torch.device("cpu"),
        free=lambda indices: None,
    )
    scheduler.req_to_token_pool = req_to_token_pool
    scheduler.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    scheduler.tree_cache = StreamingSession(
        RadixCache.create_simulated(mock_allocator=token_to_kv_pool_allocator)
    )
    scheduler.tree_cache.req_to_token_pool = req_to_token_pool
    scheduler.tree_cache.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    scheduler.session_controller = SessionController(scheduler.tree_cache)
    scheduler.model_config = SimpleNamespace(
        vocab_size=200_000,
        rvq=16,
        delay_tokens_len=12,
    )
    scheduler.max_req_input_len = 4096
    # Admission defaults mirror the ServerArgs used by the realtime stage.
    # The tests exercise request construction, not queue limiting or priority
    # scheduling, so both features are explicitly disabled here.
    scheduler.max_queued_requests = None
    scheduler.enable_priority_scheduling = False
    scheduler.enable_overlap = False
    scheduler.enable_async_decode = False
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    scheduler.inbox = Queue()
    scheduler.outbox = Queue()
    scheduler.is_entry_rank = True
    scheduler.model_worker = SimpleNamespace(model_info=lambda: {})
    scheduler.tp_rank = 0
    scheduler.tp_size = 1
    scheduler._engine_paused = False
    scheduler.request_build_max_workers = 0
    scheduler.server_args = SimpleNamespace(
        model_path="fake-model",
        load_format="auto",
        weight_version="test",
    )
    scheduler._result_adapter = lambda data: {"phase": data.turn_state.phase.value}
    return scheduler


def _set_limits(
    scheduler: MossTTSRealtimeScheduler,
    **overrides: Any,
) -> MossTTSRealtimeResourceLimits:
    values = scheduler._moss_tts_realtime_limits.model_dump()
    values.update(overrides)
    limits = MossTTSRealtimeResourceLimits(**values)
    scheduler._moss_tts_realtime_limits = limits
    scheduler._max_active_turns = limits.max_active_turns
    scheduler._input_idle_timeout_s = limits.input_idle_timeout_s
    scheduler._turn_timeout_s = limits.turn_timeout_s
    scheduler._session_idle_ttl_s = limits.session_idle_ttl_s
    return limits


def _payload(
    token_ids: tuple[int, ...] = (),
    *,
    input_done: bool = False,
    request_id: str = "request-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prefetched_chunks=[],
        prefetched_stream_done=False,
        data={
            "initial_token_ids": list(token_ids),
            "input_done": input_done,
        },
    )


def _wire_update(
    *,
    seq_no: int,
    token_ids: tuple[int, ...] = (),
    byte_count: int = 0,
    input_done: bool = False,
    request_id: str = "request-1",
    session_id: str = "session-1",
    turn_id: str = "turn-1",
) -> InputUpdateMessage:
    return InputUpdateMessage(
        request_id=request_id,
        session_id=session_id,
        turn_id=turn_id,
        seq_no=seq_no,
        token_ids=token_ids,
        byte_count=byte_count,
        input_done=input_done,
    )


def _base_rows() -> torch.Tensor:
    return torch.tensor(
        [[10, *([MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID] * 16)]],
        dtype=torch.long,
    )


def _request_data(
    initial_token_ids: tuple[int, ...],
    *,
    input_done: bool,
    max_new_tokens: int = 32,
    keep_session: bool = True,
    session_id: str = "session-1",
    turn_id: str = "turn-1",
    turn_index: int = 0,
    prompt_rows: torch.Tensor | None = None,
) -> MossTTSRealtimeRequestData:
    rows = _base_rows() if prompt_rows is None else prompt_rows.detach().clone()
    state = MossTTSRealtimeState(
        session_id=session_id,
        turn_id=turn_id,
        turn_index=turn_index,
        initial_token_ids=list(initial_token_ids),
        input_done=input_done,
        keep_session=keep_session,
        generation_kwargs={"max_new_tokens": max_new_tokens},
    )
    return MossTTSRealtimeRequestData(
        input_ids=torch.tensor(
            build_moss_tts_realtime_row_cache_key_ids(rows),
            dtype=torch.long,
        ),
        max_new_tokens=max_new_tokens,
        state=state,
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
        prompt_rows=rows,
        initial_token_ids=initial_token_ids,
        provisional_output_id=MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID,
    )


def _seed_successful_session_turn(
    scheduler: MossTTSRealtimeScheduler,
    *,
    prior_output_ids: tuple[int, ...] | None = None,
    request_id: str = "request-1",
    session_id: str = "session-1",
    turn_id: str = "turn-1",
    req_pool_idx: int = 1,
) -> tuple[Any, Any, tuple[tuple[int, ...], ...], int]:
    first = _request_data(
        tuple(range(12)),
        input_done=False,
        session_id=session_id,
        turn_id=turn_id,
    )
    scheduler._finalize_built_request(
        _payload(tuple(range(12)), request_id=request_id),
        False,
        first,
    )
    req = first.req
    session_state = first.session_state
    turn = first.turn_state
    assert req is not None
    assert session_state is not None
    assert turn is not None
    session = req.session
    assert session is not None

    generated_row = (77, *tuple(range(1, 17)))
    generated_key = build_moss_tts_realtime_row_cache_key(generated_row)
    committed_rows = turn.ledger.rows + (generated_row,)
    req.output_ids[:] = array(
        "q",
        (
            prior_output_ids
            if prior_output_ids is not None
            else (generated_key, MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID)
        ),
    )
    req.finished_reason = FINISH_MATCHED_TOKEN(MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID)
    req.req_pool_idx = req_pool_idx
    req.kv_committed_len = len(committed_rows)
    req.kv = ReqKvInfo(
        kv_allocated_len=len(committed_rows),
        swa_evicted_seqlen=0,
    )
    slot = SessionSlot()
    slot.save_from_req(req, is_first=True)
    scheduler.tree_cache.slots[session.session_id] = slot
    session.finish_req(req)

    session_state.committed_rows = committed_rows
    session_state.ledger_revision = 1
    session_state.successful_turns = 1
    session_state.active_turn_id = None
    session_state.warm_session_id = session.session_id
    session_state.warm_kv_length = len(committed_rows)
    first.lifecycle_finalized = True
    scheduler._moss_tts_realtime_requests.pop(req.rid, None)
    return session_state, session, committed_rows, generated_key
