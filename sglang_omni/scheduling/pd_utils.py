# SPDX-License-Identifier: Apache-2.0
"""Utilities shared by the explicit Prefill and Decode schedulers."""

from __future__ import annotations

import _thread
import inspect
import logging
import queue
import threading
from array import array
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Protocol

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.comm.kv_transfer import KVBufferRegion, KVPool
from sglang_omni.proto.request import StagePayload
from sglang_omni.proto.session import find_session_operation
from sglang_omni.scheduling.pd_receiver import (
    DecodeContinuation,
    ReservedKV,
    WakeupNotifier,
    page_indices_from_heads,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

logger = logging.getLogger(__name__)

SEPARATE_KV_ALLOCATIONS_POOL_TYPE_NAME = (
    "sglang_omni.model_runner.sglang_model_runner.SeparateKVAllocationsPool"
)


def serialize_kv_allocator(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    lock: _thread.RLock | None = None,
    sync_cuda_streams: bool = False,
) -> _thread.RLock:
    """Synchronize the existing allocator across threads and CUDA streams.

    Wrap concrete allocator methods in place so concrete types and existing aliases survive; an RLock covers nested calls while optional CUDA event ordering serializes cross-stream free-list mutations without touching CUDA streams on host queries.
    """
    if lock is None:
        lock = threading.RLock()
    else:
        pass

    use_cuda_sync = (
        sync_cuda_streams
        and torch.cuda.is_available()
        and str(allocator.device).startswith("cuda")
    )
    sync_event: torch.cuda.Event | None = None
    if use_cuda_sync:
        with torch.cuda.device(allocator.device):
            sync_event = torch.cuda.Event()
    else:
        pass
    last_stream: torch.cuda.Stream | None = None
    gpu_sync_depth = 0

    @contextmanager
    def gpu_sync_scope() -> Generator[None, None, None]:
        nonlocal gpu_sync_depth, last_stream
        with lock:
            if sync_event is not None and gpu_sync_depth == 0:
                current_stream = torch.cuda.current_stream(allocator.device)
                if last_stream is not None and current_stream != last_stream:
                    current_stream.wait_event(sync_event)
                else:
                    pass
            else:
                pass
            gpu_sync_depth += 1
            try:
                yield
            finally:
                gpu_sync_depth -= 1
                if sync_event is not None and gpu_sync_depth == 0:
                    current_stream = torch.cuda.current_stream(allocator.device)
                    sync_event.record(current_stream)
                    last_stream = current_stream
                else:
                    pass

    raw_available_size = allocator.available_size
    raw_get_kvcache = allocator.get_kvcache
    raw_alloc = allocator.alloc
    raw_free = allocator.free
    raw_free_segment = allocator.free_segment
    raw_free_segments = allocator.free_segments
    raw_free_full = allocator.free_full
    raw_free_full_segment = allocator.free_full_segment
    raw_free_full_segments = allocator.free_full_segments
    raw_free_group_begin = allocator.free_group_begin
    raw_free_group_end = allocator.free_group_end
    raw_merge_and_sort_free = allocator.merge_and_sort_free
    raw_get_all_free_pages = allocator.get_all_free_pages
    raw_clear = allocator.clear
    raw_resize = allocator.resize

    def synchronized_available_size() -> int:
        with lock:
            return raw_available_size()

    def synchronized_get_kvcache() -> KVCache | None:
        with lock:
            return raw_get_kvcache()

    def synchronized_alloc(need_size: int) -> torch.Tensor | None:
        with gpu_sync_scope():
            return raw_alloc(need_size)

    def synchronized_free(free_index: torch.Tensor) -> None:
        with gpu_sync_scope():
            raw_free(free_index)

    def synchronized_free_segment(free_index: torch.Tensor, *, start_pos: int) -> None:
        with gpu_sync_scope():
            raw_free_segment(free_index, start_pos=start_pos)

    def synchronized_free_segments(
        segments: Sequence[tuple[torch.Tensor, int]],
    ) -> None:
        with gpu_sync_scope():
            raw_free_segments(segments)

    def synchronized_free_full(free_index: torch.Tensor) -> None:
        with gpu_sync_scope():
            raw_free_full(free_index)

    def synchronized_free_full_segment(
        free_index: torch.Tensor, *, start_pos: int
    ) -> None:
        with gpu_sync_scope():
            raw_free_full_segment(free_index, start_pos=start_pos)

    def synchronized_free_full_segments(
        segments: Sequence[tuple[torch.Tensor, int]],
    ) -> None:
        with gpu_sync_scope():
            raw_free_full_segments(segments)

    def synchronized_free_group_begin() -> None:
        with gpu_sync_scope():
            raw_free_group_begin()

    def synchronized_free_group_end() -> None:
        with gpu_sync_scope():
            raw_free_group_end()

    def synchronized_merge_and_sort_free() -> None:
        with gpu_sync_scope():
            raw_merge_and_sort_free()

    def synchronized_get_all_free_pages() -> torch.Tensor | None:
        with gpu_sync_scope():
            return raw_get_all_free_pages()

    def synchronized_clear() -> None:
        with gpu_sync_scope():
            raw_clear()

    def synchronized_resize(config: MemoryPoolConfig) -> None:
        with gpu_sync_scope():
            raw_resize(config)

    allocator.available_size = synchronized_available_size
    allocator.get_kvcache = synchronized_get_kvcache
    allocator.alloc = synchronized_alloc
    allocator.free = synchronized_free
    allocator.free_segment = synchronized_free_segment
    allocator.free_segments = synchronized_free_segments
    allocator.free_full = synchronized_free_full
    allocator.free_full_segment = synchronized_free_full_segment
    allocator.free_full_segments = synchronized_free_full_segments
    allocator.free_group_begin = synchronized_free_group_begin
    allocator.free_group_end = synchronized_free_group_end
    allocator.merge_and_sort_free = synchronized_merge_and_sort_free
    allocator.get_all_free_pages = synchronized_get_all_free_pages
    allocator.clear = synchronized_clear
    allocator.resize = synchronized_resize

    if isinstance(allocator, (TokenToKVPoolAllocator, PagedTokenToKVPoolAllocator)):
        raw_free_page_ids = allocator.free_page_ids

        def synchronized_free_page_ids(page_ids: torch.Tensor) -> None:
            with gpu_sync_scope():
                raw_free_page_ids(page_ids)

        allocator.free_page_ids = synchronized_free_page_ids
    else:
        pass

    if isinstance(allocator, PagedTokenToKVPoolAllocator):
        raw_alloc_extend = allocator.alloc_extend
        raw_alloc_decode = allocator.alloc_decode

        def synchronized_alloc_extend(
            prefix_lens: torch.Tensor,
            prefix_lens_cpu: torch.Tensor,
            seq_lens: torch.Tensor,
            seq_lens_cpu: torch.Tensor,
            last_loc: torch.Tensor,
            extend_num_tokens: int,
            num_new_pages: int | None = None,
        ) -> torch.Tensor | None:
            with gpu_sync_scope():
                return raw_alloc_extend(
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    num_new_pages,
                )

        def synchronized_alloc_decode(
            seq_lens: torch.Tensor,
            seq_lens_cpu: torch.Tensor,
            last_loc: torch.Tensor,
        ) -> torch.Tensor | None:
            with gpu_sync_scope():
                return raw_alloc_decode(seq_lens, seq_lens_cpu, last_loc)

        allocator.alloc_extend = synchronized_alloc_extend
        allocator.alloc_decode = synchronized_alloc_decode
    else:
        pass

    return lock


class StateBuilder(Protocol):
    def __call__(
        self, request: Req
    ) -> tuple[dict[str, object], dict[str, object] | None, list[int]]: ...


class StateRestorer(Protocol):
    def __call__(
        self,
        request: Req,
        request_data: SGLangARRequestData,
        multimodal_resume: dict[str, object] | None,
    ) -> None: ...


def continuation_from_req(
    req: Req,
    transfer_id: str,
    state_builder: StateBuilder,
) -> DecodeContinuation:
    """Snapshot a completed Prefill request without serializing runtime tensors."""

    if not req.output_ids:
        raise ValueError(f"Prefill request {req.rid!r} produced no token")
    else:
        pass
    if req.custom_logit_processor:
        raise NotImplementedError("PD does not support custom logit processors")
    else:
        pass
    request_data = req.omni_data
    if (
        req.session is not None
        or find_session_operation(request_data.stage_payload.request.metadata)
        is not None
    ):
        raise NotImplementedError(
            "PD does not transfer retained streaming-session state"
        )
    else:
        pass
    if request_data.input_embeds_are_projected or getattr(
        req,
        "_input_embeds_are_projected",
        False,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    ):
        raise NotImplementedError("PD does not support projected input embeddings")
    else:
        pass
    sampling = sampling_params_to_dict(req.sampling_params)
    if any(
        sampling.get(key) for key in ("json_schema", "regex", "ebnf", "structural_tag")
    ):
        raise NotImplementedError("PD does not support structured-output sampling")
    else:
        pass

    payload, multimodal_resume, origin_input_ids = state_builder(req)
    # note (YzXiao): omit origin_input_ids_unpadded when it aliases or equals req.origin_input_ids so Decode aliases to clean origin_input_ids rather than salted media-cache ids.
    if (
        req.origin_input_ids_unpadded is None
        or req.origin_input_ids_unpadded is req.origin_input_ids
        or list(req.origin_input_ids_unpadded) == list(req.origin_input_ids)
    ):
        origin_input_ids_unpadded = None
    else:
        origin_input_ids_unpadded = list(req.origin_input_ids_unpadded)
    return DecodeContinuation(
        request_id=req.rid,
        priority=req.priority,
        transfer_id=transfer_id,
        origin_input_ids=origin_input_ids,
        origin_input_ids_unpadded=origin_input_ids_unpadded,
        output_ids=list(req.output_ids),
        vocab_size=req.vocab_size,
        sampling_params=sampling,
        stage_payload=payload,
        eos_token_ids=list(req.eos_token_ids) if req.eos_token_ids else None,
        cached_tokens=req.cached_tokens,
        cached_tokens_device=req.cached_tokens_device,
        cached_tokens_host=req.cached_tokens_host,
        cached_tokens_storage=req.cached_tokens_storage,
        mm_image_tokens=req.mm_image_tokens,
        mm_audio_tokens=req.mm_audio_tokens,
        mm_video_tokens=req.mm_video_tokens,
        return_logprob=bool(request_data.return_logprob),
        output_token_logprobs=list(request_data.output_token_logprobs),
        top_logprobs_num=req.logprob.top_logprobs_num,
        token_ids_logprob=(
            list(req.logprob.token_ids_logprob)
            if req.logprob.token_ids_logprob is not None
            else None
        ),
        logprob_start_len=req.logprob_start_len,
        return_hidden_states=req.return_hidden_states,
        return_sampling_mask=bool(req.return_sampling_mask),
        return_routed_experts=bool(req.return_routed_experts),
        return_indexer_topk=bool(req.return_indexer_topk),
        multimodal_resume=multimodal_resume,
    )


def request_from_continuation(
    continuation: DecodeContinuation,
    allocation: ReservedKV,
    *,
    req_to_token_pool: ReqToTokenPool,
    state_restorer: StateRestorer,
) -> Req:
    """Install a transferred request as SGLang's existing PREBUILT input."""

    sampling_values = dict(continuation.sampling_params)
    if isinstance(sampling_values.get("stop_token_ids"), list):
        sampling_values["stop_token_ids"] = set(sampling_values["stop_token_ids"])
    else:
        pass
    sampling_params = SamplingParams(**sampling_values)
    request = Req(
        rid=continuation.request_id,
        priority=continuation.priority,
        origin_input_text="",
        origin_input_ids=array("q", continuation.origin_input_ids),
        origin_input_ids_unpadded=(
            array("q", continuation.origin_input_ids_unpadded)
            if continuation.origin_input_ids_unpadded is not None
            else None
        ),
        sampling_params=sampling_params,
        # Prefill logprobs are already retained by Omni request data.
        return_logprob=False,
        top_logprobs_num=continuation.top_logprobs_num,
        token_ids_logprob=continuation.token_ids_logprob,
        return_sampling_mask=continuation.return_sampling_mask,
        return_hidden_states=continuation.return_hidden_states,
        return_routed_experts=continuation.return_routed_experts,
        return_indexer_topk=continuation.return_indexer_topk,
        eos_token_ids=(
            set(continuation.eos_token_ids)
            if continuation.eos_token_ids is not None
            else None
        ),
        vocab_size=continuation.vocab_size,
    )
    request.output_ids.extend(continuation.output_ids)
    request.cached_tokens = continuation.cached_tokens
    request.already_computed = continuation.cached_tokens
    request.cached_tokens_device = continuation.cached_tokens_device
    request.cached_tokens_host = continuation.cached_tokens_host
    request.cached_tokens_storage = continuation.cached_tokens_storage
    request.mm_image_tokens = continuation.mm_image_tokens
    request.mm_audio_tokens = continuation.mm_audio_tokens
    request.mm_video_tokens = continuation.mm_video_tokens
    request.logprob_start_len = continuation.logprob_start_len

    payload = StagePayload.from_dict(continuation.stage_payload)
    request_data = SGLangARRequestData(
        input_ids=torch.tensor(continuation.origin_input_ids, dtype=torch.long),
        output_ids=request.output_ids,
        req=request,
        stage_payload=payload,
        max_new_tokens=sampling_params.max_new_tokens,
        temperature=float(sampling_params.temperature),
        top_p=float(sampling_params.top_p),
        top_k=sampling_params.top_k,
        repetition_penalty=float(sampling_params.repetition_penalty),
        return_logprob=continuation.return_logprob,
        output_token_logprobs=list(continuation.output_token_logprobs),
    )
    request.omni_data = request_data
    state_restorer(request, request_data, continuation.multimodal_resume)
    if request.tokenizer is None and (
        sampling_params.stop_strs or sampling_params.stop_regex_strs
    ):
        raise ValueError("PD state_restorer must provide a tokenizer for stop strings")
    else:
        pass

    if req_to_token_pool.alloc([request]) is None:
        raise RuntimeError("decode request pool row credit violated")
    else:
        pass
    # note (YzXiao): hand off the pre-reserved row credit to the real pool row atomically so free_rows stays invariant.
    allocation.release_row_credit()
    if allocation.slots.is_cuda:
        # note (YzXiao): slots were allocated on allocation_stream; record compute stream usage so the caching allocator cannot recycle them mid-write.
        allocation.slots.record_stream(
            torch.cuda.current_stream(allocation.slots.device)
        )
    else:
        pass
    try:
        req_to_token_pool.write(
            (request.kv.req_pool_idx, slice(0, allocation.sequence_length)),
            allocation.slots[: allocation.sequence_length],
        )
    except Exception:
        req_to_token_pool.free(request)
        raise
    request.prefix_len = allocation.sequence_length
    request.kv.kv_committed_len = allocation.sequence_length
    request.kv.kv_allocated_len = allocation.sequence_length
    request.extend_end = allocation.sequence_length
    request._omni_terminal_claimed = False  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    request._coalesce_enqueue_t = 0.0  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    return request


def sampling_params_to_dict(params: SamplingParams) -> dict[str, object]:
    allowed = inspect.signature(type(params)).parameters
    values = {name: getattr(params, name) for name in allowed if hasattr(params, name)}
    custom = values.get("custom_params")
    if isinstance(custom, dict):
        # note (Yue Yin): Legacy custom parameters can retain a live Req; it must not cross the PD transport.
        values["custom_params"] = {
            key: value for key, value in custom.items() if key != "__req__"
        }
    else:
        pass
    return values


@contextmanager
def defer_first_token_finish(reqs: list[Req]) -> Generator[None, None, None]:
    """Let normal Prefill accounting run while Decode owns stop decisions."""

    saved = []
    for req in reqs:
        params = req.sampling_params
        values = (
            params.max_new_tokens,
            params.ignore_eos,
            params.stop_strs,
            params.stop_regex_strs,
        )
        saved.append((params, values))
        params.max_new_tokens = max(params.max_new_tokens, len(req.output_ids) + 2)
        params.ignore_eos = True
        params.stop_strs = []
        params.stop_regex_strs = []
    try:
        yield
    finally:
        for params, values in saved:
            (
                params.max_new_tokens,
                params.ignore_eos,
                params.stop_strs,
                params.stop_regex_strs,
            ) = values


def build_kv_pool(token_to_kv_pool: KVCache, *, pool_id: str) -> KVPool:
    """Build a transfer-registrable KVPool view over an SGLang KV cache."""
    page_size = token_to_kv_pool.page_size
    if page_size < 1:
        raise ValueError(f"SGLang KV pool page_size must be >= 1, got {page_size}")
    else:
        pass

    type_qualname = (
        f"{type(token_to_kv_pool).__module__}.{type(token_to_kv_pool).__qualname__}"
    )
    if isinstance(token_to_kv_pool, MHATokenToKVPool):
        tensors = tuple(
            token_to_kv_pool._pd_registerable_tensors()  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        if (
            type(token_to_kv_pool) is MHATokenToKVPool
            or type_qualname == SEPARATE_KV_ALLOCATIONS_POOL_TYPE_NAME
        ):
            # note (YzXiao): SeparateKVAllocationsPool is byte-identical to MHATokenToKVPool, so compiled and eager stages stay layout-compatible.
            pool_family = "MHATokenToKVPool"
        else:
            pool_family = type_qualname
        layout_tag = f"layout={token_to_kv_pool.kv_cache_layout}"
    elif isinstance(token_to_kv_pool, MLATokenToKVPool):
        tensors = tuple(token_to_kv_pool.kv_buffer)
        if type(token_to_kv_pool) is MLATokenToKVPool:
            pool_family = "MLATokenToKVPool"
        else:
            pool_family = type_qualname
        layout_tag = "layout=mla"
    else:
        registerable_getter = getattr(
            token_to_kv_pool,
            "_pd_registerable_tensors",  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            None,
        )
        if callable(registerable_getter):
            tensors = tuple(registerable_getter())
        else:
            tensors = tuple(
                tensor
                for layer_id in range(token_to_kv_pool.layer_num)
                for tensor in (
                    token_to_kv_pool.get_key_buffer(layer_id),
                    token_to_kv_pool.get_value_buffer(layer_id),
                )
            )
        pool_family = type_qualname
        layout_tag = "layout=default"

    buffer_pointers, _, page_byte_sizes = token_to_kv_pool.get_contiguous_buf_infos()
    if (
        not tensors
        or len(tensors) != len(page_byte_sizes)
        or len(tensors) != len(buffer_pointers)
    ):
        raise ValueError("SGLang KV pool exposed incompatible buffer metadata")
    else:
        pass
    tensor_pointers = [tensor.data_ptr() for tensor in tensors]
    expected_pointers = list(buffer_pointers)
    if tensor_pointers != expected_pointers:
        raise ValueError(
            "SGLang KV pool registerable tensors do not match contiguous buffer pointers"
        )
    else:
        pass
    dtype_tag = (
        f"dtype={token_to_kv_pool.dtype}:store_dtype={token_to_kv_pool.store_dtype}"
    )
    return KVPool(
        pool_id=pool_id,
        layout_id=f"{pool_family}:{layout_tag}:{dtype_tag}:page_size={page_size}",
        page_size=page_size,
        buffers=tuple(
            KVBufferRegion(
                name=f"kv_buffer.{index}",
                tensor=tensor,
                # note (YzXiao): get_contiguous_buf_infos already scales page_byte_sizes by page_size.
                bytes_per_page=bytes_per_page,
            )
            for index, (tensor, bytes_per_page) in enumerate(
                zip(tensors, page_byte_sizes)
            )
        ),
    )


def request_page_indices(
    req_to_token_pool: ReqToTokenPool,
    req: Req,
    *,
    page_size: int = 1,
) -> tuple[int, ...]:
    """Return physical page indices backing a Prefill request's prompt tokens."""
    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")
    else:
        pass
    if req.kv.req_pool_idx is None:
        raise RuntimeError(f"request {req.rid!r} has no KV mapping")
    else:
        pass
    sequence_length = len(req.origin_input_ids)
    if sequence_length <= 0:
        raise ValueError("PD cannot transfer an empty prompt")
    else:
        pass
    page_heads = req_to_token_pool.req_to_token[
        req.kv.req_pool_idx, :sequence_length:page_size
    ].tolist()
    expected_pages = (sequence_length + page_size - 1) // page_size
    return page_indices_from_heads(
        page_heads,
        expected_pages=expected_pages,
        page_size=page_size,
        context_label=f"request {req.rid!r}",
    )


class SGLangKVLease:
    """Keep source pages owned until the receiver ACKs the copy."""

    def __init__(
        self,
        req: Req | None,
        due_releases: queue.SimpleQueue[Req],
        *,
        wakeup_notifier: WakeupNotifier | None = None,
    ) -> None:
        self.req: Req | None = req
        self.due_releases = due_releases
        self.wakeup_notifier = wakeup_notifier
        self.lock = threading.Lock()

    def release(self) -> None:
        with self.lock:
            req = self.req
            self.req = None
        if req is None:
            return
        else:
            pass
        # The comm thread acknowledges ownership; only the scheduler thread
        # may mutate its request table and prefix cache.
        self.due_releases.put(req)
        if self.wakeup_notifier is not None:
            self.wakeup_notifier.notify_pd_wakeup()
        else:
            pass
