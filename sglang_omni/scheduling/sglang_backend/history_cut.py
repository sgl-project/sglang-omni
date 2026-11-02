# SPDX-License-Identifier: Apache-2.0
"""Drop whole units of retained history from an idle SGLang streaming session, in place.

This module is the only code that edits SGLang's streaming session for a history cut: its committed
token arrays and lengths, and the session's KV row, as SGLang 0.5.22 lays them out. After a cut,
ARSessionBridge checks that SGLang put exactly the expected history ahead of the next unit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from sglang.srt.session.session_controller import Session
from sglang.srt.utils.common import ceil_align


class RetainedKV(Protocol):
    req_pool_idx: int | None
    cache_protected_len: int
    kv_committed_len: int
    kv_allocated_len: int


class RequestRows(Protocol):
    req_to_token: torch.Tensor


class SessionRows(Protocol):
    req_to_token_pool: RequestRows
    page_size: int

    def free_kv_row(self, kv: RetainedKV, ranges: list[tuple[int, int]]) -> None:
        pass


class LayerKVCache(Protocol):
    size: int
    page_size: int
    start_layer: int
    layer_num: int

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        pass

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        pass


@dataclass(frozen=True)
class HistoryCut:
    """Retained session positions to drop before the next unit: [start, end) ranges, ascending and apart."""

    ranges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        bounds = [bound for cut_range in self.ranges for bound in cut_range]
        if (
            not bounds
            or bounds[0] < 0
            or any(a >= b for a, b in zip(bounds, bounds[1:]))
        ):
            raise ValueError(f"invalid history cut {self.ranges}")
        else:
            pass

    @property
    def length(self) -> int:
        return sum(end - start for start, end in self.ranges)

    def new_position(self, position: int) -> int:
        """Where a position kept by the cut lands once the cut is applied."""
        return position - sum(
            min(end, position) - start for start, end in self.ranges if start < position
        )

    def drops(self, start: int, end: int) -> bool:
        """Whether [start, end) goes with the cut; it may not straddle a cut bound."""
        for cut_start, cut_end in self.ranges:
            if cut_start <= start and end <= cut_end:
                return True
            else:
                assert (
                    end <= cut_start or cut_end <= start
                ), f"[{start}, {end}) crosses the history cut {self.ranges}"
        return False


def retained_history_length(native_session: Session) -> int:
    """Tokens Session.create_req puts ahead of the next request: the committed ids, then the last outputs up to max_new_tokens."""
    if not native_session.req_nodes:
        return 0
    else:
        [last_node] = native_session.req_nodes.values()
        last_request = last_node.req
        return native_session.committed_origin_len + len(
            last_request.output_ids[: last_request.sampling_params.max_new_tokens]
        )


def cut_token_ids(native_session: Session, cut: HistoryCut) -> None:
    """Drop the cut from the committed token arrays that the next request extends in place."""
    [last_node] = native_session.req_nodes.values()
    last_request = last_node.req
    assert cut.ranges[-1][1] <= min(
        native_session.committed_origin_len,
        native_session.committed_unpadded_len,
        native_session.committed_fill_len,
    ), f"history cut {cut.ranges} leaves the committed token ids"
    # note (0xtoward): The unpadded ids usually alias the origin ids, so each array is cut once.
    token_id_arrays = {
        id(token_ids): token_ids
        for token_ids in (
            last_request.origin_input_ids,
            last_request.origin_input_ids_unpadded,
            last_request.full_untruncated_fill_ids,
        )
    }
    for token_ids in token_id_arrays.values():
        for start, end in reversed(cut.ranges):
            del token_ids[start:end]
    native_session.committed_origin_len -= cut.length
    native_session.committed_unpadded_len -= cut.length
    native_session.committed_fill_len -= cut.length


def cut_retained_kv(
    rows: SessionRows,
    kv_cache: LayerKVCache,
    kv: RetainedKV,
    cut: HistoryCut,
    shift_key_positions: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    """Move the KV kept after the cut down the session row, with each key re-encoded at its new position, and free the row tail.

    The row keeps its slots, so pages stay whole and captured graphs keep reading the same buffers.
    """
    first_start = cut.ranges[0][0]
    assert (
        first_start >= kv.cache_protected_len
    ), f"history cut at {first_start} reaches the tree-owned prefix of {kv.cache_protected_len} tokens"
    assert (
        cut.ranges[-1][1] <= kv.kv_committed_len
    ), f"history cut {cut.ranges} passes the committed KV of {kv.kv_committed_len} tokens"
    kept_length = kv.kv_committed_len - cut.length
    row = rows.req_to_token_pool.req_to_token[kv.req_pool_idx]
    kept_ends = [start for start, _ in cut.ranges[1:]] + [kv.kv_committed_len]
    kept_positions = torch.cat(
        [
            torch.arange(end, kept_end, device=row.device)
            for (_, end), kept_end in zip(cut.ranges, kept_ends)
        ]
    )
    distances = kept_positions - torch.arange(
        first_start, kept_length, device=row.device
    )
    source_slots = row[kept_positions].to(torch.int64)
    target_slots = row[first_start:kept_length].to(torch.int64)
    for layer_id in range(
        kv_cache.start_layer, kv_cache.start_layer + kv_cache.layer_num
    ):
        key_buffer = kv_cache.get_key_buffer(layer_id)
        value_buffer = kv_cache.get_value_buffer(layer_id)
        assert (
            key_buffer.shape[0] == kv_cache.size + kv_cache.page_size
        ), "history cut needs one KV row per token slot"
        # note (0xtoward): An FP8 cache is uint8 storage behind float8 views, and index_copy_ has no float8 kernel, so its rows move as bytes.
        key_rows, value_rows = (
            (key_buffer.view(torch.uint8), value_buffer.view(torch.uint8))
            if key_buffer.dtype.itemsize == 1
            else (key_buffer, value_buffer)
        )
        shifted_keys = shift_key_positions(
            key_rows.index_select(0, source_slots).view(key_buffer.dtype), distances
        )
        key_rows.index_copy_(0, target_slots, shifted_keys.view(key_rows.dtype))
        value_rows.index_copy_(
            0, target_slots, value_rows.index_select(0, source_slots)
        )
    tail_start = ceil_align(kept_length, rows.page_size)
    if tail_start < kv.kv_allocated_len:
        rows.free_kv_row(kv, [(tail_start, kv.kv_allocated_len)])
    else:
        pass
    kv.kv_allocated_len = kept_length
    kv.kv_committed_len = kept_length
