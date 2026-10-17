# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from array import array
from dataclasses import replace

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

from sglang_omni.models.qwen3_omni.prompt_hidden import PromptHiddenStates
from sglang_omni.models.qwen3_omni.request_builders import QwenThinkerRequestData
from sglang_omni.scheduling.sglang_backend.evict_heap_radix_cache import (
    EvictHeapRadixCache,
)


class QwenPromptHiddenRadixCache(EvictHeapRadixCache):
    def __init__(self, params: CacheInitParams) -> None:
        self.prompt_hidden_states: dict[TreeNode, PromptHiddenStates] = {}
        super().__init__(params)

    def reset(self) -> None:
        self.prompt_hidden_states.clear()
        super().reset()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        capture = (
            params.req.omni_data.prompt_hidden_capture
            if params.req is not None
            and isinstance(params.req.omni_data, QwenThinkerRequestData)
            else None
        )
        if capture is None:
            return super().match_prefix(params)
        elif capture.cacheable_prefix_length is not None:
            params = replace(params, key=params.key[: capture.cacheable_prefix_length])
        else:
            pass

        result = super().match_prefix(params)
        missing_positions = capture.positions[
            ~capture.received & (capture.positions < len(result.device_indices))
        ]
        node = result.last_device_node
        end_position = len(result.device_indices)
        while node is not self.root_node and missing_positions.numel():
            start_position = end_position - len(node.key)
            states = self.prompt_hidden_states.get(node)
            if states is not None:
                missing_positions = missing_positions[
                    ~torch.isin(missing_positions, states.positions + start_position)
                ]
            else:
                pass
            node = node.parent
            end_position = start_position

        if missing_positions.numel():
            result = super().match_prefix(
                replace(params, key=params.key[: int(missing_positions[0])])
            )
        else:
            pass

        node = result.last_device_node
        end_position = len(result.device_indices)
        while node is not self.root_node:
            start_position = end_position - len(node.key)
            states = self.prompt_hidden_states.get(node)
            if states is not None:
                capture.add_rows(
                    positions=states.positions + start_position,
                    hidden_states=states.hidden_states,
                )
            else:
                pass
            node = node.parent
            end_position = start_position
        return result

    def _insert_cache(  # noqa: leading-underscore
        self,
        req: Req,
        token_ids: array[int],
        *,
        chunked: bool = False,
        key_limit: int | None = None,
        split_prompt: bool = False,
    ) -> tuple[RadixKey, torch.Tensor, int]:
        capture = (
            req.omni_data.prompt_hidden_capture
            if isinstance(req.omni_data, QwenThinkerRequestData)
            else None
        )
        if capture is not None and capture.cacheable_prefix_length is not None:
            safe_prefix_length = (
                capture.cacheable_prefix_length // self.page_size * self.page_size
            )
            key_limit = (
                safe_prefix_length
                if key_limit is None
                else min(key_limit, safe_prefix_length)
            )
        else:
            pass
        radix_key, kv_indices, prefix_length = super()._insert_cache(
            req,
            token_ids,
            chunked=chunked,
            key_limit=key_limit,
            split_prompt=split_prompt,
        )
        if capture is None or capture.hidden_states is None:
            return radix_key, kv_indices, prefix_length
        else:
            pass

        result = super().match_prefix(MatchPrefixParams(key=radix_key))
        node = result.last_device_node
        end_position = len(result.device_indices)
        while node is not self.root_node:
            start_position = end_position - len(node.key)
            stored = self.prompt_hidden_states.get(node)
            new_rows = (
                capture.received
                & (capture.positions >= start_position)
                & (capture.positions < end_position)
            )
            if stored is not None:
                new_rows &= ~torch.isin(
                    capture.positions - start_position, stored.positions
                )
            else:
                pass
            if bool(new_rows.any()):
                positions = capture.positions[new_rows] - start_position
                hidden_states = capture.hidden_states[new_rows]
                if stored is not None:
                    positions = torch.cat((stored.positions, positions))
                    hidden_states = torch.cat((stored.hidden_states, hidden_states))
                    positions, row_order = positions.sort()
                    hidden_states = hidden_states[row_order]
                else:
                    pass
                self.prompt_hidden_states[node] = PromptHiddenStates(
                    positions=positions, hidden_states=hidden_states
                )
            else:
                pass
            node = node.parent
            end_position = start_position
        return radix_key, kv_indices, prefix_length

    def _split_node(  # noqa: leading-underscore
        self, key: RadixKey, child: TreeNode, split_len: int
    ) -> TreeNode:
        stored = self.prompt_hidden_states.pop(child, None)
        prefix_node = super()._split_node(key, child, split_len)
        if stored is None:
            return prefix_node
        else:
            pass
        for node, selected_rows, offset in (
            (prefix_node, stored.positions < split_len, 0),
            (child, stored.positions >= split_len, split_len),
        ):
            if bool(selected_rows.any()):
                self.prompt_hidden_states[node] = PromptHiddenStates(
                    positions=stored.positions[selected_rows] - offset,
                    hidden_states=stored.hidden_states[selected_rows],
                )
            else:
                pass
        return prefix_node

    def _delete_leaf(self, node: TreeNode) -> None:  # noqa: leading-underscore
        self.prompt_hidden_states.pop(node, None)
        super()._delete_leaf(node)  # noqa: leading-underscore

    def evict(self, params: EvictParams) -> EvictResult:
        if self.eviction_policy == "lru":
            return super().evict(params)
        else:
            return RadixCache.evict(self, params)
