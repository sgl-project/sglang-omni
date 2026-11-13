# SPDX-License-Identifier: Apache-2.0
"""The thinker's hidden states at multimodal prompt positions, kept per KV slot in pinned host memory, so a prefix-cache hit returns the rows its cached KV was computed with."""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.pool_host.common import (  # noqa: leading-underscore  # the allocator's own unregister
    ALLOC_MEMORY_FUNCS,
    HostTensorAllocator,
    _cuda_host_unregister,
    alloc_with_host_register,
)

logger = logging.getLogger(__name__)

BYTES_PER_GIB = 1 << 30


class PromptHiddenRequest(Req):
    """A request whose prefill writes the thinker's hidden states at its multimodal prompt positions to the prompt hidden store."""

    prompt_hidden_positions: torch.Tensor
    is_prompt_hidden_handed_off: bool = False

    def prompt_hidden_offsets(
        self, prefix_length: int, extend_length: int
    ) -> torch.Tensor:
        """Offsets, within an extend of extend_length tokens after prefix_length, of the positions it computes."""
        extend_bounds = torch.tensor([prefix_length, prefix_length + extend_length])
        start, end = torch.searchsorted(
            self.prompt_hidden_positions, extend_bounds
        ).tolist()
        return self.prompt_hidden_positions[start:end] - prefix_length


@dataclass(kw_only=True)
class StagedPromptHidden:
    """One forward's rows and their KV slot ids on their way to pinned host memory."""

    slot_ids: torch.Tensor
    hidden_states: torch.Tensor
    copy_done: torch.cuda.Event


class PromptHiddenStore:
    """Hidden-state rows indexed by KV slot id, read through the slots a request's KV holds."""

    def __init__(
        self,
        *,
        slot_count: int,
        hidden_size: int,
        dtype: torch.dtype,
        req_to_token_pool: ReqToTokenPool,
        device: str,
    ) -> None:
        allocate = ALLOC_MEMORY_FUNCS[device]
        self.hidden_states: torch.Tensor | None = allocate(
            (slot_count, hidden_size),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
            allocator=HostTensorAllocator(),
        )
        self.is_host_registered = allocate is alloc_with_host_register
        self.req_to_token_pool = req_to_token_pool
        self.device_module = torch.get_device_module(device)
        self.staged: collections.deque[StagedPromptHidden] = collections.deque()
        logger.info(
            f"Prompt hidden store: {slot_count} slots of {hidden_size} {dtype}, "
            f"{self.hidden_states.nbytes / BYTES_PER_GIB:.2f} GiB pinned"
        )

    def stage(self, slot_ids: torch.Tensor, hidden_states: torch.Tensor) -> None:
        """Copy a forward's rows and their slot ids to host behind the forward on its stream."""
        staged = StagedPromptHidden(
            slot_ids=torch.empty(slot_ids.shape, dtype=slot_ids.dtype, pin_memory=True),
            hidden_states=torch.empty(
                hidden_states.shape, dtype=hidden_states.dtype, pin_memory=True
            ),
            copy_done=self.device_module.Event(),
        )
        staged.slot_ids.copy_(slot_ids, non_blocking=True)
        staged.hidden_states.copy_(hidden_states, non_blocking=True)
        staged.copy_done.record()
        self.staged.append(staged)
        self.store_copied(wait=False)

    def store_copied(self, *, wait: bool) -> None:
        """Write staged rows into their slots in forward order, so a reused slot keeps its last forward's row."""
        while self.staged and (wait or self.staged[0].copy_done.query()):
            staged = self.staged.popleft()
            staged.copy_done.synchronize()
            self.hidden_states[staged.slot_ids] = staged.hidden_states

    def destroy(self) -> None:
        """Release the host registration the allocator made, so a buffer mapped later at the same address can register."""
        if self.is_host_registered:
            _cuda_host_unregister(self.hidden_states)
        else:
            pass
        self.hidden_states = None

    def read(self, request_pool_index: int, positions: torch.Tensor) -> torch.Tensor:
        """The rows at a request's prompt positions, through the slots its KV holds now."""
        req_to_token = self.req_to_token_pool.req_to_token
        slot_ids = req_to_token[
            request_pool_index, positions.to(req_to_token.device)
        ].cpu()
        self.store_copied(wait=True)
        return self.hidden_states[slot_ids]
