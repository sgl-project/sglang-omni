# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of the routed expert cache: the decode LRU policy and expert copies."""

from __future__ import annotations

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

__all__ = [
    "HAS_TRITON",
    "copy_missing_kernel",
    "decode_policy_kernel",
    "gather_needed_kernel",
]

if HAS_TRITON:
    # Above every packed (last use, slot) key, so slots that hold a needed expert
    # sort after every eviction candidate.
    NOT_A_CANDIDATE_ORDER_KEY = tl.constexpr(2**63 - 1)

    @triton.jit
    def decode_policy_kernel(
        routed_experts: tl.tensor,
        slot_ids: tl.tensor,
        expert_to_slot: tl.tensor,
        slot_to_expert: tl.tensor,
        slot_last_used: tl.tensor,
        clock: tl.tensor,
        miss_experts: tl.tensor,
        miss_slots: tl.tensor,
        miss_count: tl.tensor,
        cache_statistics: tl.tensor,
        needed_flags: tl.tensor,
        victim_order: tl.tensor,
        routed_count: tl.tensor,
        EXPERT_COUNT: tl.constexpr,
        SLOT_COUNT: tl.constexpr,
        ROUTED_BLOCK: tl.constexpr,
        EXPERT_BLOCK: tl.constexpr,
        SLOT_BLOCK: tl.constexpr,
        SLOT_BITS: tl.constexpr,
    ) -> None:
        """One program per layer: the missing experts, in ascending id order, take the
        slots holding no needed expert, least recently used and then lowest first;
        every needed slot is stamped with the clock and each routed id gets its slot."""
        routed_offsets = tl.arange(0, ROUTED_BLOCK)
        expert_offsets = tl.arange(0, EXPERT_BLOCK)
        slot_offsets = tl.arange(0, SLOT_BLOCK)
        is_routed = routed_offsets < routed_count
        is_expert = expert_offsets < EXPERT_COUNT
        is_slot = slot_offsets < SLOT_COUNT
        routed_expert = tl.load(
            routed_experts + routed_offsets, mask=is_routed, other=0
        ).to(tl.int32)
        # note (zengjiahua): the policy goes through global scratch with barriers in
        # between instead of pairwise comparisons, so its cost is linear in the experts
        # and slots plus one sort of the slots.
        tl.store(needed_flags + expert_offsets, 0, mask=is_expert)
        tl.debug_barrier()
        tl.store(needed_flags + routed_expert, 1, mask=is_routed)
        tl.debug_barrier()
        is_needed = tl.load(needed_flags + expert_offsets, mask=is_expert, other=0) != 0
        old_slot = tl.load(expert_to_slot + expert_offsets, mask=is_expert, other=-1)
        is_missing = is_needed & (old_slot < 0)
        missing_rank = tl.cumsum(is_missing.to(tl.int32), axis=0) - is_missing.to(
            tl.int32
        )
        missing_total = tl.sum(is_missing.to(tl.int32), axis=0)
        held_expert = tl.load(slot_to_expert + slot_offsets, mask=is_slot, other=0)
        holds_needed = tl.load(needed_flags + held_expert, mask=is_slot, other=1) != 0
        last_used = tl.load(slot_last_used + slot_offsets, mask=is_slot, other=0)
        # note (zengjiahua): one sort of packed keys ranks the candidates oldest first
        # with the lower slot breaking ties, the order the reference policy evicts in.
        order_key = tl.where(
            is_slot & ~holds_needed,
            (last_used << SLOT_BITS) | slot_offsets.to(tl.int64),
            NOT_A_CANDIDATE_ORDER_KEY,
        )
        tl.store(victim_order + slot_offsets, tl.sort(order_key))
        tl.debug_barrier()
        victim_slot = (
            tl.load(victim_order + missing_rank, mask=is_missing, other=0)
            & ((1 << SLOT_BITS) - 1)
        ).to(tl.int32)
        evicted_expert = tl.load(slot_to_expert + victim_slot, mask=is_missing, other=0)
        tl.debug_barrier()
        tl.store(expert_to_slot + evicted_expert, -1, mask=is_missing)
        tl.debug_barrier()
        tl.store(expert_to_slot + expert_offsets, victim_slot, mask=is_missing)
        tl.store(slot_to_expert + victim_slot, expert_offsets, mask=is_missing)
        tl.store(miss_experts + missing_rank, expert_offsets, mask=is_missing)
        tl.store(miss_slots + missing_rank, victim_slot, mask=is_missing)
        tl.store(miss_count, missing_total)
        tl.debug_barrier()
        current_step = tl.load(clock)
        routed_slot = tl.load(expert_to_slot + routed_expert, mask=is_routed, other=0)
        tl.store(slot_last_used + routed_slot, current_step, mask=is_routed)
        tl.store(slot_ids + routed_offsets, routed_slot, mask=is_routed)
        tl.store(clock, current_step + 1)
        # note (zengjiahua): the statistics row is lookups, decode misses, prefill misses.
        tl.atomic_add(cache_statistics, tl.sum(is_needed.to(tl.int64), axis=0))
        tl.atomic_add(cache_statistics + 1, missing_total.to(tl.int64))

    @triton.jit
    def copy_missing_kernel(
        host_words: tl.tensor,
        slot_words: tl.tensor,
        miss_experts: tl.tensor,
        miss_slots: tl.tensor,
        miss_count: tl.tensor,
        WORDS_PER_EXPERT: tl.constexpr,
        BLOCK_WORDS: tl.constexpr,
    ) -> None:
        """Program (miss rank, block): one block of a missing expert, host to slot."""
        miss_rank = tl.program_id(0)
        is_miss = miss_rank < tl.load(miss_count)
        expert_index = tl.load(miss_experts + miss_rank, mask=is_miss, other=0).to(
            tl.int64
        )
        slot_index = tl.load(miss_slots + miss_rank, mask=is_miss, other=0).to(tl.int64)
        word_offsets = tl.program_id(1) * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
        is_word = (word_offsets < WORDS_PER_EXPERT) & is_miss
        expert_word_block = tl.load(
            host_words + expert_index * WORDS_PER_EXPERT + word_offsets, mask=is_word
        )
        tl.store(
            slot_words + slot_index * WORDS_PER_EXPERT + word_offsets,
            expert_word_block,
            mask=is_word,
        )

    @triton.jit
    def gather_needed_kernel(
        host_words: tl.tensor,
        slot_words: tl.tensor,
        scratch_words: tl.tensor,
        is_needed: tl.tensor,
        expert_to_slot: tl.tensor,
        WORDS_PER_EXPERT: tl.constexpr,
        BLOCK_WORDS: tl.constexpr,
    ) -> None:
        """Program (expert, block): one block of a needed expert into its scratch row,
        from its slot when resident and from host memory otherwise."""
        expert_index = tl.program_id(0).to(tl.int64)
        is_expert_needed = tl.load(is_needed + expert_index) != 0
        slot_index = tl.load(expert_to_slot + expert_index).to(tl.int64)
        word_offsets = tl.program_id(1) * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
        is_word = (word_offsets < WORDS_PER_EXPERT) & is_expert_needed
        from_slot = tl.load(
            slot_words + tl.maximum(slot_index, 0) * WORDS_PER_EXPERT + word_offsets,
            mask=is_word & (slot_index >= 0),
        )
        from_host = tl.load(
            host_words + expert_index * WORDS_PER_EXPERT + word_offsets,
            mask=is_word & (slot_index < 0),
        )
        tl.store(
            scratch_words + expert_index * WORDS_PER_EXPERT + word_offsets,
            tl.where(slot_index >= 0, from_slot, from_host),
            mask=is_word,
        )

else:
    pass
