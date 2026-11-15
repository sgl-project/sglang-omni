# SPDX-License-Identifier: Apache-2.0
"""repetition_penalty for speculative verify rows, exactly as plain decode applies it.

Verify row j follows input token j of a request's block, where token 0 is the committed anchor and tokens 1 onward are drafts. Plain decode at that position penalizes every committed output token and drafts 1 to j, and the row only commits when all of those drafts were accepted, so penalizing them is exact.
"""

from __future__ import annotations

import torch


def apply_verify_repetition_penalty(
    verify_logits: torch.Tensor,
    verify_input_ids: torch.Tensor,
    committed_token_presence: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """Scale every verify row in place as plain decode would at its position.

    Args:
        verify_logits: [requests, block, vocab].
        verify_input_ids: [requests, block] int64, the anchor then the drafts.
        committed_token_presence: [requests, vocab] bool, the committed output tokens.
        repetition_penalties: [requests] float32.
    """
    request_count, block_size, _ = verify_logits.shape
    penalties = repetition_penalties[:, None]
    committed_scales = torch.where(committed_token_presence, penalties, 1.0)[:, None, :]
    verify_logits[:] = torch.where(
        verify_logits < 0,
        verify_logits * committed_scales,
        verify_logits / committed_scales,
    )
    positions = torch.arange(block_size, device=verify_logits.device)
    # [row, input token]: drafts 1 to j precede row j.
    is_draft_before_row = (positions[None, :] >= 1) & (
        positions[None, :] <= positions[:, None]
    )
    is_same_token = verify_input_ids[:, :, None] == verify_input_ids[:, None, :]
    is_in_drafts_before_row = (
        is_same_token[:, None, :, :] & is_draft_before_row[None, :, None, :]
    ).any(dim=-1)
    is_uncommitted = ~committed_token_presence.gather(1, verify_input_ids)
    # note (wilsonzheng0327): duplicate draft tokens index one logit, and every copy carries the same value, so the scatter is deterministic.
    needs_draft_scale = is_in_drafts_before_row & is_uncommitted[:, None, :]
    token_index = verify_input_ids[:, None, :].expand(
        request_count, block_size, block_size
    )
    draft_logits = verify_logits.gather(2, token_index)
    draft_penalties = penalties[:, :, None]
    scaled_draft_logits = torch.where(
        draft_logits < 0,
        draft_logits * draft_penalties,
        draft_logits / draft_penalties,
    )
    verify_logits.scatter_(
        2,
        token_index,
        torch.where(needs_draft_scale, scaled_draft_logits, draft_logits).to(
            verify_logits.dtype
        ),
    )


def record_committed_tokens(
    committed_token_presence: torch.Tensor,
    request_slots: torch.Tensor,
    committed_token_ids: torch.Tensor,
    committed_lengths: torch.Tensor,
) -> None:
    """Mark each request's leading committed tokens, without a host sync.

    Args:
        committed_token_presence: [slots, vocab + 1] bool; the last column absorbs the uncommitted tail.
        request_slots: [requests] int64.
        committed_token_ids: [requests, width] int64.
        committed_lengths: [requests], the committed prefix of each row of committed_token_ids.
    """
    padding_column = committed_token_presence.shape[1] - 1
    is_committed = (
        torch.arange(committed_token_ids.shape[1], device=committed_token_ids.device)[
            None, :
        ]
        < committed_lengths[:, None]
    )
    token_columns = torch.where(is_committed, committed_token_ids, padding_column)
    committed_token_presence[
        request_slots[:, None].expand_as(token_columns), token_columns
    ] = True
