# SPDX-License-Identifier: Apache-2.0
"""SenseNova U1 hybrid prefill attention helpers.

U1 assigns all ``<IMG_CONTEXT>`` tokens from one image span the same temporal
index. Text rows remain causal; image rows may also attend to every token in
their image span. The resulting dense mask is equivalent to the official
``create_block_causal_mask(indexes[0])`` implementation and is used as the
native Python reference before a fused backend grows the same row policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class ImageSpan:
    """Contiguous image-token span sharing one U1 temporal index."""

    start: int
    end: int
    t_index: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def as_dict(self) -> dict[str, int]:
        return {
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "t_index": self.t_index,
        }


def _as_t_index(t_indexes: torch.Tensor) -> torch.Tensor:
    if t_indexes.ndim == 2:
        if t_indexes.shape[0] != 3:
            raise ValueError(
                "Expected U1 indexes with shape (3, L) when passing a 2D tensor."
            )
        return t_indexes[0]
    if t_indexes.ndim != 1:
        raise ValueError("Expected t_indexes with shape (L,) or indexes with shape (3, L).")
    return t_indexes


def build_image_token_tag_from_input_ids(
    input_ids: torch.Tensor,
    img_context_token_id: int,
) -> torch.Tensor:
    """Return a bool tag for per-token U1 image context rows."""

    ids = input_ids.reshape(-1)
    return ids == int(img_context_token_id)


def build_image_token_tag_from_t_indexes(t_indexes: torch.Tensor) -> torch.Tensor:
    """Infer image rows from repeated temporal indexes.

    In U1 understanding prefixes, repeated temporal indexes are produced by
    image context spans. This helper is useful for synthetic tests where token
    ids are not available.
    """

    t_index = _as_t_index(t_indexes)
    if t_index.numel() == 0:
        return torch.empty_like(t_index, dtype=torch.bool)
    _, inverse, counts = torch.unique(t_index, sorted=False, return_inverse=True, return_counts=True)
    return counts[inverse] > 1


def build_image_spans(
    t_indexes: torch.Tensor,
    image_token_tag: torch.Tensor | None = None,
) -> list[ImageSpan]:
    """Summarize contiguous image spans for evidence and debugging."""

    t_index = _as_t_index(t_indexes).detach().cpu()
    if image_token_tag is None:
        tag = build_image_token_tag_from_t_indexes(t_index)
    else:
        tag = image_token_tag.reshape(-1).detach().cpu().bool()
    if tag.numel() != t_index.numel():
        raise ValueError("image_token_tag length must match t_indexes length.")

    spans: list[ImageSpan] = []
    start: int | None = None
    cur_t: int | None = None
    for pos, is_image in enumerate(tag.tolist()):
        pos_t = int(t_index[pos].item())
        if is_image and start is None:
            start = pos
            cur_t = pos_t
        elif is_image and cur_t != pos_t:
            spans.append(ImageSpan(start=start, end=pos, t_index=int(cur_t)))
            start = pos
            cur_t = pos_t
        elif not is_image and start is not None:
            spans.append(ImageSpan(start=start, end=pos, t_index=int(cur_t)))
            start = None
            cur_t = None
    if start is not None:
        spans.append(ImageSpan(start=start, end=tag.numel(), t_index=int(cur_t)))
    return spans


def build_m_block_summary(
    image_token_tag: torch.Tensor,
    *,
    block_m: int,
) -> list[dict[str, Any]]:
    """Return the M-block OR-reduction described by U1's inference docs."""

    if block_m <= 0:
        raise ValueError("block_m must be positive.")
    tag = image_token_tag.reshape(-1).detach().cpu().bool()
    rows: list[dict[str, Any]] = []
    for start in range(0, tag.numel(), block_m):
        end = min(start + block_m, tag.numel())
        block = tag[start:end]
        rows.append(
            {
                "start": start,
                "end": end,
                "has_image": bool(block.any().item()),
                "image_rows": [start + i for i, v in enumerate(block.tolist()) if v],
            }
        )
    return rows


def build_u1_hybrid_allowed_matrix(
    t_indexes: torch.Tensor,
    image_token_tag: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the boolean U1 hybrid attention matrix with shape ``(L, L)``."""

    t_index = _as_t_index(t_indexes)
    length = t_index.numel()
    positions = torch.arange(length, device=t_index.device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    same_t = t_index.unsqueeze(0) == t_index.unsqueeze(1)
    if image_token_tag is None:
        return causal | same_t

    tag = image_token_tag.reshape(-1).to(device=t_index.device, dtype=torch.bool)
    if tag.numel() != length:
        raise ValueError("image_token_tag length must match t_indexes length.")
    same_image_span = same_t & tag.unsqueeze(0) & tag.unsqueeze(1)
    return causal | same_image_span


def build_u1_hybrid_backend_mask(
    indexes: torch.Tensor,
    image_token_tag: torch.Tensor,
    extend_seq_lens: list[int] | torch.Tensor,
    extend_prefix_lens: list[int] | torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Build a flattened backend custom mask for SGLang prefill attention.

    SGLang's extend kernels receive only this round's query/KV tensors plus a
    cached prefix. They consume one flat mask per request with row-major shape
    ``q_len x (prefix_len + q_len)``. Prefix columns are already before every
    query row, so they remain visible; current-chunk columns use U1's hybrid
    causal/image-span policy.
    """

    if indexes.ndim != 2 or indexes.shape[0] != 3:
        raise ValueError("indexes must have shape (3, total_extend_tokens).")
    tag = image_token_tag.reshape(-1).to(device=indexes.device, dtype=torch.bool)
    if tag.numel() != indexes.shape[1]:
        raise ValueError("image_token_tag length must match indexes length.")

    if isinstance(extend_seq_lens, torch.Tensor):
        seq_lens = [int(x) for x in extend_seq_lens.detach().cpu().tolist()]
    else:
        seq_lens = [int(x) for x in extend_seq_lens]
    if extend_prefix_lens is None:
        prefix_lens = [0] * len(seq_lens)
    elif isinstance(extend_prefix_lens, torch.Tensor):
        prefix_lens = [int(x) for x in extend_prefix_lens.detach().cpu().tolist()]
    else:
        prefix_lens = [int(x) for x in extend_prefix_lens]
    if len(seq_lens) != len(prefix_lens):
        raise ValueError("extend_seq_lens and extend_prefix_lens length mismatch.")

    parts: list[torch.Tensor] = []
    indptr = [0]
    token_offset = 0
    for q_len, prefix_len in zip(seq_lens, prefix_lens):
        if q_len < 0 or prefix_len < 0:
            raise ValueError("extend lengths and prefix lengths must be non-negative.")
        if q_len == 0:
            indptr.append(indptr[-1])
            continue
        req_indexes = indexes[:, token_offset : token_offset + q_len]
        req_tag = tag[token_offset : token_offset + q_len]
        if req_indexes.shape[1] != q_len or req_tag.numel() != q_len:
            raise ValueError(
                "indexes/image_token_tag do not cover all extend tokens."
            )

        t = req_indexes[0]
        key_pos = torch.arange(q_len, device=indexes.device)
        query_pos = torch.arange(q_len, device=indexes.device)
        causal = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        same_t = t.unsqueeze(1) == t.unsqueeze(0)
        same_image_span = (
            same_t
            & req_tag.unsqueeze(1)
            & req_tag.unsqueeze(0)
        )
        current_allowed = causal | same_image_span
        if prefix_len:
            prefix_allowed = torch.ones(
                (q_len, prefix_len), dtype=torch.bool, device=indexes.device
            )
            allowed = torch.cat([prefix_allowed, current_allowed], dim=1)
        else:
            allowed = current_allowed
        flat = allowed.reshape(-1).to(dtype=torch.uint8)
        parts.append(flat)
        indptr.append(indptr[-1] + int(flat.numel()))
        token_offset += q_len

    if token_offset != indexes.shape[1]:
        raise ValueError("extend lengths do not consume all provided indexes tokens.")

    indptr_tensor = torch.tensor(indptr, dtype=torch.int64, device=indexes.device)
    if not parts:
        return None, indptr_tensor
    mask = torch.cat(parts, dim=0)
    if not bool(image_token_tag.reshape(-1).to(dtype=torch.bool).any().item()):
        return None, indptr_tensor
    return mask, indptr_tensor


def create_u1_hybrid_mask(
    index: torch.Tensor,
    *,
    image_token_tag: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create U1's additive prefill attention mask with shape ``(1, 1, L, L)``."""

    t_index = _as_t_index(index)
    allowed = build_u1_hybrid_allowed_matrix(t_index, image_token_tag=image_token_tag)
    mask_dtype = dtype or torch.get_default_dtype()
    zero = torch.zeros((), device=t_index.device, dtype=mask_dtype)
    neg_inf = torch.full((), float("-inf"), device=t_index.device, dtype=mask_dtype)
    return torch.where(allowed[None, None, :, :], zero, neg_inf)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def u1_hybrid_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    t_indexes: torch.Tensor,
    *,
    image_token_tag: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense reference attention for U1 hybrid prefill rows.

    Inputs follow the HF attention convention ``(B, H, L, D)`` and the output
    follows the official U1 eager path convention ``(B, L, H, D)``.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key and value must all have shape (B, H, L, D).")
    if key.shape[:3] != value.shape[:3] or key.shape[-1] != value.shape[-1]:
        raise ValueError("key and value shapes are incompatible.")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key batch or head_dim mismatch.")
    if query.shape[2] != key.shape[2]:
        raise ValueError("U1 prefill dense reference expects query and key lengths to match.")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be a multiple of key/value heads.")

    num_key_value_groups = query.shape[1] // key.shape[1]
    key_states = _repeat_kv(key, num_key_value_groups)
    value_states = _repeat_kv(value, num_key_value_groups)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scale
    attn_weights = attn_weights + create_u1_hybrid_mask(
        t_indexes,
        image_token_tag=image_token_tag,
        dtype=attn_weights.dtype,
    )
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights
