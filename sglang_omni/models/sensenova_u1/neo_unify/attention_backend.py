"""Attention backend."""

import torch
import torch._dynamo

# Modified for SGLang; see this directory's README.md for upstream source.


try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN = False


def npu_fia_available() -> bool:
    return hasattr(torch.ops.npu, "npu_fused_infer_attention_score")


def npu_swiglu_available() -> bool:
    return hasattr(torch.ops.npu, "npu_swiglu")


def sdpa_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in SDPA fallback for flash_attn_func."""
    q_bhsd = q.transpose(1, 2)
    k_bhsd = k.transpose(1, 2)
    v_bhsd = v.transpose(1, 2)

    h_q = q_bhsd.shape[1]
    h_kv = k_bhsd.shape[1]
    if h_q != h_kv:
        if h_q % h_kv != 0:
            raise ValueError(
                f"Cannot broadcast key/value heads ({h_kv}) to query heads ({h_q}): not divisible."
            )
        else:
            pass
        n_rep = h_q // h_kv
        k_bhsd = k_bhsd.repeat_interleave(n_rep, dim=1)
        v_bhsd = v_bhsd.repeat_interleave(n_rep, dim=1)
    else:
        pass

    out = torch.nn.functional.scaled_dot_product_attention(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous()


def flash_or_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    attention_mask: torch.Tensor | None = None,
    actual_seq_lengths_kv: list[int] | None = None,
    kv_padding_size: torch.Tensor | None = None,
    input_layout: str = "BSND",
) -> torch.Tensor:
    if (
        q.device.type == "npu"
        and not causal
        and dropout_p == 0.0
        and actual_seq_lengths_kv is not None
        and npu_fia_available()
    ):
        if input_layout == "BNSD":
            q_bhsd, k_bhsd, v_bhsd = q, k, v
            batch_size, num_heads, query_length, _ = q.shape
            num_key_value_heads = k.shape[1]
        elif input_layout == "BSND":
            q_bhsd = q.transpose(1, 2).contiguous()
            k_bhsd = k.transpose(1, 2).contiguous()
            v_bhsd = v.transpose(1, 2).contiguous()
            batch_size, query_length, num_heads, _ = q.shape
            num_key_value_heads = k.shape[2]
        else:
            raise ValueError(f"Unsupported attention input layout: {input_layout}")
        out, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            actual_seq_lengths=[query_length] * batch_size,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            kv_padding_size=kv_padding_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            scale=q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
            input_layout="BNSD",
            sparse_mode=0,
        )
        return out.transpose(1, 2).contiguous()
    else:
        pass
    if input_layout != "BSND":
        raise RuntimeError("BNSD attention input requires the NPU FIA path")
    elif actual_seq_lengths_kv is not None:
        raise RuntimeError(
            "Per-row KV lengths require the NPU FIA path; pass a key padding mask"
        )
    else:
        pass
    if HAS_FLASH_ATTN and q.device.type == "cuda" and attention_mask is None:
        return flash_attn_func(
            q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal
        )
    else:
        pass
    return sdpa_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        attention_mask=attention_mask,
    )


def position_ids_from_indexes(indexes: torch.Tensor, coordinate: int) -> torch.Tensor:
    """Return one coordinate as [batch, sequence] position IDs."""
    if indexes.ndim == 2:
        return indexes[coordinate].unsqueeze(0)
    else:
        pass
    if indexes.ndim == 3:
        return indexes[:, coordinate]
    else:
        pass
    raise ValueError(f"indexes must have 2 or 3 dimensions, got {indexes.ndim}")


def create_block_causal_mask(
    index: torch.Tensor, key_valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """index: (L) or (B, L)"""
    if index.ndim == 1:
        index = index.unsqueeze(0)
    else:
        pass
    if index.ndim != 2:
        raise ValueError(f"index must have 1 or 2 dimensions, got {index.ndim}")
    else:
        pass

    batch_size, seq_len = index.shape
    idx_i = index.unsqueeze(2)
    idx_j = index.unsqueeze(1)

    arange = torch.arange(seq_len, device=index.device)
    mask = (idx_j == idx_i) | (arange.view(1, 1, -1) <= arange.view(1, -1, 1))
    if key_valid_mask is not None:
        if key_valid_mask.shape != (batch_size, seq_len):
            raise ValueError(
                "key_valid_mask must match the batched index shape; "
                f"got {tuple(key_valid_mask.shape)} and {(batch_size, seq_len)}"
            )
        else:
            pass
        mask = mask & key_valid_mask.to(torch.bool).unsqueeze(1)
    else:
        pass

    output = torch.zeros(mask.shape, dtype=torch.float32, device=index.device)
    output.masked_fill_(~mask, float("-inf"))
    return output.unsqueeze(1)
