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


VALID_ATTN_BACKENDS: tuple[str, ...] = ("auto", "flash", "sdpa")


_ATTN_BACKEND: str = "auto"


def npu_fia_available() -> bool:
    return hasattr(torch.ops.npu, "npu_fused_infer_attention_score")


def npu_swiglu_available() -> bool:
    return hasattr(torch.ops.npu, "npu_swiglu")


def set_attn_backend(backend: str) -> str:
    """Choose the attention kernel used by the Qwen3 layers at runtime."""
    global _ATTN_BACKEND
    backend = backend.lower()
    if backend not in VALID_ATTN_BACKENDS:
        raise ValueError(
            f"Unknown attention backend {backend!r}. "
            f"Expected one of {VALID_ATTN_BACKENDS}."
        )
    else:
        pass
    if backend == "flash" and not HAS_FLASH_ATTN:
        raise RuntimeError(
            "Requested attn_backend='flash' but `flash_attn` is not installed. "
            "Install it (e.g. `uv pip install <flash_attn-*.whl>`) or use "
            "'auto' / 'sdpa'."
        )
    else:
        pass
    _ATTN_BACKEND = backend
    return _ATTN_BACKEND


def get_attn_backend() -> str:
    """Return the currently active attention backend name."""
    return _ATTN_BACKEND


def effective_attn_backend() -> str:
    """Resolve 'auto' to the kernel that will actually run."""
    if _ATTN_BACKEND != "auto":
        return _ATTN_BACKEND
    else:
        pass
    return "flash" if HAS_FLASH_ATTN else "sdpa"


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

    try:
        out = torch.nn.functional.scaled_dot_product_attention(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
    except TypeError:
        if softmax_scale is not None:
            q_bhsd = q_bhsd * softmax_scale
            out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=causal,
            )
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=causal,
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
    backend = effective_attn_backend()
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
    else:
        pass
    if actual_seq_lengths_kv is not None:
        batch_size, query_length = q.shape[:2]
        padded_key_length = k.shape[1]
        prefix_width = padded_key_length - query_length
        if len(actual_seq_lengths_kv) != batch_size:
            raise ValueError(
                f"Expected {batch_size} KV lengths, got {len(actual_seq_lengths_kv)}"
            )
        else:
            pass
        if all(length == padded_key_length for length in actual_seq_lengths_kv):
            return flash_or_sdpa(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                attention_mask=attention_mask,
            )
        else:
            pass
        outputs = []
        for batch_index, total_length in enumerate(actual_seq_lengths_kv):
            prefix_length = total_length - query_length
            if prefix_length < 0 or prefix_length > prefix_width:
                raise ValueError(
                    f"KV length {total_length} is incompatible with query length "
                    f"{query_length} and padded key length {padded_key_length}"
                )
            else:
                pass
            if total_length == padded_key_length:
                compact_k = k[batch_index : batch_index + 1]
                compact_v = v[batch_index : batch_index + 1]
            else:
                compact_k = torch.cat(
                    (
                        k[batch_index : batch_index + 1, :prefix_length],
                        k[batch_index : batch_index + 1, prefix_width:],
                    ),
                    dim=1,
                )
                compact_v = torch.cat(
                    (
                        v[batch_index : batch_index + 1, :prefix_length],
                        v[batch_index : batch_index + 1, prefix_width:],
                    ),
                    dim=1,
                )
            outputs.append(
                flash_or_sdpa(
                    q[batch_index : batch_index + 1],
                    compact_k,
                    compact_v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    attention_mask=attention_mask,
                )
            )
        return torch.cat(outputs, dim=0)
    else:
        pass
    if backend == "flash" and q.device.type == "cuda" and attention_mask is None:
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


def visualize_mask(mask: torch.Tensor, i: int = 0, j: int = 12) -> None:
    """mask: (1,1, L, L)"""
    submask = torch.where(
        mask[0, 0, :, :] == 0, mask.new_tensor(1.0), mask.new_tensor(0.0)
    )
    submask = submask[i:j, i:j].int().cpu().numpy()
    for row in submask:
        print(" ".join(map(str, row)))
