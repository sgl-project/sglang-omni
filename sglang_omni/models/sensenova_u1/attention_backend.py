# SPDX-License-Identifier: Apache-2.0
"""SenseNova U1 adapters for the upstream SGLang Triton attention backend."""

from __future__ import annotations

import inspect
import types
from typing import Any

import torch


CUSTOM_MASK_DENSE_ATTENTION_CALLS = 0


def _custom_mask_dense_attention_fwd(
    q_extend: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    o_extend: torch.Tensor,
    qo_indptr: torch.Tensor,
    custom_mask: torch.Tensor,
    mask_indptr: torch.Tensor,
    k_scale: float,
    v_scale: float,
    sm_scale: float | None,
) -> None:
    """Exact short custom-mask path for requests without cached prefix KV."""

    global CUSTOM_MASK_DENSE_ATTENTION_CALLS
    CUSTOM_MASK_DENSE_ATTENTION_CALLS += 1

    scale = sm_scale or 1.0 / (q_extend.shape[-1] ** 0.5)
    batch_size = int(qo_indptr.shape[0]) - 1
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]
    for seq_idx in range(batch_size):
        q_start = int(qo_indptr[seq_idx].item())
        q_end = int(qo_indptr[seq_idx + 1].item())
        q_len = q_end - q_start
        if q_len == 0:
            continue
        mask_start = int(mask_indptr[seq_idx].item())
        mask_end = int(mask_indptr[seq_idx + 1].item())
        expected_mask_len = q_len * q_len
        if mask_end - mask_start != expected_mask_len:
            raise RuntimeError(
                "U1 dense custom-mask attention only supports no-prefix masks: "
                f"mask_len={mask_end - mask_start}, q_len={q_len}"
            )

        q = q_extend[q_start:q_end]
        k = k_extend[q_start:q_end]
        v = v_extend[q_start:q_end]
        if kv_group_num != 1:
            k = k.repeat_interleave(kv_group_num, dim=1)
            v = v.repeat_interleave(kv_group_num, dim=1)

        allowed = custom_mask[mask_start:mask_end].view(q_len, q_len).bool()
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float())
        scores.mul_(float(scale) * float(k_scale))
        scores.masked_fill_(
            ~allowed.to(device=scores.device, dtype=torch.bool).unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", probs, v)
        if float(v_scale) != 1.0:
            out = out * float(v_scale)
        o_extend[q_start:q_end].copy_(out.to(dtype=o_extend.dtype))


def _build_extend_wrapper(original):
    signature = inspect.signature(original)

    def u1_extend_attention_fwd(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        custom_mask = values["custom_mask"]
        if (
            custom_mask is not None
            and int(values["max_len_extend"]) <= 512
            and values["kv_indices"].numel() == 0
            and not values["skip_prefix"]
            and not values["skip_extend"]
            and values["lse_extend"] is None
            and values["sinks"] is None
            and values["score_mod"] is None
            and values["aux_tensors"] is None
            and int(values["sliding_window_size"]) == -1
            and float(values["logit_cap"]) == 0.0
            and int(values["xai_temperature_len"]) <= 0
            and int(values["page_size"]) == 1
        ):
            _custom_mask_dense_attention_fwd(
                values["q_extend"],
                values["k_extend"],
                values["v_extend"],
                values["o_extend"],
                values["qo_indptr"],
                custom_mask,
                values["mask_indptr"],
                values["k_scale"],
                values["v_scale"],
                values["sm_scale"],
            )
            return None

        if custom_mask is not None:
            # The custom mask already contains the complete U1 causal/image row
            # policy. Disabling the kernel's separate causal truncation keeps
            # current-image KV visible across Triton M blocks.
            values["is_causal"] = False
        return original(*bound.args, **bound.kwargs)

    return u1_extend_attention_fwd


def _inject_custom_mask_metadata(backend: Any, forward_batch: Any) -> None:
    custom_mask = getattr(forward_batch, "cross_attention_custom_mask", None)
    if custom_mask is None:
        return
    metadata = getattr(backend, "forward_metadata", None)
    if metadata is None:
        raise RuntimeError("U1 Triton metadata was not initialized")
    batch_size = int(forward_batch.batch_size)
    seq_mask_len = forward_batch.extend_seq_lens * (
        forward_batch.extend_prefix_lens + forward_batch.extend_seq_lens
    )
    mask_indptr = backend.mask_indptr
    mask_indptr[1 : batch_size + 1] = torch.cumsum(
        seq_mask_len[:batch_size],
        dim=0,
    )
    metadata.custom_mask = custom_mask
    metadata.mask_indptr = mask_indptr[: batch_size + 1]


def install_sensenova_u1_triton_attention_adapter(model_runner: Any) -> None:
    """Install U1 custom-mask behavior without modifying the SGLang checkout."""

    from sglang.kernels.ops.attention import extend_attention as extend_module
    from sglang.srt.layers.attention import triton_backend

    original_attr = "_sensenova_u1_original_extend_attention_fwd"
    if not hasattr(extend_module, original_attr):
        original = extend_module.extend_attention_fwd
        setattr(extend_module, original_attr, original)
        wrapper = _build_extend_wrapper(original)
        extend_module.extend_attention_fwd = wrapper
        triton_backend.extend_attention_fwd = wrapper

    backend = model_runner.attn_backend
    if type(backend).__name__ != "TritonAttnBackend":
        return
    if getattr(backend, "_sensenova_u1_metadata_adapter_installed", False):
        return

    original_init = backend.init_forward_metadata

    def init_forward_metadata_with_u1(self, forward_batch):
        result = original_init(forward_batch)
        _inject_custom_mask_metadata(self, forward_batch)
        return result

    backend.init_forward_metadata = types.MethodType(
        init_forward_metadata_with_u1,
        backend,
    )
    backend._sensenova_u1_metadata_adapter_installed = True


__all__ = [
    "CUSTOM_MASK_DENSE_ATTENTION_CALLS",
    "install_sensenova_u1_triton_attention_adapter",
]
