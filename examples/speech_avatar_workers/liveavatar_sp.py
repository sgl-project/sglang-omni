# SPDX-License-Identifier: Apache-2.0
"""Ulysses sequence parallelism for Live Avatar's streaming DiT forward (one block against its rolling KV cache).

The authors' code carries a sequence-parallel path (``sp_attn_forward_s2v``)
that never learned the rolling KV cache, the sink's own rope or a world that
holds more than the DiT ranks. ``install`` replaces it on a loaded
``CausalWanModel_S2V``: each rank of a group of ``size`` ranks keeps
``1/size`` of the block's tokens through the norms, the projections, the
cross-attention and the FFN; for self-attention an all-to-all trades tokens
for heads, so each rank attends with ``num_heads/size`` heads over the whole
block and the head slice of the KV cache it owns. The cache keeps all heads
(each rank fills only its slice) so the sink prefill, which runs unsharded on
every rank, writes it exactly as before.

The sink prefill (``sink_flag``) and anything with ``size == 1`` take the
authors' path unchanged.
"""

from __future__ import annotations

import types

import torch
import torch.distributed as dist


class SequenceParallel:
    def __init__(self, group, size: int, rank: int) -> None:
        self.group = group
        self.size = size
        self.rank = rank
        self.active = True
        self.global_len = None

    # -------- collectives --------

    def chunk(self, tensor, dim: int):
        """This rank's share of `tensor` along `dim`, padded to a multiple of the group size."""
        length = tensor.size(dim)
        self.global_len = length
        pad = (self.size - length % self.size) % self.size
        if pad:
            shape = list(tensor.shape)
            shape[dim] = pad
            tensor = torch.cat([tensor, tensor.new_zeros(shape)], dim=dim)
        else:
            pass
        return tensor.chunk(self.size, dim=dim)[self.rank]

    def gather(self, tensor, dim: int):
        parts = [torch.empty_like(tensor) for _ in range(self.size)]
        dist.all_gather(parts, tensor.contiguous(), group=self.group)
        out = torch.cat(parts, dim=dim)
        if self.global_len is not None:
            out = out.narrow(dim, 0, self.global_len)
        else:
            pass
        return out.contiguous()

    def tokens_to_heads(self, x):
        """[b, s_local, n, d] -> [b, s_local * size, n / size, d]."""
        b, s, n, d = x.shape
        send = x.view(b, s, self.size, n // self.size, d).permute(2, 0, 1, 3, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        return recv.permute(1, 0, 2, 3, 4).reshape(b, self.size * s, n // self.size, d)

    def heads_to_tokens(self, x):
        """[b, s_local * size, n / size, d] -> [b, s_local, n, d]."""
        b, s_all, h, d = x.shape
        s = s_all // self.size
        send = x.view(b, self.size, s, h, d).permute(1, 0, 2, 3, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        return recv.permute(1, 2, 0, 3, 4).reshape(b, s, self.size * h, d)


def install(model, group, size: int, rank: int) -> SequenceParallel:
    """Make `model` (a CausalWanModel_S2V) run its streaming forward sequence-parallel over `group`."""
    from liveavatar.models.wan import causal_model_s2v as cm

    sp = SequenceParallel(group, size, rank)
    original_attn = cm.CausalWanS2VSelfAttention.forward

    def pad_chunk(tensor, sp_size, dim, pad_value=0):
        length = tensor.size(dim)
        return sp.chunk(tensor, dim), length

    def gather_forward(input, dim, sp_size=None, group=None):
        return sp.gather(input, dim)

    def get_rank():
        return sp.rank

    cm.pad_chunk = pad_chunk
    cm.gather_forward = gather_forward
    cm.get_rank = get_rank

    def attn_forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, current_end=0, sp_size=None, seg_idx=None, freqs_cond=None):
        if not sp.active or kv_cache is None or seg_idx[1] - seg_idx[0] <= 0:
            return original_attn(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, current_end, sp_size, seg_idx, freqs_cond)
        else:
            pass
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        q = cm.causal_rope_apply(q, grid_sizes, freqs).type_as(v)
        k = cm.causal_rope_apply(k, grid_sizes, freqs).type_as(v)
        q = sp.tokens_to_heads(q)
        k = sp.tokens_to_heads(k)
        v = sp.tokens_to_heads(v)
        heads = n // sp.size
        h0, h1 = sp.rank * heads, (sp.rank + 1) * heads
        block_len = sp.global_len
        active_start = 0
        if current_start >= kv_cache["k"].shape[1]:
            current_start = current_start % kv_cache["k"].shape[1]
            active_end = kv_cache["k"].shape[1]
        else:
            active_end = current_start + block_len
        cond_end = int(kv_cache["cond_end"])
        kv_cache["k"][:, current_start : current_start + block_len, h0:h1] = k[:, :block_len]
        kv_cache["v"][:, current_start : current_start + block_len, h0:h1] = v[:, :block_len]
        out = cm.attention(
            q=q[:, :block_len],
            k=torch.cat(
                [
                    kv_cache["k"][:, active_start:active_end, h0:h1],
                    cm.causal_rope_apply_cond(kv_cache["cond_k"][:, :cond_end, h0:h1], None, freqs_cond[:, :, h0:h1]).type_as(v),
                ],
                dim=1,
            ),
            v=torch.cat(
                [
                    kv_cache["v"][:, active_start:active_end, h0:h1],
                    kv_cache["cond_v"][:, :cond_end, h0:h1],
                ],
                dim=1,
            ),
            k_lens=torch.tensor(active_end - active_start + cond_end).repeat(b),
            window_size=self.window_size,
        )
        if out.shape[1] < q.shape[1]:
            out = torch.cat([out, out.new_zeros(b, q.shape[1] - out.shape[1], heads, d)], dim=1)
        else:
            pass
        out = sp.heads_to_tokens(out.type_as(v))
        return self.o(out.flatten(2))

    for block in model.blocks:
        block.self_attn.forward = types.MethodType(attn_forward, block.self_attn)
        block.sp_size = size
    model.sp_size = size
    model.use_context_parallel = True

    original_sink = model._forward_sink

    def forward_sink(*args, **kwargs):
        # The sink prefill (a reference frame and motion latents, no noisy tokens) runs whole on every rank.
        sp.active = False
        model.use_context_parallel = False
        try:
            return original_sink(*args, **kwargs)
        finally:
            sp.active = True
            model.use_context_parallel = True

    model._forward_sink = forward_sink
    return sp
