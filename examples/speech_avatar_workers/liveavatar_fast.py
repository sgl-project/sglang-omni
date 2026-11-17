# SPDX-License-Identifier: Apache-2.0
"""A compile-friendly re-implementation of Live Avatar's streaming DiT step (one block against its rolling KV cache).

The authors' ``CausalWanModel_S2V._forward_inference`` breaks the compiled
graph several times per layer: ``Tensor.item()`` on the segment index, on the
sink's length (a CUDA tensor, so a host sync per layer), varlen flash
attention with tensor lengths and a rope table rebuilt on the host for every
forward. On one GPU that leaves the step CPU-bound for a sixth of its time; on
a sequence-parallel group, where each rank's GPU work shrinks, the host
becomes the bottleneck.

``StreamStep`` computes the same thing with the model's own weights:

* the per-block rope tables (by frame offset) and the sink's (by its random
  roll-out) are built once and cached; the text context's cross-attention
  keys and values are projected once per layer;
* the sink's length lives on the host;
* every attention is ``scaled_dot_product_attention`` on full sequences;
* with a ``SequenceParallel`` group, each rank keeps its share of the block's
  tokens and the all-to-alls and all-gathers are functional collectives, so
  the 40 layers trace into one graph.

The sink prefill keeps the authors' path (``model(..., sink_flag=True)``).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def sdpa(q, k, v):
    """[b, s, h, d] attention by torch's scaled_dot_product_attention (its default backend: FA2 on H200)."""
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def cudnn(q, k, v):
    """[b, s, h, d] attention by cuDNN's fused kernel, the one the authors' attention() picks."""
    out = torch.ops.aten._scaled_dot_product_cudnn_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None, False, 0.0, False, False
    )[0]
    return out.transpose(1, 2)


class StreamStep:
    def __init__(self, model, sp, lfb: int, frame_seq_length: int, h_patches: int, w_patches: int, lat_motion_frames: int, motion_frames: int, compile: bool = True) -> None:
        self.model = model
        self.sp = sp
        self.lfb = lfb
        self.fsl = frame_seq_length
        self.h_patches = h_patches
        self.w_patches = w_patches
        self.lat_motion_frames = lat_motion_frames
        self.motion_frames = motion_frames
        self.length = lfb * frame_seq_length
        self.heads = model.num_heads
        self.head_dim = model.dim // model.num_heads
        self.freqs_cache: Dict[int, torch.Tensor] = {}
        self.cond_freqs_cache: Dict[int, torch.Tensor] = {}
        self.text_kv: Optional[List[tuple]] = None
        self.inject = dict(model.audio_injector.injected_block_id)
        size = sp.size if sp is not None else 1
        rank = sp.rank if sp is not None else 0
        if self.length % size:
            raise ValueError(f"a block's {self.length} tokens do not split over {size} ranks")
        else:
            pass
        self.size, self.rank = size, rank
        self.local = self.length // size
        self.local_start = rank * self.local
        # Self-attention over [b, s, heads, d] tensors; liveavatar_sgld_poc.py swaps in SGLang Diffusion's.
        self.attend = cudnn
        self.core = torch.compile(self._core, dynamic=None) if compile else self._core

    # -------- host-side tables --------

    def _grid(self, frames: int):
        g = torch.tensor([[frames, self.h_patches, self.w_patches]], dtype=torch.long)
        return [[torch.zeros_like(g), g, g]]

    def block_freqs(self, frame_offset: int):
        from liveavatar.models.wan.causal_s2v_utils import rollout_grid_sizes
        from liveavatar.models.wan.wan_2_2.modules.s2v.s2v_utils import rope_precompute

        if frame_offset not in self.freqs_cache:
            model = self.model
            dummy = torch.empty(1, self.length, self.heads, self.head_dim, dtype=torch.bfloat16, device=model.patch_embedding.weight.device)
            freqs = rope_precompute(dummy, rollout_grid_sizes(self._grid(self.lfb), frame_offset), model.freqs, start=None)
            freqs = freqs[:, self.local_start : self.local_start + self.local]
            if len(self.freqs_cache) >= 4:
                self.freqs_cache.pop(next(iter(self.freqs_cache)))
            self.freqs_cache[frame_offset] = (freqs.real.float().contiguous(), freqs.imag.float().contiguous())
        else:
            pass
        return self.freqs_cache[frame_offset]

    def cond_freqs(self, rollout: int):
        from liveavatar.models.wan.causal_s2v_utils import rollout_grid_sizes
        from liveavatar.models.wan.wan_2_2.modules.s2v.s2v_utils import rope_precompute

        if rollout not in self.cond_freqs_cache:
            model = self.model
            dummy = torch.empty(model.rope_cache["cond_shape"], dtype=torch.bfloat16, device=model.patch_embedding.weight.device)
            freqs = rope_precompute(dummy, rollout_grid_sizes(model.rope_cache["grid_sizes"], rollout), model.freqs, start=None)
            heads = self.heads // self.size
            freqs = freqs[:, :, self.rank * heads : (self.rank + 1) * heads]
            if len(self.cond_freqs_cache) >= 64:
                self.cond_freqs_cache.pop(next(iter(self.cond_freqs_cache)))
            self.cond_freqs_cache[rollout] = (freqs.real.float().contiguous(), freqs.imag.float().contiguous())
        else:
            pass
        return self.cond_freqs_cache[rollout]

    def text_keys(self, context):
        """Each layer's cross-attention keys and values of the (fixed) prompt."""
        if self.text_kv is None:
            model = self.model
            with torch.no_grad():
                ctx = model.text_embedding(
                    torch.stack([torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))]) for u in context])
                )
                kv = []
                for block in model.blocks:
                    attn = block.cross_attn
                    k = attn.norm_k(attn.k(ctx)).view(1, -1, self.heads, self.head_dim).transpose(1, 2).contiguous()
                    v = attn.v(ctx).view(1, -1, self.heads, self.head_dim).transpose(1, 2).contiguous()
                    kv.append((k, v))
            self.text_kv = kv
        else:
            pass
        return self.text_kv

    # -------- the step --------

    def __call__(self, latents, t, audio_input, cond_states, context, kv_cache: list, current_start: int):
        """The model's velocity [16, F, h, w] for `latents` [16, F, h, w] at timestep `t` [1, F]."""
        model = self.model
        frame_offset = current_start // self.fsl
        # The authors draw the sink's rope offset at every forward; keep the draw (and so the caller's seeding).
        relative_dist = random.randint(4, 30)
        rollout = max(0, frame_offset - (30 - relative_dist))
        cache_len = kv_cache[0]["k"].shape[1]
        write = current_start % cache_len
        active_end = cache_len if current_start >= cache_len else current_start + self.length
        cond_end = int(kv_cache[0]["cond_end"]) if "cond_end_host" not in kv_cache[0] else kv_cache[0]["cond_end_host"]
        text_kv = self.text_keys(context)
        audio = torch.cat([audio_input[..., 0:1].repeat(1, 1, 1, self.motion_frames), audio_input], dim=-1)
        audio_res = model.casual_audio_encoder(audio)
        audio_res = tuple(a.to(model.dtype) for a in audio_res)
        if model.enbale_adain:
            audio_global, audio_emb = audio_res
            audio_global = audio_global[:, self.lat_motion_frames :]
        else:
            audio_emb, audio_global = audio_res, None
        audio_emb = audio_emb[:, self.lat_motion_frames :]
        return self.core(
            latents,
            t,
            audio_emb,
            audio_global,
            cond_states,
            self.block_freqs(frame_offset),
            self.cond_freqs(rollout),
            text_kv,
            [(layer["k"], layer["v"], layer["cond_k"], layer["cond_v"]) for layer in kv_cache],
            write,
            active_end,
            cond_end,
        )

    # -------- traced --------

    def _rope(self, x, freqs):
        """x [b, s, n, d] by the rope table (cos, sin) [b, s, n, d/2]; real arithmetic so inductor fuses it (the tables are fp16-exact)."""
        b, s, n, d = x.shape
        cos, sin = freqs
        xr = x.float().reshape(b, s, n, -1, 2)
        a, c = xr[..., 0], xr[..., 1]
        return torch.stack([a * cos - c * sin, a * sin + c * cos], dim=-1).flatten(3)

    def _tokens_to_heads(self, x):
        import torch.distributed._functional_collectives as funcol

        b, s, n, d = x.shape
        send = x.reshape(b, s, self.size, n // self.size, d).permute(2, 0, 1, 3, 4).contiguous()
        recv = funcol.all_to_all_single(send, None, None, group=self.sp.group)
        recv = funcol.wait_tensor(recv)
        return recv.permute(1, 0, 2, 3, 4).reshape(b, self.size * s, n // self.size, d)

    def _heads_to_tokens(self, x):
        import torch.distributed._functional_collectives as funcol

        b, s_all, h, d = x.shape
        s = s_all // self.size
        send = x.reshape(b, self.size, s, h, d).permute(1, 0, 2, 3, 4).contiguous()
        recv = funcol.all_to_all_single(send, None, None, group=self.sp.group)
        recv = funcol.wait_tensor(recv)
        return recv.permute(1, 2, 0, 3, 4).reshape(b, s, self.size * h, d)

    def _gather(self, x):
        import torch.distributed._functional_collectives as funcol

        return funcol.wait_tensor(funcol.all_gather_tensor(x.contiguous(), 1, group=self.sp.group))

    def _self_attn(self, attn, x, freqs, cache, cond_freqs, write: int, active_end: int, cond_end: int):
        b, s = x.shape[:2]
        n, d = self.heads, self.head_dim
        q = attn.norm_q(attn.q(x)).view(b, s, n, d)
        k = attn.norm_k(attn.k(x)).view(b, s, n, d)
        v = attn.v(x).view(b, s, n, d)
        q = self._rope(q, freqs).type_as(v)
        k = self._rope(k, freqs).type_as(v)
        if self.size > 1:
            q, k, v = self._tokens_to_heads(q), self._tokens_to_heads(k), self._tokens_to_heads(v)
        else:
            pass
        heads = n // self.size
        h0 = self.rank * heads
        cache_k, cache_v, cond_k, cond_v = cache
        cache_k[:, write : write + self.length, h0 : h0 + heads] = k
        cache_v[:, write : write + self.length, h0 : h0 + heads] = v
        sink_k = self._rope(cond_k[:, :cond_end, h0 : h0 + heads], (cond_freqs[0][:, :cond_end], cond_freqs[1][:, :cond_end])).type_as(v)
        keys = torch.cat([cache_k[:, :active_end, h0 : h0 + heads], sink_k], dim=1)
        values = torch.cat([cache_v[:, :active_end, h0 : h0 + heads], cond_v[:, :cond_end, h0 : h0 + heads]], dim=1)
        out = self.attend(q, keys, values)
        if self.size > 1:
            out = self._heads_to_tokens(out.contiguous())
        else:
            pass
        return attn.o(out.flatten(2))

    def _cross_attn(self, attn, x, k, v):
        b = x.shape[0]
        q = attn.norm_q(attn.q(x)).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        return attn.o(out.flatten(2))

    def _inject_audio(self, idx: int, hidden, audio_emb, audio_global):
        """The authors' after_transformer_block on the whole block (gathered when sequence-parallel)."""
        model = self.model
        injector = model.audio_injector
        aid = self.inject[idx]
        full = self._gather(hidden) if self.size > 1 else hidden
        frames = self.lfb
        x = full.reshape(frames, self.fsl, full.shape[-1])
        if model.enbale_adain and model.adain_mode == "attn_norm":
            temb = audio_global.to(model.dtype).reshape(frames, -1, audio_global.shape[-1])[:, 0]
            x = injector.injector_adain_layers[aid](x, temb=temb)
        else:
            x = injector.injector_pre_norm_feat[aid](x)
        context = audio_emb.reshape(frames, -1, audio_emb.shape[-1])
        attn = injector.injector[aid]
        q = attn.norm_q(attn.q(x)).view(frames, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.norm_k(attn.k(context)).view(frames, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
        v = attn.v(context).view(frames, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
        out = attn.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))
        residual = out.reshape(1, frames * self.fsl, -1)
        if self.size > 1:
            residual = residual[:, self.local_start : self.local_start + self.local]
        else:
            pass
        return hidden + residual

    def _core(self, latents, t, audio_emb, audio_global, cond_states, freqs, cond_freqs, text_kv, caches, write: int, active_end: int, cond_end: int):
        model = self.model
        bf16 = torch.bfloat16
        x = model.patch_embedding(latents.unsqueeze(0)) + model.cond_encoder(cond_states)
        x = x.flatten(2).transpose(1, 2)
        x = x + model.trainable_cond_mask(torch.zeros([1, x.shape[1]], dtype=torch.long, device=x.device)).to(x.dtype)
        if self.size > 1:
            x = x[:, self.local_start : self.local_start + self.local]
        else:
            pass
        x = x.to(bf16)
        from liveavatar.models.wan.wan_2_2.modules.model import sinusoidal_embedding_1d

        with torch.autocast("cuda", dtype=torch.float32):
            e = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t.flatten()).float())
            e0 = model.time_projection(e).unflatten(1, (6, model.dim))  # [F, 6, dim]
        # Each local token's frame.
        frame_of = (torch.arange(self.local, device=x.device) + self.local_start) // self.fsl
        for idx, block in enumerate(model.blocks):
            mod = (block.modulation.float() + e0)[frame_of]  # [L, 6, dim]
            m = [mod[:, i].unsqueeze(0) for i in range(6)]
            norm_x = block.norm1(x).float() * (1 + m[1]) + m[0]
            y = self._self_attn(block.self_attn, norm_x.to(bf16), freqs, caches[idx], cond_freqs, write, active_end, cond_end)
            x = x.float() + y.float() * m[2]
            k, v = text_kv[idx]
            x = x + self._cross_attn(block.cross_attn, block.norm3(x.to(bf16)).to(bf16), k, v).float()
            y = block.ffn((block.norm2(x).float() * (1 + m[4]) + m[3]).to(bf16))
            x = (x + y.float() * m[5]).to(bf16)
            if idx in self.inject:
                x = self._inject_audio(idx, x, audio_emb, audio_global)
            else:
                pass
        if self.size > 1:
            x = self._gather(x)
        else:
            pass
        x = model.head(x, e)
        grid = torch.tensor([[self.lfb, self.h_patches, self.w_patches]])
        return model.unpatchify(x, grid)[0]
