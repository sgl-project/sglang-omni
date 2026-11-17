# SPDX-License-Identifier: Apache-2.0
"""Sequence parallel for SoulX-LiveAct over any number of GPUs that divides its 40 heads, with the KV cache split by heads.

The authors' model_memory_sp.py splits the KV cache by sequence and only
works on two GPUs: each rank keeps half of the six memory frames and half of
the current block, the memory update is written per rank, and every step
sends the whole cache through Ulysses' all-to-all. This module keeps the
same model and the same arithmetic but stores the cache after the
all-to-all, as the full sequence of the rank's heads:

- the memory update becomes one rule over whole frames (below), the same on
  every rank;
- only the block's own q, k and v go through the all-to-all, not the cache;
- the sequence split of the block (6 or 8 latent frames) no longer has to
  fall on frame boundaries, so 4 GPUs work as well as 2.

The cache holds 14 latent frames of keys and values before RoPE: six memory
frames m0..m5 at RoPE positions 0..5 and the current block's eight frames
c0..c7 at positions 6..13 (block 0, the reference block, fills m0..m5). From
block 2 on, before a block writes its own frames, the memory takes in the
previous block's frames as the two-rank code does:

    m2 <- conv(m2, c0, c1, c2, c3)    m3 <- conv(m3, m4, m5, c4, c5)
    m4 <- c6                           m5 <- c7

where conv is the model's memory_proj_k (a depthwise Conv1d of kernel and
stride 5 over the tokens of the five frames laid end to end; the authors use
it for the values too).

A block may also run as two halves of four latents (0.8 s chunks): the
first half (positions 6..9) takes in the memory update and attends to
m0..m5 and itself, the second (10..13) to m0..m5, the first half's frames
of the same step and itself.

`patch(model_module, world_size)` swaps the forwards of the module's
WanSelfAttention and SingleStreamAttention classes; `kv_cache_shape` gives
the per-step, per-layer cache tensor shape.
"""

from __future__ import annotations


import torch
import torch.distributed as dist
import torch.nn.functional as F

MEMORY_FRAMES = 6
WINDOW_LATENTS = 14


def kv_cache_shape(frame_len: int, world_size: int, heads: int = 40, head_dim: int = 128):
    return [1, WINDOW_LATENTS * frame_len, heads // world_size, head_dim]


def _seq_to_heads(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """[B, S/P, H, D] (this rank's tokens, every head) -> [B, S, H/P, D] (every token, this rank's heads)."""
    if world_size == 1:
        return x
    b, s, h, d = x.shape
    x = x.reshape(b, s, world_size, h // world_size, d).permute(2, 0, 1, 3, 4).contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)
    # out[r] holds rank r's tokens of this rank's heads
    return out.permute(1, 0, 2, 3, 4).reshape(b, world_size * s, h // world_size, d)


def _heads_to_seq(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """[B, S, H/P, D] -> [B, S/P, H, D]."""
    if world_size == 1:
        return x
    b, s, h, d = x.shape
    x = x.reshape(b, world_size, s // world_size, h, d).permute(1, 0, 2, 3, 4).contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)
    # out[r] holds this rank's tokens of rank r's heads
    return out.permute(1, 2, 0, 3, 4).reshape(b, s // world_size, world_size * h, d)


_ROPE = {}


def _rope_table(freqs: torch.Tensor, positions: tuple, h: int, w: int, device) -> torch.Tensor:
    """complex64 [len(positions) * h * w, 1, 64]: the model's 3-axis RoPE for frames at the given temporal positions."""
    key = (positions, h, w, str(device))
    table = _ROPE.get(key)
    if table is None:
        c = freqs.shape[1]
        f0, f1, f2 = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        f = len(positions)
        idx = torch.tensor(positions, device=f0.device)
        table = (
            torch.cat(
                [
                    f0[idx].view(f, 1, 1, -1).expand(f, h, w, -1),
                    f1[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    f2[:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(device=device, dtype=torch.complex64)
        )
        _ROPE[key] = table
    return table


def _rope(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """x [1, S, H, 128] bf16 -> RoPE'd, same dtype (computed in fp32; the authors use fp64)."""
    s, n = x.shape[1], x.shape[2]
    xc = torch.view_as_complex(x[0].float().reshape(s, n, -1, 2))
    return torch.view_as_real(xc * table).flatten(2).unsqueeze(0).type_as(x)


ATTENTION = "sage"


@torch.compiler.disable
def _attention(q, k, v):
    """Self-attention, [B, S, H, D]: SageAttention 2's INT8 QK / FP8 PV kernel (the authors' choice), or FlashAttention 3 in BF16.

    FlashAttention 3 is as fast here (2.34 vs 2.37 ms for 9360 x 16380 tokens, 20 heads on an H200), exact (cos 0.999999 to SDPA, Sage
    0.99926), and safe to capture in a CUDA graph; Sage's kernels are not (a replayed graph returns garbage). Kept out of torch.compile:
    FlashAttention 3 traced by Dynamo crashed the process.
    """
    if ATTENTION == "fa3":
        # FlashAttention 3's own kernel op: its Python wrapper goes through a custom op named flash_attn_3::_flash_attn_forward, which
        # diffusers (imported first by the model) registers under the same name with another signature.
        import flash_attn_3._C  # noqa: F401

        return torch.ops.flash_attn_3.fwd(
            q, k, v, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, q.shape[-1] ** -0.5, False, -1, -1, 0, 0.0, True, None, 1, None, 0,
        )[0]
    else:
        from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90

        return sageattn_qk_int8_pv_fp8_cuda_sm90(
            q, k, v, tensor_layout="NHD", pv_accum_dtype="fp32+fp32"
        )


def _compress(conv_weight: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """memory_proj_k over five frames of one rank's heads: frames [1, 5 fs, h, d] -> [1, fs, h, d].

    The conv is depthwise over the full model's 5120 channels; this rank's
    heads are channels [rank * h * d, (rank + 1) * h * d).
    """
    b, n, h, d = frames.shape
    x = frames.reshape(b, n, h * d).transpose(1, 2)
    y = F.conv1d(x, conv_weight, stride=5, groups=h * d)
    return y.reshape(b, h, d, n // 5).permute(0, 3, 1, 2)


def update_memory(cache: torch.Tensor, fs: int, weight: torch.Tensor) -> None:
    """The memory update before a block from 2 on, in place on one rank's cache [1, 14 fs, h, d] (see the module docstring)."""
    m = MEMORY_FRAMES * fs
    new2 = _compress(weight, torch.cat([cache[:, 2 * fs:3 * fs], cache[:, m:m + 4 * fs]], dim=1))
    new3 = _compress(weight, torch.cat([cache[:, 3 * fs:6 * fs], cache[:, m + 4 * fs:m + 6 * fs]], dim=1))
    cache[:, 2 * fs:3 * fs].copy_(new2)
    cache[:, 3 * fs:4 * fs].copy_(new3)
    cache[:, 4 * fs:6 * fs].copy_(cache[:, m + 6 * fs:m + 8 * fs])


def self_attn_forward(self, x, seq_lens, grid_sizes, freqs, sp_size, sp_rank, kv_cache={}, start_idx=None, end_idx=None,
                      update_cache=False):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)
    q = _seq_to_heads(q, sp_size)
    k = _seq_to_heads(k, sp_size)
    v = _seq_to_heads(v, sp_size)
    _, gh, gw = (int(t) for t in grid_sizes[0])
    fs = gh * gw
    hp = n // sp_size
    kc, vc = kv_cache["k"], kv_cache["v"]
    if update_cache:
        weight = self.memory_proj_k.weight[sp_rank * hp * d:(sp_rank + 1) * hp * d]
        update_memory(kc, fs, weight)
        update_memory(vc, fs, weight)
    frames = q.shape[1] // fs
    if start_idx == 0:
        kc[:, :frames * fs] = k
        vc[:, :frames * fs] = v
        keys, values = kc[:, :frames * fs], vc[:, :frames * fs]
        qpos = tuple(range(frames))
        kpos = qpos
    else:
        # a block's latents from position `first` (6, or 10 for a block's second half) attend to the memory, the block's latents before them and themselves
        first = start_idx // fs
        end = first + frames
        kc[:, first * fs:end * fs] = k
        vc[:, first * fs:end * fs] = v
        keys, values = kc[:, :end * fs], vc[:, :end * fs]
        qpos = tuple(range(first, end))
        kpos = tuple(range(end))
    out = _attention(
        _rope(q, _rope_table(freqs, qpos, gh, gw, q.device)),
        _rope(keys, _rope_table(freqs, kpos, gh, gw, q.device)),
        values,
    )
    out = _heads_to_seq(out, sp_size)
    return self.o(out.flatten(2)), None


def audio_attn_forward(self, x, encoder_hidden_states, sp_size, sp_rank, shape=None, start_f=0) -> torch.Tensor:
    """SingleStreamAttention: every token attends to the 32 audio tokens of its own latent frame; the rank's tokens may start or end mid-frame."""
    from model_liveact import model_memory_sp as base

    encoder_hidden_states = encoder_hidden_states.squeeze(0)
    _, gh, gw = (int(t) for t in shape)
    fs = gh * gw
    s = x.size(1)
    first = sp_rank * s
    encoder_kv = self.kv_linear(encoder_hidden_states)
    b_e, n_a, _ = encoder_kv.shape
    encoder_kv = encoder_kv.view(b_e, n_a, 2, self.num_heads, self.head_dim)
    q_all = self.q_linear(x)
    outs = []
    pos, last = first, first + s
    while pos < last:
        frame = pos // fs
        if pos % fs == 0 and last - pos >= fs:
            nf = (last - pos) // fs
            end = pos + nf * fs
        else:
            nf = 1
            end = min((frame + 1) * fs, last)
        seg = q_all[:, pos - first:end - first]
        q = seg.reshape(nf, (end - pos) // nf, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = encoder_kv[start_f + frame:start_f + frame + nf].permute(2, 0, 3, 1, 4)
        ek, ev = kv.unbind(0)
        if base.USE_SAGEATTN:
            o = base.sageattn(q.contiguous(), ek.contiguous(), ev.contiguous(), tensor_layout="HND")
        else:
            o = F.scaled_dot_product_attention(q, ek, ev)
        outs.append(o.transpose(1, 2).reshape(1, end - pos, -1))
        pos = end
    out = self.proj(torch.cat(outs, dim=1))
    if start_f == 0 and first < fs:
        # the reference frame takes no speech (the two-rank code zeroes rank 0's first frame)
        out[:, : fs - first] = 0
    return out


def patch(module, world_size: int, attention: str = "sage") -> None:
    """Swap model_memory_sp's attention forwards for the head-split cache ones; the block's zeroing of frame 0 moves into audio_attn_forward.

    attention "fa3" also takes the text, image and speech cross-attentions off Sage (to PyTorch's SDPA), for CUDA graphs.
    """
    global ATTENTION
    ATTENTION = attention
    module.WanSelfAttention.forward = self_attn_forward
    module.SingleStreamAttention.forward = audio_attn_forward
    if attention == "fa3":
        module.USE_SAGEATTN = False
    else:
        pass
