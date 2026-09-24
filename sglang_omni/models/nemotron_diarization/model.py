# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only Nemotron 3 Sortformer.

Adapted from NVIDIA-NeMo/Speech revision
2c1a2f91d64566b5d391b83df42f9ab4cd810adb: transformer_encoder.py,
sortformer_diar_models.py, features.py, subsampling.py and multi_head_attention.py.
Omni changes: fixed checkpoint architecture, single-recording inference, plain
PyTorch modules, and explicit request-local cache ownership.
"""

from collections import OrderedDict
from collections.abc import Generator
from threading import local

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache

# Use the compiler's deterministic reduction policy without changing other models.
_compiled_attention = torch.compile(
    flex_attention,
    dynamic=True,
    options={"deterministic": True, "triton.autotune_pointwise": False},
)


class Featurizer(nn.Module):
    def __init__(self):
        super().__init__()
        # These buffers are restored from the checkpoint, like VoiceChat's mel
        # filters. Diarization uses centered STFT rather than causal left padding.
        self.register_buffer("window", torch.empty(400))
        self.register_buffer("fb", torch.empty(1, 128, 257))

    def forward(self, waveform):
        length = waveform.shape[1] // 160
        emphasized = torch.cat(
            [waveform[:, :1], waveform[:, 1:] - 0.97 * waveform[:, :-1]], dim=1
        )
        spectrum = torch.stft(
            emphasized,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        # Preserve the reference's sqrt/square rounding.
        magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1))
        features = torch.log(torch.matmul(self.fb, magnitude.pow(2)) + 2**-24)
        features[:, :, length:] = 0
        return features, length


class Preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.featurizer = Featurizer()

    def forward(self, waveform):
        return self.featurizer(waveform)


class FeatureStacking(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8 * 128, 512, bias=False)

    def forward(self, features):
        x = features.transpose(1, 2)
        x = F.pad(x, (0, 0, 0, -x.shape[1] % 8))
        return self.proj(x.reshape(1, -1, 8 * 128))


@triton.jit(do_not_specialize=["q_batch", "k_batch", "o_batch"])
def rotary_kernel(
    q,
    k,
    cos,
    sin,
    q_out,
    k_out,
    q_batch,
    q_head: tl.constexpr,
    q_frame: tl.constexpr,
    k_batch,
    k_head: tl.constexpr,
    k_frame: tl.constexpr,
    o_batch,
    o_head: tl.constexpr,
    o_frame: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    frame = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    column = tl.arange(0, block)
    rotated = (column + head_dim // 2) % head_dim
    sign = tl.where(column < head_dim // 2, -1.0, 1.0)
    valid = column < head_dim
    c = tl.load(cos + frame * head_dim + column, valid, other=0)
    s = tl.load(sin + frame * head_dim + column, valid, other=0)
    q_base = batch * q_batch + head * q_head + frame * q_frame
    k_base = batch * k_batch + head * k_head + frame * k_frame
    out_base = batch * o_batch + head * o_head + frame * o_frame
    q_value = tl.load(q + q_base + column, valid, other=0)
    q_other = tl.load(q + q_base + rotated, valid, other=0)
    k_value = tl.load(k + k_base + column, valid, other=0)
    k_other = tl.load(k + k_base + rotated, valid, other=0)
    tl.store(q_out + out_base + column, q_value * c + (sign * q_other) * s, valid)
    tl.store(k_out + out_base + column, k_value * c + (sign * k_other) * s, valid)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply NeOX rotation with separate FP32 products and addition."""
    batch, heads, frames, head_dim = q.shape
    if not q.is_cuda:
        return tuple(
            x * cos
            + torch.cat([-x[..., head_dim // 2 :], x[..., : head_dim // 2]], dim=-1)
            * sin
            for x in (q, k)
        )
    else:
        q_out, k_out = (
            torch.empty(
                (batch, frames, heads, head_dim), device=q.device, dtype=q.dtype
            ).transpose(1, 2)
            for _ in range(2)
        )
        with torch.cuda.device(q.device):
            # note (Richard Wang): Fusion changes the FP32 reference rounding.
            rotary_kernel[(frames, heads, batch)](
                q,
                k,
                cos,
                sin,
                q_out,
                k_out,
                *q.stride()[:3],
                *k.stride()[:3],
                *q_out.stride()[:3],
                head_dim,
                triton.next_power_of_2(head_dim),
                enable_fp_fusion=False,
            )
        return q_out, k_out


class RotaryEmbedding(nn.Module):
    def __init__(self, max_frames: int):
        super().__init__()
        # note (Richard Wang): CPU tables preserve device-independent rounding.
        frequencies = 1.0 / (
            10000 ** (torch.arange(0, 64, 2, device="cpu").float() / 64)
        )
        angles = torch.outer(
            torch.arange(max_frames, device="cpu").float(), frequencies
        )
        angles = torch.cat([angles, angles], dim=-1)
        cos, sin = torch.empty_like(angles), torch.empty_like(angles)
        for angle, cos_row, sin_row in zip(angles, cos, sin):
            torch.cos(angle, out=cos_row)
            torch.sin(angle, out=sin_row)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, q, k):
        cos, sin = self.cos[: q.shape[2]], self.sin[: q.shape[2]]
        return apply_rotary(q, k, cos, sin)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_qkv = nn.Linear(512, 1536, bias=False)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x, rotary, mask):
        qkv = self.w_qkv(x).view(1, x.shape[1], 3, 8, 64).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = rotary(q, k)
        attention = _compiled_attention if x.is_cuda else flex_attention
        if not x.is_cuda:
            kernel_options = None
        elif x.shape[1] < 128:
            # Keep short-window padding independent of the first compiled length.
            kernel_options = {"BLOCK_M": 16}
        else:
            kernel_options = None
        out = attention(q, k, v, block_mask=mask, kernel_options=kernel_options)
        return self.out_proj(out.transpose(1, 2).contiguous().view(1, x.shape[1], 512))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # Identity occupies the checkpoint's training-dropout index.
        self.net = nn.Sequential(
            nn.Linear(512, 2048), nn.GELU(), nn.Identity(), nn.Linear(2048, 512)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(512)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(512)
        self.ffn = FeedForward()

    def forward(self, x, rotary, mask):
        x = x + self.attn(self.norm1(x), rotary, mask)
        return x + self.ffn(self.norm2(x))


class TransformerEncoder(nn.Module):
    def __init__(self, max_frames: int):
        super().__init__()
        self.pre_encode = FeatureStacking()
        self.pos_enc = RotaryEmbedding(max_frames)
        self.embed_norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(31)])
        self.final_norm = nn.LayerNorm(512)
        self._mask_cache = local()

    def attention_mask(self, embeddings, valid_length):
        device = embeddings.device
        stream = (
            torch.cuda.current_stream(device).cuda_stream
            if embeddings.is_cuda
            else None
        )
        key = (embeddings.shape[1], valid_length, device, stream)
        cache = getattr(self._mask_cache, "entries", None)
        if cache is None:
            cache = self._mask_cache.entries = OrderedDict()
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        lengths = torch.tensor([valid_length], device=embeddings.device)

        def padding_mask(b, h, q_idx, kv_idx):
            return kv_idx < lengths[b]

        size = embeddings.shape[1]
        # note (Richard Wang): Derive prefix blocks without a dense token mask.
        block_size = 128
        blocks = torch.arange((size + block_size - 1) // block_size, device="cpu")
        full = ((blocks[:, None] + 1) * block_size <= size) & (
            (blocks[None, :] + 1) * block_size <= valid_length
        )
        partial = (blocks[None, :] * block_size < valid_length) & ~full
        mask = BlockMask.from_kv_blocks(
            partial.sum(-1).to(torch.int32)[None, None],
            partial.to(torch.int32)
            .argsort(descending=True, stable=True)
            .to(torch.int32)[None, None],
            full.sum(-1).to(torch.int32)[None, None],
            full.to(torch.int32)
            .argsort(descending=True, stable=True)
            .to(torch.int32)[None, None],
            BLOCK_SIZE=block_size,
            mask_mod=padding_mask,
            seq_lengths=(size, size),
            compute_q_blocks=False,
        ).to(device)
        # Keep immutable mask metadata on its creation stream.
        cache[key] = mask
        if len(cache) > 128:
            cache.popitem(last=False)
        return mask

    def forward(self, embeddings, valid_length, tail=None):
        mask = self.attention_mask(embeddings, valid_length)
        x = self.embed_norm(embeddings)
        for layer in self.layers:
            x = layer(x, self.pos_enc, mask)
        encoded = self.final_norm(x)
        return encoded if tail is None else tail(encoded)


class SortformerModules(nn.Module):
    def __init__(self):
        super().__init__()
        # Retained for strict loading: this legacy head is stored in the archive
        # but is unused by the high-resolution forward path.
        self.hidden_to_spks = nn.Linear(384, 8)
        self.first_hidden_to_hidden = nn.Linear(192, 192)
        self.single_hidden_to_spks = nn.Linear(192, 8)
        self.encoder_proj = nn.Linear(512, 192)
        self.subpixel_upsample = nn.Conv1d(192, 192 * 8, 3, padding=1)
        self.learnable_sil_emb = nn.Parameter(torch.empty(512))

    def forward(self, embeddings, valid_length):
        hidden = self.encoder_proj(embeddings)
        hidden = self.subpixel_upsample(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden.reshape(1, -1, 192)
        hidden = F.relu(self.first_hidden_to_hidden(F.relu(hidden)))
        preds = torch.sigmoid(self.single_hidden_to_spks(hidden))
        mask = (
            torch.arange(embeddings.shape[1], device=embeddings.device) < valid_length
        )
        return preds * mask.repeat_interleave(8)[None, :, None]


class NemotronDiarizationModel(nn.Module):
    def __init__(self, *, profile: tuple[int, int, int, int, int]):
        super().__init__()
        self.preprocessor = Preprocessor()
        self.encoder = TransformerEncoder(sum(profile[:4]))
        self.sortformer_modules = SortformerModules()
        self.profile = profile

    @torch.inference_mode()
    def forward(self, waveform):
        """Return [1, time, 8] probabilities at 10 ms, with fresh state per call."""
        return torch.cat([output for output, _ in self.iter_chunks(waveform)], dim=1)

    def iter_chunks(
        self, waveform: torch.Tensor
    ) -> Generator[tuple[torch.Tensor, bool], None, None]:
        """Yield each upload window's probabilities and whether it is the last."""
        features, length = self.preprocessor(waveform)
        del waveform
        features = features[:, :, :length]
        if length == 0:
            yield features.new_empty(1, 0, 8), True
            return
        cache_size, fifo_size, chunk_size, right_context, update_period = self.profile
        state = SpeakerCache(
            features,
            cache_size=cache_size,
            fifo_size=fifo_size,
            update_period=update_period,
        )
        for start in range(0, length, chunk_size * 8):
            # A cooperative request can resume on another settled worker stream.
            if features.is_cuda:
                stream = torch.cuda.current_stream(features.device)
                for tensor in (features, state.cache, state.fifo, state.cache_preds):
                    tensor.record_stream(stream)
            end = min(start + chunk_size * 8, length)
            right = min(right_context * 8, length - end)
            yield (
                self.forward_chunk(features[:, :, start : end + right], state, right)[
                    :, : end - start
                ],
                end == length,
            )

    def forward_chunk(self, features, state: SpeakerCache, right: int):
        """Advance one recording's cache; right context is in 10 ms frames."""
        chunk = self.encoder.pre_encode(features)
        prefix_length = state.cache.shape[1] + state.fifo.shape[1]
        embeddings = torch.cat([state.cache, state.fifo, chunk], dim=1)
        valid_length = prefix_length + (features.shape[2] + 7) // 8

        def head(encoded):
            predictions = self.sortformer_modules(encoded, valid_length)
            pooled = F.avg_pool1d(predictions.transpose(1, 2), 8, 8)
            return predictions, pooled.transpose(1, 2)

        predictions, low_resolution = self.encoder(embeddings, valid_length, head)
        count = chunk.shape[1] - (right + 7) // 8
        output = predictions[:, prefix_length * 8 : (prefix_length + count) * 8]
        state.update(
            chunk[:, :count], low_resolution, self.sortformer_modules.learnable_sil_emb
        )
        return output
