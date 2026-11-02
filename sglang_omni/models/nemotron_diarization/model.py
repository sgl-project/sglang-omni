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

# note (Richard Wang): Use deterministic compiler reductions without changing other models.
compiled_flex_attention = torch.compile(
    flex_attention,
    dynamic=True,
    options={"deterministic": True, "triton.autotune_pointwise": False},
)
MASK_BLOCK_FRAMES = 128


class Featurizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # note (Ryan Stewart): Restored from the checkpoint, for a centered rather than causal STFT.
        self.register_buffer("window", torch.empty(400))
        self.register_buffer("fb", torch.empty(1, 128, 257))

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        frame_count = waveform.shape[1] // 160
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
        # note (Ryan Stewart): Preserve the reference's sqrt/square rounding.
        magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1))
        features = torch.log(torch.matmul(self.fb, magnitude.pow(2)) + 2**-24)
        features[:, :, frame_count:] = 0
        return features, frame_count


class Preprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.featurizer = Featurizer()

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        return self.featurizer(waveform)


class FeatureStacking(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8 * 128, 512, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        frames = features.transpose(1, 2)
        frames = F.pad(frames, (0, 0, 0, -frames.shape[1] % 8))
        return self.proj(frames.reshape(1, -1, 8 * 128))


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
    cosine = tl.load(cos + frame * head_dim + column, valid, other=0)
    sine = tl.load(sin + frame * head_dim + column, valid, other=0)
    q_base = batch * q_batch + head * q_head + frame * q_frame
    k_base = batch * k_batch + head * k_head + frame * k_frame
    out_base = batch * o_batch + head * o_head + frame * o_frame
    q_value = tl.load(q + q_base + column, valid, other=0)
    q_other = tl.load(q + q_base + rotated, valid, other=0)
    k_value = tl.load(k + k_base + column, valid, other=0)
    k_other = tl.load(k + k_base + rotated, valid, other=0)
    tl.store(
        q_out + out_base + column, q_value * cosine + (sign * q_other) * sine, valid
    )
    tl.store(
        k_out + out_base + column, k_value * cosine + (sign * k_other) * sine, valid
    )


class RotaryEmbedding(nn.Module):
    def __init__(self, max_frames: int) -> None:
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

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply NeOX rotation with separate FP32 products and addition."""
        batch, heads, frames, head_dim = q.shape
        # note (Richard Wang): The kernel reads table rows by frame without a bound check.
        assert frames <= len(self.cos), f"{frames} frames exceed the rotary table"
        size = (batch, heads, frames, head_dim)
        stride = (frames * heads * head_dim, head_dim, heads * head_dim, 1)
        q_out, k_out = (q.new_empty_strided(size, stride) for _ in range(2))
        with torch.cuda.device(q.device):
            # note (Richard Wang): Fusion changes the FP32 reference rounding.
            rotary_kernel[(frames, heads, batch)](
                q,
                k,
                self.cos,
                self.sin,
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


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w_qkv = nn.Linear(512, 1536, bias=False)
        self.out_proj = nn.Linear(512, 512)

    def forward(
        self, hidden_states: torch.Tensor, rotary: RotaryEmbedding, mask: BlockMask
    ) -> torch.Tensor:
        qkv = (
            self.w_qkv(hidden_states)
            .view(1, hidden_states.shape[1], 3, 8, 64)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = rotary(q, k)
        # note (Richard Wang): Tile by window length only and keep the 64-row FP32 summation order.
        block_rows = 16 if hidden_states.shape[1] < MASK_BLOCK_FRAMES else 64
        attention_output = compiled_flex_attention(
            q, k, v, block_mask=mask, kernel_options={"BLOCK_M": block_rows}
        )
        return self.out_proj(
            attention_output.transpose(1, 2)
            .contiguous()
            .view(1, hidden_states.shape[1], 512)
        )


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # note (Ryan Stewart): Identity holds the checkpoint's training-dropout index.
        self.net = nn.Sequential(
            nn.Linear(512, 2048), nn.GELU(), nn.Identity(), nn.Linear(2048, 512)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(512)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(512)
        self.ffn = FeedForward()

    def forward(
        self, hidden_states: torch.Tensor, rotary: RotaryEmbedding, mask: BlockMask
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), rotary, mask
        )
        return hidden_states + self.ffn(self.norm2(hidden_states))


def block_mask_rows(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-row block counts and the block indices that come first."""
    return (
        blocks.sum(-1).to(torch.int32)[None, None],
        blocks.to(torch.int32)
        .argsort(descending=True, stable=True)
        .to(torch.int32)[None, None],
    )


class TransformerEncoder(nn.Module):
    def __init__(self, max_frames: int) -> None:
        super().__init__()
        self.pre_encode = FeatureStacking()
        self.pos_enc = RotaryEmbedding(max_frames)
        self.embed_norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(31)])
        self.final_norm = nn.LayerNorm(512)
        # note (Richard Wang): Per-thread masks stay on that worker thread's stream.
        self.thread_masks = local()

    def attention_mask(self, frame_count: int, valid_length: int) -> BlockMask:
        masks = getattr(self.thread_masks, "masks", None)
        if masks is None:
            masks = self.thread_masks.masks = OrderedDict()
        else:
            pass
        mask_key = (frame_count, valid_length)
        if mask_key in masks:
            masks.move_to_end(mask_key)
            return masks[mask_key]
        else:
            pass
        device = self.final_norm.weight.device
        valid_lengths = torch.tensor([valid_length], device=device)

        def padding_mask(
            batch: torch.Tensor,
            head: torch.Tensor,
            query_index: torch.Tensor,
            key_value_index: torch.Tensor,
        ) -> torch.Tensor:
            return key_value_index < valid_lengths[batch]

        # note (Richard Wang): Derive prefix blocks without a dense token mask.
        blocks = torch.arange(-(-frame_count // MASK_BLOCK_FRAMES), device="cpu")
        full = ((blocks[:, None] + 1) * MASK_BLOCK_FRAMES <= frame_count) & (
            (blocks[None, :] + 1) * MASK_BLOCK_FRAMES <= valid_length
        )
        partial = (blocks[None, :] * MASK_BLOCK_FRAMES < valid_length) & ~full
        masks[mask_key] = BlockMask.from_kv_blocks(
            *block_mask_rows(partial),
            *block_mask_rows(full),
            BLOCK_SIZE=MASK_BLOCK_FRAMES,
            mask_mod=padding_mask,
            seq_lengths=(frame_count, frame_count),
            compute_q_blocks=False,
        ).to(device)
        if len(masks) > 128:
            masks.popitem(last=False)
        else:
            pass
        return masks[mask_key]

    def forward(self, embeddings: torch.Tensor, valid_length: int) -> torch.Tensor:
        mask = self.attention_mask(embeddings.shape[1], valid_length)
        hidden_states = self.embed_norm(embeddings)
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.pos_enc, mask)
        return self.final_norm(hidden_states)


class SortformerModules(nn.Module):
    def __init__(
        self,
        *,
        fc_d_model: int = 512,
        tf_d_model: int = 192,
        speakers: int = 8,
        upsample: int = 8,
    ) -> None:
        super().__init__()
        self.tf_d_model = tf_d_model
        self.speakers = speakers
        self.upsample = upsample
        # note (Ryan Stewart): Unused legacy head that strict loading needs from the archive.
        self.hidden_to_spks = nn.Linear(2 * tf_d_model, speakers)
        self.first_hidden_to_hidden = nn.Linear(tf_d_model, tf_d_model)
        self.single_hidden_to_spks = nn.Linear(tf_d_model, speakers)
        self.encoder_proj = nn.Linear(fc_d_model, tf_d_model)
        self.subpixel_upsample = nn.Conv1d(
            tf_d_model, tf_d_model * upsample, 3, padding=1
        )
        self.learnable_sil_emb = nn.Parameter(torch.empty(fc_d_model))

    def forward(
        self, embeddings: torch.Tensor, valid_length: int | torch.Tensor
    ) -> torch.Tensor:
        """valid_length is an int, or per-row lengths shaped (batch, 1)."""
        hidden = self.encoder_proj(embeddings)
        hidden = self.subpixel_upsample(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden.reshape(embeddings.shape[0], -1, self.tf_d_model)
        hidden = F.relu(self.first_hidden_to_hidden(F.relu(hidden)))
        probabilities = torch.sigmoid(self.single_hidden_to_spks(hidden))
        mask = (
            torch.arange(embeddings.shape[1], device=embeddings.device) < valid_length
        )
        return probabilities * mask.repeat_interleave(self.upsample, dim=-1)[..., None]


class NemotronDiarizationModel(nn.Module):
    def __init__(self, *, profile: tuple[int, int, int, int, int]) -> None:
        super().__init__()
        self.preprocessor = Preprocessor()
        self.encoder = TransformerEncoder(sum(profile[:4]))
        self.sortformer_modules = SortformerModules()
        self.profile = profile

    def iter_chunks(
        self, waveform: torch.Tensor
    ) -> Generator[tuple[torch.Tensor, bool], None, None]:
        """Yield each upload window's 10 ms probabilities and whether it is the last."""
        features, frame_count = self.preprocessor(waveform)
        del waveform
        features = features[:, :, :frame_count]
        cache_size, fifo_size, chunk_size, right_context, update_period = self.profile
        state = SpeakerCache(
            features,
            cache_size=cache_size,
            fifo_size=fifo_size,
            update_period=update_period,
        )
        if frame_count == 0:
            yield features.new_empty(1, 0, 8), True
        else:
            pass
        for start_frame in range(0, frame_count, chunk_size * 8):
            if features.is_cuda:
                # note (Richard Wang): The next window may run on another worker stream.
                stream = torch.cuda.current_stream(features.device)
                for tensor in (features, state.cache, state.fifo, state.cache_preds):
                    tensor.record_stream(stream)
            else:
                pass
            end_frame = min(start_frame + chunk_size * 8, frame_count)
            right_frames = min(right_context * 8, frame_count - end_frame)
            yield (
                self.forward_chunk(
                    features[:, :, start_frame : end_frame + right_frames],
                    state,
                    right_frames,
                )[:, : end_frame - start_frame],
                end_frame == frame_count,
            )

    def forward_chunk(
        self, features: torch.Tensor, state: SpeakerCache, right_frames: int
    ) -> torch.Tensor:
        """Advance one recording's cache, with right context in 10 ms frames."""
        chunk = self.encoder.pre_encode(features)
        prefix_length = state.cache.shape[1] + state.fifo.shape[1]
        embeddings = torch.cat([state.cache, state.fifo, chunk], dim=1)
        valid_length = prefix_length + (features.shape[2] + 7) // 8
        encoded = self.encoder(embeddings, valid_length)
        predictions = self.sortformer_modules(encoded, valid_length)
        low_resolution = F.avg_pool1d(predictions.transpose(1, 2), 8, 8).transpose(1, 2)
        kept_frames = chunk.shape[1] - (right_frames + 7) // 8
        output = predictions[:, prefix_length * 8 : (prefix_length + kept_frames) * 8]
        state.update(
            chunk[:, :kept_frames],
            low_resolution,
            self.sortformer_modules.learnable_sil_emb,
        )
        return output
