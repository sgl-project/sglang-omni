# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only Nemotron 3 Sortformer.

Adapted from NVIDIA-NeMo/Speech revision
2c1a2f91d64566b5d391b83df42f9ab4cd810adb: transformer_encoder.py,
sortformer_diar_models.py, features.py, subsampling.py and multi_head_attention.py.
Omni changes: fixed checkpoint architecture, plain PyTorch modules, explicit
request-local cache ownership, and batched compact chunk execution.
"""

from collections import OrderedDict, defaultdict
from collections.abc import Generator, Sequence
from threading import local

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache
from sglang_omni.utils.device import device_guard, supports_device_streams

# note (Richard Wang): Use deterministic compiler reductions without changing other models.
compiled_flex_attention = torch.compile(
    flex_attention,
    dynamic=True,
    options={"deterministic": True, "triton.autotune_pointwise": False},
)
MASK_BLOCK_FRAMES = 128
MEL_FRAMES_PER_ENCODER_FRAME = 8


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
        return self.proj(frames.reshape(features.shape[0], -1, 8 * 128))


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
        with device_guard(q.device):
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
            .view(hidden_states.shape[0], hidden_states.shape[1], 3, 8, 64)
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
            .view(hidden_states.shape[0], hidden_states.shape[1], 512)
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
    if blocks.ndim == 2:
        blocks = blocks.unsqueeze(0)
    else:
        pass
    return (
        blocks.sum(-1).to(torch.int32)[:, None],
        blocks.to(torch.int32)
        .argsort(descending=True, stable=True)
        .to(torch.int32)[:, None],
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

    def attention_mask(
        self, frame_count: int, valid_length: int | Sequence[int]
    ) -> BlockMask:
        masks = getattr(self.thread_masks, "masks", None)
        if masks is None:
            masks = self.thread_masks.masks = OrderedDict()
        else:
            pass
        valid_encoder_lengths = (
            (valid_length,) if isinstance(valid_length, int) else tuple(valid_length)
        )
        if (
            not valid_encoder_lengths
            or min(valid_encoder_lengths) < 1
            or max(valid_encoder_lengths) > frame_count
        ):
            raise ValueError(
                "Attention valid_encoder_lengths must be in [1, frame_count]"
            )
        else:
            pass
        padding_mask_key = (frame_count, valid_encoder_lengths)
        if padding_mask_key in masks:
            masks.move_to_end(padding_mask_key)
            return masks[padding_mask_key]
        else:
            pass
        device = self.final_norm.weight.device
        encoder_length_tensor = torch.tensor(valid_encoder_lengths, device=device)

        def padding_mask(
            batch: torch.Tensor,
            head: torch.Tensor,
            query_index: torch.Tensor,
            key_value_index: torch.Tensor,
        ) -> torch.Tensor:
            return key_value_index < encoder_length_tensor[batch]

        # note (Richard Wang): Derive prefix blocks without a dense token mask.
        blocks = torch.arange(-(-frame_count // MASK_BLOCK_FRAMES), device="cpu")
        host_encoder_lengths = torch.tensor(valid_encoder_lengths, device="cpu")
        full = ((blocks[None, :, None] + 1) * MASK_BLOCK_FRAMES <= frame_count) & (
            (blocks[None, None, :] + 1) * MASK_BLOCK_FRAMES
            <= host_encoder_lengths[:, None, None]
        )
        partial = (
            blocks[None, None, :] * MASK_BLOCK_FRAMES
            < host_encoder_lengths[:, None, None]
        ) & ~full
        masks[padding_mask_key] = BlockMask.from_kv_blocks(
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
        return masks[padding_mask_key]

    def forward(
        self, embeddings: torch.Tensor, valid_length: int | Sequence[int]
    ) -> torch.Tensor:
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
        mask = (
            torch.arange(embeddings.shape[1], device=embeddings.device) < valid_length
        )
        hidden = self.encoder_proj(embeddings)
        # note (eranik): Projection bias must also be zeroed before convolution reads a padded neighbor.
        hidden = hidden.masked_fill(~mask[..., None], 0.0)
        hidden = self.subpixel_upsample(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden.reshape(embeddings.shape[0], -1, self.tf_d_model)
        hidden = F.relu(self.first_hidden_to_hidden(F.relu(hidden)))
        probabilities = torch.sigmoid(self.single_hidden_to_spks(hidden))
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
            if supports_device_streams(features.device):
                # note (Richard Wang): The next window may run on another worker stream.
                stream = torch.get_device_module(features.device).current_stream(
                    features.device
                )
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

    def forward_chunks(
        self,
        mel_features: Sequence[torch.Tensor],
        speaker_caches: Sequence[SpeakerCache],
        right_context_mel_frames: Sequence[int],
    ) -> list[torch.Tensor]:
        """Advance independent recordings with compact, right-padded encoder rows."""
        batch_size = len(mel_features)
        if (
            batch_size == 0
            or len(speaker_caches) != batch_size
            or len(right_context_mel_frames) != batch_size
        ):
            raise ValueError(
                "Features, states and right contexts need equal nonzero lengths"
            )
        else:
            pass
        if any(
            chunk_mel_features.ndim != 3
            or chunk_mel_features.shape[:2] != (1, 128)
            or chunk_mel_features.shape[2] == 0
            for chunk_mel_features in mel_features
        ):
            raise ValueError("Each chunk must have shape (1,128,positive frames)")
        else:
            pass
        maximum_mel_frames = max(
            chunk_mel_features.shape[2] for chunk_mel_features in mel_features
        )
        packed_mel_features = torch.cat(
            [
                F.pad(
                    chunk_mel_features,
                    (0, maximum_mel_frames - chunk_mel_features.shape[2]),
                )
                for chunk_mel_features in mel_features
            ]
        )
        chunk_embeddings = self.encoder.pre_encode(packed_mel_features)
        chunk_encoder_frames = [
            (chunk_mel_features.shape[2] + MEL_FRAMES_PER_ENCODER_FRAME - 1)
            // MEL_FRAMES_PER_ENCODER_FRAME
            for chunk_mel_features in mel_features
        ]
        prefix_encoder_frames = [
            speaker_cache.cache.shape[1] + speaker_cache.fifo.shape[1]
            for speaker_cache in speaker_caches
        ]
        valid_encoder_frames = [
            prefix_encoder_frame_count + chunk_encoder_frame_count
            for prefix_encoder_frame_count, chunk_encoder_frame_count in zip(
                prefix_encoder_frames, chunk_encoder_frames
            )
        ]
        maximum_encoder_frames = max(valid_encoder_frames)
        # note (eranik): Padding only after each complete prefix preserves rotary positions.
        packed_embeddings = torch.cat(
            [
                F.pad(
                    torch.cat(
                        [
                            speaker_cache.cache,
                            speaker_cache.fifo,
                            chunk_embeddings[
                                row_index : row_index + 1, :chunk_encoder_frame_count
                            ],
                        ],
                        dim=1,
                    ),
                    (0, 0, 0, maximum_encoder_frames - valid_encoder_frames[row_index]),
                )
                for row_index, (speaker_cache, chunk_encoder_frame_count) in enumerate(
                    zip(speaker_caches, chunk_encoder_frames)
                )
            ]
        )
        encoded_embeddings = self.encoder(packed_embeddings, valid_encoder_frames)
        encoder_length_tensor = torch.tensor(
            valid_encoder_frames, device=encoded_embeddings.device
        )[:, None]
        predictions = self.sortformer_modules(encoded_embeddings, encoder_length_tensor)
        downsampled_predictions = F.avg_pool1d(
            predictions.transpose(1, 2),
            MEL_FRAMES_PER_ENCODER_FRAME,
            MEL_FRAMES_PER_ENCODER_FRAME,
        ).transpose(1, 2)
        kept_encoder_frames = [
            chunk_encoder_frame_count
            - (right_frames + MEL_FRAMES_PER_ENCODER_FRAME - 1)
            // MEL_FRAMES_PER_ENCODER_FRAME
            for chunk_encoder_frame_count, right_frames in zip(
                chunk_encoder_frames, right_context_mel_frames
            )
        ]
        if min(kept_encoder_frames) < 1:
            raise ValueError(
                "Each offline step needs at least one central encoder frame"
            )
        else:
            pass
        outputs = [
            predictions[
                row_index : row_index + 1,
                prefix_encoder_frame_count
                * MEL_FRAMES_PER_ENCODER_FRAME : (
                    prefix_encoder_frame_count + kept_encoder_frame_count
                )
                * MEL_FRAMES_PER_ENCODER_FRAME,
            ]
            for row_index, (
                prefix_encoder_frame_count,
                kept_encoder_frame_count,
            ) in enumerate(zip(prefix_encoder_frames, kept_encoder_frames))
        ]
        cache_groups: dict[tuple[int, int, bool, int, int, int, int], list[int]] = (
            defaultdict(list)
        )
        for row_index, (speaker_cache, kept_encoder_frame_count) in enumerate(
            zip(speaker_caches, kept_encoder_frames)
        ):
            cache_geometry = (
                speaker_cache.cache.shape[1],
                speaker_cache.fifo.shape[1],
                speaker_cache.compressed,
                speaker_cache.cache_size,
                speaker_cache.fifo_size,
                speaker_cache.update_period,
                kept_encoder_frame_count,
            )
            cache_groups[cache_geometry].append(row_index)
        silence_embedding = self.sortformer_modules.learnable_sil_emb
        # note (eranik): Only matching geometries and compression phases can share the native cache update.
        for row_indices in cache_groups.values():
            if len(row_indices) == 1:
                row_index = row_indices[0]
                speaker_caches[row_index].update(
                    chunk_embeddings[
                        row_index : row_index + 1, : kept_encoder_frames[row_index]
                    ],
                    downsampled_predictions[
                        row_index : row_index + 1, : valid_encoder_frames[row_index]
                    ],
                    silence_embedding,
                )
            else:
                first_cache = speaker_caches[row_indices[0]]
                merged_cache = SpeakerCache(
                    chunk_embeddings,
                    cache_size=first_cache.cache_size,
                    fifo_size=first_cache.fifo_size,
                    update_period=first_cache.update_period,
                    batch=len(row_indices),
                )
                merged_cache.cache = torch.cat(
                    [speaker_caches[row_index].cache for row_index in row_indices]
                )
                merged_cache.fifo = torch.cat(
                    [speaker_caches[row_index].fifo for row_index in row_indices]
                )
                merged_cache.cache_preds = torch.cat(
                    [speaker_caches[row_index].cache_preds for row_index in row_indices]
                )
                merged_cache.compressed = first_cache.compressed
                kept_encoder_frame_count = kept_encoder_frames[row_indices[0]]
                required_encoder_frames = (
                    first_cache.cache.shape[1]
                    + first_cache.fifo.shape[1]
                    + kept_encoder_frame_count
                )
                merged_cache.update(
                    torch.cat(
                        [
                            chunk_embeddings[
                                row_index : row_index + 1, :kept_encoder_frame_count
                            ]
                            for row_index in row_indices
                        ]
                    ),
                    torch.cat(
                        [
                            downsampled_predictions[
                                row_index : row_index + 1, :required_encoder_frames
                            ]
                            for row_index in row_indices
                        ]
                    ),
                    silence_embedding,
                )
                for cache_row_index, row_index in enumerate(row_indices):
                    # note (eranik): Fresh storage keeps request views safe across later slot refills.
                    speaker_caches[row_index].cache = merged_cache.cache[
                        cache_row_index : cache_row_index + 1
                    ]
                    speaker_caches[row_index].fifo = merged_cache.fifo[
                        cache_row_index : cache_row_index + 1
                    ]
                    speaker_caches[row_index].cache_preds = merged_cache.cache_preds[
                        cache_row_index : cache_row_index + 1
                    ]
                    speaker_caches[row_index].compressed = merged_cache.compressed
        return outputs
