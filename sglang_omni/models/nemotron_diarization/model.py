# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only Nemotron 3 Sortformer.

Adapted from NVIDIA-NeMo/Speech revision
2c1a2f91d64566b5d391b83df42f9ab4cd810adb: transformer_encoder.py,
sortformer_diar_models.py, features.py, subsampling.py and multi_head_attention.py.
Omni changes: fixed checkpoint architecture, single-recording inference, plain
PyTorch modules, and explicit request-local cache ownership.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache

_compiled_attention = torch.compile(flex_attention, dynamic=True)


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


class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # NeMo computes transcendental functions on CPU for device-independent
        # rounding. The bounded speaker cache keeps every chunk below 5000 frames.
        frequencies = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
        angles = torch.outer(torch.arange(5000).float(), frequencies)
        angles = torch.cat([angles, angles], dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        cos, sin = self.cos[: x.shape[2]], self.sin[: x.shape[2]]
        rotated = torch.cat([-x[..., 32:], x[..., :32]], dim=-1)
        return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_qkv = nn.Linear(512, 1536, bias=False)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x, rotary, mask):
        qkv = self.w_qkv(x).view(1, x.shape[1], 3, 8, 64).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = _compiled_attention if x.is_cuda else flex_attention
        out = attention(rotary(q), rotary(k), v, block_mask=mask)
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
    def __init__(self):
        super().__init__()
        self.pre_encode = FeatureStacking()
        self.pos_enc = RotaryEmbedding()
        self.embed_norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(31)])
        self.final_norm = nn.LayerNorm(512)

    def forward(self, embeddings, valid_length):
        lengths = torch.tensor([valid_length], device=embeddings.device)

        def padding_mask(b, h, q_idx, kv_idx):
            return kv_idx < lengths[b]

        size = embeddings.shape[1]
        mask = create_block_mask(
            padding_mask, B=1, H=1, Q_LEN=size, KV_LEN=size, device=embeddings.device
        )
        x = self.embed_norm(embeddings)
        for layer in self.layers:
            x = layer(x, self.pos_enc, mask)
        return self.final_norm(x)


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
        self.encoder = TransformerEncoder()
        self.sortformer_modules = SortformerModules()
        self.profile = profile

    @torch.inference_mode()
    def forward(self, waveform):
        """Return [1, time, 8] probabilities at 10 ms, with fresh state per call."""
        features, length = self.preprocessor(waveform)
        features = features[:, :, :length]
        if length == 0:
            return waveform.new_empty(1, 0, 8)
        cache_size, fifo_size, chunk_size, right_context, update_period = self.profile
        state = SpeakerCache(
            features,
            cache_size=cache_size,
            fifo_size=fifo_size,
            update_period=update_period,
        )
        outputs = []
        for start in range(0, features.shape[2], chunk_size * 8):
            end = min(start + chunk_size * 8, features.shape[2])
            right = min(right_context * 8, features.shape[2] - end)
            outputs.append(
                self.forward_chunk(features[:, :, start : end + right], state, right)
            )
        predictions = torch.cat(outputs, dim=1)[:, : features.shape[2]]
        predictions[:, length:] = 0
        return predictions

    def forward_chunk(self, features, state: SpeakerCache, right: int):
        """Advance one recording's cache; right context is in 10 ms frames."""
        chunk = self.encoder.pre_encode(features)
        prefix_length = state.cache.shape[1] + state.fifo.shape[1]
        embeddings = torch.cat([state.cache, state.fifo, chunk], dim=1)
        valid_length = prefix_length + (features.shape[2] + 7) // 8
        encoded = self.encoder(embeddings, valid_length)
        predictions = self.sortformer_modules(encoded, valid_length)
        count = chunk.shape[1] - (right + 7) // 8
        output = predictions[:, prefix_length * 8 : (prefix_length + count) * 8]
        low_resolution = F.avg_pool1d(predictions.transpose(1, 2), 8, 8).transpose(1, 2)
        state.update(
            chunk[:, :count], low_resolution, self.sortformer_modules.learnable_sil_emb
        )
        return output
