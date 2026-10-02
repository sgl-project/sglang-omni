# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie heads on top of the Nemotron-H backbone.

The backbone produces one hidden state per frame; these heads turn it into a
stacked acoustic frame through a small causal local transformer that samples
one codebook at a time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sglang.srt.layers.sampler import multinomial_with_seed
from torch import nn

from sglang_omni.models.easymagpie_tts.hf_config import EasyMagpieTTSConfig


class LocalAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("local transformer width must divide evenly into heads")
        else:
            pass
        self.heads = heads
        self.head_dim = width // heads
        self.qkv_net = nn.Linear(width, width * 3, bias=False)
        self.o_net = nn.Linear(width, width, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, _ = values.shape
        qkv = self.qkv_net(values).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
        )
        return self.o_net(attended.transpose(1, 2).reshape(batch, length, -1))


class PointwiseConv(nn.Module):
    """Kernel-1 convolution stored as a Linear under the checkpoint's conv name."""

    def __init__(self, source: int, target: int) -> None:
        super().__init__()
        self.conv = nn.Linear(source, target, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(values)


class LocalFeedForward(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = PointwiseConv(width, width * 4)
        self.o_net = PointwiseConv(width * 4, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.o_net(F.gelu(self.proj(values), approximate="tanh"))


class LocalLayer(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(width, bias=False)
        self.self_attention = LocalAttention(width, heads)
        self.norm_pos_ff = nn.LayerNorm(width, bias=False)
        self.pos_ff = LocalFeedForward(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.self_attention(self.norm_self(values))
        return values + self.pos_ff(self.norm_pos_ff(values))


class LocalTransformer(nn.Module):
    def __init__(self, config: EasyMagpieTTSConfig) -> None:
        super().__init__()
        width = config.local_transformer_hidden_dim
        self.position_embeddings = nn.Embedding(config.num_stacked_codebooks + 2, width)
        self.layers = nn.ModuleList(
            LocalLayer(width, config.local_transformer_n_heads)
            for _ in range(config.local_transformer_n_layers)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        positions = self.position_embeddings.weight[: values.shape[1]]
        values = values + positions.unsqueeze(0)
        for layer in self.layers:
            values = layer(values)
        return values


class EasyMagpieTTSHeads(nn.Module):
    """Every non-backbone talker weight in the converted checkpoint."""

    def __init__(self, config: EasyMagpieTTSConfig) -> None:
        super().__init__()
        self.config = config
        width = config.embedding_dim
        vocab = config.codebook_vocab_size
        self.text_embedding = nn.Embedding(config.text_vocab_size, width)
        self.phoneme_embeddings = nn.ModuleList(
            nn.Embedding(config.phoneme_vocab_size, width)
            for _ in range(config.phoneme_stacking_factor)
        )
        self.phoneme_final_proj = nn.Linear(
            width, config.phoneme_vocab_size * config.phoneme_stacking_factor
        )
        self.audio_embeddings = nn.ModuleList(
            nn.Embedding(vocab, width) for _ in range(config.num_stacked_codebooks)
        )
        self.local_transformer = LocalTransformer(config)
        self.local_transformer_out_projections = nn.ModuleList(
            nn.Linear(width, vocab) for _ in range(config.num_stacked_codebooks)
        )
        forbidden = torch.zeros(vocab, dtype=torch.bool)
        forbidden[config.codebook_size :] = True
        forbidden[config.audio_eos_id] = False
        self.register_buffer("forbidden_code_mask", forbidden, persistent=False)

    def embed_audio_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """Average the per-codebook embeddings of one stacked frame."""
        codebooks = len(self.audio_embeddings)
        if codes.ndim != 2 or codes.shape[1] != codebooks:
            raise ValueError(
                f"expected codes [frames, {codebooks}], got {tuple(codes.shape)}"
            )
        else:
            pass
        values = sum(
            table(codes[:, index]) for index, table in enumerate(self.audio_embeddings)
        )
        return values / codebooks

    def embed_phonemes(self, phoneme_tokens: torch.Tensor) -> torch.Tensor:
        """Average the per-channel embeddings of one stacked phoneme row."""
        channels = self.config.phoneme_stacking_factor
        if phoneme_tokens.ndim != 2 or phoneme_tokens.shape[1] != channels:
            raise ValueError(f"phoneme_tokens must have shape [tokens, {channels}]")
        else:
            pass
        values = sum(
            table(phoneme_tokens[:, index])
            for index, table in enumerate(self.phoneme_embeddings)
        )
        return values / channels

    def predict_phonemes(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy phoneme row, replaced by UNK when the head is underconfident."""
        config = self.config
        logits = self.phoneme_final_proj(hidden_states).view(
            hidden_states.shape[0],
            config.phoneme_stacking_factor,
            config.phoneme_vocab_size,
        )
        phonemes = logits.argmax(dim=-1)
        threshold = config.phoneme_confidence_unk_threshold
        if threshold > 0:
            confidence = logits.float().softmax(dim=-1).amax(dim=-1)
            underconfident = (confidence < threshold).any(dim=1, keepdim=True)
            is_eos = (phonemes == config.phoneme_eos_id).any(dim=1, keepdim=True)
            return torch.where(
                underconfident & ~is_eos,
                torch.full_like(phonemes, config.phoneme_unk_id),
                phonemes,
            )
        else:
            return phonemes

    def sample_codes(
        self,
        hidden_states: torch.Tensor,
        *,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        seeds: torch.Tensor,
        positions: torch.Tensor,
        max_top_k: int,
    ) -> torch.Tensor:
        """Sample one stacked frame with request-local temperature, top-k and seed."""
        batch = hidden_states.shape[0]
        codebooks = len(self.audio_embeddings)
        sample_k = min(max_top_k, self.config.codebook_vocab_size)
        values = hidden_states.new_zeros(batch, codebooks, hidden_states.shape[-1])
        values[:, 0] = hidden_states
        ranks = torch.arange(sample_k, device=hidden_states.device).unsqueeze(0)
        codes = []
        for index, head in enumerate(self.local_transformer_out_projections):
            local_hidden = self.local_transformer(values[:, : index + 1])[:, -1]
            logits = head(local_hidden).masked_fill(
                self.forbidden_code_mask, -torch.inf
            )
            scores, token_ids = (
                logits.float().div(temperatures.unsqueeze(1)).topk(sample_k, dim=-1)
            )
            scores.masked_fill_(ranks >= top_ks.unsqueeze(1), -torch.inf)
            sampled = sample_seeded_rows(scores, seeds, positions + index)
            code = token_ids.gather(1, sampled).squeeze(1)
            codes.append(code)
            if index + 1 < codebooks:
                values[:, index + 1] = self.audio_embeddings[index](code)
            else:
                pass
        return torch.stack(codes, dim=1)


def sample_seeded_rows(
    scores: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Draw one index per row, reproducible from the row's seed and position."""
    if scores.device.type == "cuda":
        return multinomial_with_seed(scores, seeds, positions).view(-1, 1)
    else:
        rows = []
        for row in range(scores.shape[0]):
            generator = torch.Generator(device=scores.device)
            generator.manual_seed(int(seeds[row]) ^ int(positions[row]))
            rows.append(
                torch.multinomial(scores[row].softmax(dim=-1), 1, generator=generator)
            )
        return torch.stack(rows, dim=0)


__all__ = ["EasyMagpieTTSHeads", "LocalTransformer"]
