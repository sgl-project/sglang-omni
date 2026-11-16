# SPDX-License-Identifier: Apache-2.0
"""The ELLSA 2 Standalone generator: one 0.8 s block of motion codes from the retained audio and motion history.

Each block re-encodes the whole retained history of both audio streams and
runs every one of its six steps as a full forward pass without a KV cache,
under the same bf16 autocast scopes as the reference, so a block's codes
equal the reference's for the same inputs.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import LlamaConfig, Qwen2Config, Qwen2Model

from sglang_omni.models.ellsa2.audio_encoder import CausalAudioEncoder
from sglang_omni.models.ellsa2.rvq import CODEBOOK_SIZE

# The release's block: 0.8 s of 16 kHz audio per stream, read as ten embeddings, and six motion steps of 18 codes (six upper body, six hands, four expression, two mouth) out of 2048 each.
BLOCK_SAMPLES = 12800
AUDIO_EMBEDDINGS_PER_BLOCK = 10
BLOCK_STEPS = 6
NUM_CODEBOOKS = 18
# The connector joins four encoder frames into one backbone embedding.
CONNECTOR_FRAMES = 4
CONNECTOR_WIDTH = 4096
# The release's per-codebook weights on the summed motion embeddings: six upper-body, six hand, four expression, two mouth codebooks.
MOTION_EMBED_WEIGHTS = (
    (1.0, 0.5, 0.25, 0.1, 0.05, 0.025)
    + (1.0, 0.5, 0.25, 0.1, 0.05, 0.025)
    + (1.0, 0.5, 0.25, 0.1)
    + (1.0, 0.5)
)
# --max-history-sec 12 and --reset-history-chunks 3: at most 15 blocks, the current one included; past that, the last 3 completed blocks stay.
MAX_HISTORY_BLOCKS = 15
RETAINED_BLOCKS = 3
PROMPT_HEAD = (
    "<|im_start|>system\n"
    "You are a co-speech gesture and facial expression generation model.<|im_end|>\n"
    "<|im_start|>user\n"
    "Generate co-speech gesture and facial expression tokens from the interleaved speaker and listener audio stream.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
AUDIO_START = "<|vision_start|>"
AUDIO_END = "<|vision_end|>"


@dataclass(frozen=True, kw_only=True)
class PromptTokens:
    """Token ids of the prompt's text head and of the markers around each audio block."""

    head: torch.Tensor
    audio_start: int
    audio_end: int


@dataclass(kw_only=True)
class HistoryBlock:
    """One retained block: both streams' float32 16 kHz audio and, once generated, its codes [BLOCK_STEPS, NUM_CODEBOOKS]."""

    target_audio: np.ndarray
    partner_audio: np.ndarray
    codes: torch.Tensor | None = None


def retained_history(history: list[HistoryBlock]) -> list[HistoryBlock]:
    """The completed blocks a new block is generated after: all of them, or the last RETAINED_BLOCKS once one more would exceed MAX_HISTORY_BLOCKS."""
    if len(history) + 1 > MAX_HISTORY_BLOCKS:
        return history[-RETAINED_BLOCKS:]
    else:
        return history


class Ellsa2Generator(nn.Module):
    """Qwen2 backbone, causal audio encoder, connector and the 18 motion codebooks' embeddings and heads, named as in the checkpoint."""

    def __init__(
        self,
        backbone_config: Qwen2Config,
        encoder_config: LlamaConfig,
        cudnn_attention: bool = False,
    ) -> None:
        super().__init__()
        self.cudnn_attention = cudnn_attention
        hidden_size = backbone_config.hidden_size
        self.backbone = Qwen2Model(backbone_config)
        self.audio_encoder = CausalAudioEncoder(encoder_config)
        self.encoder_width = encoder_config.hidden_size
        self.ln_audio = nn.LayerNorm(self.encoder_width)
        self.connector = nn.Sequential(
            nn.Linear(self.encoder_width * CONNECTOR_FRAMES, CONNECTOR_WIDTH),
            nn.ReLU(),
            nn.Linear(CONNECTOR_WIDTH, hidden_size),
        )
        self.motion_embeds = nn.ModuleList(
            [nn.Embedding(CODEBOOK_SIZE, hidden_size) for _ in range(NUM_CODEBOOKS)]
        )
        self.motion_heads = nn.ModuleList(
            [nn.Linear(hidden_size, CODEBOOK_SIZE) for _ in range(NUM_CODEBOOKS)]
        )
        self.motion_embed_weights = nn.Buffer(
            torch.tensor(MOTION_EMBED_WEIGHTS), persistent=False
        )

    def restore_rope_precision(self) -> None:
        """Recompute the backbone's rotary frequencies in float32 on its device after a cast to bf16.

        The checkpoint was trained with float32 Qwen2 frequencies; the audio
        encoder's frequencies stay as the cast left them, as in the reference.
        """
        config = self.backbone.config
        head_dim = config.hidden_size // config.num_attention_heads
        device = self.backbone.rotary_emb.inv_freq.device
        self.backbone.rotary_emb.inv_freq = 1.0 / (
            config.rope_parameters["rope_theta"]
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / head_dim
            )
        )

    def autocast(self) -> torch.autocast:
        device = self.motion_embed_weights.device
        return torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        )

    def attention_backends(self) -> AbstractContextManager[None]:
        if self.cudnn_attention:
            return nullcontext()
        else:
            return sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            )

    def encode_audio(self, features: torch.Tensor) -> torch.Tensor:
        """Log-mel features [1, frames, bins] → [1, ceil(((frames - 1) // 2) / CONNECTOR_FRAMES), hidden]."""
        with self.autocast():
            hidden = self.ln_audio(self.audio_encoder(features))
            hidden = F.pad(hidden, (0, 0, 0, -hidden.shape[1] % CONNECTOR_FRAMES))
            return self.connector(
                hidden.view(1, -1, self.encoder_width * CONNECTOR_FRAMES)
            )

    def motion_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes [1, steps, NUM_CODEBOOKS] → the weighted sum of their codebook embeddings [1, steps, hidden]; a negative code adds nothing."""
        levels = []
        for level, embedding in enumerate(self.motion_embeds):
            level_codes = codes[:, :, level]
            embedded = embedding(level_codes.clamp(min=0, max=CODEBOOK_SIZE - 1))
            embedded = embedded.masked_fill((level_codes < 0).unsqueeze(-1), 0.0)
            levels.append(
                embedded * self.motion_embed_weights[level].to(embedded.dtype)
            )
        return torch.stack(levels, dim=0).sum(dim=0)

    def block_inputs(
        self,
        prompt: PromptTokens,
        target_embeddings: torch.Tensor,
        partner_embeddings: torch.Tensor,
        known_codes: torch.Tensor,
        style_steps: int,
        num_blocks: int,
    ) -> torch.Tensor:
        """The prompt head, the style's motion, then per block both streams' audio and, for completed blocks, their motion: [1, length, hidden]."""
        markers = self.backbone.embed_tokens(
            torch.tensor(
                [prompt.audio_start, prompt.audio_end], device=known_codes.device
            )
        )
        motion = self.motion_embeddings(known_codes.unsqueeze(0))[0]
        parts = [self.backbone.embed_tokens(prompt.head), motion[:style_steps]]
        for block in range(num_blocks):
            audio = slice(
                block * AUDIO_EMBEDDINGS_PER_BLOCK,
                (block + 1) * AUDIO_EMBEDDINGS_PER_BLOCK,
            )
            parts += [
                markers[:1],
                target_embeddings[0, audio],
                markers[1:],
                markers[:1],
                partner_embeddings[0, audio],
                markers[1:],
            ]
            if block < num_blocks - 1:
                start = style_steps + block * BLOCK_STEPS
                parts.append(motion[start : start + BLOCK_STEPS])
            else:
                pass
        return torch.cat(parts, dim=0).unsqueeze(0)

    @torch.no_grad()
    def generate_block(
        self,
        prompt: PromptTokens,
        target_features: torch.Tensor,
        partner_features: torch.Tensor,
        known_codes: torch.Tensor,
        style_steps: int,
        num_blocks: int,
    ) -> torch.Tensor:
        """Greedy codes [BLOCK_STEPS, NUM_CODEBOOKS] of the last of num_blocks blocks, from both streams' features over all of them and the codes known before it (style, then completed blocks)."""
        # Note (Junnan Li): Without cudnn_attention the backbone's SDPA leaves cuDNN out: a block takes about 96 ms instead of 105 ms on an H100, so one GPU serves 8 Sessions, and the codes repeat across processes. With it SDPA keeps torch's own choice, cuDNN included, as the reference does, which matches the reference's codes bit for bit under its libraries.
        embeds = self.block_inputs(
            prompt,
            self.encode_audio(target_features),
            self.encode_audio(partner_features),
            known_codes,
            style_steps,
            num_blocks,
        )
        attention_mask = torch.ones(
            embeds.shape[:2], dtype=torch.long, device=embeds.device
        )
        steps = []
        for _ in range(BLOCK_STEPS):
            with self.autocast(), self.attention_backends():
                hidden = self.backbone(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state[:, -1:, :]
                logits = [head(hidden) for head in self.motion_heads]
            step = torch.stack(
                [
                    level_logits.squeeze(1).float().argmax(dim=-1)
                    for level_logits in logits
                ],
                dim=-1,
            )
            steps.append(step)
            embeds = torch.cat(
                [embeds, self.motion_embeddings(step.unsqueeze(1))], dim=1
            )
            attention_mask = F.pad(attention_mask, (0, 1), value=1)
        return torch.cat(steps, dim=0)


__all__ = [
    "AUDIO_END",
    "AUDIO_START",
    "BLOCK_SAMPLES",
    "BLOCK_STEPS",
    "NUM_CODEBOOKS",
    "PROMPT_HEAD",
    "Ellsa2Generator",
    "HistoryBlock",
    "PromptTokens",
    "retained_history",
]
