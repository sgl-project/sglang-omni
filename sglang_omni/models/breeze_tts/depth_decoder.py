# SPDX-License-Identifier: Apache-2.0
"""Breeze's Llama-style depth decoder, using the pinned Transformers primitives.

Forward/token layout follows breezeblue-ai/breeze-tts (Apache-2.0), revision
43e2ea1595297c4059477e2e4a300653761c759b. Depth KV state lasts one audio frame,
not one speech request; no mutable cache is stored on the module.
"""

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel

from .sampling import SamplingConfig, apply_cfg, sample_logits


class BreezeDepthDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = LlamaConfig(**config)
        cfg._attn_implementation = "sdpa"
        self.config = cfg
        self.num_codebooks = config["num_codebooks"]
        self.audio_embed_size = config["audio_embed_size"]
        self.model = LlamaModel(cfg)
        self.model.embed_tokens = nn.Embedding(
            self.num_codebooks * cfg.vocab_size, self.audio_embed_size
        )
        self.model.inputs_embeds_projector = nn.Linear(
            self.audio_embed_size, cfg.hidden_size, bias=False
        )
        backbone_hidden_size = config["backbone_hidden_size"]
        if backbone_hidden_size != self.audio_embed_size:
            raise ValueError(
                "Breeze-TTS-2 requires matching backbone/audio embedding sizes"
            )
        self.codebooks_head = nn.Module()
        self.codebooks_head.weight = nn.Parameter(
            torch.empty(self.num_codebooks - 1, cfg.hidden_size, cfg.vocab_size)
        )

    def embed_frames(self, codes: torch.Tensor) -> torch.Tensor:
        offsets = (
            torch.arange(self.num_codebooks, device=codes.device)
            * self.config.vocab_size
        )
        return self.model.embed_tokens(codes.long() + offsets).sum(dim=-2)

    @torch.no_grad()
    def decode_frame(
        self,
        hidden: torch.Tensor,
        first_code: torch.Tensor,
        params: SamplingConfig,
        generator: torch.Generator,
        *,
        codebook_size: int = 2048,
    ) -> torch.Tensor:
        # [backbone hidden, c0] prefill predicts c1 with head 0. Every later
        # position embeds c(k) from its own codebook before predicting c(k+1).
        token = first_code.reshape(1).expand(2)
        embeds = torch.stack((hidden, self.model.embed_tokens(token)), dim=1)
        codes = [first_code.reshape(())]
        cache = None
        for codebook in range(1, self.num_codebooks):
            output = self.model(
                inputs_embeds=self.model.inputs_embeds_projector(embeds),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            logits = (
                output.last_hidden_state[:, -1].float()
                @ self.codebooks_head.weight[codebook - 1].float()
            )
            code = sample_logits(
                apply_cfg(logits, params.cfg_scale),
                params,
                generator,
                codebook_size=codebook_size,
            )
            codes.append(code.reshape(()))
            embeds = self.model.embed_tokens(
                code.expand(2) + codebook * self.config.vocab_size
            ).unsqueeze(1)
        return torch.stack(codes)
