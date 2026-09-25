# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Flow for MiniCPM-o."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.components.token2wav.conformer import (
    UpsampleConformerEncoderV2,
    make_pad_mask,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT


class CausalConditionalCFM(torch.nn.Module):

    def __init__(self, estimator: DiT, inference_cfg_rate: float = 0.7) -> None:
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.out_channels = estimator.out_channels
        self.register_buffer(
            "rand_noise",
            torch.randn([1, self.out_channels, 50 * 600]),
            persistent=False,
        )

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        caches: list[dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        t = t_span[0].expand(batch_size)
        dt = t_span[1] - t_span[0]
        assert self.inference_cfg_rate > 0, "inference_cfg_rate better > 0"
        mask_in = torch.cat([mask, mask], dim=0)
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        for step in range(1, len(t_span)):
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            dphi_dt = self.estimator.forward(
                x_in,
                mask_in if caches is None else None,
                mu_in,
                t_in,
                spks_in,
                cond_in,
                cache=caches[step - 1] if caches is not None else None,
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (
                1.0 + self.inference_cfg_rate
            ) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]
            else:
                pass
        return x

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        caches: list[dict[str, torch.Tensor]] | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        if n_timesteps <= 0:
            raise ValueError("n_timesteps must be positive")
        else:
            pass
        if offset + mu.size(2) > self.rand_noise.size(2):
            raise ValueError(
                "Combined reference and generated audio exceed 600 seconds"
            )
        else:
            pass
        z = (
            self.rand_noise[:, :, offset : offset + mu.size(2)]
            .expand(mu.size(0), -1, -1)
            .clone()
            * temperature
        )
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span, mu, mask, spks, cond, caches)

    @torch.inference_mode()
    def forward_chunk(
        self,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        cnn_cache: torch.Tensor | None = None,
        att_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offset = att_cache.shape[4] if att_cache is not None else 0
        caches = (
            [{} for _ in range(n_timesteps)]
            if att_cache is None
            else [
                {"cnn": cnn_cache[index], "attention": att_cache[index]}
                for index in range(n_timesteps)
            ]
        )
        result = self.forward(
            mu,
            torch.ones_like(mu[:, :1]),
            spks,
            cond,
            n_timesteps,
            temperature,
            caches,
            offset,
        )
        return (
            result,
            torch.stack([cache["cnn"] for cache in caches]),
            torch.stack([cache["attention"] for cache in caches]),
        )


class CausalMaskedDiffWithXvec(torch.nn.Module):

    def __init__(
        self,
        encoder: UpsampleConformerEncoderV2,
        decoder: CausalConditionalCFM,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: Literal["mel"] = "mel",
        vocab_size: int = 6561,
    ) -> None:
        super().__init__()
        if output_type != "mel":
            raise ValueError("MiniCPM-o flow output must be mel")
        else:
            pass
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.pre_lookahead_len = int(encoder.pre_lookahead_layer.pre_lookahead_len)
        self.up_rate = int(encoder.up_layer.stride)
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_dim, output_size)
        self.decoder = decoder

    @torch.inference_mode()
    def inference(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        assert token.shape[0] == prompt_token.shape[0], (
            f"flow batch size mismatch: token={token.shape[0]} "
            f"prompt_token={prompt_token.shape[0]}"
        )
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        token_len = prompt_token_len + token_len
        token = torch.concat([prompt_token, token], dim=1)
        token_mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * token_mask
        h, _ = self.encoder.forward(token, token_len)
        frame_mask = (~make_pad_mask(token_len * self.up_rate, h.shape[1])).to(h)
        h = self.encoder_proj(h) * frame_mask.unsqueeze(-1)
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]
        conds = torch.zeros_like(h)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2).contiguous()
        feat = self.decoder.forward(
            mu=h.transpose(1, 2).contiguous(),
            mask=frame_mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat

    @torch.inference_mode()
    def setup_cache(
        self,
        token: torch.Tensor,
        mel: torch.Tensor,
        spk: torch.Tensor,
        n_timesteps: int = 10,
    ) -> dict[str, torch.Tensor]:
        assert (token.shape[1] - self.pre_lookahead_len) * self.up_rate == mel.shape[
            1
        ], (token.shape, mel.shape)
        _, cache = self.inference_chunk(
            token, spk, None, n_timesteps=n_timesteps, prompt_feat=mel
        )
        return cache

    @torch.inference_mode()
    def inference_chunk(
        self,
        token: torch.Tensor,
        spk: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
        last_chunk: bool = False,
        n_timesteps: int = 10,
        prompt_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        conformer_cnn_cache = (
            cache["conformer_cnn_cache"] if cache is not None else None
        )
        conformer_att_cache = (
            cache["conformer_att_cache"] if cache is not None else None
        )
        estimator_cnn_cache = (
            cache["estimator_cnn_cache"] if cache is not None else None
        )
        estimator_att_cache = (
            cache["estimator_att_cache"] if cache is not None else None
        )
        spk = F.normalize(spk, dim=1)
        spk = self.spk_embed_affine_layer(spk)
        token = self.input_embedding(token)
        h, conformer_cnn_cache, conformer_att_cache = self.encoder.forward_chunk(
            xs=token,
            last_chunk=last_chunk,
            cnn_cache=conformer_cnn_cache,
            att_cache=conformer_att_cache,
        )
        h = self.encoder_proj(h)
        cond = torch.zeros_like(h) if prompt_feat is None else prompt_feat
        feat, estimator_cnn_cache, estimator_att_cache = self.decoder.forward_chunk(
            mu=h.transpose(1, 2).contiguous(),
            spks=spk,
            cond=cond.transpose(1, 2).contiguous(),
            n_timesteps=n_timesteps,
            temperature=1.0,
            cnn_cache=estimator_cnn_cache,
            att_cache=estimator_att_cache,
        )
        new_cache = {
            "conformer_cnn_cache": conformer_cnn_cache,
            "conformer_att_cache": conformer_att_cache,
            "estimator_cnn_cache": estimator_cnn_cache,
            "estimator_att_cache": estimator_att_cache,
        }
        return (feat, new_cache)
