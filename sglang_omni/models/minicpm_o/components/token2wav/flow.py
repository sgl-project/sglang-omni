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
"""Generate mel features from codec tokens and speaker conditioning with flow matching."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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
                x_in, mask_in, mu_in, t_in, spks_in, cond_in
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
    ) -> torch.Tensor:
        if n_timesteps <= 0:
            raise ValueError("n_timesteps must be positive")
        else:
            pass
        if mu.size(2) > self.rand_noise.size(2):
            raise ValueError(
                "Combined reference and generated audio exceed 600 seconds"
            )
        else:
            pass
        z = (
            self.rand_noise[:, :, : mu.size(2)].expand(mu.size(0), -1, -1).clone()
            * temperature
        )
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span, mu, mask, spks, cond)


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
        token_lens: Sequence[int],
        prompt_token: torch.Tensor,
        prompt_token_lens: Sequence[int],
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
        batch_size = token.shape[0]
        # note(liuqihao): place padding after both sequences to preserve each prompt boundary.
        combined_rows = []
        for i in range(batch_size):
            prompt_row = prompt_token[i, : prompt_token_lens[i]]
            generated_row = token[i, : token_lens[i]]
            combined_rows.append(torch.cat([prompt_row, generated_row]))
        combined = pad_sequence(combined_rows, batch_first=True)
        combined_lengths = torch.tensor(
            [
                prompt + generated
                for prompt, generated in zip(prompt_token_lens, token_lens, strict=True)
            ],
            dtype=torch.int32,
            device=token.device,
        )
        valid_tokens = ~make_pad_mask(combined_lengths, combined.shape[1])
        token_mask = valid_tokens.unsqueeze(-1)
        token_mask = token_mask.to(embedding)
        nonnegative_tokens = torch.clamp(combined, min=0)
        token = self.input_embedding(nonnegative_tokens) * token_mask
        h, _ = self.encoder.forward(token, combined_lengths)
        valid_frames = ~make_pad_mask(combined_lengths * self.up_rate, h.shape[1])
        frame_mask = valid_frames.to(h)
        h = self.encoder_proj(h) * frame_mask.unsqueeze(-1)
        conds = torch.zeros_like(h)
        for i, prompt_len in enumerate(prompt_token_lens):
            prompt_frames = prompt_len * self.up_rate
            conds[i, :prompt_frames] = prompt_feat[i, :prompt_frames]
        conds = conds.transpose(1, 2).contiguous()
        feat = self.decoder.forward(
            mu=h.transpose(1, 2).contiguous(),
            mask=frame_mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
        )
        generated = []
        for i in range(batch_size):
            first_generated_frame = prompt_token_lens[i] * self.up_rate
            generated_frame_count = token_lens[i] * self.up_rate
            generated.append(
                feat[
                    i,
                    :,
                    first_generated_frame : first_generated_frame
                    + generated_frame_count,
                ]
            )
        time_major_rows = [row.transpose(0, 1) for row in generated]
        padded_rows = pad_sequence(time_major_rows, batch_first=True)
        return padded_rows.transpose(1, 2)
