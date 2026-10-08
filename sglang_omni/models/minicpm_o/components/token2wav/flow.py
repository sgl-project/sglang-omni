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
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.minicpm_o.components.token2wav.conformer import (
    UpsampleConformerEncoderV2,
    make_pad_mask,
)
from sglang_omni.models.minicpm_o.components.token2wav.conformer_state import (
    ConformerState,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT, DiTState


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
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        states: list[DiTState] | None = None,
    ) -> tuple[torch.Tensor, list[DiTState] | None]:
        """Integrate the flow; streaming passes one estimator state per step."""
        batch_size = x.size(0)
        t = t_span[0].expand(batch_size)
        dt = t_span[1] - t_span[0]
        assert self.inference_cfg_rate > 0, "inference_cfg_rate better > 0"
        paired_mask = torch.cat([mask, mask], dim=0)
        paired_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        paired_speaker_embeddings = torch.cat(
            [speaker_embeddings, torch.zeros_like(speaker_embeddings)], dim=0
        )
        paired_mel_conditioning = torch.cat(
            [mel_conditioning, torch.zeros_like(mel_conditioning)], dim=0
        )
        next_states: list[DiTState] | None = None if states is None else []
        for step in range(1, len(t_span)):
            paired_sample = torch.cat([x, x], dim=0)
            paired_timesteps = torch.cat([t, t], dim=0)
            if states is None:
                conditional_derivative = self.estimator.forward(
                    paired_sample,
                    paired_mask,
                    paired_mu,
                    paired_timesteps,
                    paired_speaker_embeddings,
                    paired_mel_conditioning,
                )
            else:
                conditional_derivative, next_state = self.estimator.forward_chunk(
                    paired_sample,
                    paired_mu,
                    paired_timesteps,
                    paired_speaker_embeddings,
                    paired_mel_conditioning,
                    states[step - 1],
                )
                next_states.append(next_state)
            conditional_derivative, unconditional_derivative = torch.split(
                conditional_derivative, [x.size(0), x.size(0)], dim=0
            )
            guided_derivative = (
                (1.0 + self.inference_cfg_rate) * conditional_derivative
                - self.inference_cfg_rate * unconditional_derivative
            )
            x = x + dt * guided_derivative
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]
            else:
                pass
        return x, next_states

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        states: list[DiTState] | None = None,
        offset: int = 0,
    ) -> tuple[torch.Tensor, list[DiTState] | None]:
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
        return self.solve_euler(
            z, t_span, mu, mask, speaker_embeddings, mel_conditioning, states
        )

    @torch.inference_mode()
    def forward_chunk(
        self,
        mu: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        convolution_cache: torch.Tensor | None = None,
        attention_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attention_cache is None:
            offset = 0
            states = [DiTState() for _ in range(n_timesteps)]
        else:
            assert convolution_cache is not None
            offset = attention_cache.shape[4]
            states = [
                DiTState(
                    convolution=convolution_cache[index],
                    attention=attention_cache[index],
                )
                for index in range(n_timesteps)
            ]
        result, next_states = self.forward(
            mu,
            torch.ones_like(mu[:, :1]),
            speaker_embeddings,
            mel_conditioning,
            n_timesteps,
            temperature,
            states,
            offset,
        )
        assert next_states is not None
        return (
            result,
            torch.stack([state.convolution for state in next_states]),
            torch.stack([state.attention for state in next_states]),
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
        self.speaker_embedding_projection = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_dim, output_size)
        self.decoder = decoder

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_token_lengths: torch.Tensor,
        prompt_mel: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        assert speech_tokens.shape[0] == prompt_tokens.shape[0], (
            f"flow batch size mismatch: speech_tokens={speech_tokens.shape[0]} "
            f"prompt_tokens={prompt_tokens.shape[0]}"
        )
        speaker_embeddings = F.normalize(speaker_embeddings, dim=1)
        speaker_embeddings = self.speaker_embedding_projection(speaker_embeddings)
        prompt_row_lengths = prompt_token_lengths.tolist()
        generated_row_lengths = token_lengths.tolist()
        combined_tokens = pad_sequence(
            [
                torch.cat(
                    [prompt_tokens[i, :prompt_length], speech_tokens[i, :token_length]]
                )
                for i, (prompt_length, token_length) in enumerate(
                    zip(prompt_row_lengths, generated_row_lengths, strict=True)
                )
            ],
            batch_first=True,
        )
        combined_token_lengths = prompt_token_lengths + token_lengths
        token_mask = (
            (~make_pad_mask(combined_token_lengths))
            .unsqueeze(-1)
            .to(speaker_embeddings)
        )
        embedded_tokens = (
            self.input_embedding(torch.clamp(combined_tokens, min=0)) * token_mask
        )
        hidden_states, _ = self.encoder.forward(embedded_tokens, combined_token_lengths)
        frame_mask = (
            ~make_pad_mask(
                combined_token_lengths * self.up_rate, hidden_states.shape[1]
            )
        ).to(hidden_states)
        hidden_states = self.encoder_proj(hidden_states) * frame_mask.unsqueeze(-1)
        mel_conditioning = torch.zeros_like(hidden_states)
        for i, prompt_length in enumerate(prompt_row_lengths):
            prompt_frames = prompt_length * self.up_rate
            mel_conditioning[i, :prompt_frames] = prompt_mel[i, :prompt_frames]
        mel_conditioning = mel_conditioning.transpose(1, 2).contiguous()
        predicted_mel, _ = self.decoder.forward(
            mu=hidden_states.transpose(1, 2).contiguous(),
            mask=frame_mask.unsqueeze(1),
            speaker_embeddings=speaker_embeddings,
            mel_conditioning=mel_conditioning,
            n_timesteps=n_timesteps,
        )
        generated = [
            predicted_mel[
                i,
                :,
                prompt_length
                * self.up_rate : (prompt_length + token_length)
                * self.up_rate,
            ]
            for i, (prompt_length, token_length) in enumerate(
                zip(prompt_row_lengths, generated_row_lengths, strict=True)
            )
        ]
        return pad_sequence(
            [row.transpose(0, 1) for row in generated], batch_first=True
        ).transpose(1, 2)

    @torch.inference_mode()
    def setup_cache(
        self,
        token_ids: torch.Tensor,
        prompt_mel: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        n_timesteps: int = 10,
    ) -> dict[str, torch.Tensor]:
        assert (
            token_ids.shape[1] - self.pre_lookahead_len
        ) * self.up_rate == prompt_mel.shape[1], (token_ids.shape, prompt_mel.shape)
        _, cache = self.inference_chunk(
            token_ids,
            speaker_embeddings,
            None,
            n_timesteps=n_timesteps,
            prompt_mel=prompt_mel,
        )
        return cache

    @torch.inference_mode()
    def inference_chunk(
        self,
        token_ids: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
        is_last_chunk: bool = False,
        n_timesteps: int = 10,
        prompt_mel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        conformer_convolution_cache = (
            cache["conformer_convolution_cache"] if cache is not None else None
        )
        conformer_attention_cache = (
            cache["conformer_attention_cache"] if cache is not None else None
        )
        estimator_convolution_cache = (
            cache["estimator_convolution_cache"] if cache is not None else None
        )
        estimator_attention_cache = (
            cache["estimator_attention_cache"] if cache is not None else None
        )
        speaker_embeddings = F.normalize(speaker_embeddings, dim=1)
        speaker_embeddings = self.speaker_embedding_projection(speaker_embeddings)
        embedded_tokens = self.input_embedding(token_ids)
        conformer_state = ConformerState.from_packed(
            conformer_convolution_cache,
            conformer_attention_cache,
            len(self.encoder.encoders),
            self.encoder.up_layer.stride,
        )
        hidden_states, conformer_state = self.encoder.forward_chunk(
            xs=embedded_tokens,
            is_last_chunk=is_last_chunk,
            state=conformer_state,
        )
        conformer_convolution_cache, conformer_attention_cache = (
            conformer_state.to_packed(self.encoder.up_layer.stride)
        )
        hidden_states = self.encoder_proj(hidden_states)
        mel_conditioning = (
            torch.zeros_like(hidden_states) if prompt_mel is None else prompt_mel
        )
        predicted_mel, estimator_convolution_cache, estimator_attention_cache = (
            self.decoder.forward_chunk(
                mu=hidden_states.transpose(1, 2).contiguous(),
                speaker_embeddings=speaker_embeddings,
                mel_conditioning=mel_conditioning.transpose(1, 2).contiguous(),
                n_timesteps=n_timesteps,
                temperature=1.0,
                convolution_cache=estimator_convolution_cache,
                attention_cache=estimator_attention_cache,
            )
        )
        new_cache = {
            "conformer_convolution_cache": conformer_convolution_cache,
            "conformer_attention_cache": conformer_attention_cache,
            "estimator_convolution_cache": estimator_convolution_cache,
            "estimator_attention_cache": estimator_attention_cache,
        }
        return (predicted_mel, new_cache)
