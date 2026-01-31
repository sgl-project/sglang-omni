# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni model definitions for multi-stage pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.models.weight_utils import default_weight_loader
from sglang_omni.models.utils.sampling import sample_top_k_top_p

from .modules import (
    AudioEncoderLayer,
    CausalConv1d,
    CausalTransConv1d,
    Code2WavConvNeXtBlock,
    Code2WavDecoderBlock,
    Code2WavPreTransformer,
    DecoderLayer,
    MoeDecoderLayer,
    MRoPE,
    SinusoidsPositionEmbedding,
    SnakeBeta,
)


# ---- Output Types ----


@dataclass
class ThinkerOutput:
    """Output from Thinker module (language model)."""

    sequences: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    logits: torch.Tensor | None = None
    past_key_values: Any = None
    last_hidden_state: torch.Tensor | None = None


@dataclass
class TalkerOutput:
    """Output from Talker module (audio codec generator)."""

    codec_tokens: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    logits: torch.Tensor | None = None
    past_key_values: Any = None


@dataclass
class Code2WavOutput:
    """Output from Code2Wav module (vocoder)."""

    waveform: torch.Tensor


def _build_causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_length, seq_length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)


def _maybe_apply_causal_mask(
    attention_mask: torch.Tensor | None,
    seq_length: int,
    device: torch.device,
    *,
    allow_none: bool,
) -> torch.Tensor | None:
    if attention_mask is None:
        return _build_causal_mask(seq_length, device) if allow_none else None
    if attention_mask.dim() == 2:
        return _build_causal_mask(seq_length, device)
    return attention_mask


def _build_position_ids(
    seq_length: int, device: torch.device, start: int = 0
) -> torch.Tensor:
    return torch.arange(start, start + seq_length, device=device).unsqueeze(0)


# ---- Audio Encoder ----


class Qwen3OmniAudioEncoder(nn.Module):
    """Qwen3-Omni audio encoder for extracting audio features."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        audio_config = config.get("audio_config", config)

        self.d_model = audio_config.get("d_model", 1280)
        self.num_mel_bins = audio_config.get("num_mel_bins", 128)
        self.max_source_positions = audio_config.get("max_source_positions", 1500)
        self.encoder_layers = audio_config.get("encoder_layers", 32)
        self.encoder_attention_heads = audio_config.get("encoder_attention_heads", 20)
        self.encoder_ffn_dim = audio_config.get("encoder_ffn_dim", 5120)
        self.output_dim = audio_config.get("output_dim", 3584)
        self.downsample_hidden_size = audio_config.get("downsample_hidden_size", 480)
        self.n_window = audio_config.get("n_window", 100)
        self.n_window_infer = audio_config.get("n_window_infer", 400)
        self.conv_chunksize = audio_config.get("conv_chunksize", 500)

        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, self.d_model
        )

        self.conv2d1 = nn.Conv2d(1, self.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(self.downsample_hidden_size, self.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(self.downsample_hidden_size, self.downsample_hidden_size, 3, 2, padding=1)

        conv_out_dim = self.downsample_hidden_size * ((((self.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
        self.conv_out = nn.Linear(conv_out_dim, self.d_model, bias=False)

        self.layers = nn.ModuleList([
            AudioEncoderLayer(
                d_model=self.d_model,
                encoder_attention_heads=self.encoder_attention_heads,
                encoder_ffn_dim=self.encoder_ffn_dim,
            )
            for _ in range(self.encoder_layers)
        ])

        self.ln_post = nn.LayerNorm(self.d_model)
        self.proj1 = nn.Linear(self.d_model, self.d_model)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(self.d_model, self.output_dim)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    def _build_chunk_lengths(self, feature_lens: torch.Tensor) -> torch.Tensor:
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        total_chunks = int(chunk_num.sum().item())
        chunk_lengths = torch.full(
            (total_chunks,),
            self.n_window * 2,
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2
        return chunk_lengths

    def _pad_chunk_features(
        self, input_features: torch.Tensor, chunk_lengths: torch.Tensor
    ) -> torch.Tensor:
        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        return nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)

    def _build_chunk_mask(self, chunk_lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
        feature_lens_after_cnn = self._get_feat_extract_output_lengths(chunk_lengths)
        return nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=device) for length in feature_lens_after_cnn],
            batch_first=True,
        )

    def _apply_conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        return x

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        chunk_lengths = self._build_chunk_lengths(feature_lens)
        padded_feature = self._pad_chunk_features(input_features, chunk_lengths)
        padded_mask_after_cnn = self._build_chunk_mask(chunk_lengths, padded_feature.device)

        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = self._apply_conv_stack(chunk)
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding(padded_embed.shape[1])
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states.unsqueeze(0)).squeeze(0)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name in params_dict:
                default_weight_loader(params_dict[name], loaded_weight)


# ---- Thinker - Language Model ----


class Qwen3OmniThinker(nn.Module):
    """Qwen3-Omni Thinker module - the main language model with MoE support."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        thinker_config = config.get("thinker_config", config)
        text_config = thinker_config.get("text_config", thinker_config)

        self.hidden_size = text_config.get("hidden_size", 4096)
        self.num_hidden_layers = text_config.get("num_hidden_layers", 32)
        self.vocab_size = text_config.get("vocab_size", 152064)
        self.num_attention_heads = text_config.get("num_attention_heads", 32)
        self.num_key_value_heads = text_config.get("num_key_value_heads", 8)
        self.rms_norm_eps = text_config.get("rms_norm_eps", 1e-6)
        self.max_position_embeddings = text_config.get("max_position_embeddings", 32768)
        self.rope_theta = text_config.get("rope_theta", 10000.0)
        self.hidden_act = text_config.get("hidden_act", "silu")
        self.head_dim = text_config.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.use_qk_norm = text_config.get("use_qk_norm", False)

        # MoE config
        self.num_experts = text_config.get("num_experts", 0)
        self.num_experts_per_tok = text_config.get("num_experts_per_tok", 8)
        self.moe_intermediate_size = text_config.get("moe_intermediate_size", 768)
        self.shared_expert_intermediate_size = text_config.get("shared_expert_intermediate_size", 0)
        self.norm_topk_prob = text_config.get("norm_topk_prob", True)

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([
            MoeDecoderLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                num_experts=self.num_experts,
                num_experts_per_tok=self.num_experts_per_tok,
                moe_intermediate_size=self.moe_intermediate_size,
                shared_expert_intermediate_size=self.shared_expert_intermediate_size,
                head_dim=self.head_dim,
                rms_norm_eps=self.rms_norm_eps,
                hidden_act=self.hidden_act,
                norm_topk_prob=self.norm_topk_prob,
                use_qk_norm=self.use_qk_norm,
            )
            for _ in range(self.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.rotary_emb = MRoPE(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        audio_config = thinker_config.get("audio_config")
        self.audio_tower = Qwen3OmniAudioEncoder(audio_config) if audio_config else None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> ThinkerOutput:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        seq_length = inputs_embeds.shape[1]

        if position_ids is None:
            position_ids = _build_position_ids(seq_length, inputs_embeds.device)

        attention_mask = _maybe_apply_causal_mask(
            attention_mask, seq_length, inputs_embeds.device, allow_none=False
        )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_head(hidden_states)

        return ThinkerOutput(
            sequences=logits.argmax(dim=-1),
            hidden_states=all_hidden_states,
            logits=logits,
            past_key_values=new_past_key_values if use_cache else None,
            last_hidden_state=hidden_states,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Load weights with simple name matching."""
        params_dict = dict(self.named_parameters())
        loaded_count = 0
        skipped_count = 0

        for name, loaded_weight in weights:
            if "talker" in name or "code2wav" in name:
                continue

            name = name.replace("thinker.", "").replace("model.", "")

            if "rotary_emb.inv_freq" in name or "visual" in name:
                skipped_count += 1
                continue

            if name in params_dict:
                default_weight_loader(params_dict[name], loaded_weight)
                loaded_count += 1
            else:
                skipped_count += 1

        print(f"[Qwen3OmniThinker] Loaded {loaded_count} weights, skipped {skipped_count}")


# ---- Talker - Audio Codec Generator ----


class TalkerProjection(nn.Module):
    """Projection layer for Talker with checkpoint-compatible naming."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(in_features, in_features, bias=True)
        self.linear_fc2 = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_fc1(x)
        x = self.act(x)
        x = self.linear_fc2(x)
        return x


class TalkerCodePredictor(nn.Module):
    """Code predictor for Talker (predicts remaining codec groups)."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 1024)
        self.num_hidden_layers = config.get("num_hidden_layers", 5)
        self.vocab_size = config.get("vocab_size", 2048)
        self.num_attention_heads = config.get("num_attention_heads", 16)
        self.num_key_value_heads = config.get("num_key_value_heads", 8)
        self.intermediate_size = config.get("intermediate_size", 3072)
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-6)
        self.hidden_act = config.get("hidden_act", "silu")
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.use_qk_norm = config.get("use_qk_norm", True)
        self.num_code_groups = config.get("num_code_groups", 16)
        self.max_position_embeddings = config.get("max_position_embeddings", 32768)
        self.rope_theta = config.get("rope_theta", 10000.0)
        rope_params = config.get("rope_scaling") or config.get("rope_parameters") or {}
        self.mrope_section = rope_params.get("mrope_section")

        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                rms_norm_eps=self.rms_norm_eps,
                use_qk_norm=self.use_qk_norm,
            )
            for _ in range(self.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.rotary_emb = MRoPE(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            mrope_section=self.mrope_section,
        )
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(self.vocab_size, self.hidden_size) for _ in range(self.num_code_groups - 1)]
        )
        self.lm_head = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.vocab_size, bias=False) for _ in range(self.num_code_groups - 1)]
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        generation_steps: int | None = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if generation_steps is None:
            generation_steps = 0
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            embed_layer = self.codec_embedding[generation_steps]
            inputs_embeds = embed_layer(input_ids)

        seq_length = inputs_embeds.shape[1]
        attention_mask = _maybe_apply_causal_mask(
            attention_mask, seq_length, inputs_embeds.device, allow_none=True
        )

        hidden_states = inputs_embeds
        position_ids = _build_position_ids(seq_length, inputs_embeds.device)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=None,
                use_cache=False,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head[generation_steps](hidden_states)
        if return_hidden_states:
            return logits, hidden_states
        return logits

    def generate_residual_codes(
        self,
        past_hidden: torch.Tensor,
        last_id_hidden: torch.Tensor,
        top_k: int = 50,
        top_p: float = 0.8,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate residual codec groups for each token."""
        if past_hidden.dim() != 3 or last_id_hidden.dim() != 3:
            raise ValueError("past_hidden and last_id_hidden must be [1, seq, hidden]")

        past_hidden = past_hidden.squeeze(0)
        last_id_hidden = last_id_hidden.squeeze(0)
        inputs_embeds = torch.stack([past_hidden, last_id_hidden], dim=1)

        residual_codes = []
        residual_hiddens: list[torch.Tensor] = []
        for step in range(self.num_code_groups - 1):
            logits, hidden_states = self.forward(
                inputs_embeds=inputs_embeds, generation_steps=step, return_hidden_states=True
            )
            next_token = sample_top_k_top_p(logits[:, -1, :], top_k, top_p)
            next_embed = self.codec_embedding[step](next_token).unsqueeze(1)
            residual_hiddens.append(next_embed)
            residual_codes.append(next_token)
            if step < self.num_code_groups - 2:
                inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

        return torch.stack(residual_codes, dim=0), residual_hiddens


class Qwen3OmniTalker(nn.Module):
    """Qwen3-Omni Talker module - generates audio codec tokens with MoE support."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        talker_config = config.get("talker_config", config)
        text_config = talker_config.get("text_config", talker_config)

        self.hidden_size = text_config.get("hidden_size", 1024)
        self.num_hidden_layers = text_config.get("num_hidden_layers", 8)
        self.vocab_size = text_config.get("vocab_size", 8194)
        self.num_attention_heads = text_config.get("num_attention_heads", 16)
        self.num_key_value_heads = text_config.get("num_key_value_heads", 4)
        self.rms_norm_eps = text_config.get("rms_norm_eps", 1e-6)
        self.hidden_act = text_config.get("hidden_act", "silu")
        self.head_dim = text_config.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.use_qk_norm = text_config.get("use_qk_norm", True)
        self.max_position_embeddings = text_config.get("max_position_embeddings", 65536)
        self.rope_theta = text_config.get("rope_theta", 1000000.0)
        rope_params = text_config.get("rope_scaling") or text_config.get("rope_parameters") or {}
        self.mrope_section = rope_params.get("mrope_section")

        # MoE config
        self.num_experts = text_config.get("num_experts", 0)
        self.num_experts_per_tok = text_config.get("num_experts_per_tok", 6)
        self.moe_intermediate_size = text_config.get("moe_intermediate_size", 384)
        self.shared_expert_intermediate_size = text_config.get("shared_expert_intermediate_size", 0)
        self.norm_topk_prob = text_config.get("norm_topk_prob", True)

        self.codec_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([
            MoeDecoderLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                num_experts=self.num_experts,
                num_experts_per_tok=self.num_experts_per_tok,
                moe_intermediate_size=self.moe_intermediate_size,
                shared_expert_intermediate_size=self.shared_expert_intermediate_size,
                head_dim=self.head_dim,
                rms_norm_eps=self.rms_norm_eps,
                hidden_act=self.hidden_act,
                norm_topk_prob=self.norm_topk_prob,
                use_qk_norm=self.use_qk_norm,
            )
            for _ in range(self.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.codec_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.rotary_emb = MRoPE(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            mrope_section=self.mrope_section,
        )

        talker_cfg = config.get("talker_config", {})
        thinker_hidden_size = talker_cfg.get("thinker_hidden_size", 2048)
        self.text_projection = TalkerProjection(thinker_hidden_size, self.hidden_size)
        self.hidden_projection = TalkerProjection(thinker_hidden_size, self.hidden_size)

        code_predictor_config = talker_config.get("code_predictor_config", {})
        self.code_predictor = TalkerCodePredictor(code_predictor_config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.codec_embedding

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        inputs_embeds: torch.Tensor | None = None,
        thinker_hidden_state: torch.Tensor | None = None,
        use_cache: bool = False,
        full_input_ids: torch.Tensor | None = None,
        thinker_embed: torch.Tensor | None = None,
        thinker_hidden: torch.Tensor | None = None,
        multimodal_mask: torch.Tensor | None = None,
        tts_token_embeds: torch.Tensor | None = None,
        speaker: str | None = None,
        **kwargs,
    ) -> TalkerOutput:
        if full_input_ids is not None and thinker_embed is not None and thinker_hidden is not None:
            return self._generate_audio(
                full_input_ids=full_input_ids,
                thinker_embed=thinker_embed,
                thinker_hidden=thinker_hidden,
                multimodal_mask=multimodal_mask,
                tts_token_embeds=tts_token_embeds,
                speaker=speaker,
                **kwargs,
            )

        if inputs_embeds is None:
            inputs_embeds = self.codec_embedding(input_ids)

        if thinker_hidden_state is not None and past_key_values is None:
            projected_hidden = self.hidden_projection(thinker_hidden_state)
            inputs_embeds = torch.cat([projected_hidden, inputs_embeds], dim=1)

        seq_length = inputs_embeds.shape[1]

        attention_mask = _maybe_apply_causal_mask(
            attention_mask, seq_length, inputs_embeds.device, allow_none=True
        )

        if position_ids is None:
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0:
                past_len = past_key_values[0][0].shape[2]
            position_ids = _build_position_ids(
                seq_length, inputs_embeds.device, start=past_len
            )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.codec_head(hidden_states)

        return TalkerOutput(
            codec_tokens=logits.argmax(dim=-1),
            hidden_states=hidden_states,
            logits=logits,
            past_key_values=new_past_key_values if use_cache else None,
        )

    def _generate_audio(
        self,
        full_input_ids: torch.Tensor,
        thinker_embed: torch.Tensor,
        thinker_hidden: torch.Tensor,
        multimodal_mask: torch.Tensor | None = None,
        tts_token_embeds: torch.Tensor | None = None,
        speaker: str | None = None,
        **kwargs,
    ) -> TalkerOutput:
        """Generate codec tokens using complete thinker features.
        
        This method is called when full thinker features are provided,
        executing the complete codec token generation pipeline.
        The generated codec tokens are then passed to Code2Wav for waveform synthesis.
        """
        from sglang_omni.models.utils.sampling import sample_logits

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        config = self.config
        talker_config = config.get("talker_config", config)
        text_config = talker_config.get("text_config", talker_config)
        code2wav_config = config.get("code2wav_config", {})

        codec_vocab_size = text_config.get("vocab_size", 8194)
        codec_bos_id = talker_config.get("codec_bos_id", 0)
        codec_eos_id = talker_config.get("codec_eos_token_id", codec_vocab_size - 1)
        codebook_size = code2wav_config.get("codebook_size", 2048)
        suppress_tokens = [i for i in range(codebook_size, codec_vocab_size) if i != codec_eos_id]

        im_start_token_id = config.get("im_start_token_id", 151644)
        system_token_id = config.get("system_token_id", 8948)
        user_token_id = config.get("user_token_id", 872)
        assistant_token_id = config.get("assistant_token_id", 77091)
        tts_pad_token_id = config.get("tts_pad_token_id", 151671)
        tts_bos_token_id = config.get("tts_bos_token_id", 151672)
        tts_eos_token_id = config.get("tts_eos_token_id", 151673)
        speaker_map = talker_config.get("speaker_id", {})
        hidden_size = text_config.get("hidden_size", 1024)

        speaker = (speaker or "Ethan").lower()
        speaker_id = speaker_map.get(speaker)
        if speaker_id is None:
            speaker_id = list(speaker_map.values())[0] if speaker_map else 0

        full_input_ids = full_input_ids.to(device)
        thinker_embed = thinker_embed.to(device)
        thinker_hidden = thinker_hidden.to(device)
        if multimodal_mask is not None:
            multimodal_mask = multimodal_mask.to(device, dtype=torch.bool)
        else:
            multimodal_mask = torch.zeros_like(full_input_ids, dtype=torch.bool)

        if tts_token_embeds is None:
            tts_tokens = torch.tensor(
                [[tts_bos_token_id, tts_eos_token_id, tts_pad_token_id]],
                device=device,
                dtype=full_input_ids.dtype,
            )
            tts_token_embeds = self.codec_embedding(tts_tokens)

        tts_bos_embed, tts_eos_embed, tts_pad_embed = (
            self.text_projection(tts_token_embeds.to(device)).chunk(3, dim=1)
        )

        im_start_indexes = torch.nonzero(full_input_ids[0] == im_start_token_id).squeeze()
        if im_start_indexes.dim() == 0:
            im_start_indexes = im_start_indexes.unsqueeze(0)
        im_start_indexes = torch.cat(
            (im_start_indexes, torch.tensor([full_input_ids.shape[-1]], device=device, dtype=full_input_ids.dtype)),
            dim=-1,
        )

        talker_input_embeds = []
        talker_input_ids = []
        trailing_text_hidden = None

        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i].item()
            segment_end_index = im_start_indexes[i + 1].item()
            role_token = full_input_ids[0][im_start_index + 1].item()

            if role_token == system_token_id:
                continue
            elif role_token == user_token_id:
                user_talker_part = torch.empty(
                    (1, segment_end_index - im_start_index, hidden_size),
                    device=device,
                    dtype=dtype,
                )
                user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index]
                if user_mm_mask.any():
                    user_thinker_hidden_mm = thinker_hidden[:, im_start_index:segment_end_index][user_mm_mask]
                    mm_hidden = self.hidden_projection(user_thinker_hidden_mm).to(thinker_hidden.device)
                    user_talker_part[user_mm_mask] = mm_hidden
                user_thinker_embed = thinker_embed[:, im_start_index:segment_end_index][~user_mm_mask]
                user_text_hidden = self.text_projection(user_thinker_embed).to(thinker_hidden.device)
                user_talker_part[~user_mm_mask] = user_text_hidden
                talker_input_embeds.append(user_talker_part)
                talker_input_ids.append(full_input_ids[:, im_start_index:segment_end_index])

            elif role_token == assistant_token_id and i == len(im_start_indexes) - 2:
                assistant_hidden = self.text_projection(
                    thinker_embed[:, im_start_index:segment_end_index]
                ).to(tts_pad_embed.device)
                assistant_text_hidden = torch.cat(
                    (assistant_hidden[:, :3], tts_pad_embed.expand(-1, 4, -1), tts_bos_embed, assistant_hidden[:, 3:4]),
                    dim=1,
                )
                codec_special_tokens = torch.tensor(
                    [[
                        talker_config.get("codec_nothink_id", 0),
                        talker_config.get("codec_think_bos_id", 0),
                        talker_config.get("codec_think_eos_id", 0),
                        speaker_id,
                        talker_config.get("codec_pad_id", 0),
                        codec_bos_id,
                    ]],
                    device=tts_pad_embed.device,
                    dtype=torch.long,
                )
                assistant_codec_hidden = torch.cat(
                    (
                        torch.zeros((1, 3, hidden_size), device=tts_pad_embed.device, dtype=dtype),
                        self.get_input_embeddings()(codec_special_tokens).to(tts_pad_embed.device),
                    ),
                    dim=1,
                )
                trailing_text_hidden = torch.cat((assistant_hidden[:, 4:], tts_eos_embed), dim=1)
                input_embeds = assistant_text_hidden + assistant_codec_hidden
                input_ids_segment = torch.full(
                    (1, assistant_text_hidden.shape[1]),
                    fill_value=tts_pad_token_id,
                    dtype=torch.long,
                    device=assistant_text_hidden.device,
                )
                talker_input_embeds.append(input_embeds)
                talker_input_ids.append(input_ids_segment)

        if not talker_input_embeds:
            return TalkerOutput(codec_tokens=torch.tensor([[]], device=device))

        talker_input_embed = torch.cat([embed.to(device) for embed in talker_input_embeds], dim=1)
        talker_input_id = torch.cat([ids.to(device) for ids in talker_input_ids], dim=1)

        text_token_count = trailing_text_hidden.shape[1] if trailing_text_hidden is not None else 0
        max_audio_from_text = text_token_count * 10 + 200
        max_new_tokens = min(kwargs.get("max_new_tokens", 4096), max_audio_from_text)

        temperature = kwargs.get("temperature", 0.9)
        top_k = kwargs.get("top_k", 50)
        top_p = kwargs.get("top_p", 1.0)
        repetition_penalty = kwargs.get("repetition_penalty", 1.1)
        code_top_k = kwargs.get("code_top_k", 50)
        code_top_p = kwargs.get("code_top_p", 0.8)
        do_sample = temperature > 0

        attention_mask = torch.ones((1, talker_input_embed.shape[1]), device=device, dtype=torch.long)
        position_ids, rope_deltas = _get_rope_index(talker_input_id, attention_mask)

        outputs = self._forward_step(
            inputs_embeds=talker_input_embed,
            attention_mask=attention_mask,
            use_cache=True,
            position_ids=position_ids,
        )
        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        generated_token_ids: list[int] = []

        prefill_suppress = (
            suppress_tokens + [codec_eos_id]
            if trailing_text_hidden is not None and trailing_text_hidden.shape[1] > 1
            else suppress_tokens
        )
        next_token = sample_logits(
            logits, do_sample, temperature, top_k, top_p, prefill_suppress, repetition_penalty, generated_token_ids
        )
        generated_token_ids.append(next_token.item())
        codec_ids = [next_token]
        residual_codes_list: list[torch.Tensor] = []
        generation_step = 0
        hidden_states = outputs.hidden_states

        for _ in range(max_new_tokens):
            last_id_hidden = self.codec_embedding(next_token.view(1, 1))
            residual_codes, residual_hiddens = self.code_predictor.generate_residual_codes(
                hidden_states[:, -1:, :], last_id_hidden, top_k=code_top_k, top_p=code_top_p
            )
            residual_codes_list.append(residual_codes)

            if residual_hiddens:
                last_group_idx = residual_codes.shape[0] - 1
                last_residual_ids = residual_codes[last_group_idx].unsqueeze(0)
                last_residual_hidden = self.code_predictor.codec_embedding[last_group_idx](last_residual_ids)
                residual_hiddens = (
                    residual_hiddens[:-1] + [last_residual_hidden]
                    if len(residual_hiddens) > 1
                    else [last_residual_hidden]
                )

            codec_hiddens = torch.cat([last_id_hidden] + residual_hiddens, dim=1)
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)

            if trailing_text_hidden is not None and generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed

            past_len = past_kv[0][0].shape[2] if past_kv is not None else 0
            delta = (past_len + rope_deltas).to(inputs_embeds.device)
            step_pos_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            step_pos_ids = step_pos_ids.view(1, -1).expand(delta.shape[0], -1)
            step_pos_ids = (step_pos_ids + delta).unsqueeze(0).expand(3, -1, -1)

            outputs = self._forward_step(
                inputs_embeds=inputs_embeds, past_key_values=past_kv, use_cache=True, position_ids=step_pos_ids
            )
            past_kv = outputs.past_key_values
            hidden_states = outputs.hidden_states
            logits = outputs.logits[:, -1, :]

            step_suppress = (
                suppress_tokens + [codec_eos_id]
                if trailing_text_hidden is not None and generation_step < trailing_text_hidden.shape[1] - 1
                else suppress_tokens
            )
            next_token = sample_logits(
                logits, do_sample, temperature, top_k, top_p, step_suppress, repetition_penalty, generated_token_ids
            )
            generation_step += 1

            if next_token.item() == codec_eos_id:
                break
            codec_ids.append(next_token)
            generated_token_ids.append(next_token.item())
            if len(codec_ids) >= max_new_tokens:
                break

        codec_ids_tensor = torch.stack(codec_ids, dim=1)
        if len(residual_codes_list) < codec_ids_tensor.shape[1]:
            last_id_hidden = self.codec_embedding(codec_ids_tensor[:, -1:])
            residual_codes, _ = self.code_predictor.generate_residual_codes(
                hidden_states[:, -1:, :], last_id_hidden, top_k=code_top_k, top_p=code_top_p
            )
            residual_codes_list.append(residual_codes)

        residual_codes = torch.cat(residual_codes_list[: codec_ids_tensor.shape[1]], dim=1)
        codec_tensor = torch.cat([codec_ids_tensor.unsqueeze(1), residual_codes.unsqueeze(0)], dim=1)

        return TalkerOutput(
            codec_tokens=codec_tensor,
            hidden_states=hidden_states,
            logits=None,
            past_key_values=None,
        )

    def _forward_step(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
    ) -> TalkerOutput:
        """Single forward step for autoregressive generation."""
        seq_length = inputs_embeds.shape[1]

        attention_mask = _maybe_apply_causal_mask(
            attention_mask, seq_length, inputs_embeds.device, allow_none=True
        )

        if position_ids is None:
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0:
                past_len = past_key_values[0][0].shape[2]
            position_ids = _build_position_ids(
                seq_length, inputs_embeds.device, start=past_len
            )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.codec_head(hidden_states)

        return TalkerOutput(
            codec_tokens=logits.argmax(dim=-1),
            hidden_states=hidden_states,
            logits=logits,
            past_key_values=new_past_key_values if use_cache else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Load weights with simple name matching."""
        params_dict = dict(self.named_parameters())
        loaded_count = 0
        skipped_count = 0

        for name, loaded_weight in weights:
            if "thinker" in name or "code2wav" in name:
                continue

            name = name.replace("talker.", "").replace("model.", "")

            if "rotary_emb.inv_freq" in name:
                skipped_count += 1
                continue

            if name in params_dict:
                default_weight_loader(params_dict[name], loaded_weight)
                loaded_count += 1
            else:
                skipped_count += 1

        print(f"[Qwen3OmniTalker] Loaded {loaded_count} weights, skipped {skipped_count}")


# ---- Code2Wav - Vocoder ----


class Qwen3OmniCode2Wav(nn.Module):
    """Qwen3-Omni Code2Wav module - converts codec tokens to waveform (HiFi-GAN vocoder)."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        code2wav_config = config.get("code2wav_config", config)

        self.hidden_size = code2wav_config.get("hidden_size", 1024)
        self.codebook_size = code2wav_config.get("codebook_size", 2048)
        self.num_quantizers = code2wav_config.get("num_quantizers", 16)
        self.decoder_dim = code2wav_config.get("decoder_dim", 1536)
        self.upsample_rates = code2wav_config.get("upsample_rates", [8, 5, 4, 3])
        self.upsampling_ratios = code2wav_config.get("upsampling_ratios", [2, 2])
        self.sample_rate = code2wav_config.get("sample_rate", 24000)

        self.total_upsample = math.prod(self.upsample_rates + self.upsampling_ratios)

        self.code_embedding = nn.Embedding(
            self.codebook_size * self.num_quantizers, self.hidden_size
        )
        self.register_buffer(
            "code_offset",
            torch.arange(self.num_quantizers).view(1, -1, 1) * self.codebook_size,
            persistent=False,
        )

        self.pre_transformer = Code2WavPreTransformer(code2wav_config)

        upsample_blocks: list[nn.ModuleList] = []
        for factor in self.upsampling_ratios:
            upsample_blocks.append(
                nn.ModuleList([
                    CausalTransConv1d(self.hidden_size, self.hidden_size, factor, factor),
                    Code2WavConvNeXtBlock(self.hidden_size),
                ])
            )
        self.upsample = nn.ModuleList(upsample_blocks)

        decoder_layers: list[nn.Module] = []
        decoder_layers.append(CausalConv1d(self.hidden_size, self.decoder_dim, 7))

        current_dim = self.decoder_dim
        for rate in self.upsample_rates:
            out_dim = current_dim // 2
            decoder_layers.append(Code2WavDecoderBlock(current_dim, out_dim, rate))
            current_dim = out_dim

        output_dim = current_dim
        decoder_layers.append(SnakeBeta(output_dim))
        decoder_layers.append(CausalConv1d(output_dim, 1, 7))

        self.decoder = nn.ModuleList(decoder_layers)

    def forward(
        self,
        codes: torch.Tensor | None = None,
        codec_tokens: torch.Tensor | None = None,
        chunk_size: int = 300,
        left_context_size: int = 25,
        **kwargs,
    ) -> Code2WavOutput:
        """Convert codec tokens to waveform.
        
        Args:
            codes: Codec tokens tensor (legacy parameter name)
            codec_tokens: Codec tokens tensor (preferred parameter name)
            chunk_size: Size of each chunk for chunked decoding
            left_context_size: Context size for overlapping chunks
        """
        if codec_tokens is not None:
            codes = codec_tokens
        
        if codes is None:
            raise ValueError("Either 'codes' or 'codec_tokens' must be provided")
        
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)

        if codes.shape[1] != self.num_quantizers:
            codes = codes.transpose(1, 2)

        if codes.shape[-1] > chunk_size:
            waveform = self.chunked_decode(codes, chunk_size, left_context_size)
            return Code2WavOutput(waveform=waveform)

        hidden = self.code_embedding(codes + self.code_offset).mean(1)

        hidden = self.pre_transformer(hidden)
        hidden = hidden.transpose(1, 2)

        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)

        wav = hidden
        for block in self.decoder:
            wav = block(wav)

        wav = wav.clamp(min=-1, max=1)
        return Code2WavOutput(waveform=wav)

    def chunked_decode(
        self, codes: torch.Tensor, chunk_size: int = 300, left_context_size: int = 25
    ) -> torch.Tensor:
        """Decode codes in chunks with overlapping context for longer sequences."""
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = min(left_context_size, start_index)
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_output = self(codes_chunk)
            wav_chunk = wav_output.waveform
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Load weights with simple name matching."""
        params_dict = dict(self.named_parameters())
        loaded_count = 0
        skipped_count = 0

        for name, loaded_weight in weights:
            if "thinker" in name or "talker" in name:
                continue

            name = name.replace("code2wav.", "")

            if name in params_dict:
                default_weight_loader(params_dict[name], loaded_weight)
                loaded_count += 1
            else:
                skipped_count += 1

        print(f"[Qwen3OmniCode2Wav] Loaded {loaded_count} weights, skipped {skipped_count}")


# ---- Model Registry ----

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "thinker": Qwen3OmniThinker,
    "talker": Qwen3OmniTalker,
    "code2wav": Qwen3OmniCode2Wav,
    "audio_encoder": Qwen3OmniAudioEncoder,
}

WEIGHT_PREFIX: dict[str, str] = {
    "thinker": "thinker.",
    "talker": "talker.",
    "code2wav": "code2wav.",
    "audio_encoder": "thinker.audio_tower.",
}


def get_model_class(name: str) -> type[nn.Module]:
    """Get model class by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


def get_weight_prefix(name: str) -> str:
    """Get weight prefix for a model."""
    if name not in WEIGHT_PREFIX:
        raise ValueError(f"Unknown model: {name}")
    return WEIGHT_PREFIX[name]


# ---- Generation Utilities ----


def _get_rope_index(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE position indices for M-RoPE (3D positional encoding)."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    position_ids = attention_mask.float().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
    max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
    mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
    return position_ids, mrope_position_deltas


def compute_thinker_features_for_talker(
    model: Qwen3OmniThinker,
    input_ids: torch.Tensor,
    output_ids: list[int],
    config: dict[str, Any],
) -> dict[str, torch.Tensor | None]:
    """Compute hidden states and embeddings needed by Talker.

    This performs an additional forward pass through Thinker to extract:
    - thinker_embed: Input embeddings (layer 0)
    - thinker_hidden: Hidden states at accept_hidden_layer
    - multimodal_mask: Mask for multimodal tokens
    - tts_token_embeds: Embeddings for TTS special tokens

    Returns dict with tensors on CPU for transfer to Talker stage.
    """
    model_device = next(model.parameters()).device
    input_ids_device = input_ids.to(model_device)
    output_ids_tensor = torch.tensor(
        output_ids, device=model_device, dtype=input_ids_device.dtype
    )
    full_input_ids = torch.cat([input_ids_device, output_ids_tensor], dim=0).unsqueeze(0)

    talker_config = config.get("talker_config", {})
    accept_hidden_layer = talker_config.get("accept_hidden_layer", 24)

    with torch.no_grad():
        outputs = model(
            input_ids=full_input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    thinker_embed = outputs.hidden_states[0].detach()
    thinker_hidden = outputs.hidden_states[accept_hidden_layer].detach()

    thinker_cfg = config.get("thinker_config", config)
    audio_token_id = thinker_cfg.get("audio_token_id")
    image_token_id = thinker_cfg.get("image_token_id")
    video_token_id = thinker_cfg.get("video_token_id")
    multimodal_mask = torch.zeros_like(full_input_ids, dtype=torch.bool)
    for token_id in (audio_token_id, image_token_id, video_token_id):
        if token_id is not None:
            multimodal_mask |= full_input_ids == token_id

    tts_bos = config.get("tts_bos_token_id", 151672)
    tts_eos = config.get("tts_eos_token_id", 151673)
    tts_pad = config.get("tts_pad_token_id", 151671)
    tts_tokens = torch.tensor(
        [[tts_bos, tts_eos, tts_pad]], device=model_device, dtype=full_input_ids.dtype
    )
    tts_token_embeds = model.get_input_embeddings()(tts_tokens).detach()

    return {
        "full_input_ids": full_input_ids.cpu(),
        "thinker_embed": thinker_embed.cpu(),
        "thinker_hidden": thinker_hidden.cpu(),
        "multimodal_mask": multimodal_mask.cpu(),
        "tts_token_embeds": tts_token_embeds.cpu(),
    }


def run_talker_generation(
    data: dict[str, Any],
    params: dict[str, Any],
    talker: Qwen3OmniTalker,
    code2wav: Qwen3OmniCode2Wav,
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Run talker generation to produce audio codec tokens and waveform."""
    from sglang_omni.models.utils.sampling import sample_logits

    config = config or {}
    talker_config = config.get("talker_config", config)
    text_config = talker_config.get("text_config", talker_config)
    code2wav_config = config.get("code2wav_config", {})

    codec_vocab_size = text_config.get("vocab_size", 8194)
    codec_bos_id = talker_config.get("codec_bos_id", 0)
    codec_eos_id = talker_config.get("codec_eos_token_id", codec_vocab_size - 1)
    codebook_size = code2wav_config.get("codebook_size", 2048)
    suppress_tokens = [i for i in range(codebook_size, codec_vocab_size) if i != codec_eos_id]

    im_start_token_id = config.get("im_start_token_id", 151644)
    system_token_id = config.get("system_token_id", 8948)
    user_token_id = config.get("user_token_id", 872)
    assistant_token_id = config.get("assistant_token_id", 77091)
    tts_pad_token_id = config.get("tts_pad_token_id", 151671)
    tts_bos_token_id = config.get("tts_bos_token_id", 151672)
    tts_eos_token_id = config.get("tts_eos_token_id", 151673)
    speaker_map = talker_config.get("speaker_id", {})
    hidden_size = text_config.get("hidden_size", 1024)

    talker_device = next(talker.parameters()).device if talker else device
    talker_dtype = next(talker.parameters()).dtype if talker else torch.float16

    def _ensure_tensor(value: Any, dev: str, dtype: torch.dtype | None = None) -> torch.Tensor | None:
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        return value.to(device=dev, dtype=dtype) if dtype else value.to(device=dev)

    full_input_ids = data.get("full_input_ids")
    thinker_embed = data.get("thinker_embed")
    thinker_hidden = data.get("thinker_hidden")
    multimodal_mask = data.get("multimodal_mask")
    tts_token_embeds = data.get("tts_token_embeds")
    speaker = (data.get("speaker") or "Ethan").lower()
    speaker_id = speaker_map.get(speaker)

    if speaker_id is None:
        raise ValueError(f"Unknown speaker: {speaker}")
    if full_input_ids is None or thinker_embed is None or thinker_hidden is None:
        raise ValueError("Missing thinker features for talker")

    full_input_ids = _ensure_tensor(full_input_ids, device, dtype=torch.long)
    thinker_embed = _ensure_tensor(thinker_embed, device)
    thinker_hidden = _ensure_tensor(thinker_hidden, device)
    multimodal_mask = (
        _ensure_tensor(multimodal_mask, device, dtype=torch.bool)
        if multimodal_mask is not None
        else torch.zeros_like(full_input_ids, dtype=torch.bool)
    )
    tts_token_embeds = _ensure_tensor(tts_token_embeds, device)

    if tts_token_embeds is None:
        tts_tokens = torch.tensor(
            [[tts_bos_token_id, tts_eos_token_id, tts_pad_token_id]],
            device=device,
            dtype=full_input_ids.dtype,
        )
        tts_token_embeds = talker.get_input_embeddings()(tts_tokens)

    tts_bos_embed, tts_eos_embed, tts_pad_embed = (
        talker.text_projection(tts_token_embeds.to(device)).chunk(3, dim=1)
    )

    im_start_indexes = torch.nonzero(full_input_ids[0] == im_start_token_id).squeeze()
    if im_start_indexes.dim() == 0:
        im_start_indexes = im_start_indexes.unsqueeze(0)
    im_start_indexes = torch.cat(
        (im_start_indexes, torch.tensor([full_input_ids.shape[-1]], device=device, dtype=full_input_ids.dtype)),
        dim=-1,
    )

    talker_input_embeds = []
    talker_input_ids = []
    trailing_text_hidden = None

    for i in range(len(im_start_indexes) - 1):
        im_start_index = im_start_indexes[i].item()
        segment_end_index = im_start_indexes[i + 1].item()
        role_token = full_input_ids[0][im_start_index + 1].item()

        if role_token == system_token_id:
            continue
        elif role_token == user_token_id:
            user_talker_part = torch.empty(
                (1, segment_end_index - im_start_index, hidden_size),
                device=talker_device,
                dtype=talker_dtype,
            )
            user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index]
            if user_mm_mask.any():
                user_thinker_hidden_mm = thinker_hidden[:, im_start_index:segment_end_index][user_mm_mask]
                mm_hidden = talker.hidden_projection(user_thinker_hidden_mm).to(thinker_hidden.device)
                user_talker_part[user_mm_mask] = mm_hidden
            user_thinker_embed = thinker_embed[:, im_start_index:segment_end_index][~user_mm_mask]
            user_text_hidden = talker.text_projection(user_thinker_embed).to(thinker_hidden.device)
            user_talker_part[~user_mm_mask] = user_text_hidden
            talker_input_embeds.append(user_talker_part)
            talker_input_ids.append(full_input_ids[:, im_start_index:segment_end_index])

        elif role_token == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_hidden = talker.text_projection(
                thinker_embed[:, im_start_index:segment_end_index]
            ).to(tts_pad_embed.device)
            assistant_text_hidden = torch.cat(
                (assistant_hidden[:, :3], tts_pad_embed.expand(-1, 4, -1), tts_bos_embed, assistant_hidden[:, 3:4]),
                dim=1,
            )
            codec_special_tokens = torch.tensor(
                [[
                    talker_config.get("codec_nothink_id", 0),
                    talker_config.get("codec_think_bos_id", 0),
                    talker_config.get("codec_think_eos_id", 0),
                    speaker_id,
                    talker_config.get("codec_pad_id", 0),
                    codec_bos_id,
                ]],
                device=tts_pad_embed.device,
                dtype=torch.long,
            )
            assistant_codec_hidden = torch.cat(
                (
                    torch.zeros((1, 3, hidden_size), device=tts_pad_embed.device, dtype=talker_dtype),
                    talker.get_input_embeddings()(codec_special_tokens).to(tts_pad_embed.device),
                ),
                dim=1,
            )
            trailing_text_hidden = torch.cat((assistant_hidden[:, 4:], tts_eos_embed), dim=1)
            input_embeds = assistant_text_hidden + assistant_codec_hidden
            input_ids = torch.full(
                (1, assistant_text_hidden.shape[1]),
                fill_value=tts_pad_token_id,
                dtype=torch.long,
                device=assistant_text_hidden.device,
            )
            talker_input_embeds.append(input_embeds)
            talker_input_ids.append(input_ids)

    talker_input_embed = torch.cat([embed.to(device) for embed in talker_input_embeds], dim=1)
    talker_input_id = torch.cat([ids.to(device) for ids in talker_input_ids], dim=1)

    text_token_count = trailing_text_hidden.shape[1] if trailing_text_hidden is not None else 0
    max_audio_from_text = text_token_count * 10 + 200
    max_new_tokens = min(data.get("max_audio_tokens", 4096), max_audio_from_text)

    temperature = params.get("talker_temperature", 0.9)
    top_k = params.get("talker_top_k", 50)
    top_p = params.get("talker_top_p", 1.0)
    repetition_penalty = params.get("talker_repetition_penalty", 1.1)
    code_top_k = params.get("code_top_k", 50)
    code_top_p = params.get("code_top_p", 0.8)
    do_sample = temperature > 0

    attention_mask = torch.ones((1, talker_input_embed.shape[1]), device=device, dtype=torch.long)
    position_ids, rope_deltas = _get_rope_index(talker_input_id, attention_mask)

    outputs = talker(
        inputs_embeds=talker_input_embed,
        attention_mask=attention_mask,
        use_cache=True,
        position_ids=position_ids,
    )
    past_kv = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    generated_token_ids: list[int] = []

    prefill_suppress = (
        suppress_tokens + [codec_eos_id]
        if trailing_text_hidden is not None and trailing_text_hidden.shape[1] > 1
        else suppress_tokens
    )
    next_token = sample_logits(
        logits, do_sample, temperature, top_k, top_p, prefill_suppress, repetition_penalty, generated_token_ids
    )
    generated_token_ids.append(next_token.item())
    codec_ids = [next_token]
    residual_codes_list: list[torch.Tensor] = []
    generation_step = 0
    hidden_states = outputs.hidden_states

    for _ in range(max_new_tokens):
        last_id_hidden = talker.codec_embedding(next_token.view(1, 1))
        residual_codes, residual_hiddens = talker.code_predictor.generate_residual_codes(
            hidden_states[:, -1:, :], last_id_hidden, top_k=code_top_k, top_p=code_top_p
        )
        residual_codes_list.append(residual_codes)

        if residual_hiddens:
            last_group_idx = residual_codes.shape[0] - 1
            last_residual_ids = residual_codes[last_group_idx].unsqueeze(0)
            last_residual_hidden = talker.code_predictor.codec_embedding[last_group_idx](last_residual_ids)
            residual_hiddens = (
                residual_hiddens[:-1] + [last_residual_hidden]
                if len(residual_hiddens) > 1
                else [last_residual_hidden]
            )

        codec_hiddens = torch.cat([last_id_hidden] + residual_hiddens, dim=1)
        inputs_embeds = codec_hiddens.sum(1, keepdim=True)

        if trailing_text_hidden is not None and generation_step < trailing_text_hidden.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

        past_len = past_kv[0][0].shape[2] if past_kv is not None else 0
        delta = (past_len + rope_deltas).to(inputs_embeds.device)
        step_pos_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        step_pos_ids = step_pos_ids.view(1, -1).expand(delta.shape[0], -1)
        step_pos_ids = (step_pos_ids + delta).unsqueeze(0).expand(3, -1, -1)

        outputs = talker(
            inputs_embeds=inputs_embeds, past_key_values=past_kv, use_cache=True, position_ids=step_pos_ids
        )
        past_kv = outputs.past_key_values
        hidden_states = outputs.hidden_states
        logits = outputs.logits[:, -1, :]

        step_suppress = (
            suppress_tokens + [codec_eos_id]
            if trailing_text_hidden is not None and generation_step < trailing_text_hidden.shape[1] - 1
            else suppress_tokens
        )
        next_token = sample_logits(
            logits, do_sample, temperature, top_k, top_p, step_suppress, repetition_penalty, generated_token_ids
        )
        generation_step += 1

        if next_token.item() == codec_eos_id:
            break
        codec_ids.append(next_token)
        generated_token_ids.append(next_token.item())
        if len(codec_ids) >= max_new_tokens:
            break

    codec_ids = torch.stack(codec_ids, dim=1)
    if len(residual_codes_list) < codec_ids.shape[1]:
        last_id_hidden = talker.codec_embedding(codec_ids[:, -1:])
        residual_codes, _ = talker.code_predictor.generate_residual_codes(
            hidden_states[:, -1:, :], last_id_hidden, top_k=code_top_k, top_p=code_top_p
        )
        residual_codes_list.append(residual_codes)

    import numpy as np
    residual_codes = torch.cat(residual_codes_list[: codec_ids.shape[1]], dim=1)
    codec_tensor = torch.cat([codec_ids.unsqueeze(1), residual_codes.unsqueeze(0)], dim=1)

    waveform_tensor = code2wav.chunked_decode(codec_tensor, chunk_size=300, left_context_size=25)
    waveform_tensor = waveform_tensor.float().detach().squeeze()
    waveform = waveform_tensor.cpu().numpy()

    return {
        "output_ids": data.get("output_ids", []),
        "codec_tokens": codec_ids.squeeze(0).tolist(),
        "waveform": waveform,
        "sample_rate": code2wav.sample_rate,
    }


# ---- Result Builders ----


def create_thinker_result_builder(
    model: Qwen3OmniThinker | None = None,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Create result builder for Thinker stage.

    When return_audio is True, computes additional hidden states for Talker.
    Otherwise, just passes through the output_ids.
    """
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from sglang_omni.proto import StagePayload

    config = config or {}

    def result_builder(payload: "StagePayload", result: Any) -> "StagePayload":
        output_ids = list(result.output_ids)
        last_hidden_state = getattr(result, "last_hidden_state", None)
        params = payload.request.params if payload.request else {}
        return_audio = params.get("return_audio", False)

        input_ids = payload.data.get("input_ids") if isinstance(payload.data, dict) else None

        if return_audio and input_ids is not None and model is not None:
            features = compute_thinker_features_for_talker(
                model, input_ids, output_ids, config
            )
            payload.data = {
                "output_ids": output_ids,
                "last_hidden_state": last_hidden_state,
                "return_audio": return_audio,
                "speaker": params.get("speaker", "Ethan"),
                **features,
            }
        else:
            payload.data = {
                "output_ids": output_ids,
                "last_hidden_state": last_hidden_state,
                "return_audio": return_audio,
                "speaker": params.get("speaker", "Ethan"),
            }
        return payload

    return result_builder


def create_talker_result_builder(**kwargs: Any):
    """Create result builder for Talker stage."""
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from sglang_omni.proto import StagePayload

    def result_builder(payload: "StagePayload", result: Any) -> "StagePayload":
        if not isinstance(payload.data, dict):
            payload.data = {}
        
        if isinstance(result, TalkerOutput):
            payload.data["model_output"] = result
            if result.codec_tokens is not None:
                payload.data["codec_tokens"] = result.codec_tokens
        elif isinstance(result, dict):
            payload.data.update(result)
        else:
            payload.data["model_output"] = result
        return payload

    return result_builder


def create_code2wav_result_builder(**kwargs: Any):
    """Create result builder for Code2Wav stage."""
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from sglang_omni.proto import StagePayload

    def result_builder(payload: "StagePayload", result: Any) -> "StagePayload":
        if not isinstance(payload.data, dict):
            payload.data = {}
        
        if isinstance(result, Code2WavOutput):
            payload.data["model_output"] = result
            payload.data["waveform"] = result.waveform
            payload.data["sample_rate"] = 24000
        elif isinstance(result, dict):
            payload.data.update(result)
        else:
            payload.data["model_output"] = result
            if hasattr(result, "waveform"):
                payload.data["waveform"] = result.waveform
                payload.data["sample_rate"] = 24000
        return payload

    return result_builder
