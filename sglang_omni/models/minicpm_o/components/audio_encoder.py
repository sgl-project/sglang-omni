# SPDX-License-Identifier: Apache-2.0
"""Whisper audio encoding with chunked-causal attention and MiniCPM projection."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

from sglang_omni.models.minicpm_o.hf_config import MiniCPMOConfig
from sglang_omni.models.weight_loader import (
    load_weights_by_prefix,
    resolve_dtype,
    resolve_model_path,
)

# note (MayDomine): finite mask values avoid NaNs on fully masked padding rows.
MASK_MIN = -1e9

logger = logging.getLogger(__name__)
AudioKVCache = list[tuple[torch.Tensor, torch.Tensor]]

QKV_SHARDS = {"q_proj": 0, "k_proj": 1, "v_proj": 2}


def audio_config_object(config: PretrainedConfig) -> PretrainedConfig:
    audio_config = config.audio_config
    if isinstance(audio_config, dict):
        return PretrainedConfig.from_dict(audio_config)
    else:
        pass
    return audio_config


def chunked_causal_mask(
    size: int, chunk_size: int, device: torch.device
) -> torch.Tensor:
    """Allow attention within the current chunk and to all preceding chunks."""
    frame = torch.arange(size, device=device)
    visible_end = (frame // chunk_size + 1) * chunk_size
    return frame[None, :] < visible_end[:, None]


def feature_lens_after_conv(input_lengths: torch.Tensor) -> torch.Tensor:
    """Valid frame counts after the encoder's stride-2 conv2."""
    return (input_lengths - 1) // 2 + 1


def feature_lens_after_pooling(
    input_lengths: torch.Tensor, pool_step: int
) -> torch.Tensor:
    """Valid frame counts after pooling."""
    after_cnn = feature_lens_after_conv(input_lengths)
    after_pool = (after_cnn - pool_step) // pool_step + 1
    return after_pool.to(dtype=torch.int32)


def min_mel_frames(pool_step: int) -> int:
    """Fewest mel frames the pooling stage accepts (one pooled frame)."""
    return 2 * pool_step - 1


class MiniCPMWhisperEncoderAttention(nn.Module):
    """Whisper self-attention with fused QKV and an additive SDPA mask."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        # note (MayDomine): Whisper's K projection is bias-free.
        with torch.no_grad():
            self.qkv_proj.bias[self.embed_dim : 2 * self.embed_dim].zero_()
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def reshape_heads(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        return states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_values: AudioKVCache | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        key, value = self.reshape_heads(key), self.reshape_heads(value)
        if past_key_values is not None:
            if layer_index < len(past_key_values):
                previous_key, previous_value = past_key_values[layer_index]
                key = torch.cat((previous_key, key), dim=2)
                value = torch.cat((previous_value, value), dim=2)
                past_key_values[layer_index] = (key, value)
            else:
                past_key_values.append((key, value))
        else:
            pass
        attn_output = F.scaled_dot_product_attention(
            self.reshape_heads(query),
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.embed_dim,
        )
        return self.out_proj(attn_output)


class MiniCPMWhisperEncoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.self_attn = MiniCPMWhisperEncoderAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_values: AudioKVCache | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attn_mask, past_key_values, layer_index
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return residual + hidden_states


class MiniCPMWhisperEncoder(nn.Module):
    """Standard Whisper encoder stack driven by an external additive mask."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            config.num_mel_bins,
            config.d_model,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = nn.ModuleList(
            [MiniCPMWhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_features: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_values: AudioKVCache | None = None,
        prefix_extra_frames: int = 0,
        suffix_extra_frames: int = 0,
    ) -> torch.Tensor:
        hidden_states = input_features.to(
            device=self.conv1.weight.device, dtype=self.conv1.weight.dtype
        )
        hidden_states = F.gelu(self.conv1(hidden_states))
        hidden_states = F.gelu(self.conv2(hidden_states))
        # note (Junnan Li): Context mel frames affect convolution, but never enter KV.
        prefix_rows = (prefix_extra_frames + 1) // 2
        suffix_rows = (suffix_extra_frames + 1) // 2
        hidden_states = hidden_states[:, :, prefix_rows:]
        if suffix_rows:
            hidden_states = hidden_states[:, :, :-suffix_rows]
        else:
            pass
        hidden_states = hidden_states.permute(0, 2, 1)

        position = past_key_values[0][0].shape[2] if past_key_values else 0
        embed_pos = self.embed_positions.weight[
            position : position + hidden_states.shape[1]
        ]
        hidden_states = hidden_states + embed_pos.to(hidden_states.device)

        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, attn_mask, past_key_values, layer_index
            )
        return self.layer_norm(hidden_states)


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=True)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(audio_features)))


def fuse_qkv(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fuse QKV checkpoint projections, filling the absent K bias with zeros."""
    fused: dict[str, torch.Tensor] = {}
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in state_dict.items():
        stem, _, leaf = name.rpartition(".")
        base, _, projection = stem.rpartition(".")
        if projection in QKV_SHARDS and base.endswith("self_attn"):
            pending.setdefault(f"{base}.qkv_proj.{leaf}", {})[projection] = tensor
        else:
            fused[name] = tensor
    for target, shards in pending.items():
        if target.endswith(".bias") and "k_proj" not in shards:
            shards["k_proj"] = torch.zeros_like(shards["q_proj"])
        else:
            pass
        fused[target] = torch.cat(
            [shards["q_proj"], shards["k_proj"], shards["v_proj"]], dim=0
        )
    return fused


class MiniCPMOAudioEncoder(nn.Module):
    """Encode and pool audio features into thinker embeddings."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        model_dir = str(resolve_model_path(model_path))
        config = MiniCPMOConfig.from_pretrained(model_dir)
        self.device = torch.device(device)
        self.dtype = torch_dtype

        audio_config = audio_config_object(config)
        self.apm = MiniCPMWhisperEncoder(audio_config)
        apm_state = fuse_qkv(load_weights_by_prefix(model_dir, prefix=("apm.",)))
        self.apm.load_state_dict(apm_state, strict=True)

        projector = MultiModalProjector(
            in_dim=int(audio_config.d_model), out_dim=int(config.hidden_size)
        )
        projector.load_state_dict(
            load_weights_by_prefix(model_dir, prefix=("audio_projection_layer.",)),
            strict=True,
        )
        self.audio_projection_layer = projector

        self.eval()
        self.to(device=self.device, dtype=torch_dtype)

        self.audio_pool_step = int(config.audio_pool_step)
        self.audio_avg_pooler = nn.AvgPool1d(
            self.audio_pool_step, stride=self.audio_pool_step
        )
        # note (MayDomine): stride-2 convolution yields 50 frames per second.
        self.chunk_num_frame = int(float(config.audio_chunk_length) * 50)
        self.chunk_mask_cache: tuple[int, torch.Tensor] | None = None

    def cached_chunk_mask(self, size: int) -> torch.Tensor:
        if self.chunk_mask_cache is None or self.chunk_mask_cache[0] != size:
            self.chunk_mask_cache = (
                size,
                chunked_causal_mask(size, self.chunk_num_frame, self.device),
            )
        else:
            pass
        return self.chunk_mask_cache[1]

    @torch.no_grad()
    def forward(
        self,
        *,
        audio_features: torch.Tensor | None = None,
        audio_feature_lens: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Return (sum(pooled_lens), hidden) embeddings in audio-chunk order."""
        if (
            audio_features is None
            or audio_features.numel() == 0
            or audio_feature_lens is None
        ):
            return {}
        else:
            pass
        wavforms = audio_features.to(self.device, dtype=self.dtype)
        lens_cpu = audio_feature_lens.to("cpu")
        lens = audio_feature_lens.to(self.device)

        # note (wenyao): a short trailing segment contributes zero pooled tokens.
        frame_limit = min_mel_frames(self.audio_pool_step)
        if int(lens_cpu.max()) < frame_limit:
            shortest = int(lens_cpu.min())
            raise ValueError(
                f"MiniCPM-o accepts audio up to {frame_limit} mel frames "
                f"minimum, but the shortest segment has only {shortest}; "
                "send a longer clip"
            )
        else:
            pass

        _, _, max_mel_seq_len = wavforms.shape
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1

        # note (MayDomine): convolution sees padding before the attention mask does.
        mel_range = torch.arange(max_mel_seq_len, device=self.device)
        wavforms = wavforms.masked_fill(
            mel_range[None, None, :] >= lens[:, None, None], 0.0
        )

        # note (MayDomine): validity lengths must account for convolution stride.
        seq_range = torch.arange(max_seq_len, device=self.device)
        lens_after_conv = feature_lens_after_conv(lens)
        valid = seq_range[None, :] < lens_after_conv[:, None]
        allowed = self.cached_chunk_mask(max_seq_len)[None, :, :] & valid[:, None, :]
        attn_mask = torch.where(allowed, 0.0, MASK_MIN).to(self.dtype)
        attn_mask = attn_mask.unsqueeze(1)

        audio_states = self.apm(wavforms, attn_mask)
        audio_embeds = self.audio_projection_layer(audio_states)

        audio_embeds = audio_embeds.transpose(1, 2)
        audio_embeds = self.audio_avg_pooler(audio_embeds)
        audio_embeds = audio_embeds.transpose(1, 2)

        # note (MayDomine): host-side lengths avoid per-sample device synchronization.
        pooled_lens = feature_lens_after_pooling(lens_cpu, self.audio_pool_step)
        pool_range = torch.arange(audio_embeds.shape[1], device=self.device)
        keep = pool_range[None, :] < pooled_lens.to(self.device)[:, None]
        return {"audio_embeds": audio_embeds[keep]}

    @torch.no_grad()
    def forward_streaming(
        self,
        audio_features: torch.Tensor,
        audio_feature_lens: torch.Tensor,
        past_key_values: AudioKVCache | None,
        prefix_extra_frames: int,
        suffix_extra_frames: int,
    ) -> tuple[torch.Tensor, AudioKVCache]:
        """Encode one mel chunk while retaining only its non-context attention KV."""
        if audio_features.shape[0] != 1:
            raise ValueError("streaming audio encoding supports batch_size=1")
        else:
            pass
        cache = [] if past_key_values is None else past_key_values
        past_length = cache[0][0].shape[2] if cache else 0
        convolution_length = (audio_features.shape[-1] + 1) // 2
        if past_length + convolution_length >= self.apm.embed_positions.num_embeddings:
            logger.warning(f"Resetting audio encoder KV at {past_length} frames")
            cache = []
            past_length = 0
        else:
            pass
        current_length = (
            convolution_length
            - (prefix_extra_frames + 1) // 2
            - (suffix_extra_frames + 1) // 2
        )
        if current_length < self.audio_pool_step:
            raise ValueError("streaming audio chunk is too short after context removal")
        else:
            pass
        attention_mask = torch.zeros(
            (1, 1, current_length, past_length + current_length),
            dtype=self.dtype,
            device=self.device,
        )
        states = self.apm(
            audio_features,
            attention_mask,
            cache,
            prefix_extra_frames,
            suffix_extra_frames,
        )
        embeds = self.audio_projection_layer(states)
        embeds = self.audio_avg_pooler(embeds.transpose(1, 2)).transpose(1, 2)
        pooled_length = feature_lens_after_pooling(
            audio_feature_lens, self.audio_pool_step
        ).item()
        return embeds[0, :pooled_length], cache
