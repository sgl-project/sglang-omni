# SPDX-License-Identifier: Apache-2.0
"""Audio encoder component for MiniCPM-o.

Native Whisper encoder variant (structure follows
``sglang_omni.models.whisper_asr.sglang_model.WhisperEncoder``) plus the
MiniCPM audio projection. The only semantic difference from a standard
Whisper encoder is the attention mask: a chunked-causal mask (each frame
attends to every frame up to the end of its own chunk) combined with the
variable-length padding mask, applied as an additive SDPA mask. The encoder
output goes through a two-layer projection (``audio_projection_layer.``),
average pooling, then per-audio trimming to the pooled feature lengths.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PretrainedConfig
from transformers.activations import ACT2FN

from sglang_omni.models.weight_loader import (
    load_weights_by_prefix,
    resolve_dtype,
    resolve_model_path,
)

logger = logging.getLogger(__name__)

# Additive-mask fill value. A large-but-finite negative keeps fully masked
# (padding) rows NaN-free through softmax; -inf rows are kernel-dependent.
_MASK_MIN = -1e9

_QKV_SHARDS = {"q_proj": 0, "k_proj": 1, "v_proj": 2}


def _audio_config_object(config: PretrainedConfig) -> PretrainedConfig:
    audio_config = config.audio_config
    if isinstance(audio_config, dict):
        return PretrainedConfig.from_dict(audio_config)
    return audio_config


def _chunked_causal_mask(
    size: int, chunk_size: int, device: torch.device
) -> torch.Tensor:
    """Boolean ``(size, size)`` mask where frame ``i`` attends to
    ``[0, (i // chunk_size + 1) * chunk_size)``: bidirectional within its own
    chunk plus every preceding chunk (streaming whisper convention)."""
    frame = torch.arange(size, device=device)
    visible_end = (frame // chunk_size + 1) * chunk_size
    return frame[None, :] < visible_end[:, None]


class MiniCPMWhisperEncoderAttention(nn.Module):
    """Whisper encoder self-attention with a fused qkv projection and an
    additive SDPA mask (the chunked-causal + padding mask)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        # Whisper K projections have no bias. The zero K shard preserves that
        # checkpoint structure while issuing one GEMM for all three projections.
        with torch.no_grad():
            self.qkv_proj.bias[self.embed_dim : 2 * self.embed_dim].zero_()
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        return states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self, hidden_states: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        # TODO(perf): replace the dense additive mask with a maskless varlen
        # backend. The chunked-causal mask decomposes into per-chunk
        # bidirectional varlen attention (query chunk j attends kv [0,
        # (j+1)*chunk), causal=False), so FA2 varlen / FlexAttention
        # (mask_mod: kv_idx < (q_idx // chunk + 1) * chunk) / flashinfer
        # ragged prefill can all express it without materializing (B,1,T,T).
        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        attn_output = F.scaled_dot_product_attention(
            self._shape(query),
            self._shape(key),
            self._shape(value),
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
    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = MiniCPMWhisperEncoderAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]

    def forward(
        self, hidden_states: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return residual + hidden_states


class MiniCPMWhisperEncoder(nn.Module):
    """Standard Whisper encoder stack driven by an external additive mask."""

    def __init__(self, config) -> None:
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
        self, input_features: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = input_features.to(
            device=self.conv1.weight.device, dtype=self.conv1.weight.dtype
        )
        hidden_states = F.gelu(self.conv1(hidden_states))
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight[: hidden_states.shape[1]]
        hidden_states = hidden_states + embed_pos.to(hidden_states.device)

        # TODO(perf): no CUDA graph yet. whisper_asr has a
        # WhisperEncoderCudaGraphRunner precedent; variable-length mels can be
        # bucketed by (batch, padded T) since the mask keeps padding correct.
        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask)
        return self.layer_norm(hidden_states)


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=True)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(audio_features)))


def _fuse_qkv(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fuse per-layer ``{q,k,v}_proj`` checkpoint weights into ``qkv_proj``.

    Whisper K projections ship without bias; the K bias shard stays at its
    zero initialization.
    """
    fused: dict[str, torch.Tensor] = {}
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in state_dict.items():
        stem, _, leaf = name.rpartition(".")
        base, _, projection = stem.rpartition(".")
        if projection in _QKV_SHARDS and base.endswith("self_attn"):
            pending.setdefault(f"{base}.qkv_proj.{leaf}", {})[projection] = tensor
        else:
            fused[name] = tensor
    for target, shards in pending.items():
        if target.endswith(".bias") and "k_proj" not in shards:
            shards["k_proj"] = torch.zeros_like(shards["q_proj"])
        fused[target] = torch.cat(
            [shards["q_proj"], shards["k_proj"], shards["v_proj"]], dim=0
        )
    return fused


class MiniCPMOAudioEncoder(nn.Module):
    """Native whisper encoder (``apm.``) + projection
    (``audio_projection_layer.``)."""

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
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self._device = torch.device(device)
        self._dtype = torch_dtype

        audio_config = _audio_config_object(config)
        self.apm = MiniCPMWhisperEncoder(audio_config)
        apm_state = _fuse_qkv(load_weights_by_prefix(model_dir, prefix=("apm.",)))
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
        self.to(device=self._device, dtype=torch_dtype)

        self.audio_pool_step = int(config.audio_pool_step)
        self.audio_avg_pooler = nn.AvgPool1d(
            self.audio_pool_step, stride=self.audio_pool_step
        )
        # Generate path uses chunked attention (audio_chunk_length seconds,
        # 50 frames/sec after the conv downsample).
        self.chunk_num_frame = int(float(config.audio_chunk_length) * 50)
        self._chunk_mask_cache: tuple[int, torch.Tensor] | None = None

    def _feature_lens_after_pooling(self, input_lengths: torch.Tensor) -> torch.Tensor:
        after_cnn = (input_lengths - 1) // 2 + 1
        after_pool = (after_cnn - self.audio_pool_step) // self.audio_pool_step + 1
        return after_pool.to(dtype=torch.int32)

    def _cached_chunk_mask(self, size: int) -> torch.Tensor:
        if self._chunk_mask_cache is None or self._chunk_mask_cache[0] != size:
            self._chunk_mask_cache = (
                size,
                _chunked_causal_mask(size, self.chunk_num_frame, self._device),
            )
        return self._chunk_mask_cache[1]

    @torch.no_grad()
    def forward(
        self,
        *,
        audio_features: torch.Tensor | None = None,
        audio_feature_lens: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of mel spectrograms.

        Args:
            audio_features: ``(num_chunks, 80, max_mel_len)`` mel features.
            audio_feature_lens: ``(num_chunks,)`` valid mel lengths.

        Returns:
            ``audio_embeds``: flat ``(sum(pooled_lens), hidden)`` rows in chunk
            order, matching the placeholder token layout.
        """
        if (
            audio_features is None
            or audio_features.numel() == 0
            or audio_feature_lens is None
        ):
            return {}
        wavforms = audio_features.to(self._device, dtype=self._dtype)
        lens_cpu = audio_feature_lens.to("cpu")
        lens = audio_feature_lens.to(self._device)

        _, _, max_mel_seq_len = wavforms.shape
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1

        seq_range = torch.arange(max_seq_len, device=self._device)
        valid = seq_range[None, :] < lens[:, None]  # (B, T) key validity
        allowed = self._cached_chunk_mask(max_seq_len)[None, :, :] & valid[:, None, :]
        attn_mask = torch.where(allowed, 0.0, _MASK_MIN).to(self._dtype)
        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T, T)

        audio_states = self.apm(wavforms, attn_mask)
        audio_embeds = self.audio_projection_layer(audio_states)

        audio_embeds = audio_embeds.transpose(1, 2)
        audio_embeds = self.audio_avg_pooler(audio_embeds)
        audio_embeds = audio_embeds.transpose(1, 2)

        # Trim each chunk to its pooled length in one masked select; lengths
        # stay host-side so no per-sample GPU→CPU sync is needed.
        pooled_lens = self._feature_lens_after_pooling(lens_cpu)
        pool_range = torch.arange(audio_embeds.shape[1], device=self._device)
        keep = pool_range[None, :] < pooled_lens.to(self._device)[:, None]
        return {"audio_embeds": audio_embeds[keep]}
