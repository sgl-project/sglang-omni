# SPDX-License-Identifier: Apache-2.0
"""Whisper audio encoding with chunked-causal attention and MiniCPM projection."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang_omni.models.minicpm_o.components.whisper_encoder import (
    AudioEncoderState,
    MiniCPMWhisperEncoder,
)
from sglang_omni.models.minicpm_o.hf_config import MiniCPMOConfig
from sglang_omni.models.weight_loader import (
    load_weights_by_prefix,
    resolve_dtype,
    resolve_model_path,
)

# note (MayDomine): finite mask values avoid NaNs on fully masked padding rows.
MASK_MIN = -1e9

logger = logging.getLogger(__name__)

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

        audio_states, _ = self.apm(wavforms, attn_mask)
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
        state: AudioEncoderState | None = None,
        prefix_extra_frames: int = 0,
        suffix_extra_frames: int = 0,
    ) -> tuple[torch.Tensor, AudioEncoderState]:
        """Encode one mel chunk while retaining only its non-context attention KV."""
        if audio_features.shape[0] != 1:
            raise ValueError("streaming audio encoding supports batch_size=1")
        else:
            pass
        state = AudioEncoderState() if state is None else state
        past_length = state.past_length
        convolution_length = (audio_features.shape[-1] + 1) // 2
        if past_length + convolution_length >= self.apm.embed_positions.num_embeddings:
            logger.warning(f"Resetting audio encoder KV at {past_length} frames")
            state = AudioEncoderState()
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
        states, new_state = self.apm(
            audio_features,
            attention_mask,
            state,
            prefix_extra_frames,
            suffix_extra_frames,
        )
        embeds = self.audio_projection_layer(states)
        embeds = self.audio_avg_pooler(embeds.transpose(1, 2)).transpose(1, 2)
        pooled_length = feature_lens_after_pooling(
            audio_feature_lens, self.audio_pool_step
        ).item()
        assert new_state is not None
        return embeds[0, :pooled_length], new_state
