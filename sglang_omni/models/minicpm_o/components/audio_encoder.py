# SPDX-License-Identifier: Apache-2.0
"""Audio encoder component for MiniCPM-o.

Wraps the checkpoint's remote-code whisper encoder (``apm.``) and projection
(``audio_projection_layer.``). The forward pass mirrors the remote code's
``get_audio_embedding`` generate path: additive attention mask with a chunked
causal mask (``audio_chunk_length``), last hidden state → projection → avg
pool, then per-audio trimming to the pooled feature lengths.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang_omni.models.weight_loader import (
    load_module,
    resolve_dtype,
    resolve_model_path,
)

logger = logging.getLogger(__name__)


def _subsequent_chunk_mask(
    size: int, chunk_size: int, device: torch.device
) -> torch.Tensor:
    """Chunked-causal mask: each frame attends to all frames up to the end of
    its own chunk (streaming whisper convention from the remote code)."""
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, :ending] = True
    return ret


def _shim_whisper_attention_returns(apm: nn.Module) -> None:
    """Pad WhisperAttention returns to the 3-tuple the remote code unpacks.

    The remote ``MiniCPMWhisperEncoderLayer`` was written against transformers
    <=4.x where attention returned ``(out, weights, past_kv)``; v5 returns
    ``(out, weights)``.
    """
    for layer in apm.layers:
        attn = layer.self_attn
        orig_forward = attn.forward

        def _forward(*args, _orig=orig_forward, **kwargs):
            out = _orig(*args, **kwargs)
            if isinstance(out, tuple) and len(out) == 2:
                return (*out, None)
            return out

        attn.forward = _forward


class MiniCPMOAudioEncoder(nn.Module):
    """Whisper encoder + projection extracted from the remote code."""

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

        # Mirrors the remote code's init_audio_module. FA2 breaks the
        # cu_seqlens path in this encoder, so it pins sdpa.
        whisper_cls = get_class_from_dynamic_module(
            "modeling_minicpmo.MiniCPMWhisperEncoder", model_dir
        )
        projector_cls = get_class_from_dynamic_module(
            "modeling_minicpmo.MultiModalProjector", model_dir
        )

        audio_config = config.audio_config
        audio_config._attn_implementation = "sdpa"
        apm = whisper_cls(audio_config)
        _shim_whisper_attention_returns(apm)
        self.apm = load_module(
            apm, model_dir, prefix=("apm.",), dtype=torch_dtype, device=device
        )

        audio_output_dim = int(audio_config.encoder_ffn_dim // 4)
        projector = projector_cls(in_dim=audio_output_dim, out_dim=config.hidden_size)
        self.audio_projection_layer = load_module(
            projector,
            model_dir,
            prefix=("audio_projection_layer.",),
            dtype=torch_dtype,
            device=device,
        )

        self.audio_pool_step = int(config.audio_pool_step)
        self.audio_avg_pooler = nn.AvgPool1d(
            self.audio_pool_step, stride=self.audio_pool_step
        )
        self.audio_chunk_length = float(config.audio_chunk_length)
        self.audio_encoder_layer = -1

    def _feature_lens_after_pooling(self, input_lengths: torch.Tensor) -> torch.Tensor:
        after_cnn = (input_lengths - 1) // 2 + 1
        after_pool = (after_cnn - self.audio_pool_step) // self.audio_pool_step + 1
        return after_pool.to(dtype=torch.int32)

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
        if audio_features is None or audio_features.numel() == 0:
            return {}
        wavforms = audio_features.to(self._device, dtype=self._dtype)
        audio_feature_lens = audio_feature_lens.to(self._device)

        batch_size, _, max_mel_seq_len = wavforms.shape
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1

        seq_range = (
            torch.arange(0, max_seq_len, device=self._device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
        padding_mask = seq_range >= lengths_expand
        mask_bool = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )

        # Generate path uses chunked attention (audio_chunk_length seconds,
        # 50 frames/sec after the conv downsample).
        chunk_num_frame = int(self.audio_chunk_length * 50)
        chunk_mask = _subsequent_chunk_mask(max_seq_len, chunk_num_frame, self._device)
        mask_bool = torch.logical_or(mask_bool, torch.logical_not(chunk_mask))

        audio_attention_mask = torch.zeros(
            mask_bool.shape, dtype=self._dtype, device=self._device
        )
        audio_attention_mask[mask_bool] = float("-inf")

        audio_states = self.apm(
            wavforms, output_hidden_states=True, attention_mask=audio_attention_mask
        ).hidden_states[self.audio_encoder_layer]
        audio_embeds = self.audio_projection_layer(audio_states)

        audio_embeds = audio_embeds.transpose(1, 2)
        audio_embeds = self.audio_avg_pooler(audio_embeds)
        audio_embeds = audio_embeds.transpose(1, 2)

        pooled_lens = self._feature_lens_after_pooling(audio_feature_lens)
        flat = torch.cat(
            [audio_embeds[i, : pooled_lens[i], :] for i in range(batch_size)], dim=0
        )
        return {"audio_embeds": flat}
