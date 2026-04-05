# SPDX-License-Identifier: Apache-2.0
"""Audio encoder component for Ming-Omni."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.models.ming_omni.components.common import load_ming_config
from sglang_omni.models.weight_loader import load_module, resolve_dtype

logger = logging.getLogger(__name__)

# Weight prefixes in the Ming-Omni checkpoint
AUDIO_TOWER_PREFIXES = ("audio.",)
AUDIO_PROJ_PREFIXES = ("linear_proj_audio.",)


class Transpose(nn.Module):
    """Helper module to transpose tensor dimensions."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class MingAudioEncoder(nn.Module):
    """Whisper-based audio encoder with Conv1d projection for Ming-Omni.

    Takes mel-spectrogram features and produces embeddings compatible with
    the BailingMoeV2 LLM hidden size (4096).

    Architecture:
        mel [B, T, 128] -> WhisperEncoder [B, T, 1280]
        -> Conv1d(1280, 4096, k=3, s=2) + MLP -> [B, T', 4096]
        -> F.normalize(dim=-1)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str | None = None,
    ):
        super().__init__()
        self._model_path = model_path
        self._device = device
        self._dtype = resolve_dtype(dtype) if dtype else torch.bfloat16

        config = load_ming_config(model_path)
        audio_cfg = config.audio_config
        whisper_cfg = audio_cfg.whisper_encoder_config
        llm_hidden_size = config.llm_config.hidden_size

        # Build whisper encoder
        self.audio_tower = self._build_whisper_encoder(whisper_cfg)

        # Build projection: Conv1d + Transpose + MLP layers
        audio_encoder_proj = nn.Conv1d(
            whisper_cfg.n_state,
            llm_hidden_size,
            kernel_size=audio_cfg.ds_kernel_size,
            stride=audio_cfg.ds_stride,
            padding=audio_cfg.ds_kernel_size // 2,
        )

        mlp_modules: list[nn.Module] = [audio_encoder_proj, Transpose(-1, -2)]
        for _ in range(1, config.mlp_depth):
            mlp_modules.append(nn.GELU())
            mlp_modules.append(nn.Linear(llm_hidden_size, llm_hidden_size))
        mlp_modules.append(Transpose(-1, -2))
        self.projection = nn.Sequential(*mlp_modules)

        self._norm_query_embeds = audio_cfg.norm_query_embeds

        # Load weights
        self._load_weights()

        # Move to target device/dtype
        self.to(device=self._device, dtype=self._dtype)
        self.eval()

    def _build_whisper_encoder(self, whisper_cfg: Any) -> nn.Module:
        """Build a WhisperAudioEncoder from config."""
        try:
            from whisper.model import AudioEncoder

            encoder = AudioEncoder(
                n_mels=whisper_cfg.n_mels,
                n_ctx=whisper_cfg.n_ctx,
                n_state=whisper_cfg.n_state,
                n_head=whisper_cfg.n_head,
                n_layer=whisper_cfg.n_layer,
            )
            return encoder
        except ImportError:
            raise ImportError(
                "whisper package is required for Ming-Omni. "
                "Install with: pip install openai-whisper"
            )

    def _load_weights(self) -> None:
        """Load audio tower and projection weights from checkpoint."""
        logger.info("Loading Ming audio encoder weights from %s", self._model_path)
        load_module(
            self.audio_tower,
            self._model_path,
            prefix=AUDIO_TOWER_PREFIXES,
            dtype=self._dtype,
            device="cpu",
        )
        load_module(
            self.projection,
            self._model_path,
            prefix=AUDIO_PROJ_PREFIXES,
            dtype=self._dtype,
            device="cpu",
        )

    def forward(
        self,
        audio_feats: torch.Tensor,
        audio_feats_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode mel-spectrogram features to LLM-compatible embeddings.

        Args:
            audio_feats: Mel features [B, T, n_mels] (concatenated segments per batch).
            audio_feats_lengths: Segment lengths [B, N] for unwrapping.

        Returns:
            Dict with:
                audio_embeds: Projected embeddings [B, T', hidden_size]
                audio_embed_lengths: Output lengths [B, N]
        """
        # Whisper encoder expects [B, T, n_mels] and we process segments independently
        # Unwrap segments for per-segment encoding
        if audio_feats_lengths is not None and audio_feats_lengths.dim() == 2:
            # Unwrap batch-concatenated format
            segments, seg_lengths = self._unwrap_feats(audio_feats, audio_feats_lengths)
        else:
            segments = audio_feats
            seg_lengths = torch.tensor(
                [audio_feats.shape[1]], device=audio_feats.device, dtype=torch.long
            )

        with torch.no_grad():
            # Whisper encoder forward: [B, T, n_mels] -> [B, T, n_state]
            # The whisper encoder expects [B, T, n_mels], transposes internally
            encoded = self._whisper_forward(segments)

            # Project: [B, n_state, T] -> Conv1d -> [B, hidden, T'] -> transpose -> [B, T', hidden]
            projected = self.projection(encoded.transpose(-1, -2)).transpose(-1, -2)

            # Compute output lengths after Conv1d downsampling
            config = load_ming_config(self._model_path)
            audio_cfg = config.audio_config
            out_lengths = self._compute_output_lengths(
                seg_lengths,
                audio_cfg.ds_kernel_size,
                audio_cfg.ds_stride,
            )

            # Normalize if configured
            if self._norm_query_embeds:
                projected = F.normalize(projected, dim=-1)

        # Re-wrap to batch-concatenated format if needed
        if audio_feats_lengths is not None and audio_feats_lengths.dim() == 2:
            projected, _, out_lengths = self._wrap_feats(
                projected, audio_feats_lengths, out_lengths
            )

        return {
            "audio_embeds": projected,
            "audio_embed_lengths": out_lengths,
        }

    def _whisper_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run whisper encoder forward pass with variable-length support."""
        # x: [B, T, n_mels]
        x = x.transpose(1, 2)  # [B, n_mels, T]
        x = F.gelu(self.audio_tower.conv1(x))
        x = F.gelu(self.audio_tower.conv2(x))
        x = x.permute(0, 2, 1)  # [B, T', n_state]
        # Add positional embedding (variable length)
        pos_emb = self.audio_tower.positional_embedding[: x.shape[1], :]
        x = (x + pos_emb).to(x.dtype)
        for block in self.audio_tower.blocks:
            x = block(x)
        x = self.audio_tower.ln_post(x)
        return x

    @staticmethod
    def _compute_output_lengths(
        input_lengths: torch.Tensor,
        kernel_size: int,
        stride: int,
    ) -> torch.Tensor:
        """Compute output sequence lengths after Conv1d projection."""
        padding = kernel_size // 2
        return (input_lengths - kernel_size + 2 * padding) // stride + 1

    @staticmethod
    def _unwrap_feats(
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unwrap batch-concatenated features into per-segment rows."""
        feat_segs = []
        feat_seg_lengths = []
        for i in range(feats_lengths.shape[0]):
            feat_index = 0
            for j in range(feats_lengths.shape[1]):
                feat_len = feats_lengths[i, j].item()
                if feat_len == 0:
                    break
                feat_segs.append(feats[i, feat_index : feat_index + feat_len])
                feat_seg_lengths.append(feat_len)
                feat_index += feat_len
        feat_segs_batch = torch.nn.utils.rnn.pad_sequence(feat_segs, batch_first=True)
        feat_seg_lengths_t = torch.tensor(
            feat_seg_lengths, dtype=torch.long, device=feats.device
        )
        return feat_segs_batch.to(feats.device), feat_seg_lengths_t

    @staticmethod
    def _wrap_feats(
        feat_segs: torch.Tensor,
        feats_lengths: torch.Tensor,
        feats_seg_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Wrap per-segment features back to batch-concatenated format."""
        feat_idx = 0
        feats_buffer = []
        feats_lengths_buffer = []
        for i in range(feats_lengths.shape[0]):
            feat_buffer = []
            feat_lengths_buffer = []
            for j in range(feats_lengths.shape[1]):
                feat_len = feats_lengths[i, j].item()
                if feat_len == 0:
                    break
                out_len = feats_seg_lengths[feat_idx].item()
                feat_buffer.append(feat_segs[feat_idx, :out_len])
                feat_lengths_buffer.append(out_len)
                feat_idx += 1
            feats_buffer.append(torch.cat(feat_buffer))
            feats_lengths_buffer.append(
                torch.tensor(feat_lengths_buffer, dtype=torch.long)
            )
        result = torch.nn.utils.rnn.pad_sequence(feats_buffer, batch_first=True)
        new_lengths = torch.nn.utils.rnn.pad_sequence(
            feats_lengths_buffer, batch_first=True
        )
        return (
            result.to(feat_segs.device),
            None,
            new_lengths.to(feats_lengths.device),
        )
