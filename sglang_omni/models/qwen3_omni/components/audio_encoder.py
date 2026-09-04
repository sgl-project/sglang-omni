# SPDX-License-Identifier: Apache-2.0
"""Audio encoder component for Qwen3-Omni."""

from __future__ import annotations

import logging
from types import MethodType

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.audio_layer_graph import (
    AudioLayerGraphRunner,
)
from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.weight_loader import load_module, resolve_dtype
from sglang_omni.utils import instantiate_module

logger = logging.getLogger(__name__)

AUDIO_TOWER_PREFIX = ("thinker.audio_tower.", "audio_tower.")
AUDIO_TOWER_CLASS = hf_modeling.Qwen3OmniMoeAudioEncoder


def _build_audio_tower(
    model_path: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    audio_cfg = thinker_cfg.audio_config
    audio_tower = instantiate_module(AUDIO_TOWER_CLASS, audio_cfg)
    return load_module(
        audio_tower,
        model_path,
        prefix=AUDIO_TOWER_PREFIX,
        dtype=torch_dtype,
        device=device,
        strict=True,
    )


def pack_padded_audio_features(
    input_features: torch.Tensor,
    feature_attention_mask: torch.Tensor,
    audio_feature_lengths: torch.Tensor,
) -> torch.Tensor:
    """Concatenate each row's valid frames along time, giving ``[mel, sum(len)]``.

    Note (wenyao): both mask sources here are prefix masks, which is what makes
    the slice path valid; an interior hole would need the gather instead.
    """
    mask = feature_attention_mask.bool()
    lengths = audio_feature_lengths.to(torch.long).view(-1)
    steps = torch.arange(mask.shape[-1], device=mask.device)
    if not torch.equal(mask, steps.unsqueeze(0) < lengths.unsqueeze(1)):
        return (
            input_features.permute(0, 2, 1)[mask.to(input_features.device)]
            .permute(1, 0)
            .contiguous()
        )

    return torch.cat(
        [row[:, :length] for row, length in zip(input_features, lengths.tolist())],
        dim=-1,
    ).contiguous()


class _SegmentSplits:
    """Per-request attention segment sizes, shared by every encoder layer."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: list[int] | None = None


def _forward_with_shared_segments(self, hidden_states, cu_seqlens, **kwargs):
    splits = self._omni_segment_splits.value
    if splits is None or sum(splits) != hidden_states.shape[0]:
        # Note (wenyao): a stale or mismatched split would silently corrupt
        # attention rather than fail, so fall back instead of trusting it.
        return self._omni_unshared_forward(hidden_states, cu_seqlens, **kwargs)

    seq_length, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface = hf_modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, hf_modeling.eager_attention_forward
    )
    qkv_splits = [
        torch.split(tensor, splits, dim=2)
        for tensor in (query_states, key_states, value_states)
    ]
    attn_outputs = [
        attention_interface(
            self,
            q,
            k,
            v,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )[0]
        for q, k, v in zip(*qkv_splits)
    ]
    attn_output = torch.cat(attn_outputs, dim=1).reshape(seq_length, -1).contiguous()
    return self.out_proj(attn_output)


def _share_segment_splits(tower: nn.Module, splits: _SegmentSplits) -> None:
    # Note (wenyao): the stock attention derives its split sizes with a
    # device-to-host copy, so every one of the 32 layers stalls on the same
    # value; that sync is also what makes the stack uncapturable.
    for layer in tower.layers:
        attention = layer.self_attn
        attention._omni_segment_splits = splits
        attention._omni_unshared_forward = attention.forward
        attention.forward = MethodType(_forward_with_shared_segments, attention)


class _GraphedLayerStack(nn.Module):
    """Stands in for the whole layer list so the tower's loop runs once."""

    def __init__(self, layers: nn.ModuleList, runner, splits: _SegmentSplits) -> None:
        super().__init__()
        self._layers = layers
        self._runner = runner
        self._splits = splits

    def __iter__(self):
        yield self

    def __len__(self) -> int:
        return 1

    def forward(self, hidden_states, cu_seqlens, **kwargs):
        segments = self._splits.value
        if segments is not None:
            replayed = self._runner.maybe_replay(hidden_states, cu_seqlens, segments)
            if replayed is not None:
                return (replayed,)
        for layer in self._layers:
            hidden_states = layer(hidden_states, cu_seqlens, **kwargs)[0]
        return (hidden_states,)


class Qwen3OmniAudioEncoder(nn.Module):
    """Audio tower extracted from the HF thinker."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
        enable_layer_cuda_graph: bool = False,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        thinker_cfg = load_thinker_config(model_path)
        self._device = torch.device(device)
        self.audio_tower = _build_audio_tower(
            model_path,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )
        self._downsample_lengths = hf_modeling._get_feat_extract_output_lengths
        self._segment_splits = _SegmentSplits()
        _share_segment_splits(self.audio_tower, self._segment_splits)
        self._layer_graph_runner = None
        if enable_layer_cuda_graph and self._device.type == "cuda":
            self._enable_layer_cuda_graph()

    def _enable_layer_cuda_graph(self) -> None:
        tower = self.audio_tower
        chunk_tokens = int(
            self._downsample_lengths(torch.tensor([tower.n_window * 2])).item()
        )
        window = chunk_tokens * (tower.n_window_infer // (tower.n_window * 2))
        runner = AudioLayerGraphRunner(tower, device=self._device, window=window)
        runner.capture_all()
        if not runner.has_graphs:
            logger.warning("audio layer CUDA graphs unavailable; staying eager")
            return
        self._layer_graph_runner = runner
        tower.layers = _GraphedLayerStack(tower.layers, runner, self._segment_splits)

    def forward(
        self,
        *,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = pack_padded_audio_features(
                input_features.to(device=self._device, dtype=self.audio_tower.dtype),
                feature_attention_mask,
                audio_feature_lengths,
            )
        if audio_feature_lengths is None:
            raise ValueError(
                "audio_feature_lengths or feature_attention_mask is required"
            )

        audio_feature_lengths = audio_feature_lengths.to(self._device, dtype=torch.long)
        input_features = input_features.to(
            device=self._device, dtype=self.audio_tower.dtype
        )
        tower = self.audio_tower
        padded_feature, chunk_lengths = hf_modeling.chunk_and_pad_features(
            input_features, audio_feature_lengths, tower.n_window
        )
        valid_indices = hf_modeling.get_valid_indices(chunk_lengths)
        cu_seqlens = hf_modeling.get_audio_cu_seqlens(
            chunk_lengths,
            audio_feature_lengths,
            tower.n_window_infer,
            tower.n_window,
        )
        self._segment_splits.value = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        try:
            outputs = tower(
                input_features,
                feature_lens=audio_feature_lengths,
                padded_feature=padded_feature,
                chunk_lengths=chunk_lengths,
                valid_indices=valid_indices,
                cu_seqlens=cu_seqlens,
            )
        finally:
            self._segment_splits.value = None
        audio_embeds = outputs.last_hidden_state
        audio_output_lengths = self._downsample_lengths(audio_feature_lengths)
        return {
            "audio_embeds": audio_embeds,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_output_lengths": audio_output_lengths,
        }
