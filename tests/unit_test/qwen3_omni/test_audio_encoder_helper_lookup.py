# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni audio encoder constructor against the pinned Transformers module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe

from sglang_omni.models.qwen3_omni.components import audio_encoder


class FakeAudioTower(nn.Module):
    """Just enough tower for the constructor's segment-split wiring."""

    def __init__(self) -> None:
        super().__init__()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        self.layers = nn.ModuleList([layer])
        self.dtype = torch.float32


def fake_load_thinker_config(model_path: str) -> object:
    return object()


def fake_build_audio_tower(
    model_path: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    return FakeAudioTower()


def test_constructor_binds_pinned_downsample_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Checkpoint loading stays mocked; the helper lookup must run for real so a
    # renamed Transformers symbol fails here instead of at server startup.
    monkeypatch.setattr(audio_encoder, "load_thinker_config", fake_load_thinker_config)
    monkeypatch.setattr(audio_encoder, "build_audio_tower", fake_build_audio_tower)

    encoder = audio_encoder.Qwen3OmniAudioEncoder("unused-model-path", device="cpu")

    pinned_helper = modeling_qwen3_omni_moe._get_feat_extract_output_lengths
    assert encoder._downsample_lengths is pinned_helper
