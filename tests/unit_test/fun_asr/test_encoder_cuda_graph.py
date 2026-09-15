# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang_omni.models.fun_asr.encoder_cuda_graph import _bucket_batch, _bucket_t
from sglang_omni.models.fun_asr.sglang_model import FunAsrNanoForConditionalGeneration
from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)


def test_bucket_batch_rounds_up_within_max() -> None:
    assert _bucket_batch(1, 8) == 1
    assert _bucket_batch(2, 8) == 2
    assert _bucket_batch(3, 8) == 4
    assert _bucket_batch(5, 8) == 8
    assert _bucket_batch(8, 8) == 8
    # max_batch not a power of two: fall through to max itself
    assert _bucket_batch(5, 6) == 6
    # over the max -> no bucket
    assert _bucket_batch(9, 8) is None


def test_bucket_t_rounds_up_to_step() -> None:
    assert _bucket_t(1) == 64
    assert _bucket_t(64) == 64
    assert _bucket_t(65) == 128
    assert _bucket_t(500) == 512
    # beyond the 30s ceiling -> no bucket
    assert _bucket_t(513) is None


class _EagerTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.calls: list[tuple] = []

    def forward(self, xs, mask):
        self.calls.append((xs.shape, None if mask is None else mask.shape))
        return xs


class _EagerProjector(nn.Module):
    def __init__(self, llm_dim: int = 4) -> None:
        super().__init__()
        self.llm_dim = llm_dim

    def forward(self, enc_out, mask):
        b, t, _ = enc_out.shape
        t_out = int(fun_asr_low_frame_rate_length(t))
        return torch.arange(b * t_out * self.llm_dim, dtype=torch.float32).reshape(
            b, t_out, self.llm_dim
        )


def _model_with(runner) -> FunAsrNanoForConditionalGeneration:
    model = object.__new__(FunAsrNanoForConditionalGeneration)
    nn.Module.__init__(model)
    model.audio_tower = _EagerTower()
    model.multi_modal_projector = _EagerProjector()
    if runner is not None:
        model.encoder_cuda_graph_runner = runner
    return model


def _item(num_frames: int) -> SimpleNamespace:
    return SimpleNamespace(
        feature=torch.randn(1, 560, num_frames),
        feature_attention_mask=torch.ones(1, num_frames, dtype=torch.long),
    )


def test_get_audio_feature_routes_through_graph_runner() -> None:
    observed = {}

    class _Runner:
        def run(self, xs, lengths):
            observed["xs_shape"] = tuple(xs.shape)
            observed["lengths"] = list(lengths)
            b = xs.shape[0]
            t_out = int(fun_asr_low_frame_rate_length(xs.shape[1]))
            return torch.ones(b, t_out, 4)

    model = _model_with(_Runner())
    out = model.get_audio_feature([_item(17), _item(9)])

    assert observed["xs_shape"] == (2, 17, 560)
    assert observed["lengths"] == [17, 9]
    expected_rows = int(fun_asr_low_frame_rate_length(17)) + int(
        fun_asr_low_frame_rate_length(9)
    )
    assert out.shape == (expected_rows, 4)
    # eager tower must not have run
    assert model.audio_tower.calls == []


def test_get_audio_feature_falls_back_to_eager_when_runner_declines() -> None:
    class _DecliningRunner:
        def run(self, xs, lengths):
            return None

    model = _model_with(_DecliningRunner())
    out = model.get_audio_feature([_item(17), _item(9)])

    # eager path ran, with a mask (batched input)
    assert len(model.audio_tower.calls) == 1
    xs_shape, mask_shape = model.audio_tower.calls[0]
    assert tuple(xs_shape) == (2, 17, 560)
    assert tuple(mask_shape) == (2, 1, 17)
    expected_rows = int(fun_asr_low_frame_rate_length(17)) + int(
        fun_asr_low_frame_rate_length(9)
    )
    assert out.shape == (expected_rows, 4)


def test_get_audio_feature_without_runner_matches_previous_behavior() -> None:
    model = _model_with(None)
    out = model.get_audio_feature([_item(12)])

    # single unpadded item keeps the maskless fast path
    assert model.audio_tower.calls == [((1, 12, 560), None)]
    assert out.shape == (int(fun_asr_low_frame_rate_length(12)), 4)
