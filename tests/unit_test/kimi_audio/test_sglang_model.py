# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.kimi_audio.sglang_model import (
    KimiAudioForTextGeneration,
    KimiAudioTextModel,
)


def _model() -> KimiAudioTextModel:
    model = object.__new__(KimiAudioTextModel)
    model.text_blank_id = 99
    return model


def test_prefill_metadata_flattens_parallel_streams() -> None:
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        mm_inputs=[
            SimpleNamespace(
                mm_items=[
                    SimpleNamespace(
                        text_input_ids=torch.tensor([1, 2]),
                        continuous_mask=torch.tensor([False, True]),
                        feature=torch.ones((1, 4)),
                    )
                ]
            ),
            SimpleNamespace(
                mm_items=[
                    SimpleNamespace(
                        text_input_ids=torch.tensor([3]),
                        continuous_mask=torch.tensor([False]),
                        feature=torch.empty((0, 4)),
                    )
                ]
            ),
        ],
        extend_seq_lens_cpu=[2, 1],
        extend_prefix_lens_cpu=[0, 0],
    )

    audio_ids, text_ids, mask, features = _model()._prefill_streams(
        forward_batch, torch.tensor([11, 12, 13])
    )

    assert audio_ids.tolist() == [11, 12, 13]
    assert text_ids.tolist() == [1, 2, 3]
    assert mask.tolist() == [False, True, False]
    assert features.shape == (1, 4)


def test_prefill_metadata_rejects_length_mismatch() -> None:
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        mm_inputs=[
            SimpleNamespace(
                mm_items=[
                    SimpleNamespace(
                        text_input_ids=torch.tensor([1]),
                        continuous_mask=torch.tensor([False]),
                        feature=torch.empty((0, 4)),
                    )
                ]
            )
        ],
        extend_seq_lens_cpu=[1],
        extend_prefix_lens_cpu=[0],
    )

    with pytest.raises(ValueError, match="extend lengths"):
        _model()._prefill_streams(forward_batch, torch.tensor([11, 12]))


def test_prefill_streams_rebuild_retracted_prompt_and_generated_tail() -> None:
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        mm_inputs=[
            SimpleNamespace(
                mm_items=[
                    SimpleNamespace(
                        text_input_ids=torch.tensor([1, 2, 3]),
                        continuous_mask=torch.tensor([False, True, False]),
                        feature=torch.tensor([[7.0, 8.0]]),
                    )
                ]
            )
        ],
        extend_seq_lens_cpu=[3],
        extend_prefix_lens_cpu=[1],
    )

    audio_ids, text_ids, mask, features = _model()._prefill_streams(
        forward_batch, torch.tensor([12, 13, 55])
    )

    assert audio_ids.tolist() == [12, 13, 99]
    assert text_ids.tolist() == [2, 3, 55]
    assert mask.tolist() == [True, False, False]
    assert features.tolist() == [[7.0, 8.0]]


def test_load_weights_rejects_missing_required_parameters() -> None:
    model = object.__new__(KimiAudioForTextGeneration)
    model.named_parameters = lambda: [
        ("required.weight", torch.nn.Parameter(torch.zeros(1)))
    ]

    with pytest.raises(RuntimeError, match="required.weight"):
        model.load_weights([])


def test_load_weights_rejects_incomplete_fused_parameter() -> None:
    model = object.__new__(KimiAudioForTextGeneration)
    parameter = torch.nn.Parameter(torch.zeros(1))
    parameter.weight_loader = lambda *args: None
    name = "model.layers.0.self_attn.qkv_proj.weight"
    model.named_parameters = lambda: [(name, parameter)]

    with pytest.raises(RuntimeError, match="missing shards.*k.*v"):
        model.load_weights([("model.layers.0.self_attn.q_proj.weight", torch.ones(1))])
