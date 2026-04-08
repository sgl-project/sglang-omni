from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_s2_pro.runtime.s2pro_sglang_ar import (
    S2ProSGLangRequestData,
    _resolve_sampling_state,
)
from sglang_omni.models.fishaudio_s2_pro.sglang_model import (
    _select_semantic_token_with_fallback,
)


def test_resolve_sampling_state_uses_request_defaults_without_repetition() -> None:
    data = S2ProSGLangRequestData(
        input_ids=[],
        _previous_semantic_tokens=[11, 12, 13],
    )

    sampling_state = _resolve_sampling_state(
        data,
        ras_window=16,
    )

    assert sampling_state.use_ras is False
    assert sampling_state.previous_tokens == [11, 12, 13]


def test_resolve_sampling_state_switches_to_ras_on_recent_duplicate_window() -> None:
    data = S2ProSGLangRequestData(
        input_ids=[],
        _previous_semantic_tokens=[101, 103, 101],
    )

    sampling_state = _resolve_sampling_state(
        data,
        ras_window=16,
    )

    assert sampling_state.use_ras is True
    assert sampling_state.previous_tokens == [101, 103, 101]


def test_resolve_sampling_state_uses_last_history_window() -> None:
    data = S2ProSGLangRequestData(
        input_ids=[],
        _previous_semantic_tokens=list(range(20)),
    )

    sampling_state = _resolve_sampling_state(
        data,
        ras_window=16,
    )

    assert sampling_state.use_ras is False
    assert sampling_state.previous_tokens == list(range(4, 20))


def test_select_semantic_token_with_fallback_only_changes_collapsing_rows() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.8],
            [0.1, 0.9, 0.8],
        ],
        dtype=torch.float32,
    )
    fallback_mask = torch.tensor([True, False], dtype=torch.bool)
    previous_tokens = torch.tensor(
        [
            [1, 0, 0],
            [2, 0, 0],
        ],
        dtype=torch.long,
    )
    previous_mask = torch.tensor(
        [
            [True, False, False],
            [True, False, False],
        ],
        dtype=torch.bool,
    )

    selected = _select_semantic_token_with_fallback(
        logits,
        use_ras_mask=fallback_mask,
        previous_tokens=previous_tokens,
        previous_mask=previous_mask,
    )

    assert selected.tolist() == [2, 1]


def test_select_semantic_token_with_fallback_keeps_greedy_without_history() -> None:
    logits = torch.tensor([[0.1, 0.9, 0.8]], dtype=torch.float32)
    selected = _select_semantic_token_with_fallback(
        logits,
        use_ras_mask=torch.tensor([True], dtype=torch.bool),
        previous_tokens=torch.zeros((1, 3), dtype=torch.long),
        previous_mask=torch.zeros((1, 3), dtype=torch.bool),
    )

    assert selected.tolist() == [1]
