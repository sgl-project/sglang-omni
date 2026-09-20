# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import WhisperTimeStampLogitsProcessor

from sglang_omni.models.whisper_asr.timestamp_logit_processor import (
    WhisperTimestampLogitProcessor,
)

_EOS_TOKEN_ID = 2
_TEXT_TOKEN_ID = 3
_NO_TIMESTAMPS_TOKEN_ID = 5
_TIMESTAMP_BEGIN_ID = _NO_TIMESTAMPS_TOKEN_ID + 1
_MAX_INITIAL_TIMESTAMP_INDEX = 2
_PROMPT = [10, 10, 10]
_VOCAB_SIZE = 12


def _run_processors(
    history: list[int], logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    config = SimpleNamespace(
        eos_token_id=_EOS_TOKEN_ID,
        bos_token_id=1,
        no_timestamps_token_id=_NO_TIMESTAMPS_TOKEN_ID,
        max_initial_timestamp_index=_MAX_INITIAL_TIMESTAMP_INDEX,
    )
    transformers_processor = WhisperTimeStampLogitsProcessor(
        config, begin_index=len(_PROMPT)
    )
    transformers_scores = transformers_processor(
        torch.tensor([_PROMPT + history]), logits.clone()
    )

    sglang_scores = WhisperTimestampLogitProcessor()(
        logits.clone(),
        [
            {
                "segment_timestamps": True,
                "timestamp_begin_id": _TIMESTAMP_BEGIN_ID,
                "no_timestamps_token_id": _NO_TIMESTAMPS_TOKEN_ID,
                "eos_token_id": _EOS_TOKEN_ID,
                "max_initial_timestamp_index": _MAX_INITIAL_TIMESTAMP_INDEX,
                "__req__": SimpleNamespace(output_ids=history),
            }
        ],
    )
    return transformers_scores, sglang_scores


@pytest.mark.parametrize(
    "history",
    [
        [],
        [_TIMESTAMP_BEGIN_ID],
        [_TIMESTAMP_BEGIN_ID, _TIMESTAMP_BEGIN_ID],
        [_TIMESTAMP_BEGIN_ID, _TEXT_TOKEN_ID],
        [_TIMESTAMP_BEGIN_ID, _TEXT_TOKEN_ID, _TEXT_TOKEN_ID],
        [_TIMESTAMP_BEGIN_ID, _TEXT_TOKEN_ID, _TIMESTAMP_BEGIN_ID + 1],
        [
            _TIMESTAMP_BEGIN_ID,
            _TEXT_TOKEN_ID,
            _TIMESTAMP_BEGIN_ID + 1,
            _TIMESTAMP_BEGIN_ID + 1,
        ],
        [_TIMESTAMP_BEGIN_ID + 2, _TEXT_TOKEN_ID],
    ],
)
def test_timestamp_processor_matches_transformers_masks_and_greedy_token(
    history: list[int],
) -> None:
    generator = torch.Generator().manual_seed(100 + len(history))
    logits = torch.randn((1, _VOCAB_SIZE), generator=generator)

    transformers_scores, sglang_scores = _run_processors(history, logits)

    assert torch.equal(
        torch.isneginf(sglang_scores), torch.isneginf(transformers_scores)
    )
    assert (
        sglang_scores.argmax(dim=-1).item() == transformers_scores.argmax(dim=-1).item()
    )


def test_timestamp_processor_forbids_non_monotonic_timestamp_candidates() -> None:
    history = [_TIMESTAMP_BEGIN_ID + 2, _TEXT_TOKEN_ID]
    logits = torch.zeros((1, _VOCAB_SIZE))

    transformers_scores, sglang_scores = _run_processors(history, logits)

    assert torch.isneginf(
        sglang_scores[0, _TIMESTAMP_BEGIN_ID : _TIMESTAMP_BEGIN_ID + 2]
    ).all()
    assert torch.equal(
        torch.isneginf(sglang_scores), torch.isneginf(transformers_scores)
    )


def test_timestamp_processor_preserves_eos_after_closing_timestamp() -> None:
    history = [_TIMESTAMP_BEGIN_ID, _TEXT_TOKEN_ID, _TIMESTAMP_BEGIN_ID + 1]
    logits = torch.full((1, _VOCAB_SIZE), -10.0)
    logits[0, _EOS_TOKEN_ID] = 10.0

    transformers_scores, sglang_scores = _run_processors(history, logits)

    assert not torch.isneginf(sglang_scores[0, _EOS_TOKEN_ID])
    assert torch.equal(
        torch.isneginf(sglang_scores), torch.isneginf(transformers_scores)
    )
    assert sglang_scores.argmax(dim=-1).item() == _EOS_TOKEN_ID


@pytest.mark.parametrize("timestamp_dominates", [True, False])
def test_timestamp_processor_matches_transformers_probability_rule(
    timestamp_dominates: bool,
) -> None:
    history = [_TIMESTAMP_BEGIN_ID, _TEXT_TOKEN_ID]
    logits = torch.full((1, _VOCAB_SIZE), -10.0)
    if timestamp_dominates:
        logits[0, _TIMESTAMP_BEGIN_ID:] = 0.0
    else:
        logits[0, _TEXT_TOKEN_ID] = 10.0

    transformers_scores, sglang_scores = _run_processors(history, logits)

    assert torch.equal(
        torch.isneginf(sglang_scores), torch.isneginf(transformers_scores)
    )
    assert (
        sglang_scores.argmax(dim=-1).item() == transformers_scores.argmax(dim=-1).item()
    )
    assert (
        torch.isneginf(sglang_scores[0, :_TIMESTAMP_BEGIN_ID]).all().item()
        is timestamp_dominates
    )
