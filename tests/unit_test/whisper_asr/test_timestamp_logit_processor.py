# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import WhisperTimeStampLogitsProcessor

from sglang_omni.models.whisper_asr.timestamp_logit_processor import (
    WhisperTimestampLogitProcessor,
    apply_timestamp_rules,
    apply_timestamp_rules_on_device,
)

EOS_TOKEN_ID = 2
TEXT_TOKEN_ID = 3
NO_TIMESTAMPS_TOKEN_ID = 5
TIMESTAMP_BEGIN_ID = NO_TIMESTAMPS_TOKEN_ID + 1
MAX_INITIAL_TIMESTAMP_INDEX = 2
PROMPT = [10, 10, 10]
VOCAB_SIZE = 12


def run_processors(
    history: list[int], logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    config = SimpleNamespace(
        eos_token_id=EOS_TOKEN_ID,
        bos_token_id=1,
        no_timestamps_token_id=NO_TIMESTAMPS_TOKEN_ID,
        max_initial_timestamp_index=MAX_INITIAL_TIMESTAMP_INDEX,
    )
    transformers_processor = WhisperTimeStampLogitsProcessor(
        config, begin_index=len(PROMPT)
    )
    transformers_scores = transformers_processor(
        torch.tensor([PROMPT + history]), logits.clone()
    )

    sglang_scores = WhisperTimestampLogitProcessor()(
        logits.clone(),
        [
            {
                "segment_timestamps": True,
                "timestamp_begin_id": TIMESTAMP_BEGIN_ID,
                "no_timestamps_token_id": NO_TIMESTAMPS_TOKEN_ID,
                "eos_token_id": EOS_TOKEN_ID,
                "max_initial_timestamp_index": MAX_INITIAL_TIMESTAMP_INDEX,
                "__req__": SimpleNamespace(output_ids=history),
            }
        ],
    )
    return transformers_scores, sglang_scores


@pytest.mark.parametrize(
    "history",
    [
        [],
        [TIMESTAMP_BEGIN_ID],
        [TIMESTAMP_BEGIN_ID, TIMESTAMP_BEGIN_ID],
        [TIMESTAMP_BEGIN_ID, TEXT_TOKEN_ID],
        [TIMESTAMP_BEGIN_ID, TEXT_TOKEN_ID, TEXT_TOKEN_ID],
        [TIMESTAMP_BEGIN_ID, TEXT_TOKEN_ID, TIMESTAMP_BEGIN_ID + 1],
        [
            TIMESTAMP_BEGIN_ID,
            TEXT_TOKEN_ID,
            TIMESTAMP_BEGIN_ID + 1,
            TIMESTAMP_BEGIN_ID + 1,
        ],
        [TIMESTAMP_BEGIN_ID + 2, TEXT_TOKEN_ID],
    ],
)
def test_timestamp_processor_matches_transformers_masks_and_greedy_token(
    history: list[int],
) -> None:
    generator = torch.Generator().manual_seed(100 + len(history))
    logits = torch.randn((1, VOCAB_SIZE), generator=generator)

    transformers_scores, sglang_scores = run_processors(history, logits)

    assert torch.equal(
        torch.isneginf(sglang_scores), torch.isneginf(transformers_scores)
    )
    assert (
        sglang_scores.argmax(dim=-1).item() == transformers_scores.argmax(dim=-1).item()
    )


def test_timestamp_processor_preserves_eos_after_closing_timestamp() -> None:
    history = [TIMESTAMP_BEGIN_ID, TEXT_TOKEN_ID, TIMESTAMP_BEGIN_ID + 1]
    logits = torch.full((1, VOCAB_SIZE), -10.0)
    logits[0, EOS_TOKEN_ID] = 10.0

    transformers_scores, sglang_scores = run_processors(history, logits)

    assert not torch.isneginf(sglang_scores[0, EOS_TOKEN_ID])
    assert torch.equal(
        torch.isneginf(sglang_scores), torch.isneginf(transformers_scores)
    )
    assert sglang_scores.argmax(dim=-1).item() == EOS_TOKEN_ID


@pytest.mark.parametrize("timestamp_dominates", [True, False])
def test_timestamp_processor_matches_transformers_probability_rule(
    timestamp_dominates: bool,
) -> None:
    history = [TIMESTAMP_BEGIN_ID, TEXT_TOKEN_ID]
    logits = torch.full((1, VOCAB_SIZE), -10.0)
    if timestamp_dominates:
        logits[0, TIMESTAMP_BEGIN_ID:] = 0.0
    else:
        logits[0, TEXT_TOKEN_ID] = 10.0

    transformers_scores, sglang_scores = run_processors(history, logits)

    assert torch.equal(
        torch.isneginf(sglang_scores), torch.isneginf(transformers_scores)
    )
    assert (
        sglang_scores.argmax(dim=-1).item() == transformers_scores.argmax(dim=-1).item()
    )
    assert (
        torch.isneginf(sglang_scores[0, :TIMESTAMP_BEGIN_ID]).all().item()
        is timestamp_dominates
    )


def test_device_timestamp_rules_match_the_per_row_rules() -> None:
    """The speculative chain masks every row from three tokens of its history; the result must not differ."""
    generator = torch.Generator().manual_seed(0)
    params = {
        "segment_timestamps": True,
        "timestamp_begin_id": TIMESTAMP_BEGIN_ID,
        "no_timestamps_token_id": NO_TIMESTAMPS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "max_initial_timestamp_index": MAX_INITIAL_TIMESTAMP_INDEX,
    }
    histories = []
    for _ in range(400):
        length = int(torch.randint(1, 6, (1,), generator=generator))
        histories.append(
            torch.randint(0, VOCAB_SIZE, (length,), generator=generator).tolist()
        )
    logits = torch.randn(len(histories), VOCAB_SIZE, generator=generator) * 3
    expected = logits.clone()
    for row, history in enumerate(histories):
        apply_timestamp_rules(expected[row], history, params)

    actual = logits.clone()
    apply_timestamp_rules_on_device(
        actual,
        last_token=torch.tensor([history[-1] for history in histories]),
        penultimate_token=torch.tensor(
            [
                history[-2] if len(history) > 1 else TIMESTAMP_BEGIN_ID
                for history in histories
            ]
        ),
        last_timestamp=torch.tensor(
            [
                next((t for t in reversed(history) if t >= TIMESTAMP_BEGIN_ID), -1)
                for history in histories
            ]
        ),
        params=params,
    )

    torch.testing.assert_close(actual, expected)
