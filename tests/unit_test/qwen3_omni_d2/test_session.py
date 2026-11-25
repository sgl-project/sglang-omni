# SPDX-License-Identifier: Apache-2.0
"""The D2 reply state machine: RESPONSE opens, PAD ends the words, INTERRUPT closes the Talker."""

from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_omni_d2.session import (
    TurnState,
    advance_turn,
    allowed_tokens,
)

WORD, PAD, RESPONSE, INTERRUPT = 3, 5, 6, 7


def test_a_reply_opens_speaks_finishes_its_words_and_is_interrupted() -> None:
    model = SimpleNamespace(
        pad_token_id=PAD,
        response_token_id=RESPONSE,
        interrupt_token_id=INTERRUPT,
        lm_head=SimpleNamespace(out_features=8),
        device=torch.device("cpu"),
    )
    state = TurnState(
        is_responding=False, is_text_done=True, is_talker_responding=False
    )
    previous_token_id = PAD
    observed = []
    for token_id in (PAD, RESPONSE, WORD, PAD, PAD, INTERRUPT):
        allowed = allowed_tokens(state, previous_token_id, model)
        assert allowed[token_id] and allowed[INTERRUPT]
        state, reads_text, emits_audio = advance_turn(state, token_id, model)
        observed.append(
            (
                bool(allowed[WORD]),
                bool(allowed[PAD]),
                bool(allowed[RESPONSE]),
                reads_text,
                emits_audio,
            )
        )
        previous_token_id = token_id
    assert observed == [
        (False, True, True, False, False),
        # The Talker reads the RESPONSE frame but starts from codec BOS, not audio.
        (False, True, True, True, False),
        # A reply cannot end on the frame right after it opened.
        (True, False, False, True, True),
        (True, True, False, True, True),
        # After PAD the words are done while the Talker keeps speaking.
        (False, True, False, True, True),
        (False, True, False, True, False),
    ]
    assert state == TurnState(
        is_responding=False, is_text_done=True, is_talker_responding=False
    )
