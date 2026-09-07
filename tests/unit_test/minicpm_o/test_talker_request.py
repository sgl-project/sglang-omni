# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the thinker→talker tts-span slicing."""

import pytest
import torch

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_talker_request

TTS_BOS = 900
TTS_EOS = 901


def _state(prompt_ids, output_ids, hidden_dim=4):
    # hidden_seq[0] covers the last prompt position; entry i>0 covers the
    # position of output token i-1.
    hidden_seq = [
        torch.full((hidden_dim,), float(i)) for i in range(len(output_ids) + 1)
    ]
    return MiniCPMOPipelineState(
        prompt={"input_ids": torch.tensor(prompt_ids, dtype=torch.long)},
        thinker_out={
            "output_ids": list(output_ids),
            "extra_model_outputs": {"hidden_states_seq": hidden_seq},
        },
    )


def _build(state):
    return build_talker_request(
        state, tts_bos_token_id=TTS_BOS, tts_eos_token_id=TTS_EOS
    )


def test_span_between_bos_and_eos():
    # prompt ends with tts_bos; output = [t0, t1, t2, tts_eos]
    state = _state([1, 2, TTS_BOS], [10, 11, 12, TTS_EOS])
    out = _build(state)
    assert out["tts_token_ids"].tolist() == [10, 11, 12]
    # positions 3,4,5 → hidden_seq indices 1,2,3
    assert out["tts_hidden"][:, 0].tolist() == [1.0, 2.0, 3.0]


def test_span_without_eos_extends_to_end():
    state = _state([1, TTS_BOS], [10, 11])
    out = _build(state)
    assert out["tts_token_ids"].tolist() == [10, 11]
    assert out["tts_hidden"][:, 0].tolist() == [1.0, 2.0]


def test_bos_inside_output():
    # tts_bos generated mid-output; span starts after it.
    state = _state([1, 2], [5, TTS_BOS, 10, 11, TTS_EOS])
    out = _build(state)
    assert out["tts_token_ids"].tolist() == [10, 11]
    # full positions 4,5 → hidden indices 3,4
    assert out["tts_hidden"][:, 0].tolist() == [3.0, 4.0]


def test_last_bos_wins():
    state = _state([TTS_BOS, 2], [TTS_BOS, 10, TTS_EOS])
    out = _build(state)
    assert out["tts_token_ids"].tolist() == [10]


def test_no_bos_returns_empty():
    state = _state([1, 2], [10, 11])
    out = _build(state)
    assert out["tts_token_ids"].numel() == 0
    assert out["tts_hidden"].numel() == 0


def test_eos_immediately_after_bos_returns_empty():
    state = _state([1, TTS_BOS], [TTS_EOS])
    out = _build(state)
    assert out["tts_token_ids"].numel() == 0


def test_prompt_side_span_rejected():
    # bos deep inside prompt: hidden states were never captured there.
    state = _state([TTS_BOS, 7, 8], [10])
    with pytest.raises(ValueError, match="precedes first captured hidden"):
        _build(state)


def test_end_clamped_to_captured_hidden():
    # Fewer hidden entries than output tokens (e.g. final step not captured).
    state = MiniCPMOPipelineState(
        prompt={"input_ids": torch.tensor([1, TTS_BOS], dtype=torch.long)},
        thinker_out={
            "output_ids": [10, 11, 12],
            "extra_model_outputs": {
                "hidden_states_seq": [torch.zeros(4), torch.ones(4)]
            },
        },
    )
    out = _build(state)
    # hidden covers full-sequence positions 1..2 only → span clamps to [10].
    assert out["tts_token_ids"].tolist() == [10]
