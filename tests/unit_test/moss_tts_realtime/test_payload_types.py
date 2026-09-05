# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.models.moss_tts_realtime.payload_types import MossTTSRealtimeState
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_BOS_TOKEN_ID,
    AUDIO_EOS_TOKEN_ID,
    AUDIO_PAD_TOKEN_ID,
    AUDIO_VOCAB_SIZE,
    MODEL_CONFIG,
    REFERENCE_AUDIO_PAD_TOKEN_ID,
    TEXT_PAD_TOKEN_ID,
)


def _prompt_row(text_token: int = 1) -> list[int]:
    return [text_token, *([AUDIO_PAD_TOKEN_ID] * int(MODEL_CONFIG.rvq))]


def test_payload_round_trip_preserves_realtime_fields() -> None:
    state = MossTTSRealtimeState(
        session_id="session-1",
        turn_id="turn-1",
        voice="voice-1",
        ref_audio={"path": "ref.wav"},
        ref_text="reference",
        language="en",
        instructions="calm",
        turn_index=1,
        user_text="user context",
        user_audio={"path": "user.wav"},
        initial_text="assistant text",
        input_done=True,
        generation_kwargs={"temperature": 0.8, "top_k": 30},
        prompt_rows=[_prompt_row()],
        audio_codes=[
            [1] * int(MODEL_CONFIG.rvq),
            [2] * int(MODEL_CONFIG.rvq),
        ],
        stream_metadata={"stream": True, "modality": "audio_codes"},
        prompt_tokens=4,
        completion_tokens=2,
        engine_time_s=0.25,
    )

    restored = MossTTSRealtimeState.from_dict(state.to_dict())

    assert restored.to_dict() == state.to_dict()
    assert restored.sample_rate == 24000
    assert restored.turn_index == 1
    assert restored.initial_text == "assistant text"
    assert restored.prompt_rows == [_prompt_row()]
    assert restored.audio_codes == [
        [1] * int(MODEL_CONFIG.rvq),
        [2] * int(MODEL_CONFIG.rvq),
    ]


def test_payload_mutable_defaults_are_independent() -> None:
    first = MossTTSRealtimeState()
    second = MossTTSRealtimeState()

    first.initial_token_ids.append(1)
    first.generation_kwargs["seed"] = 7
    first.stream_metadata["stream"] = True

    assert second.initial_token_ids == []
    assert second.generation_kwargs == {}
    assert second.stream_metadata == {}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_rate": 0}, "sample_rate must be positive"),
        ({"sample_rate": 24000.0}, "sample_rate must be an integer"),
        ({"initial_token_ids": [-1]}, "non-negative"),
        (
            {"initial_text": "hello", "initial_token_ids": [1]},
            "mutually exclusive",
        ),
        ({"turn_index": -1}, "turn_index"),
        ({"turn_index": True}, "turn_index must be an integer"),
        ({"initial_text": 1}, "initial_text must be a string"),
        ({"input_done": 1}, "input_done must be a boolean"),
        ({"generation_kwargs": []}, "generation_kwargs must be a dictionary"),
        ({"stream_metadata": []}, "stream_metadata must be a dictionary"),
        ({"prompt_rows": [1]}, "prompt_rows rows must be sequences"),
        ({"audio_codes": "invalid"}, "audio_codes must be a rank-2"),
    ],
)
def test_payload_rejects_invalid_realtime_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        MossTTSRealtimeState(**kwargs)


def test_runtime_config_matches_checkpoint_processor_values() -> None:
    assert {
        "rvq": 16,
        "audio_pad_token": AUDIO_PAD_TOKEN_ID,
        "audio_bos_token": AUDIO_BOS_TOKEN_ID,
        "audio_eos_token": AUDIO_EOS_TOKEN_ID,
        "audio_vocab_size": AUDIO_VOCAB_SIZE,
        "reference_audio_pad": REFERENCE_AUDIO_PAD_TOKEN_ID,
        "text_pad": TEXT_PAD_TOKEN_ID,
    } == {
        "rvq": int(MODEL_CONFIG.rvq),
        "audio_pad_token": int(MODEL_CONFIG.audio_pad_token),
        "audio_bos_token": int(MODEL_CONFIG.audio_bos_token),
        "audio_eos_token": int(MODEL_CONFIG.audio_eos_token),
        "audio_vocab_size": int(MODEL_CONFIG.audio_vocab_size),
        "reference_audio_pad": int(MODEL_CONFIG.reference_audio_pad),
        "text_pad": int(MODEL_CONFIG.text_pad),
    }
