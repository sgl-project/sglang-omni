# SPDX-License-Identifier: Apache-2.0
"""GPU-free contract tests for CSM-1B preprocessing."""

from __future__ import annotations

from sglang_omni.models.csm_tts.payload_types import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    CsmTtsState,
)
from sglang_omni.models.csm_tts.stages import create_preprocessing_executor
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeTokenizerAdapter:
    def __init__(self, checkpoint_dir: str) -> None:
        del checkpoint_dir

    def tokenize_segment_text(self, speaker_id: int, text: str) -> list[int]:
        del speaker_id
        return list(range(min(len(text), 8) or 1))

    def build_prompt(self, **kwargs):
        del kwargs
        return [128000, 1, 2, 128001]


def _payload(inputs, params=None, metadata=None) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata=metadata or {},
        ),
        data={},
    )


def test_plain_tts_keeps_csm_sampling_defaults(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.models.csm_tts.stages.resolve_checkpoint", lambda p: p
    )
    monkeypatch.setattr(
        "sglang_omni.models.csm_tts.stages.CsmTokenizerAdapter", _FakeTokenizerAdapter
    )
    scheduler = create_preprocessing_executor("/unused")
    payload = scheduler._fn(
        _payload(
            "The past is just a story we tell ourselves.",
            params={},
            metadata={"tts_params": {"voice": "alloy"}},
        )
    )
    state = CsmTtsState.from_dict(payload.data)
    assert state.target_text == "The past is just a story we tell ourselves."
    assert state.speaker_id == 0
    assert state.temperature == DEFAULT_TEMPERATURE
    assert state.top_k == DEFAULT_TOP_K
    assert state.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert state.prompt_token_ids == [128000, 1, 2, 128001]


def test_empty_input_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.models.csm_tts.stages.resolve_checkpoint", lambda p: p
    )
    monkeypatch.setattr(
        "sglang_omni.models.csm_tts.stages.CsmTokenizerAdapter", _FakeTokenizerAdapter
    )
    scheduler = create_preprocessing_executor("/unused")
    try:
        scheduler._fn(_payload({"text": "   "}))
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty input must be rejected")
