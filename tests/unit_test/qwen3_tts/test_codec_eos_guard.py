# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS codec budget / miss-EOS safeguard unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.client.types import SpeechResult
from sglang_omni.models.qwen3_tts.request_builders import (
    QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS,
    Qwen3TTSSGLangRequestData,
    apply_qwen3_tts_codec_token_budget,
    apply_sglang_qwen3_tts_result,
    count_qwen3_tts_text_tokens,
    resolve_qwen3_tts_codec_token_budget,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.qwen3_tts_codec_guard import (
    Qwen3TTSCodecLimitError,
    can_retry_qwen3_tts_codec_limit,
    speech_result_exhausted_codec_budget,
)


def make_payload(inputs: object = "target") -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=inputs, params={}),
        data={},
    )


def test_scheduler_eos_at_budget_stays_stop() -> None:
    """A real EOS remains valid when it lands exactly on the token budget."""
    payload = make_payload()
    data = Qwen3TTSSGLangRequestData(
        req=SimpleNamespace(
            output_ids=[],
            finished_reason=SimpleNamespace(
                to_json=lambda: {"type": "stop", "matched": 2150}
            ),
        ),
        output_codes=[torch.tensor([1, 2]), torch.tensor([3, 4])],
        stage_payload=payload,
        max_new_tokens=2,
    )

    result = apply_sglang_qwen3_tts_result(payload, data)

    assert result.data["finish_reason"] == "stop"
    assert result.data["completion_tokens"] == 2


def test_scheduler_stop_below_budget_stays_stop() -> None:
    payload = make_payload()
    data = Qwen3TTSSGLangRequestData(
        req=SimpleNamespace(
            output_ids=[],
            finished_reason=SimpleNamespace(
                to_json=lambda: {"type": "stop", "matched": 2150}
            ),
        ),
        output_codes=[torch.tensor([1, 2])],
        stage_payload=payload,
        max_new_tokens=8,
    )

    result = apply_sglang_qwen3_tts_result(payload, data)

    assert result.data["finish_reason"] == "stop"


@pytest.mark.parametrize(
    ("configured", "text_tokens", "explicit", "expected"),
    [
        (2048, 30, False, 360),
        (2048, 10, False, 192),
        (256, 30, False, 256),
        (2048, 30, True, 2048),
        (2048, 0, False, 2048),
        (2048, None, False, 2048),
    ],
)
def test_resolve_qwen3_tts_codec_token_budget(
    configured: int,
    text_tokens: int | None,
    explicit: bool,
    expected: int,
) -> None:
    assert (
        resolve_qwen3_tts_codec_token_budget(
            configured_cap=configured,
            text_token_count=text_tokens,
            explicit_max_new_tokens=explicit,
        )
        == expected
    )


def test_apply_codec_token_budget_clamps_default_cap() -> None:
    gen_kwargs = {
        "max_new_tokens": QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS,
        "temperature": 0.9,
    }

    updated = apply_qwen3_tts_codec_token_budget(
        gen_kwargs,
        text_token_count=30,
        explicit_max_new_tokens=False,
    )

    assert updated["max_new_tokens"] == 360
    assert updated["temperature"] == 0.9
    assert gen_kwargs["max_new_tokens"] == QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS


def test_apply_codec_token_budget_preserves_explicit_cap() -> None:
    gen_kwargs = {"max_new_tokens": 4096}

    updated = apply_qwen3_tts_codec_token_budget(
        gen_kwargs,
        text_token_count=30,
        explicit_max_new_tokens=True,
    )

    assert updated["max_new_tokens"] == 4096


def test_count_text_tokens_uses_sequence_dimension() -> None:
    wrapper = SimpleNamespace(
        _tokenize_texts=lambda texts: [torch.arange(30).unsqueeze(0)]
    )

    assert count_qwen3_tts_text_tokens(wrapper, "target") == 30


def test_speech_result_exhausted_codec_budget_detects_length() -> None:
    result = SpeechResult(
        audio_bytes=b"",
        mime_type="audio/wav",
        format="wav",
        finish_reason="length",
    )
    assert speech_result_exhausted_codec_budget(result) is True


def test_speech_result_exhausted_codec_budget_ignores_stop() -> None:
    result = SpeechResult(
        audio_bytes=b"",
        mime_type="audio/wav",
        format="wav",
        finish_reason="stop",
    )
    assert speech_result_exhausted_codec_budget(result) is False


def test_can_retry_only_without_explicit_seed_or_max_new_tokens() -> None:
    assert (
        can_retry_qwen3_tts_codec_limit(
            seed=None,
            max_new_tokens=None,
        )
        is True
    )
    assert (
        can_retry_qwen3_tts_codec_limit(
            seed=7,
            max_new_tokens=None,
        )
        is False
    )
    assert (
        can_retry_qwen3_tts_codec_limit(
            seed=None,
            max_new_tokens=192,
        )
        is False
    )


def test_codec_limit_error_is_retryable() -> None:
    error = Qwen3TTSCodecLimitError("budget exhausted", output_tokens=192, limit=192)
    assert error.retryable is True
    assert error.output_tokens == 192
    assert error.limit == 192
    assert str(error) == "budget exhausted"
