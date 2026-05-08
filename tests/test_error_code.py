# SPDX-License-Identifier: Apache-2.0
"""Tests for structured error code classification and propagation."""

from __future__ import annotations

from sglang_omni.proto.messages import (
    CLIENT_ERROR_CODES,
    CompleteMessage,
    classify_error_code,
)
from sglang_omni.serve.openai_api import _is_bad_request_error


def _make_error(message: str, error_code: str | None = None) -> RuntimeError:
    exc = RuntimeError(message)
    if error_code is not None:
        exc.error_code = error_code  # type: ignore[attr-defined]
    return exc


class TestClassifyErrorCode:
    def test_prompt_too_long_matches_context_length(self) -> None:
        code = classify_error_code(
            "The input is longer than the model's context length of 8192 tokens"
        )
        assert code == "PROMPT_TOO_LONG", f"Expected PROMPT_TOO_LONG, got {code}"

    def test_prompt_too_long_matches_token_count(self) -> None:
        code = classify_error_code(
            "Requested token count exceeds the model's maximum context length"
        )
        assert code == "PROMPT_TOO_LONG", f"Expected PROMPT_TOO_LONG, got {code}"

    def test_unrecognized_error_returns_none(self) -> None:
        result = classify_error_code("Some random CUDA OOM error")
        assert result is None, f"Expected None for unrecognized error, got {result}"

    def test_empty_message_returns_none(self) -> None:
        result = classify_error_code("")
        assert result is None, f"Expected None for empty message, got {result}"


class TestCompleteMessageErrorCode:
    def test_roundtrip_with_error_code(self) -> None:
        msg = CompleteMessage(
            request_id="req-1",
            from_stage="thinker",
            success=False,
            error="prompt too long",
            error_code="PROMPT_TOO_LONG",
        )
        serialized = msg.to_dict()
        assert serialized["error_code"] == "PROMPT_TOO_LONG", (
            f"error_code must be present in serialized dict, got {serialized}"
        )
        restored = CompleteMessage.from_dict(serialized)
        assert restored.error_code == "PROMPT_TOO_LONG", (
            f"error_code must survive round-trip, got {restored.error_code}"
        )

    def test_roundtrip_without_error_code(self) -> None:
        msg = CompleteMessage(
            request_id="req-1",
            from_stage="thinker",
            success=False,
            error="relay read failed",
        )
        serialized = msg.to_dict()
        assert "error_code" not in serialized, (
            f"error_code key must be absent when None, got {serialized}"
        )
        restored = CompleteMessage.from_dict(serialized)
        assert restored.error_code is None, (
            f"error_code must be None when absent from dict, got {restored.error_code}"
        )

    def test_success_message_omits_error_code(self) -> None:
        msg = CompleteMessage(
            request_id="req-1",
            from_stage="decode",
            success=True,
            result={"text": "ok"},
        )
        serialized = msg.to_dict()
        assert "error_code" not in serialized, (
            f"error_code key must be absent for successful messages, got {serialized}"
        )


class TestIsBadRequestError:
    def test_error_code_takes_priority(self) -> None:
        exc = _make_error("something went wrong", error_code="PROMPT_TOO_LONG")
        assert _is_bad_request_error(exc) is True, (
            "PROMPT_TOO_LONG error_code must be classified as 400, even with "
            "a message that would not match legacy markers"
        )

    def test_unknown_error_code_not_bad_request(self) -> None:
        exc = _make_error("something went wrong", error_code="SOME_OTHER_CODE")
        assert _is_bad_request_error(exc) is False, (
            "Unknown error_code must not be classified as a 400"
        )

    def test_fallback_to_string_matching(self) -> None:
        exc = RuntimeError(
            "The input is longer than the model's context length"
        )
        assert _is_bad_request_error(exc) is True, (
            "Legacy phrase matching must still work when error_code is absent"
        )

    def test_fallback_no_match_returns_false(self) -> None:
        exc = RuntimeError("Internal server error")
        assert _is_bad_request_error(exc) is False, (
            "Unrecognized message without error_code must not be classified as 400"
        )

    def test_cl_error_codes_is_derived_from_patterns(self) -> None:
        assert "PROMPT_TOO_LONG" in CLIENT_ERROR_CODES, (
            f"PROMPT_TOO_LONG must be in CLIENT_ERROR_CODES, got {CLIENT_ERROR_CODES}"
        )
        assert "NONEXISTENT" not in CLIENT_ERROR_CODES, (
            f"NONEXISTENT must not appear in CLIENT_ERROR_CODES, got {CLIENT_ERROR_CODES}"
        )
