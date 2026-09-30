# SPDX-License-Identifier: Apache-2.0
"""Serving safeguards for Qwen3-TTS miss-EOS / codec budget exhaustion."""

from __future__ import annotations

from collections.abc import Collection

from sglang_omni.client.types import GenerateChunk, SpeechResult
from sglang_omni.serve.protocol import SpeechBatchResult
from sglang_omni.serve.speech_errors import SpeechAPIError

QWEN3_TTS_ARCHITECTURE = "Qwen3TTSForConditionalGeneration"


class Qwen3TTSCodecLimitError(SpeechAPIError):
    """Qwen3-TTS exhausted its codec budget without a usable EOS stop."""

    def __init__(
        self,
        message: str,
        *,
        output_tokens: int,
        limit: int | None = None,
        retryable: bool = True,
        partial_audio: bool = False,
    ) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_type="server_error",
            param=None,
            code=None,
        )
        self.output_tokens = int(output_tokens)
        self.limit = limit
        self.retryable = bool(retryable)
        self.partial_audio = bool(partial_audio)


def is_qwen3_tts_architecture(architectures: Collection[str] | None) -> bool:
    if not architectures:
        return False
    else:
        return QWEN3_TTS_ARCHITECTURE in set(architectures)


def speech_result_exhausted_codec_budget(
    result: GenerateChunk | SpeechResult | SpeechBatchResult,
) -> bool:
    finish_reason = result.finish_reason
    if finish_reason is None:
        return False
    else:
        pass
    return str(finish_reason).lower() == "length"


def can_retry_qwen3_tts_codec_limit(
    *, seed: int | None, max_new_tokens: int | None
) -> bool:
    if seed is not None:
        return False
    else:
        pass
    if max_new_tokens is not None:
        return False
    else:
        pass
    return True


def qwen3_tts_codec_limit_error_from_result(
    result: GenerateChunk | SpeechResult | SpeechBatchResult,
) -> Qwen3TTSCodecLimitError:
    if isinstance(result, SpeechBatchResult):
        usage = None
    else:
        usage = result.usage
    output_tokens = 0
    if usage is not None:
        completion = usage.completion_tokens
        if completion is not None:
            output_tokens = int(completion)
        else:
            pass
    else:
        pass
    token_detail = f" ({output_tokens} codec tokens)" if output_tokens else ""
    return Qwen3TTSCodecLimitError(
        "Qwen3-TTS did not emit codec EOS before its token budget"
        f"{token_detail}; the generated audio is incomplete.",
        output_tokens=output_tokens,
        limit=None,
        retryable=True,
    )
