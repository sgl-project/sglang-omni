# SPDX-License-Identifier: Apache-2.0
"""Speech API error mapping helpers."""

from __future__ import annotations

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.serve.speech_errors import speech_generation_error


@pytest.mark.parametrize(
    "exc",
    [QueueFullError(), RuntimeError(QueueFullError.MESSAGE)],
)
def test_speech_generation_error_maps_queue_full_to_503(exc: BaseException) -> None:
    err = speech_generation_error(exc)
    assert err.status_code == 503
    assert QueueFullError.MESSAGE in err.message


def test_speech_generation_error_keeps_other_failures_as_500() -> None:
    err = speech_generation_error(RuntimeError("cuda out of memory"))
    assert err.status_code == 500
    assert "cuda out of memory" in err.message
