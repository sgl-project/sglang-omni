# SPDX-License-Identifier: Apache-2.0
"""Regression tests for talker executor error propagation.

Covers both Ming-Omni and Qwen3-Omni talker TTS paths, ensuring common
exceptions surface to callers instead of being swallowed into silent
None-waveform fallbacks.

Reference: https://github.com/sgl-project/sglang-omni/issues/300

Author:
Xuesong Ye https://github.com/yxs
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang_omni.models.ming_omni.components.talker_executor import MingTalkerExecutor
from sglang_omni.models.qwen3_omni.components import talker_executor as qwen3_te
from sglang_omni.proto import OmniRequest, StagePayload

_INJECTED_ERRORS = [
    torch.OutOfMemoryError("CUDA OOM (injected)"),
    RuntimeError("runtime failure (injected)"),
    ValueError("invalid value (injected)"),
]


def _error_id(exc: BaseException) -> str:
    return type(exc).__name__


@pytest.mark.parametrize("exc", _INJECTED_ERRORS, ids=_error_id)
@pytest.mark.asyncio
async def test_ming_talker_propagates_errors(exc: BaseException) -> None:
    executor = MingTalkerExecutor(model_path="/fake/model/path")
    payload = StagePayload(
        request_id="t1",
        request=MagicMock(spec=OmniRequest),
        data={},
    )

    with (
        patch.object(executor, "_extract_text", return_value="hello world"),
        patch.object(executor, "_generate_speech", side_effect=exc),
        pytest.raises(type(exc)),
    ):
        await executor.add_request(payload)


@pytest.mark.parametrize("exc", _INJECTED_ERRORS, ids=_error_id)
def test_qwen3_talker_propagates_errors(exc: BaseException) -> None:
    executor = MagicMock(spec=qwen3_te.TalkerStreamingExecutor)
    executor._tts_special_cache = None
    executor._tts_bos_token_id = 0
    executor._tts_eos_token_id = 1
    executor._tts_pad_token_id = 2
    executor._resolved_model_path = "/fake/model/path"

    with (
        patch.object(
            qwen3_te,
            "_load_thinker_embedding_rows",
            side_effect=exc,
        ),
        pytest.raises(type(exc)),
    ):
        qwen3_te.TalkerStreamingExecutor._get_tts_special_embeds(executor)
