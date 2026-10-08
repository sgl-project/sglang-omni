# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o decode-stage result contract."""

from __future__ import annotations

from pathlib import Path

from sglang_omni.models.minicpm_o.merge import build_decode_result
from sglang_omni.proto import OmniRequest, StagePayload

STAGES_PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni"
    / "models"
    / "minicpm_o"
    / "stages.py"
)


class FakeTokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return "hello"


def test_build_decode_result_returns_text_and_usage() -> None:
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": False}),
        data={"engine_outputs": {"thinker": {"output_ids": [1, 2], "is_final": True}}},
    )
    result = build_decode_result(
        payload,
        tokenizer=FakeTokenizer(),
        eos_token_id=None,
        is_streaming=False,
    )
    assert result["text"] == "hello"
    assert result["events"][0]["type"] == "text_final"
    assert result["usage"]["completion_tokens"] == 2


def test_minicpm_stages_do_not_import_qwen3_omni() -> None:
    assert "qwen3_omni" not in STAGES_PATH.read_text()
